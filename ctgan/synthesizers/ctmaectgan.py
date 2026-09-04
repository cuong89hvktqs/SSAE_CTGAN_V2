"""CTMAECTGAN: MAECTGAN with independently configurable sparse perturbation.

Two perturbation paths are now controlled separately:

  ctgan_noise_type   — noise applied to CTGAN-generated samples
  real_perturb_mode  — perturbation applied to real normal samples

This separation matters because different datasets need different strategies:
  UNSW / CIC-IoT : attacks are globally different → dense real perturbation works
  CIC-IDS        : near-normal attacks (Slowloris, Bot) → sparse real perturbation

Backward compatibility
----------------------
Both parameters default to "dense", reproducing original MAECTGAN behavior.
"""

import numpy as np
import torch
import torch.nn.functional as F
from math import ceil

from ctgan.synthesizers.maectgan import MAECTGAN

_VALID_MODES = {"dense", "sparse", "directional"}


class CTMAECTGAN(MAECTGAN):
    """MAECTGAN with independently configurable sparse/dense/directional perturbation.

    Parameters
    ----------
    ctgan_noise_type : str  {"dense", "sparse", "directional"}
        Controls noise on CTGAN-generated samples.
        "dense"       — original isotropic noise on ctgan_noise_feature_ratio fraction.
        "sparse"      — symmetric noise on k features per row, rest unchanged.
        "directional" — like sparse but noise always pushes features AWAY from
                        the cached normal center (no accidental inward push).

    real_perturb_mode : str  {"dense", "sparse", "directional"}
        Controls perturbation of real normal data.
        "dense"       — original mix (cross-row shuffle) + scale (all features).
                        Best for UNSW/CIC-IoT (globally different attacks).
        "sparse"      — symmetric sparse noise on k features.
        "directional" — sparse noise, always outward from normal mean.
                        Best for CIC-IDS (Slowloris, Bot, FTP-Patator).

    ctgan_sparse_k_ratio : float  (0, 1]
        Fraction of continuous features perturbed per row in sparse/directional mode.
        k = max(1, ceil(n_cont * ctgan_sparse_k_ratio)).

    ctgan_sparse_noise_std : float  >= 0
        Std of noise applied to the k selected features in sparse/directional mode.

    score_filter_ratio : float  [0, 1]
        **Hướng A** — Fraction of pseudo-anomalies to keep each training step.
        0.0 = disabled (keep all). 0.5 = keep the 50% with highest reconstruction
        error (hardest pseudo-anomalies; discards easy ones that look too normal).
        Filtering starts only after score_filter_start_ratio of MAE training is done.

    score_filter_start_ratio : float  [0, 1]
        Fraction of MAE-active epochs to wait before pseudo-anomaly filtering starts.
        0.5 = start filtering halfway through MAE training.

    score_alpha : float  [0, 1]
        **Hướng B** — Weight for reconstruction error in the combined anomaly score:
            score = alpha * norm(re_error) + (1 - alpha) * norm(latent_norm)
        1.0 = pure reconstruction error (original behavior).
        0.5 = equal combination of both.
        0.0 = pure latent norm.

    All other parameters are identical to MAECTGAN.
    """

    def __init__(
        self,
        *args,
        ctgan_noise_type: str = "dense",
        real_perturb_mode: str = "dense",
        ctgan_sparse_k_ratio: float = 0.1,
        ctgan_sparse_noise_std: float = 0.3,
        score_filter_ratio: float = 0.0,
        score_filter_start_ratio: float = 0.5,
        score_alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if ctgan_noise_type not in _VALID_MODES:
            raise ValueError(f"ctgan_noise_type must be one of {_VALID_MODES}.")
        if real_perturb_mode not in _VALID_MODES:
            raise ValueError(f"real_perturb_mode must be one of {_VALID_MODES}.")
        if not (0.0 < ctgan_sparse_k_ratio <= 1.0):
            raise ValueError("ctgan_sparse_k_ratio must be in (0, 1].")
        if ctgan_sparse_noise_std < 0.0:
            raise ValueError("ctgan_sparse_noise_std must be >= 0.")
        if not (0.0 <= score_filter_ratio <= 1.0):
            raise ValueError("score_filter_ratio must be in [0, 1].")
        if not (0.0 <= score_filter_start_ratio <= 1.0):
            raise ValueError("score_filter_start_ratio must be in [0, 1].")
        if not (0.0 <= score_alpha <= 1.0):
            raise ValueError("score_alpha must be in [0, 1].")

        self.ctgan_noise_type = ctgan_noise_type
        self.real_perturb_mode = real_perturb_mode
        self.ctgan_sparse_k_ratio = float(ctgan_sparse_k_ratio)
        self.ctgan_sparse_noise_std = float(ctgan_sparse_noise_std)
        self.score_filter_ratio = float(score_filter_ratio)
        self.score_filter_start_ratio = float(score_filter_start_ratio)
        self.score_alpha = float(score_alpha)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_sparse_k(self, cont_idx: torch.Tensor) -> int:
        """Return the number of features to perturb given k_ratio."""
        return max(1, int(ceil(cont_idx.numel() * self.ctgan_sparse_k_ratio)))

    def _sparse_perturb(
        self,
        x: torch.Tensor,
        cont_idx: torch.Tensor,
        k: int,
        noise_std: float,
    ) -> torch.Tensor:
        """Apply Gaussian noise to k randomly-selected continuous features per row.

        Vectorized — no Python loop over batch rows.
        Features NOT in the selected k are left at their original values.

        Args:
            x         : clone of the batch tensor  (B, D), modified in-place on cont cols.
            cont_idx  : 1-D LongTensor of continuous feature indices into dim-1 of x.
            k         : number of continuous features to perturb per row.
            noise_std : standard deviation of the Gaussian noise.

        Returns:
            x with the selected k continuous features perturbed and clamped to [-1, 1].
        """
        B = x.size(0)
        n_cont = cont_idx.numel()
        if B == 0 or n_cont == 0 or k == 0 or noise_std == 0.0:
            return x

        k = min(k, n_cont)

        # Per-row random selection via top-k on uniform random scores.
        # Each row independently draws k distinct feature indices.
        scores = torch.rand(B, n_cont, device=x.device)
        _, topk_idx = scores.topk(k, dim=1)          # (B, k)

        # Binary mask: 1 where noise should be applied, 0 elsewhere.
        mask = torch.zeros(B, n_cont, device=x.device)
        mask.scatter_(1, topk_idx, 1.0)              # (B, n_cont)

        noise = noise_std * torch.randn(B, n_cont, device=x.device)
        x_cont = x[:, cont_idx] + mask * noise
        x[:, cont_idx] = torch.clamp(x_cont, -1.0, 1.0)
        return x

    def _directional_perturb(
        self,
        x: torch.Tensor,
        cont_idx: torch.Tensor,
        k: int,
        noise_std: float,
    ) -> torch.Tensor:
        """Sparse perturbation that always pushes features AWAY from normal center.

        Unlike _sparse_perturb (symmetric ±noise), this method computes the
        direction from the cached normal validation mean to the current feature
        value and applies noise only in that outward direction.

        Result: pseudo-anomaly that is "near-normal but consistently more extreme"
        — matches the signature of CIC-IDS attacks like Slowloris (very long
        duration, very low bytes) or Patator (very high connection count).
        """
        B = x.size(0)
        n_cont = cont_idx.numel()
        if B == 0 or n_cont == 0 or k == 0 or noise_std == 0.0:
            return x

        k = min(k, n_cont)

        # Per-row feature selection (same vectorized approach as _sparse_perturb)
        scores = torch.rand(B, n_cont, device=x.device)
        _, topk_idx = scores.topk(k, dim=1)
        mask = torch.zeros(B, n_cont, device=x.device)
        mask.scatter_(1, topk_idx, 1.0)

        # Direction: sign of (current_value - normal_mean)
        # Cached in MAECTGAN._validation_cont_mean after _set_validation_continuous_stats()
        mean = getattr(self, "_validation_cont_mean", None)
        if mean is not None and mean.numel() == n_cont:
            mean = mean.to(x.device)
            direction = torch.sign(x[:, cont_idx] - mean.unsqueeze(0))
        else:
            # Fallback: random direction
            direction = torch.sign(torch.randn(B, n_cont, device=x.device))

        # Ensure no zero direction (replace 0 with random ±1)
        zero_mask = direction == 0
        if zero_mask.any():
            direction[zero_mask] = (
                2.0 * torch.randint(0, 2, (int(zero_mask.sum()),), device=x.device).float() - 1.0
            )

        # Magnitude: always positive, so direction controls the sign
        magnitude = noise_std * torch.rand(B, n_cont, device=x.device)

        x_cont = x[:, cont_idx] + mask * direction * magnitude
        x[:, cont_idx] = torch.clamp(x_cont, -1.0, 1.0)
        return x

    # ─────────────────────────────────────────────────────────────────────
    # Override: Gaussian noise on CTGAN-generated samples
    # ─────────────────────────────────────────────────────────────────────

    def add_gaussian_noise_to_ctgan_samples(
        self, fake_batch, noise_std=None, feature_ratio=None
    ):
        """Dispatch to sparse or dense perturbation based on ctgan_noise_type.

        Sparse mode: apply ctgan_sparse_noise_std noise to k = max(1,
        ceil(n_cont * ctgan_sparse_k_ratio)) randomly-chosen continuous
        features per row; all other features unchanged.

        Dense mode: delegate to parent (uses ctgan_noise_std /
        ctgan_noise_feature_ratio — original MAECTGAN behavior).
        """
        if self.ctgan_noise_type in {"sparse", "directional"}:
            x = fake_batch.clone().detach()
            cont_idx = self.get_continuous_indices(device=x.device)
            if cont_idx.numel() == 0:
                return x
            k = self._get_sparse_k(cont_idx)
            perturb_fn = (
                self._directional_perturb
                if self.ctgan_noise_type == "directional"
                else self._sparse_perturb
            )
            return perturb_fn(x, cont_idx, k, self.ctgan_sparse_noise_std)

        return super().add_gaussian_noise_to_ctgan_samples(
            fake_batch, noise_std, feature_ratio
        )

    # ─────────────────────────────────────────────────────────────────────
    # Override: real-data perturbation used to build the MAE training set
    # ─────────────────────────────────────────────────────────────────────

    def make_real_perturbed_anomaly(self, real_batch):
        """Dispatch real-data perturbation based on real_perturb_mode.

        "sparse": perturb k continuous features per row, all others unchanged.
                  Use for near-normal attacks (CIC-IDS: Slowloris, Bot...).
        "dense":  delegate to parent — mix (cross-row feature shuffle) +
                  scale (multiplicative noise on ALL features).
                  Use for globally-different attacks (UNSW, CIC-IoT).
        """
        if self.real_perturb_mode in {"sparse", "directional"}:
            x = real_batch.clone()
            cont_idx = self.get_continuous_indices(device=x.device)
            if cont_idx.numel() == 0:
                return self.make_categorical_span_anomaly(real_batch)
            k = self._get_sparse_k(cont_idx)
            perturb_fn = (
                self._directional_perturb
                if self.real_perturb_mode == "directional"
                else self._sparse_perturb
            )
            return perturb_fn(x, cont_idx, k, self.ctgan_sparse_noise_std)

        return super().make_real_perturbed_anomaly(real_batch)

    # ─────────────────────────────────────────────────────────────────────
    # Hướng A: Filter pseudo-anomalies by reconstruction score
    # ─────────────────────────────────────────────────────────────────────

    def _post_assemble_pseudo_anomalies(self, fake_data, epoch, total_epochs):
        """Keep only the score_filter_ratio hardest pseudo-anomalies per training step.

        "Hard" = high reconstruction error from the current MAE, meaning the MAE
        cannot easily reconstruct them → they look like anomalies to the model.
        "Easy" = low error → the MAE reconstructs them well → they look like normal
        data despite being labeled y=1 (label noise). These are discarded.

        Filtering only activates after score_filter_start_ratio of MAE training so
        the MAE has time to warm up before its scores are used as a signal.
        """
        if self.score_filter_ratio <= 0.0 or self.score_filter_ratio >= 1.0:
            return fake_data

        # Wait until the MAE has been training for score_filter_start_ratio of its budget
        mae_start = int(total_epochs * self.mae_start_ratio)
        filter_start = mae_start + int(
            self.score_filter_start_ratio * max(1, total_epochs - mae_start)
        )
        if epoch < filter_start:
            return fake_data

        B = fake_data.size(0)
        keep_n = max(4, int(B * self.score_filter_ratio))
        if keep_n >= B:
            return fake_data

        self._mae.eval()
        with torch.no_grad():
            _, recon = self._mae.forward(fake_data)
            errors = F.mse_loss(recon, fake_data, reduction="none").mean(dim=1)
        self._mae.train()

        # Keep only the keep_n samples with the HIGHEST reconstruction error
        _, topk_idx = errors.topk(keep_n, largest=True)
        return fake_data[topk_idx]

    # ─────────────────────────────────────────────────────────────────────
    # Hướng B: Combined anomaly score (reconstruction error + latent norm)
    # ─────────────────────────────────────────────────────────────────────

    def get_score_mae(self, X, y):
        """Return combined anomaly score if score_alpha < 1.0.

        score = alpha * minmax_norm(re_error) + (1 - alpha) * minmax_norm(latent_norm)

        alpha=1.0 → pure reconstruction error (original behavior, no change).
        alpha=0.5 → equal combination, useful when latent norm carries complementary
                    signal (e.g. near-normal attacks in CIC-IDS).
        The combined score is returned as the FIRST element so that the existing
        evaluation pipeline uses it without modification.
        """
        re_errors, z_norm_scores, y_true = super().get_score_mae(X, y)

        if self.score_alpha >= 1.0:
            return re_errors, z_norm_scores, y_true

        def _minmax(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-8)

        combined = (
            self.score_alpha * _minmax(re_errors)
            + (1.0 - self.score_alpha) * _minmax(z_norm_scores)
        )
        return combined, z_norm_scores, y_true
