"""MAEAfterCTGAN — Two-phase training: CTGAN fully first, then MAE on generated pseudo-anomalies.

Pipeline
--------
Phase 1 (epochs):     Train CTGAN on normal data until convergence.
Phase 2 (once):       Generate pseudo_anomaly_ratio% of normal data size as
                      pseudo-anomaly samples from the trained CTGAN.
                      Apply critic D-score filter if filter_mode="critic".
Phase 3 (mae_epochs): Train MAE using the fixed pre-generated pseudo-anomaly pool.
                      Apply MAE reconstruction error filter if filter_mode="mae_recon".

Advantages over joint training (MAECTGAN):
  - CTGAN is fully converged before pseudo-anomalies are sampled → better quality.
  - MAE training is fully decoupled → hyperparameters can be tuned independently.
  - K% control (pseudo_anomaly_ratio) gives explicit control over class balance.
"""

import warnings
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from ctgan.synthesizers.maectgan import Discriminator, Generator, MAE


# Number of pseudo-anomaly sources for each type_train mode
_N_SOURCES = {
    "noise": 1, "std": 1, "3std": 1,
    "noise_std": 2, "noise_3std": 2, "std_3std": 2,
    "all": 3,
}


class MAEAfterCTGAN(BaseSynthesizer):
    """Two-phase training: CTGAN first, then MAE on generated pseudo-anomalies.

    Parameters
    ----------
    epochs : int
        CTGAN training epochs (Phase 1).
    mae_epochs : int
        MAE training epochs (Phase 3).  Default 100.
    pseudo_anomaly_ratio : float
        K% — number of pseudo-anomaly samples to generate, as a percentage of
        the normal training data size.
        100.0 = same number of pseudo-anomalies as normal samples (1 : 1).
        200.0 = twice as many pseudo-anomalies as normal samples.
    filter_mode : str {"none", "critic", "mae_recon"}
        "critic"    → at generation time, keep only the filter_keep_ratio fraction
                      of generated samples with the LOWEST D score (look least like
                      normal data).  Applied once during Phase 2.
        "mae_recon" → during Phase 3 MAE training, keep the filter_keep_ratio
                      fraction of pseudo-anomalies with the HIGHEST reconstruction
                      error (hardest pseudo-anomalies).  Applied every epoch.
        "none"      → no filtering.
    filter_keep_ratio : float in (0, 1]
        Fraction of pseudo-anomaly samples to keep.  Default 0.7.
    filter_start_ratio : float in [0, 1]
        (mae_recon only) Fraction of mae_epochs to wait before activating the
        MAE reconstruction filter.  0.5 → filter starts at epoch mae_epochs//2.
    ctgan_noise_std : float
        Std of Gaussian noise added to continuous features of generated samples
        (applied at generation time).  0.0 = no noise.
    """

    def __init__(
        self,
        # ── CTGAN ───────────────────────────────────────────────────────────
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        # ── MAE ─────────────────────────────────────────────────────────────
        lamda=0.1,
        margin=1.0,
        mae_loss_mode="origin",
        type_train="noise_std",
        mae_epochs=100,
        # ── Pseudo-anomaly generation ────────────────────────────────────────
        pseudo_anomaly_ratio=100.0,
        ctgan_noise_std=0.0,
        ctgan_noise_feature_ratio=1.0,
        ctgan_pseudo_anomaly_mode="origin",
        ctgan_feature_outlier_r=1.0,
        # ── Filter ──────────────────────────────────────────────────────────
        filter_mode="none",
        filter_keep_ratio=0.7,
        filter_start_ratio=0.5,
    ):
        assert batch_size % 2 == 0, "batch_size must be divisible by 2."
        if batch_size % pac != 0:
            valid_pac = max(p for p in range(1, pac + 1) if batch_size % p == 0)
            warnings.warn(
                f"MAEAfterCTGAN: batch_size={batch_size} not divisible by pac={pac}. "
                f"Auto-adjusting pac to {valid_pac}.",
                UserWarning,
            )
            pac = valid_pac

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self.lamda = lamda
        self.margin = margin
        self.mae_loss_mode = str(mae_loss_mode)
        self.type_train = str(type_train)
        self.mae_epochs = int(mae_epochs)
        self.pseudo_anomaly_ratio = float(pseudo_anomaly_ratio)
        self.ctgan_noise_std = float(ctgan_noise_std)
        self.ctgan_noise_feature_ratio = float(ctgan_noise_feature_ratio)
        self.ctgan_pseudo_anomaly_mode = (
            "feature_outlier"
            if ctgan_pseudo_anomaly_mode == "feature_outliner"
            else str(ctgan_pseudo_anomaly_mode)
        )
        self.ctgan_feature_outlier_r = float(ctgan_feature_outlier_r)
        self.filter_mode = str(filter_mode)
        self.filter_keep_ratio = float(filter_keep_ratio)
        self.filter_start_ratio = float(filter_start_ratio)

        # Validation
        if self.mae_loss_mode not in {"origin", "renormal"}:
            raise ValueError("mae_loss_mode must be 'origin' or 'renormal'.")
        if self.type_train not in _N_SOURCES:
            raise ValueError(f"type_train='{self.type_train}' is not valid.")
        if self.mae_epochs < 1:
            raise ValueError("mae_epochs must be >= 1.")
        if self.pseudo_anomaly_ratio <= 0:
            raise ValueError("pseudo_anomaly_ratio must be > 0.")
        if self.ctgan_pseudo_anomaly_mode not in {"origin", "feature_outlier"}:
            raise ValueError(
                "ctgan_pseudo_anomaly_mode must be 'origin' or 'feature_outlier'."
            )
        if self.filter_mode not in {"none", "critic", "mae_recon"}:
            raise ValueError("filter_mode must be 'none', 'critic', or 'mae_recon'.")
        if not 0.0 < self.filter_keep_ratio <= 1.0:
            raise ValueError("filter_keep_ratio must be in (0, 1].")
        if not 0.0 <= self.filter_start_ratio <= 1.0:
            raise ValueError("filter_start_ratio must be in [0, 1].")

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"
        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
        self._mae = None
        self.loss_values = None          # MAE training losses (Phase 3)
        self.ctgan_loss_values = None    # CTGAN training losses (Phase 1)
        self._on_epoch_end_callbacks = None
        self._on_epoch_best_callbacks = None
        self._validation_cont_mean = None
        self._validation_cont_std = None

    # ──────────────────────────────────────────────────────────────────────────
    # Static helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        for _ in range(10):
            transformed = functional.gumbel_softmax(
                logits, tau=tau, hard=hard, eps=eps, dim=dim
            )
            if not torch.isnan(transformed).any():
                return transformed
        raise ValueError("gumbel_softmax returning NaN.")

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                ed = st + span_info.dim
                if span_info.activation_fn == "tanh":
                    data_t.append(torch.tanh(data[:, st:ed]))
                elif span_info.activation_fn == "softmax":
                    data_t.append(self._gumbel_softmax(data[:, st:ed], tau=0.2))
                else:
                    raise ValueError(
                        f"Unexpected activation function {span_info.activation_fn}."
                    )
                st = ed
        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction="none",
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c
        loss = torch.stack(loss, dim=1)
        if not isinstance(m, torch.Tensor):
            m = torch.from_numpy(m)
        m = m.to(loss.device)
        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = [
                col for col in discrete_columns
                if col < 0 or col >= train_data.shape[1]
            ]
        else:
            raise TypeError("train_data should be pd.DataFrame or np.ndarray.")
        if invalid_columns:
            raise ValueError(f"Invalid columns found: {invalid_columns}")

    # ──────────────────────────────────────────────────────────────────────────
    # Feature index helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_continuous_indices(self, device=None):
        if self._transformer is None:
            raise RuntimeError("DataTransformer must be fitted first.")
        continuous_idx = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                ed = st + span_info.dim
                if span_info.activation_fn == "tanh":
                    continuous_idx.extend(range(st, ed))
                st = ed
        return torch.tensor(
            continuous_idx, device=device or self._device, dtype=torch.long
        )

    def get_discrete_span_slices(self):
        if self._transformer is None:
            raise RuntimeError("DataTransformer must be fitted first.")
        discrete_slices = []
        st = 0
        for column_info in self._transformer.output_info_list:
            is_discrete = (
                len(column_info) == 1 and column_info[0].activation_fn == "softmax"
            )
            for span_info in column_info:
                ed = st + span_info.dim
                if is_discrete:
                    discrete_slices.append((st, ed))
                st = ed
        return discrete_slices

    def copy_discrete_spans(self, target_batch, source_batch):
        x = target_batch.clone()
        for st, ed in self.get_discrete_span_slices():
            x[:, st:ed] = source_batch[:, st:ed]
        return x

    # ──────────────────────────────────────────────────────────────────────────
    # Pseudo-anomaly construction
    # ──────────────────────────────────────────────────────────────────────────

    def make_categorical_span_anomaly(self, real_batch, span_prob=0.3):
        x = real_batch.clone()
        discrete_slices = self.get_discrete_span_slices()
        B = x.size(0)
        if B <= 1 or not discrete_slices or span_prob <= 0.0:
            return x
        span_prob = min(1.0, float(span_prob))
        changed = False
        for st, ed in discrete_slices:
            if torch.rand((), device=x.device) >= span_prob:
                continue
            perm = torch.randperm(B, device=x.device)
            if torch.equal(perm, torch.arange(B, device=x.device)):
                perm = torch.roll(perm, shifts=1)
            x[:, st:ed] = x[perm, st:ed]
            changed = True
        if not changed:
            span_id = torch.randint(len(discrete_slices), (1,), device=x.device).item()
            st, ed = discrete_slices[span_id]
            perm = torch.randperm(B, device=x.device)
            if torch.equal(perm, torch.arange(B, device=x.device)):
                perm = torch.roll(perm, shifts=1)
            x[:, st:ed] = x[perm, st:ed]
        return x

    def make_unsw_scale_anomaly(self, real_batch, noise=0.1):
        x = real_batch.clone()
        cont_idx = self.get_continuous_indices(device=x.device)
        if cont_idx.numel() > 0:
            x[:, cont_idx] = x[:, cont_idx] * (
                1 + float(noise) * torch.randn_like(x[:, cont_idx])
            )
            x[:, cont_idx] = torch.clamp(x[:, cont_idx], -1.0, 1.0)
        return x

    def make_unsw_mix_anomaly(self, real_batch):
        B, _ = real_batch.shape
        x = real_batch.clone()
        cont_idx = self.get_continuous_indices(device=x.device)
        mask = torch.rand(cont_idx.numel(), device=x.device) < 0.3
        for j in cont_idx[mask].tolist():
            if B > 1:
                x[:, j] = x[torch.randperm(B), j]
        return x

    def make_real_perturbed_anomaly(self, real_batch):
        real_mix = self.make_unsw_mix_anomaly(real_batch)
        real_scale = self.make_unsw_scale_anomaly(real_batch)
        real_categorical = self.make_categorical_span_anomaly(real_batch)
        real_perturbed = 0.5 * real_mix + 0.5 * real_scale
        return self.copy_discrete_spans(real_perturbed, real_categorical)

    def add_gaussian_noise_to_ctgan_samples(self, fake_batch, noise_std=None, feature_ratio=None):
        x = fake_batch.clone().detach()
        cont_idx = self.get_continuous_indices(device=x.device)
        noise_std = self.ctgan_noise_std if noise_std is None else float(noise_std)
        feature_ratio = (
            self.ctgan_noise_feature_ratio if feature_ratio is None else float(feature_ratio)
        )
        if cont_idx.numel() == 0 or feature_ratio == 0.0 or noise_std == 0.0:
            return x
        num_features = max(1, int(np.ceil(cont_idx.numel() * feature_ratio)))
        selected = cont_idx[
            torch.randperm(cont_idx.numel(), device=x.device)[:num_features]
        ]
        gaussian_noise = noise_std * torch.randn_like(x[:, selected])
        x[:, selected] = torch.clamp(x[:, selected] + gaussian_noise, -1.0, 1.0)
        return x

    def make_ctgan_feature_outlier_anomaly(self, real_batch, generated_batch, r=None):
        x = real_batch.clone().detach()
        cont_idx = self.get_continuous_indices(device=x.device)
        if cont_idx.numel() == 0:
            return x
        generated_cont = generated_batch.detach().to(x.device)[:, cont_idx]
        feature_mean = generated_cont.mean(dim=0)
        feature_std = generated_cont.std(dim=0, unbiased=False)
        coefficient = self.ctgan_feature_outlier_r if r is None else float(r)
        outlier_values = feature_mean + coefficient * feature_std
        x[:, cont_idx] = torch.clamp(outlier_values.unsqueeze(0), -1.0, 1.0)
        return x

    # ──────────────────────────────────────────────────────────────────────────
    # Validation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _set_validation_continuous_stats(self, validation_data):
        cont_idx = self.get_continuous_indices(device=validation_data.device)
        if cont_idx.numel() == 0 or validation_data.size(0) == 0:
            self._validation_cont_mean = None
            self._validation_cont_std = None
            return
        validation_cont = validation_data[:, cont_idx]
        self._validation_cont_mean = validation_cont.mean(dim=0).detach()
        self._validation_cont_std = validation_cont.std(dim=0, unbiased=False).detach()

    def _get_validation_continuous_stats(self, batch, cont_idx):
        mean = getattr(self, "_validation_cont_mean", None)
        std = getattr(self, "_validation_cont_std", None)
        if mean is None or std is None or mean.numel() != cont_idx.numel():
            batch_cont = batch[:, cont_idx]
            mean = batch_cont.mean(dim=0)
            std = batch_cont.std(dim=0, unbiased=False)
        else:
            mean = mean.to(batch.device)
            std = std.to(batch.device)
        return mean, std

    def make_validation_std_anomaly(self, real_batch, scale):
        x = real_batch.clone()
        cont_idx = self.get_continuous_indices(device=x.device)
        if cont_idx.numel() == 0:
            return self.make_categorical_span_anomaly(real_batch)
        mean, feature_std = self._get_validation_continuous_stats(x, cont_idx)
        feature_std = torch.clamp(feature_std, min=1e-6)
        cont_values = x[:, cont_idx]
        direction = torch.where(
            cont_values >= mean.unsqueeze(0),
            torch.ones_like(cont_values),
            -torch.ones_like(cont_values),
        )
        x[:, cont_idx] = torch.clamp(
            cont_values + direction * float(scale) * feature_std.unsqueeze(0),
            -1.0, 1.0,
        )
        return x

    def _validation_pseudo_modes(self):
        mode_map = {
            "noise": ("noise",), "std": ("std",), "3std": ("3std",),
            "std_3std": ("std", "3std"), "noise_std": ("noise", "std"),
            "noise_3std": ("noise", "3std"), "all": ("noise", "std", "3std"),
        }
        if self.type_train not in mode_map:
            raise ValueError(f"type_train is not valid: {self.type_train}")
        return mode_map[self.type_train]

    def make_validation_pseudo_anomaly(self, normal_batch):
        modes = self._validation_pseudo_modes()
        pseudo_batches = []
        for mode_idx, mode in enumerate(modes):
            mode_batch = normal_batch[mode_idx :: len(modes)]
            if mode_batch.size(0) == 0:
                continue
            if mode == "noise":
                pseudo_batches.append(self.make_real_perturbed_anomaly(mode_batch))
            elif mode == "std":
                pseudo_batches.append(self.make_validation_std_anomaly(mode_batch, 1.0))
            elif mode == "3std":
                pseudo_batches.append(
                    self.make_validation_std_anomaly(mode_batch, sqrt(3))
                )
        if not pseudo_batches:
            return normal_batch.new_empty((0, normal_batch.size(1)))
        return torch.cat(pseudo_batches, dim=0)

    def evaluate_pseudo_validation_auc(self, val_data_loader):
        self._mae.eval()
        normal_scores, pseudo_scores = [], []
        with torch.no_grad():
            for xb, _ in val_data_loader:
                xb = xb.to(self._device)
                xb_pseudo = self.make_validation_pseudo_anomaly(xb)
                z_normal, _ = self._mae.forward(xb)
                z_pseudo, _ = self._mae.forward(xb_pseudo)
                normal_scores.append(torch.norm(z_normal, p=2, dim=1).cpu().numpy())
                pseudo_scores.append(torch.norm(z_pseudo, p=2, dim=1).cpu().numpy())
        if not normal_scores or not pseudo_scores:
            return 0.0, -float("inf")
        normal_scores = np.concatenate(normal_scores)
        pseudo_scores = np.concatenate(pseudo_scores)
        y = np.concatenate([
            np.zeros_like(normal_scores, dtype=np.int64),
            np.ones_like(pseudo_scores, dtype=np.int64),
        ])
        scores = np.concatenate([normal_scores, pseudo_scores])
        pseudo_auc = roc_auc_score(y, scores)
        pseudo_separation = float(np.mean(pseudo_scores) - np.mean(normal_scores))
        return float(pseudo_auc), pseudo_separation

    # ──────────────────────────────────────────────────────────────────────────
    # D-score helpers and filters
    # ──────────────────────────────────────────────────────────────────────────

    def _get_expanded_d_scores(self, y_fake, batch_size):
        """Expand group-level D scores (pac > 1) to per-sample scores."""
        scores = y_fake.detach().squeeze(-1)  # (B//pac,)
        if self.pac == 1:
            return scores
        expanded = scores.repeat_interleave(self.pac)
        if expanded.size(0) < batch_size:
            pad = scores.mean().expand(batch_size - expanded.size(0))
            expanded = torch.cat([expanded, pad], dim=0)
        return expanded[:batch_size]

    def _critic_filter_generation(self, fakeact, c1):
        """Filter generated samples at generation time using D scores.

        Keep the filter_keep_ratio fraction with LOWEST D score.
        No EMA — applied once to a completed CTGAN, so just use direct percentile.
        """
        if self.filter_mode != "critic" or self.filter_keep_ratio >= 1.0:
            return fakeact

        N = fakeact.size(0)
        if N < 4:
            return fakeact

        with torch.no_grad():
            if c1 is not None:
                y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = self._discriminator(fakeact)

        d_scores = self._get_expanded_d_scores(y_fake, N)
        keep_n = max(4, int(N * self.filter_keep_ratio))
        _, idx = d_scores.topk(keep_n, largest=False)  # lowest D = most anomalous
        return fakeact[idx]

    def _mae_recon_filter(self, fake_data, mae_epoch, total_mae_epochs):
        """Filter assembled pseudo-anomaly batch by MAE reconstruction error.

        Keep the filter_keep_ratio fraction with HIGHEST reconstruction error.
        Activates after filter_start_ratio * total_mae_epochs epochs.
        """
        if self.filter_mode != "mae_recon" or self.filter_keep_ratio >= 1.0:
            return fake_data

        filter_start = int(self.filter_start_ratio * total_mae_epochs)
        if mae_epoch < filter_start:
            return fake_data

        B = fake_data.size(0)
        keep_n = max(4, int(B * self.filter_keep_ratio))
        if keep_n >= B:
            return fake_data

        self._mae.eval()
        with torch.no_grad():
            _, recon = self._mae.forward(fake_data)
            errors = F.mse_loss(recon, fake_data, reduction="none").mean(dim=1)
        self._mae.train()

        _, topk_idx = errors.topk(keep_n, largest=True)
        return fake_data[topk_idx]

    # ──────────────────────────────────────────────────────────────────────────
    # Pseudo-anomaly pool generation (Phase 2)
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_pseudo_anomaly_pool(self, n_needed, use_3std=False):
        """Generate n_needed pseudo-anomaly samples from trained CTGAN.

        Generates in batches, applying noise and critic filter as configured.
        Keeps generating until n_needed samples are collected.

        Parameters
        ----------
        n_needed : int
        use_3std : bool
            If True use z ~ N(0, sqrt(3)·I); otherwise z ~ N(0, I).

        Returns
        -------
        Tensor (n_needed, data_dim) on CPU
        """
        self._generator.eval()
        self._discriminator.eval()

        std_scale = sqrt(3) if use_3std else 1.0
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + std_scale

        collected = []
        total = 0
        # Generate extra batches to account for filtering losses
        max_batches = max(
            int(np.ceil(n_needed / self._batch_size)) * 10, 50
        )

        with torch.no_grad():
            for _ in range(max_batches):
                if total >= n_needed:
                    break

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                c1 = None
                if condvec is not None:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                # Gaussian noise augmentation
                if self.ctgan_noise_std > 0.0:
                    fakeact = self.add_gaussian_noise_to_ctgan_samples(fakeact)

                # Feature outlier mode
                if self.ctgan_pseudo_anomaly_mode == "feature_outlier":
                    real_idx = np.random.randint(
                        0, self._data_sampler._data.shape[0], self._batch_size
                    )
                    real_ref = torch.tensor(
                        self._data_sampler._data[real_idx].astype("float32"),
                        device=self._device,
                    )
                    fakeact = self.make_ctgan_feature_outlier_anomaly(
                        real_ref, fakeact, r=self.ctgan_feature_outlier_r
                    )

                # Critic filter (generation-time, no EMA)
                fakeact = self._critic_filter_generation(fakeact, c1)

                collected.append(fakeact.cpu())
                total += fakeact.size(0)

        self._generator.train()
        self._discriminator.train()

        if not collected:
            raise RuntimeError(
                f"MAEAfterCTGAN: failed to generate {n_needed} pseudo-anomaly samples."
            )

        result = torch.cat(collected, dim=0)[:n_needed]
        label = "3std" if use_3std else "std"
        print(
            f"  [Phase 2] Generated {result.size(0)} pseudo-anomaly samples "
            f"(type={label}, filter_mode={self.filter_mode})"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    @random_state
    def fit(self, full_data, train_data, val_data, discrete_columns=(), epochs=None):
        print(
            "Starting MAEAfterCTGAN training...",
            " data shape:", train_data.shape,
            " CTGAN epochs:", self._epochs,
            " MAE epochs:", self.mae_epochs,
            " pseudo_anomaly_ratio:", self.pseudo_anomaly_ratio,
            " filter_mode:", self.filter_mode,
            " filter_keep_ratio:", self.filter_keep_ratio,
            " pac:", self.pac,
        )

        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                "`epochs` argument in `fit` is deprecated. "
                "Pass `epochs` to the constructor instead.",
                DeprecationWarning,
            )

        # ── Data preparation ──────────────────────────────────────────────────
        self._transformer = DataTransformer()
        self._transformer.fit(full_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        print("  Transformed data shape:", train_data.shape)

        X_val_processed = self._transformer.transform(val_data)
        y_val = np.zeros(X_val_processed.shape[0])
        val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
        self._set_validation_continuous_stats(val_tensor)
        val_dataset = TensorDataset(val_tensor, torch.tensor(y_val, dtype=torch.float32))
        val_data_loader = DataLoader(
            val_dataset, batch_size=self._batch_size, shuffle=False
        )

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )
        data_dim = self._transformer.output_dimensions

        # ── Build networks ────────────────────────────────────────────────────
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim, data_dim,
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim, pac=self.pac,
        ).to(self._device)

        self._mae = MAE(
            input_dim=data_dim,
            lamda=self.lamda,
            margin=self.margin,
            mae_loss_mode=self.mae_loss_mode,
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizerD = optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr, betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        # ══════════════════════════════════════════════════════════════════════
        # Phase 1: CTGAN Training
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n[Phase 1] Training CTGAN for {epochs} epochs...")
        ctgan_loss_rows = []
        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))

        for i in epoch_iterator:
            loss_g = loss_d = torch.tensor(0.0)

            for id_ in range(steps_per_epoch):

                # ── Discriminator step ────────────────────────────────────
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if condvec is None:
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                        fake_cat = fakeact
                        real_cat = torch.from_numpy(real.astype("float32")).to(self._device)
                    else:
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]
                        real = torch.from_numpy(real.astype("float32")).to(self._device)
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)

                    y_fake_d = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)
                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake_d))
                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # ── Generator step ────────────────────────────────────────
                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                    cross_entropy = self._cond_loss(fake, c1, m1)
                else:
                    y_fake = self._discriminator(fakeact)
                    cross_entropy = 0

                loss_g = -torch.mean(y_fake) + cross_entropy
                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            # ── Log CTGAN epoch ───────────────────────────────────────────
            g_val = loss_g.detach().cpu().item()
            d_val = loss_d.detach().cpu().item()
            ctgan_loss_rows.append({"Epoch": i, "Generator Loss": g_val, "Discriminator Loss": d_val})
            if self._verbose:
                epoch_iterator.set_description(f"G={g_val:.3f} D={d_val:.3f}")

        self.ctgan_loss_values = pd.DataFrame(ctgan_loss_rows)
        print("[Phase 1] CTGAN training complete.")

        # ══════════════════════════════════════════════════════════════════════
        # Phase 2: Generate pseudo-anomaly pool
        # ══════════════════════════════════════════════════════════════════════
        n_normal = len(train_data)
        n_pseudo = max(4, int(n_normal * self.pseudo_anomaly_ratio / 100.0))
        print(f"\n[Phase 2] Generating pseudo-anomaly pool: n_normal={n_normal}, "
              f"pseudo_anomaly_ratio={self.pseudo_anomaly_ratio}%, n_pseudo={n_pseudo}")

        needs_std = self.type_train in {"std", "noise_std", "std_3std", "all"}
        needs_3std = self.type_train in {"3std", "noise_3std", "std_3std", "all"}

        pseudo_pool_std = (
            self._generate_pseudo_anomaly_pool(n_pseudo, use_3std=False)
            if needs_std else None
        )
        pseudo_pool_3std = (
            self._generate_pseudo_anomaly_pool(n_pseudo, use_3std=True)
            if needs_3std else None
        )
        print("[Phase 2] Generation complete.")

        # ══════════════════════════════════════════════════════════════════════
        # Phase 3: MAE Training
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n[Phase 3] Training MAE for {self.mae_epochs} epochs...")

        # Keep normal data on CPU; move to device per-epoch
        normal_tensor_cpu = torch.tensor(train_data, dtype=torch.float32)
        n_sources = _N_SOURCES[self.type_train]
        # Samples per source per epoch: distribute n_pseudo equally across sources
        n_per_source = max(4, n_pseudo // n_sources)

        filter_start_epoch = int(self.filter_start_ratio * self.mae_epochs)

        self.loss_values = pd.DataFrame(columns=[
            "Epoch", "MAE Loss", "Reconstruction Loss", "Latent Loss",
            "Validation Loss", "Validation Reconstruction Loss",
            "Validation Latent Loss", "Pseudo Val AUC", "Pseudo Separation",
            "Filter Kept Ratio",
        ])

        best_pseudo_val_auc = -float("inf")
        best_pseudo_val_separation = -float("inf")

        mae_iterator = tqdm(range(self.mae_epochs), disable=(not self._verbose))

        for i in mae_iterator:
            # ── Build pseudo-anomaly for this epoch ───────────────────────
            parts = []

            if self.type_train in {"std", "noise_std", "std_3std", "all"} and pseudo_pool_std is not None:
                idx = torch.randperm(pseudo_pool_std.size(0))[:n_per_source]
                parts.append(pseudo_pool_std[idx].to(self._device))

            if self.type_train in {"3std", "noise_3std", "std_3std", "all"} and pseudo_pool_3std is not None:
                idx = torch.randperm(pseudo_pool_3std.size(0))[:n_per_source]
                parts.append(pseudo_pool_3std[idx].to(self._device))

            if self.type_train in {"noise", "noise_std", "noise_3std", "all"}:
                noise_idx = torch.randperm(normal_tensor_cpu.size(0))[:n_per_source]
                noise_batch = normal_tensor_cpu[noise_idx].to(self._device)
                parts.append(self.make_real_perturbed_anomaly(noise_batch))

            if not parts:
                continue

            fake_data = torch.cat(parts, dim=0)

            # ── MAE reconstruction filter ─────────────────────────────────
            avg_kept_ratio = np.nan
            if self.filter_mode == "mae_recon":
                n_before = fake_data.size(0)
                fake_data = self._mae_recon_filter(fake_data, i, self.mae_epochs)
                n_after = fake_data.size(0)
                if n_before > 0:
                    avg_kept_ratio = n_after / n_before

            # ── MAE training step ─────────────────────────────────────────
            # Sample equal-sized normal batch
            normal_idx = torch.randperm(normal_tensor_cpu.size(0))[:fake_data.size(0)]
            real_batch = normal_tensor_cpu[normal_idx].to(self._device)

            y_fake_mae = torch.ones(fake_data.size(0), 1, device=self._device)
            y_real_mae = torch.zeros(real_batch.size(0), 1, device=self._device)
            mae_dataset = torch.cat([fake_data, real_batch], dim=0)
            y_mae = torch.cat([y_fake_mae, y_real_mae], dim=0)

            mae_dataloader = DataLoader(
                TensorDataset(mae_dataset, y_mae),
                batch_size=self._batch_size,
                shuffle=True,
            )
            mae_loss, re_loss, latent_loss, _ = self._mae.train_mae(
                epoch=i, train_data_loader=mae_dataloader
            )

            # ── Validation ────────────────────────────────────────────────
            avg_val_total = avg_val_re = avg_val_latent = np.nan
            pseudo_val_auc = pseudo_val_separation = np.nan

            avg_val_total, threshold_re, threshold_z_0, threshold_95, threshold_95_total = (
                self._mae.validate(val_data_loader, epoch=i)
            )
            avg_val_re = getattr(self._mae, "last_validation_reconstruction_loss", np.nan)
            avg_val_latent = getattr(self._mae, "last_validation_latent_loss", np.nan)
            pseudo_val_auc, pseudo_val_separation = self.evaluate_pseudo_validation_auc(
                val_data_loader
            )

            is_best = pseudo_val_auc > best_pseudo_val_auc or (
                np.isclose(pseudo_val_auc, best_pseudo_val_auc)
                and pseudo_val_separation > best_pseudo_val_separation
            )
            if is_best:
                best_pseudo_val_auc = pseudo_val_auc
                best_pseudo_val_separation = pseudo_val_separation
                if self._on_epoch_best_callbacks is not None:
                    self._on_epoch_best_callbacks(
                        model=self, epoch=i,
                        threshold_re=threshold_re,
                        threshold_z_0=threshold_z_0,
                        threshold_95=threshold_95,
                        threshold_95_total_loss=threshold_95_total,
                    )
                print(
                    f"[BEST] mae_epoch={i} pseudo_val_auc={pseudo_val_auc:.6f} "
                    f"pseudo_separation={pseudo_val_separation:.6f}"
                )

            # ── Log ───────────────────────────────────────────────────────
            kept_str = f"{avg_kept_ratio:.3f}" if not np.isnan(avg_kept_ratio) else "N/A"
            print(
                f"MAE Epoch {i}: Loss={mae_loss} RE={re_loss} Latent={latent_loss} "
                f"PseudoAUC={pseudo_val_auc} FilterKept={kept_str}"
            )

            epoch_df = pd.DataFrame({
                "Epoch": [i],
                "MAE Loss": [mae_loss.item() if torch.is_tensor(mae_loss) else mae_loss],
                "Reconstruction Loss": [re_loss.item() if torch.is_tensor(re_loss) else re_loss],
                "Latent Loss": [latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss],
                "Validation Loss": [avg_val_total],
                "Validation Reconstruction Loss": [avg_val_re],
                "Validation Latent Loss": [avg_val_latent],
                "Pseudo Val AUC": [pseudo_val_auc],
                "Pseudo Separation": [pseudo_val_separation],
                "Filter Kept Ratio": [avg_kept_ratio],
            })

            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_df

            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)

        print("[Phase 3] MAE training complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = (
                self._data_sampler.generate_cond_from_condition_column_info(
                    condition_info, self._batch_size
                )
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for _ in range(steps):
            m = torch.zeros(self._batch_size, self._embedding_dim)
            s = m + 1
            fakez = torch.normal(mean=m, std=s).to(self._device)
            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)
            if condvec is not None:
                c1 = torch.from_numpy(condvec).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)[:n]
        return self._transformer.inverse_transform(data)

    def get_score_mae(self, X, y):
        X_processed = self._transformer.transform(X)
        score_res, latent_loss, y_true = self._mae.get_score_mae(
            X_processed, y, batch_size=self._batch_size, device=self._device
        )
        return score_res, latent_loss, y_true

    def get_latent_z_mae(self, X, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        X_processed = self._transformer.transform(X)
        X_tensor = (
            torch.tensor(X_processed, dtype=torch.float32)
            if isinstance(X_processed, np.ndarray)
            else X_processed.float()
        )
        X_tensor = X_tensor.to(self._device)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
        z_all = []
        self._mae.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device)
                z, _ = self._mae.forward(xb)
                z_all.append(z.cpu().numpy())
        return np.concatenate(z_all)

    def get_threshold_mae(self, X_val_train):
        X_val_processed = self._transformer.transform(X_val_train)
        y_val = np.zeros(X_val_processed.shape[0])
        dataset = TensorDataset(
            torch.tensor(X_val_processed, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        val_data_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
        avg_loss, threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss = (
            self._mae.validate(val_data_loader, epoch=0)
        )
        return avg_loss, threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def set_epoch_end_callback(self, callback):
        self._on_epoch_end_callbacks = callback

    def set_epoch_best_callback(self, callback):
        self._on_epoch_best_callbacks = callback
