"""CleanMAECTGAN module — MAECTGAN with two pseudo-anomaly filtering strategies.

Independent implementation (does not inherit from MAECTGAN).

Two filter modes (selected by filter_mode parameter)
-----------------------------------------------------
"critic"    — Uses the CTGAN discriminator (D) score as the signal.
              D score HIGH → sample looks like normal training data → label noise.
              Works with any pac value: group scores (pac > 1) are expanded to
              per-sample level via repeat_interleave.  An EMA threshold tracks the
              upward trend of D scores as the generator improves, making the filter
              self-calibrating.  Applied BEFORE the replay buffer so only cleaner
              pseudo-anomalies accumulate.

"mae_recon" — Uses MAE reconstruction error.  Applied AFTER assembling the full
              pseudo-anomaly batch (from buffer).  Samples with LOW reconstruction
              error look normal to the MAE → label noise → discard.  Only the
              hardest pseudo-anomalies (highest error) are kept.  Ported from
              CTMAECTGAN Hướng A.

"none"      — No filtering; behaves identically to MAECTGAN (default).

pac stays at the CTGAN default (10) to preserve training quality.
"""

import warnings
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import (
    BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU,
    Sequential, Tanh, MSELoss, Sigmoid, functional,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
# Import neural network blocks only — no class inheritance from MAECTGAN
from ctgan.synthesizers.maectgan import Discriminator, Generator, MAE


class CleanMAECTGAN(BaseSynthesizer):
    """MAECTGAN with two pseudo-anomaly filtering strategies.

    Parameters (new, compared with MAECTGAN)
    -----------------------------------------
    filter_mode : str  {"none", "critic", "mae_recon"}
        Filtering strategy.
        "none"     — no filtering (baseline, same as MAECTGAN).
        "critic"   — discard CTGAN-generated samples with high D scores
                     (they look like normal data → label noise).  Uses an EMA
                     threshold that adapts as D scores trend upward during training.
                     Applied BEFORE the replay buffer.  Works with any pac.
        "mae_recon"— discard assembled pseudo-anomaly samples with low MAE
                     reconstruction error (they are too easy → look normal).
                     Applied AFTER buffer assembly.  Ported from CTMAECTGAN.
    filter_keep_ratio : float in (0, 1]
        Fraction of pseudo-anomaly samples to KEEP each step.
        0.7 → keep 70%, discard 30% that look most normal.
        1.0 → effectively disables filtering.
    filter_start_ratio : float in [0, 1]
        Fraction of post-MAE-start epochs to wait before filtering activates.
        0.5 → filtering begins halfway between mae_start_epoch and last epoch.
    critic_ema_alpha : float in (0, 1]
        EMA smoothing factor for the D-score trend threshold (critic mode only).
        0.3 → slow adaptation (stable).  0.9 → fast (follows current batch).
    """

    def __init__(
        self,
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
        lamda=0.1,
        margin=1.0,
        mae_loss_mode="origin",
        type_train="all",
        mae_start_ratio=0.0,
        ctgan_noise_start_ratio=0.75,
        ctgan_noise_std=0.1,
        ctgan_noise_feature_ratio=1.0,
        ctgan_pseudo_anomaly_mode="origin",
        ctgan_feature_outlier_r=1.0,
        # ── filter ────────────────────────────────────────────────────────
        filter_mode="none",
        filter_keep_ratio=0.7,
        filter_start_ratio=0.5,
        critic_ema_alpha=0.3,
    ):
        assert batch_size % 2 == 0, "batch_size must be divisible by 2."
        if batch_size % pac != 0:
            valid_pac = max(p for p in range(1, pac + 1) if batch_size % p == 0)
            warnings.warn(
                f"CleanMAECTGAN: batch_size={batch_size} is not divisible by pac={pac}. "
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
        self.type_train = type_train
        self.mae_start_ratio = float(mae_start_ratio)
        self.ctgan_noise_start_ratio = float(ctgan_noise_start_ratio)
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
        self.critic_ema_alpha = float(critic_ema_alpha)

        # Validation
        if not 0.0 <= self.mae_start_ratio <= 1.0:
            raise ValueError("mae_start_ratio must be in [0, 1].")
        if not 0.0 <= self.ctgan_noise_start_ratio <= 1.0:
            raise ValueError("ctgan_noise_start_ratio must be in [0, 1].")
        if self.ctgan_noise_std < 0.0:
            raise ValueError("ctgan_noise_std must be >= 0.")
        if not 0.0 <= self.ctgan_noise_feature_ratio <= 1.0:
            raise ValueError("ctgan_noise_feature_ratio must be in [0, 1].")
        if self.ctgan_pseudo_anomaly_mode not in {"origin", "feature_outlier"}:
            raise ValueError(
                "ctgan_pseudo_anomaly_mode must be 'origin' or 'feature_outlier'."
            )
        if self.mae_loss_mode not in {"origin", "renormal"}:
            raise ValueError("mae_loss_mode must be 'origin' or 'renormal'.")
        if self.filter_mode not in {"none", "critic", "mae_recon"}:
            raise ValueError("filter_mode must be 'none', 'critic', or 'mae_recon'.")
        if not 0.0 < self.filter_keep_ratio <= 1.0:
            raise ValueError("filter_keep_ratio must be in (0, 1].")
        if not 0.0 <= self.filter_start_ratio <= 1.0:
            raise ValueError("filter_start_ratio must be in [0, 1].")
        if not 0.0 < self.critic_ema_alpha <= 1.0:
            raise ValueError("critic_ema_alpha must be in (0, 1].")

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
        self.loss_values = None
        self._on_epoch_end_callbacks = None
        self._on_epoch_best_callbacks = None
        self._validation_cont_mean = None
        self._validation_cont_std = None
        self._d_score_ema_threshold = None  # running EMA for critic filter

    # ──────────────────────────────────────────────────────────────────────────
    # Static helpers (identical to MAECTGAN)
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
        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = [
                col
                for col in discrete_columns
                if col < 0 or col >= train_data.shape[1]
            ]
        else:
            raise TypeError("train_data should be pd.DataFrame or np.ndarray.")
        if invalid_columns:
            raise ValueError(f"Invalid columns found: {invalid_columns}")

    # ──────────────────────────────────────────────────────────────────────────
    # Replay buffer helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def to_buffer(buffer, new_fake, buf_per_iter):
        new_fake = new_fake.clone().detach()
        batch_size = new_fake.shape[0]
        perm = torch.randperm(batch_size)
        idx = perm[:buf_per_iter]
        new_buffer = new_fake[idx]

        if buffer is None:
            buffer = new_buffer
        else:
            buffer = buffer.clone().detach().to(new_fake.device)
            buffer = torch.cat((buffer, new_buffer), dim=0)
            max_buffer_size = 8 * buf_per_iter
            keep_size = min(max_buffer_size, buffer.size(0))
            perm2 = torch.randperm(buffer.size(0), device=buffer.device)
            buffer = buffer[perm2[:keep_size]].detach()
        return buffer

    @staticmethod
    def from_buffer(buffer, num_of_imgs):
        buffer = buffer.clone().detach()
        perm = torch.randperm(buffer.size(0))
        idx = perm[: min(num_of_imgs, buffer.size(0))]
        return buffer[idx]

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

    def add_gaussian_noise_to_ctgan_samples(
        self, fake_batch, noise_std=None, feature_ratio=None
    ):
        x = fake_batch.clone().detach()
        cont_idx = self.get_continuous_indices(device=x.device)
        noise_std = self.ctgan_noise_std if noise_std is None else float(noise_std)
        feature_ratio = (
            self.ctgan_noise_feature_ratio
            if feature_ratio is None
            else float(feature_ratio)
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
    # Validation pseudo-anomaly helpers
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
            -1.0,
            1.0,
        )
        return x

    def _validation_pseudo_modes(self):
        mode_map = {
            "noise": ("noise",),
            "std": ("std",),
            "3std": ("3std",),
            "std_3std": ("std", "3std"),
            "noise_std": ("noise", "std"),
            "noise_3std": ("noise", "3std"),
            "all": ("noise", "std", "3std"),
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
        normal_scores = []
        pseudo_scores = []
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
        y_pseudo = np.concatenate([
            np.zeros_like(normal_scores, dtype=np.int64),
            np.ones_like(pseudo_scores, dtype=np.int64),
        ])
        score_pseudo = np.concatenate([normal_scores, pseudo_scores])
        pseudo_auc = roc_auc_score(y_pseudo, score_pseudo)
        pseudo_separation = float(np.mean(pseudo_scores) - np.mean(normal_scores))
        return float(pseudo_auc), pseudo_separation

    # ──────────────────────────────────────────────────────────────────────────
    # Pseudo-anomaly filters  ← NEW in CleanMAECTGAN
    # ──────────────────────────────────────────────────────────────────────────

    def _get_expanded_d_scores(self, y_fake, batch_size):
        """Expand group-level D scores to per-sample scores.

        With pac > 1 the discriminator returns (B//pac, 1) scores — each score
        covers pac consecutive samples.  We expand to (B,) so the critic filter
        can operate per-sample.

        Parameters
        ----------
        y_fake : Tensor (B//pac, 1)
        batch_size : int   full batch size B

        Returns
        -------
        Tensor (batch_size,)
        """
        scores = y_fake.detach().squeeze(-1)  # (B//pac,)
        if self.pac == 1:
            return scores
        expanded = scores.repeat_interleave(self.pac)  # (B_trunc,)
        if expanded.size(0) < batch_size:
            pad = scores.mean().expand(batch_size - expanded.size(0))
            expanded = torch.cat([expanded, pad], dim=0)
        return expanded[:batch_size]

    def _critic_filter(self, fakeact, d_scores, epoch, total_epochs, mae_start_epoch):
        """Filter generated samples using D-score EMA trend (critic mode).

        As the generator improves, some generated samples consistently score HIGH
        on D — the generator is pulling them toward the normal distribution.
        These are label noise (labeled anomaly but look normal).

        An EMA threshold tracks how D scores trend upward over training, making
        the filter self-calibrating without a fixed percentile.

        Applied BEFORE the replay buffer so only cleaner samples accumulate.

        Parameters
        ----------
        fakeact   : Tensor (N, D)
        d_scores  : Tensor (N,)  per-sample scores from _get_expanded_d_scores
        epoch, total_epochs, mae_start_epoch : int

        Returns
        -------
        Tensor (M, D)  M <= N
        """
        if self.filter_mode != "critic":
            return fakeact
        if self.filter_keep_ratio >= 1.0:
            return fakeact

        filter_start = mae_start_epoch + int(
            self.filter_start_ratio * max(1, total_epochs - mae_start_epoch)
        )
        if epoch < filter_start:
            return fakeact

        N = fakeact.size(0)
        if N < 4:
            return fakeact

        scores = d_scores.detach().to(fakeact.device)

        # Threshold from current batch at the keep-ratio quantile
        current_thresh = torch.quantile(scores, self.filter_keep_ratio).item()

        # EMA: tracks how the threshold trends upward as generator matures
        if self._d_score_ema_threshold is None:
            self._d_score_ema_threshold = current_thresh
        else:
            self._d_score_ema_threshold = (
                self.critic_ema_alpha * current_thresh
                + (1.0 - self.critic_ema_alpha) * self._d_score_ema_threshold
            )

        ema_thresh = torch.tensor(
            self._d_score_ema_threshold, device=fakeact.device, dtype=scores.dtype
        )

        # Keep samples with D score <= EMA threshold (look least like normal data)
        mask = scores <= ema_thresh
        keep = int(mask.sum().item())

        if keep < 4:
            k = max(4, int(N * self.filter_keep_ratio))
            k = min(k, N)
            _, idx = scores.topk(k, largest=False)
            return fakeact[idx]

        return fakeact[mask]

    def _mae_recon_filter(self, fake_data, epoch, total_epochs, mae_start_epoch):
        """Filter assembled pseudo-anomaly batch by MAE reconstruction error.

        Keep only the filter_keep_ratio fraction with the HIGHEST reconstruction
        error.  Low error → MAE can reconstruct them easily → they look normal
        → label noise.  High error → genuinely hard pseudo-anomalies → keep.

        Applied AFTER buffer assembly so buffer diversity is unaffected.

        Parameters
        ----------
        fake_data : Tensor (B, D)  assembled pseudo-anomaly batch
        epoch, total_epochs, mae_start_epoch : int

        Returns
        -------
        Tensor (M, D)  M <= B
        """
        if self.filter_mode != "mae_recon":
            return fake_data
        if self.filter_keep_ratio >= 1.0:
            return fake_data

        filter_start = mae_start_epoch + int(
            self.filter_start_ratio * max(1, total_epochs - mae_start_epoch)
        )
        if epoch < filter_start:
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

        # Keep keep_n samples with HIGHEST reconstruction error (hardest anomalies)
        _, topk_idx = errors.topk(keep_n, largest=True)
        return fake_data[topk_idx]

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    @random_state
    def fit(self, full_data, train_data, val_data, discrete_columns=(), epochs=None):
        print(
            "Starting CleanMAECTGAN training..., data shape:", train_data.shape,
            " discrete_columns:", discrete_columns,
            " epochs:", self._epochs,
            " lamda:", self.lamda,
            " margin:", self.margin,
            " pac:", self.pac,
            " filter_mode:", self.filter_mode,
            " filter_keep_ratio:", self.filter_keep_ratio,
            " filter_start_ratio:", self.filter_start_ratio,
        )
        self._d_score_ema_threshold = None  # reset EMA at the start of each fit
        fake_buff_list_NOR = None
        fake_buff_list_ANO = None
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                "`epochs` argument in `fit` method has been deprecated. "
                "Pass `epochs` to the constructor instead.",
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(full_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        print("data after transform shape:", train_data.shape)

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

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        self._mae = MAE(
            input_dim=data_dim,
            lamda=self.lamda,
            margin=self.margin,
            mae_loss_mode=self.mae_loss_mode,
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizerD = optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=[
            "Epoch", "Generator Loss", "Discriminator Loss",
            "MAE Loss", "Reconstruction Loss", "Latent Loss",
            "Validation Loss", "Validation Reconstruction Loss",
            "Validation Latent Loss", "Pseudo Val AUC", "Pseudo Separation",
            "Filter Kept Ratio",
        ])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = "Gen. ({gen:.2f}) | Disc. ({dis:.2f})"
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        best_pseudo_val_auc = -float("inf")
        best_pseudo_val_separation = -float("inf")
        mae_start_epoch = int(np.ceil(epochs * self.mae_start_ratio))
        ctgan_noise_start_epoch = int(np.ceil(epochs * self.ctgan_noise_start_ratio))
        filter_start_epoch = mae_start_epoch + int(
            self.filter_start_ratio * max(1, epochs - mae_start_epoch)
        )
        print(
            "[SCHEDULE] MAE starts after", mae_start_epoch,
            "epochs; noise starts after", ctgan_noise_start_epoch,
            "epochs; filter starts after", filter_start_epoch,
            "epochs; filter_mode=", self.filter_mode,
            "pac=", self.pac,
        )

        for i in epoch_iterator:
            mae_active = i >= mae_start_epoch
            ctgan_noise_active = mae_active and i >= ctgan_noise_start_epoch
            gaussian_noise_active = (
                ctgan_noise_active and self.ctgan_pseudo_anomaly_mode == "origin"
            )
            mae_loss = np.nan
            re_loss = np.nan
            latent_loss = np.nan
            list_latent = None
            mae_dataset = None
            epoch_kept_ratios = []

            for id_ in range(steps_per_epoch):

                # ── Discriminator step ────────────────────────────────────────
                real_data = None
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    fakez_3 = torch.normal(mean=mean, std=std * sqrt(3))
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)

                    if c1 is not None:
                        fakez = torch.cat([fakez, c1], dim=1)
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                        fakez_3 = torch.cat([fakez_3, c1], dim=1)
                    fake_3 = self._generator(fakez_3)
                    fakeact_3 = self._apply_activate(fake_3)

                    if condvec is None:
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    real = torch.from_numpy(real.astype("float32")).to(self._device)
                    real_data = real

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        fake_cat = fakeact
                        real_cat = real

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

                # ── Generator step ────────────────────────────────────────────
                fakez = torch.normal(mean=mean, std=std)
                fakez_3 = torch.normal(mean=mean, std=std * sqrt(3))
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                    fakez_3 = torch.cat([fakez_3, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                fake_3 = self._generator(fakez_3)
                fakeact_3 = self._apply_activate(fake_3)

                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)

                # ── Capture D scores for critic filter (FREE — already computed) ──
                # y_fake shape: (B//pac, 1) → expand to (B,) for per-sample filtering
                _gen_d_scores = (
                    self._get_expanded_d_scores(y_fake, self._batch_size)
                    if self.filter_mode == "critic"
                    else None
                )

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                if not mae_active:
                    continue

                # ── Build pseudo-anomaly batches ──────────────────────────────
                if self.ctgan_pseudo_anomaly_mode == "feature_outlier":
                    fakeact_for_mae_nor = self.make_ctgan_feature_outlier_anomaly(
                        real_data, fakeact, r=self.ctgan_feature_outlier_r
                    )
                    fakeact_for_mae_ano = self.make_ctgan_feature_outlier_anomaly(
                        real_data, fakeact_3, r=self.ctgan_feature_outlier_r
                    )
                else:
                    fakeact_for_mae_nor = fakeact
                    fakeact_for_mae_ano = fakeact_3
                    if gaussian_noise_active:
                        fakeact_for_mae_nor = self.add_gaussian_noise_to_ctgan_samples(
                            fakeact_for_mae_nor,
                            noise_std=self.ctgan_noise_std,
                            feature_ratio=self.ctgan_noise_feature_ratio,
                        )

                # ── Critic filter (before buffering, critic mode only) ────────
                n_before = fakeact_for_mae_nor.size(0)
                if self.filter_mode == "critic" and _gen_d_scores is not None:
                    fakeact_for_mae_nor = self._critic_filter(
                        fakeact_for_mae_nor, _gen_d_scores, i, epochs, mae_start_epoch
                    )
                    n_after = fakeact_for_mae_nor.size(0)
                    if n_before > 0:
                        epoch_kept_ratios.append(n_after / n_before)

                # ── Buffer ────────────────────────────────────────────────────
                fake_buff_list_NOR = self.to_buffer(
                    fake_buff_list_NOR, fakeact_for_mae_nor, self._batch_size
                )
                fake_buff_list_ANO = self.to_buffer(
                    fake_buff_list_ANO, fakeact_for_mae_ano, self._batch_size
                )

                real_perturbed = self.make_real_perturbed_anomaly(real_data)

                if fake_buff_list_NOR is None:
                    fakeact_form_buffer_NOR = fakeact_for_mae_nor
                else:
                    fakeact_form_buffer_NOR = self.from_buffer(
                        fake_buff_list_NOR, self._batch_size
                    )

                if fake_buff_list_ANO is None:
                    fakeact_form_buffer_ANO = fakeact_for_mae_ano
                else:
                    fakeact_form_buffer_ANO = self.from_buffer(
                        fake_buff_list_ANO, self._batch_size
                    )

                # ── Assemble MAE training batch ───────────────────────────────
                if self.type_train == "all":
                    fake_data = torch.cat(
                        [fakeact_form_buffer_ANO, real_perturbed, fakeact_form_buffer_NOR],
                        dim=0,
                    )
                elif self.type_train == "noise":
                    fake_data = real_perturbed
                elif self.type_train == "3std":
                    fake_data = fakeact_form_buffer_ANO
                elif self.type_train == "std":
                    fake_data = fakeact_form_buffer_NOR
                elif self.type_train == "std_3std":
                    fake_data = torch.cat(
                        [fakeact_form_buffer_ANO, fakeact_form_buffer_NOR], dim=0
                    )
                elif self.type_train == "noise_std":
                    fake_data = torch.cat(
                        [real_perturbed, fakeact_form_buffer_NOR], dim=0
                    )
                elif self.type_train == "noise_3std":
                    fake_data = torch.cat(
                        [real_perturbed, fakeact_form_buffer_ANO], dim=0
                    )
                else:
                    fake_data = None
                    print("type_train is not valid")

                if fake_data is None:
                    continue

                # ── MAE recon filter (after assembly, mae_recon mode only) ───
                if self.filter_mode == "mae_recon":
                    n_before_mae = fake_data.size(0)
                    fake_data = self._mae_recon_filter(
                        fake_data, i, epochs, mae_start_epoch
                    )
                    n_after_mae = fake_data.size(0)
                    if n_before_mae > 0:
                        epoch_kept_ratios.append(n_after_mae / n_before_mae)

                y_fake_mae = torch.ones(fake_data.size(0), 1, device=self._device)
                y_real_mae = torch.zeros(real_data.size(0), 1, device=self._device)
                mae_dataset = torch.cat([fake_data, real_data], dim=0)
                y_mae = torch.cat([y_fake_mae, y_real_mae], dim=0)
                mae_dataloader = DataLoader(
                    TensorDataset(mae_dataset, y_mae),
                    batch_size=self._batch_size,
                    shuffle=True,
                )
                mae_loss, re_loss, latent_loss, list_latent = self._mae.train_mae(
                    epoch=i, train_data_loader=mae_dataloader
                )

            # ── End of epoch ──────────────────────────────────────────────────
            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            avg_val_total_loss = np.nan
            avg_val_re_loss = np.nan
            avg_val_latent_loss = np.nan
            pseudo_val_auc = np.nan
            pseudo_val_separation = np.nan
            avg_kept_ratio = (
                float(np.mean(epoch_kept_ratios)) if epoch_kept_ratios else np.nan
            )

            if mae_active:
                avg_val_total_loss, threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss = (
                    self._mae.validate(val_data_loader, epoch=i)
                )
                avg_val_re_loss = getattr(
                    self._mae, "last_validation_reconstruction_loss", np.nan
                )
                avg_val_latent_loss = getattr(
                    self._mae, "last_validation_latent_loss", np.nan
                )
                pseudo_val_auc, pseudo_val_separation = (
                    self.evaluate_pseudo_validation_auc(val_data_loader)
                )
                is_best_checkpoint = (
                    pseudo_val_auc > best_pseudo_val_auc
                    or (
                        np.isclose(pseudo_val_auc, best_pseudo_val_auc)
                        and pseudo_val_separation > best_pseudo_val_separation
                    )
                )
                if is_best_checkpoint:
                    best_pseudo_val_auc = pseudo_val_auc
                    best_pseudo_val_separation = pseudo_val_separation
                    if self._on_epoch_best_callbacks is not None:
                        self._on_epoch_best_callbacks(
                            model=self,
                            epoch=i,
                            threshold_re=threshold_re,
                            threshold_z_0=threshold_z_0,
                            threshold_95=threshold_95,
                            threshold_95_total_loss=threshold_95_total_loss,
                        )
                    print(
                        f"[BEST] epoch={i} pseudo_val_auc={pseudo_val_auc:.6f} "
                        f"pseudo_separation={pseudo_val_separation:.6f} "
                        f"val_total_loss={avg_val_total_loss:.6f}"
                    )
                mae_data_shape = mae_dataset.shape if mae_dataset is not None else "N/A"
            else:
                mae_data_shape = "warmup: MAE not trained"

            epoch_loss_df = pd.DataFrame({
                "Epoch": [i],
                "Generator Loss": [generator_loss],
                "Discriminator Loss": [discriminator_loss],
                "MAE Loss": [mae_loss.item() if torch.is_tensor(mae_loss) else mae_loss],
                "Reconstruction Loss": [re_loss.item() if torch.is_tensor(re_loss) else re_loss],
                "Latent Loss": [latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss],
                "Validation Loss": [avg_val_total_loss],
                "Validation Reconstruction Loss": [avg_val_re_loss],
                "Validation Latent Loss": [avg_val_latent_loss],
                "Pseudo Val AUC": [pseudo_val_auc],
                "Pseudo Separation": [pseudo_val_separation],
                "Filter Kept Ratio": [avg_kept_ratio],
            })
            kept_str = f"{avg_kept_ratio:.3f}" if not np.isnan(avg_kept_ratio) else "N/A"
            print(
                f"Epoch {i}: Gen={generator_loss:.4f} Disc={discriminator_loss:.4f} "
                f"MAE={mae_loss} RE={re_loss} Latent={latent_loss} "
                f"PseudoAUC={pseudo_val_auc} FilterKept={kept_str}",
                "mae shape:", mae_data_shape,
            )

            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)

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
        if isinstance(X_processed, np.ndarray):
            X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        else:
            X_tensor = X_processed.float()
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
