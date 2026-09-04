"""M01AECTGAN — Joint training: CTGAN diversity + M01AE two-sided margin loss.

Pipeline (joint, per epoch inner step)
---------------------------------------
  1. CTGAN Discriminator step(s)
  2. CTGAN Generator step  → produces fakeact (in transformed space)
  3. M01AE step:
       type_train="noise"     → x_pseudo = _add_noise(real_normal)
       type_train="std"       → x_pseudo = _add_noise(fakeact)   [CTGAN sample + noise]
       type_train="noise_std" → x_pseudo = 50% noise + 50% _add_noise(fakeact)
       Before mae_start_ratio * epochs → always fallback to "noise" (CTGAN not warmed up)

Key advantages over MAECTGAN
-----------------------------
  - Two-sided margin loss (from M01AE):
        normal  → pull ‖z‖ → 0
        anomaly → pull ‖z‖ → margin  (tighter, symmetric separation)
  - _add_noise() on CTGAN samples:
        CTGAN covers diverse regions of the normal distribution;
        noise then pushes those samples OUT of the normal manifold.
        → harder and more diverse pseudo-anomalies than noise alone.
  - Early stopping on validation loss (patience-based).
  - mae_start_ratio: CTGAN warms up first before its samples are used.
  - Dual best-checkpoint selection: tracks the best model by validation loss
    (drives early stopping, as before) AND, independently, by validation AUC
    (pseudo-anomalies for validation are generated using the SAME type_train
    method as training: noise / std / noise_std). Both are saved to disk via
    the epoch-best callback (see sdv/single_table/m01aectgan.py).
"""

import copy
import warnings
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from ctgan.synthesizers.maectgan import Discriminator, Generator


class M01AECTGAN(BaseSynthesizer):
    """Joint CTGAN + M01AE training with two-sided margin loss.

    Parameters
    ----------
    embedding_dim, generator_dim, discriminator_dim,
    generator_lr, generator_decay, discriminator_lr, discriminator_decay,
    batch_size, discriminator_steps, log_frequency, verbose, epochs, pac, cuda
        Standard CTGAN parameters.

    lamda : float
        Weight of latent loss: total = re_loss + lamda * latent_loss.
    margin : float
        Target ‖z‖ for anomaly samples in the two-sided margin loss.
    mae_loss_mode : str  {"origin", "renormal"}
        "renormal" → reconstruction loss computed on normal samples only.
    mae_lr : float
        Learning rate for M01AE encoder/decoder.
    noise_std : float
        Std of Gaussian noise added to continuous features.
    noise_ratio : float  (0, 1]
        Fraction of continuous features perturbed per sample.
    noise_type : str  {"none", "global"}
        Noise type for continuous features. "global" = N(0, noise_std^2).
    cat_noise_type : str  {"none", "shuffle", "flip", "combined"}
        Noise type for categorical (one-hot) features.
    cat_noise_ratio : float  (0, 1]
        Fraction of one-hot spans perturbed per sample.

    type_train : str  {"noise", "std", "noise_std"}
        Pseudo-anomaly source:
          "noise"     → _add_noise(real_normal)       [no CTGAN samples]
          "std"       → _add_noise(ctgan_sample)      [CTGAN + noise]
          "noise_std" → 50% noise + 50% _add_noise(ctgan_sample)
    mae_start_ratio : float  [0, 1)
        Fraction of epochs CTGAN trains alone before M01AE uses its samples.
        Before this point, type_train always falls back to "noise".
    patience : int
        Early stopping patience in epochs (M01AE validation loss).
    """

    def __init__(
        self,
        # ── CTGAN ──────────────────────────────────────────────────────────
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
        # ── M01AE ───────────────────────────────────────────────────────────
        lamda=0.1,
        margin=1.0,
        mae_loss_mode="renormal",
        mae_lr=1e-3,
        noise_std=0.1,
        noise_ratio=1.0,
        noise_type="global",
        noise_r=1.0,
        noise_k=3.0,
        cat_noise_type="none",
        cat_noise_ratio=1.0,
        # ── Joint ───────────────────────────────────────────────────────────
        type_train="noise_std",
        mae_start_ratio=0.25,
        patience=50,
        # ── Misc ─────────────────────────────────────────────────────────────
        dataset="",
        path="",
    ):
        assert batch_size % 2 == 0, "batch_size must be even."
        if batch_size % pac != 0:
            valid_pac = max(p for p in range(1, pac + 1) if batch_size % p == 0)
            warnings.warn(
                f"M01AECTGAN: batch_size={batch_size} not divisible by pac={pac}. "
                f"Auto-adjusting pac to {valid_pac}.",
                UserWarning,
            )
            pac = valid_pac

        # ── CTGAN params ──────────────────────────────────────────────────
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

        # ── M01AE params ──────────────────────────────────────────────────
        self.lamda = float(lamda)
        self.margin = float(margin)
        self.mae_loss_mode = str(mae_loss_mode)
        self.mae_lr = float(mae_lr)
        self.noise_std = float(noise_std)
        self.noise_ratio = float(noise_ratio)
        self.noise_type = str(noise_type)
        self.noise_r = float(noise_r)
        self.noise_k = float(noise_k)
        self.cat_noise_type = str(cat_noise_type)
        self.cat_noise_ratio = float(cat_noise_ratio)

        # ── Joint params ──────────────────────────────────────────────────
        self.type_train = str(type_train)
        self.mae_start_ratio = float(mae_start_ratio)
        self.patience = int(patience)

        # ── Validation ────────────────────────────────────────────────────
        if self.mae_loss_mode not in {"origin", "renormal"}:
            raise ValueError("mae_loss_mode must be 'origin' or 'renormal'.")
        if self.type_train not in {"noise", "std", "noise_std"}:
            raise ValueError("type_train must be 'noise', 'std', or 'noise_std'.")
        if self.noise_type not in {"none", "global", "feature", "feature_outlier"}:
            raise ValueError("noise_type must be 'none', 'global', 'feature', or 'feature_outlier'.")
        if self.cat_noise_type not in {"none", "shuffle", "flip", "combined"}:
            raise ValueError("cat_noise_type must be 'none', 'shuffle', 'flip', or 'combined'.")
        if not 0.0 <= self.mae_start_ratio < 1.0:
            raise ValueError("mae_start_ratio must be in [0, 1).")

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"
        self._device = torch.device(device)

        # ── Network references (built in fit()) ───────────────────────────
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
        self._mae_enc = None
        self._mae_dec = None

        # Feature index tensors (extracted from transformer after fit)
        self._cont_indices = None      # LongTensor: tanh-dim indices
        self._discrete_slices = None   # list of (st, ed) for one-hot spans
        # Per-feature statistics in transformed space (for feature/feature_outlier noise)
        self._feature_means = None     # Tensor (n_cont,)
        self._feature_stds = None      # Tensor (n_cont,)

        self.loss_values = None
        self._best_thresholds = None
        self._best_thresholds_auc = None
        self._best_mae_state_auc = None
        self._on_epoch_end_callbacks = None
        self._on_epoch_best_callbacks = None

    # =========================================================================
    # Static CTGAN helpers (identical to MAECTGAN)
    # =========================================================================

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
                    raise ValueError(f"Unexpected activation: {span_info.activation_fn}")
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
            invalid = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid = [c for c in discrete_columns if c < 0 or c >= train_data.shape[1]]
        else:
            raise TypeError("train_data must be pd.DataFrame or np.ndarray.")
        if invalid:
            raise ValueError(f"Invalid columns: {invalid}")

    # =========================================================================
    # Feature index extraction from transformer
    # =========================================================================

    def _extract_feature_indices(self):
        """Populate _cont_indices (tanh dims) and _discrete_slices (softmax spans)."""
        cont = []
        discrete = []
        st = 0
        for column_info in self._transformer.output_info_list:
            is_discrete = (
                len(column_info) == 1 and column_info[0].activation_fn == "softmax"
            )
            for span_info in column_info:
                ed = st + span_info.dim
                if span_info.activation_fn == "tanh":
                    cont.extend(range(st, ed))
                elif is_discrete:
                    discrete.append((st, ed))
                st = ed
        self._cont_indices = torch.tensor(cont, dtype=torch.long, device=self._device)
        self._discrete_slices = discrete

    # =========================================================================
    # Feature statistics for transformed space (needed by feature/feature_outlier)
    # =========================================================================

    def _compute_feature_statistics(self, train_data_t):
        """Compute per-feature mean and std for tanh-dims in the transformed space.

        Called once after DataTransformer.transform() during fit().
        Results stored as Tensors on self._device.
        """
        if self._cont_indices is None or self._cont_indices.numel() == 0:
            self._feature_means = torch.empty(0, device=self._device)
            self._feature_stds  = torch.empty(0, device=self._device)
            return

        X = torch.tensor(train_data_t, dtype=torch.float32)
        cont_cpu = self._cont_indices.cpu()
        X_cont = X[:, cont_cpu]               # (N, n_cont)
        means = X_cont.mean(dim=0)            # (n_cont,)
        stds  = X_cont.std(dim=0).clamp(min=1e-6)
        self._feature_means = means.to(self._device)
        self._feature_stds  = stds.to(self._device)
        print(
            f"  Feature stats: {cont_cpu.numel()} cont dims  "
            f"mean=[{means.min():.3f},{means.max():.3f}]  "
            f"std=[{stds.min():.3f},{stds.max():.3f}]"
        )

    # =========================================================================
    # Noise generation in CTGAN transformed space
    # =========================================================================

    def _add_noise_transformed(self, x):
        """Add noise in CTGAN's transformed space — mirrors M01AE._add_noise().

        Continuous dims (tanh ∈ [-1,1]):
          'none'           → no perturbation
          'global'         → N(0, noise_std²), clamp [-1,1]
          'feature'        → per-feature N(0, (noise_std·σᵢ)²), clamp [-1,1]
          'feature_outlier'→ set to μᵢ + noise_r·noise_k·σᵢ, clamp [-1,1]

        Categorical dims (one-hot spans):
          'none'     → no perturbation
          'shuffle'  → row-shuffle within span
          'flip'     → 1 − x (invert one-hot)
          'combined' → 50/50 shuffle or flip per span

        noise_ratio / cat_noise_ratio: fraction of features perturbed
        (deterministic k-subset, same as M01AE).
        """
        x_noisy = x.clone()
        cont_idx = self._cont_indices   # LongTensor (n_cont,) on device

        # ── Continuous noise ───────────────────────────────────────────────
        if self.noise_type != "none" and cont_idx.numel() > 0:
            n_cont = cont_idx.numel()
            n_to_noise = max(1, int(n_cont * self.noise_ratio))
            perm = torch.randperm(n_cont, device=x.device)
            feat_pos = perm[:n_to_noise]          # positions within cont_idx
            selected  = cont_idx[feat_pos]        # global column indices

            if self.noise_type == "global":
                noise = torch.randn(x.size(0), n_to_noise, device=x.device) * self.noise_std
                x_noisy[:, selected] = torch.clamp(
                    x_noisy[:, selected] + noise, -1.0, 1.0
                )

            elif self.noise_type == "feature":
                if self._feature_means is None or self._feature_means.numel() == 0:
                    raise RuntimeError(
                        "Feature statistics not computed. "
                        "Ensure fit() was called with noise_type='feature'."
                    )
                stds = self._feature_stds[feat_pos]           # (n_to_noise,)
                noise = (
                    torch.randn(x.size(0), n_to_noise, device=x.device)
                    * (stds * self.noise_std).unsqueeze(0)
                )
                x_noisy[:, selected] = torch.clamp(
                    x_noisy[:, selected] + noise, -1.0, 1.0
                )

            elif self.noise_type == "feature_outlier":
                if self._feature_means is None or self._feature_means.numel() == 0:
                    raise RuntimeError(
                        "Feature statistics not computed. "
                        "Ensure fit() was called with noise_type='feature_outlier'."
                    )
                means = self._feature_means[feat_pos]         # (n_to_noise,)
                stds  = self._feature_stds[feat_pos]
                outlier = (means + self.noise_r * self.noise_k * stds).clamp(-1.0, 1.0)
                x_noisy[:, selected] = outlier.unsqueeze(0).expand(x.size(0), -1)

        # ── Categorical noise ──────────────────────────────────────────────
        if self.cat_noise_type != "none" and self._discrete_slices:
            B = x.size(0)
            if B > 1:
                n_spans = len(self._discrete_slices)
                n_to_noise_cat = max(1, int(n_spans * self.cat_noise_ratio))
                perm_spans = torch.randperm(n_spans, device=x.device).tolist()
                selected_spans = [self._discrete_slices[i] for i in perm_spans[:n_to_noise_cat]]

                for st, ed in selected_spans:
                    perm_rows = torch.randperm(B, device=x.device)
                    if torch.equal(perm_rows, torch.arange(B, device=x.device)):
                        perm_rows = torch.roll(perm_rows, 1)

                    if self.cat_noise_type == "shuffle":
                        x_noisy[:, st:ed] = x_noisy[perm_rows, st:ed]
                    elif self.cat_noise_type == "flip":
                        x_noisy[:, st:ed] = 1.0 - x_noisy[:, st:ed]
                    elif self.cat_noise_type == "combined":
                        if torch.rand(()).item() < 0.5:
                            x_noisy[:, st:ed] = x_noisy[perm_rows, st:ed]
                        else:
                            x_noisy[:, st:ed] = 1.0 - x_noisy[:, st:ed]

        return x_noisy

    # =========================================================================
    # Validation pseudo-anomaly (mirrors type_train, used for AUC-based selection)
    # =========================================================================

    def _generate_ctgan_fakeact(self, n):
        """Sample n fresh CTGAN rows (transformed space) for validation pseudo-anomalies."""
        self._generator.eval()
        with torch.no_grad():
            mean = torch.zeros(n, self._embedding_dim, device=self._device)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)
            condvec = self._data_sampler.sample_original_condvec(n)
            if condvec is not None:
                c1 = torch.from_numpy(condvec).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
        self._generator.train()
        return fakeact

    def _make_pseudo_anomaly(self, real_normal, ctgan_sample, use_ctgan):
        """Build a pseudo-anomaly batch from real_normal, following self.type_train.

        Shared by the training step and validation/AUC evaluation so that the
        model is always *validated* against the same anomaly distribution it
        is *trained* against.

          use_ctgan=False        → _add_noise(real_normal)
          type_train="std"       → _add_noise(ctgan_sample)
          type_train="noise_std" → 50% _add_noise(real_normal) + 50% _add_noise(ctgan_sample)
        """
        if not use_ctgan:
            return self._add_noise_transformed(real_normal)

        if self.type_train == "std":
            return self._add_noise_transformed(ctgan_sample)

        # noise_std: 50% from real noise, 50% from ctgan noise
        half = real_normal.size(0) // 2
        pseudo_noise = self._add_noise_transformed(real_normal[:half])
        pseudo_ctgan = self._add_noise_transformed(ctgan_sample[:half])
        return torch.cat([pseudo_noise, pseudo_ctgan], dim=0)

    def _make_validation_pseudo_anomaly(self, real_normal, use_ctgan):
        ctgan_sample = self._generate_ctgan_fakeact(real_normal.size(0)) if use_ctgan else None
        return self._make_pseudo_anomaly(real_normal, ctgan_sample, use_ctgan)

    def _evaluate_pseudo_validation_auc(self, val_loader, use_ctgan):
        """ROC-AUC separating normal validation rows from pseudo-anomalies built
        the same way as type_train (noise / std / noise_std).

        Returns (auc, separation). auc is NaN if validation has < 2 classes
        (e.g. no pseudo-anomalies could be built).
        """
        self._mae_enc.eval()
        self._mae_dec.eval()
        normal_scores, pseudo_scores = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self._device)
                normal_mask = yb.view(-1) == 0
                if not normal_mask.any():
                    continue
                xb_n = xb[normal_mask]
                xb_pseudo = self._make_validation_pseudo_anomaly(xb_n, use_ctgan)
                if xb_pseudo.size(0) == 0:
                    continue

                z_n, _ = self._mae_forward(xb_n)
                z_p, _ = self._mae_forward(xb_pseudo)
                normal_scores.append(torch.norm(z_n, p=2, dim=1).cpu().numpy())
                pseudo_scores.append(torch.norm(z_p, p=2, dim=1).cpu().numpy())

        self._mae_enc.train()
        self._mae_dec.train()

        if not normal_scores or not pseudo_scores:
            return float("nan"), float("nan")

        normal_scores = np.concatenate(normal_scores)
        pseudo_scores = np.concatenate(pseudo_scores)
        separation = float(pseudo_scores.mean() - normal_scores.mean())

        y = np.concatenate([np.zeros_like(normal_scores), np.ones_like(pseudo_scores)])
        scores = np.concatenate([normal_scores, pseudo_scores])
        if len(np.unique(y)) < 2:
            return float("nan"), separation
        return float(roc_auc_score(y, scores)), separation

    # =========================================================================
    # M01AE encoder / decoder construction
    # =========================================================================

    def _build_mae_networks(self, input_dim):
        """Build M01AE encoder + decoder with same architecture as M01AE.

        Encoder: d → 0.75d → 0.5d → sqrt(d)+1
        Decoder: sqrt(d)+1 → 0.5d → 0.75d → d  (Tanh output — matches CTGAN space)
        """
        d = input_dim
        lat = round(sqrt(d)) + 1

        self._mae_enc = nn.Sequential(
            nn.Linear(d, round(d * 0.75)),
            nn.Tanh(),
            nn.Linear(round(d * 0.75), round(d * 0.5)),
            nn.Tanh(),
            nn.Linear(round(d * 0.5), lat),
        ).to(self._device)

        self._mae_dec = nn.Sequential(
            nn.Linear(lat, round(d * 0.5)),
            nn.Tanh(),
            nn.Linear(round(d * 0.5), round(d * 0.75)),
            nn.Tanh(),
            nn.Linear(round(d * 0.75), d),
            nn.Tanh(),   # CTGAN transformed space: continuous ∈ [-1,1]
        ).to(self._device)

    def _mae_forward(self, x):
        z = self._mae_enc(x)
        x_hat = self._mae_dec(z)
        return z, x_hat

    # =========================================================================
    # M01AE two-sided margin loss
    # =========================================================================

    def _mae_loss(self, x, z, x_hat, y):
        """Two-sided margin loss (from M01AE, key difference vs MAECTGAN).

        re_loss    : MSE reconstruction (normal only if mae_loss_mode="renormal")
        latent_loss: normal → ‖z‖² → 0 ; anomaly → (‖z‖ - margin)² → 0
        total      : re_loss + lamda * latent_loss
        """
        re_per = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)

        if self.mae_loss_mode == "renormal":
            normal_mask = y.view(-1) == 0
            re_loss = (
                re_per[normal_mask].mean()
                if normal_mask.any()
                else re_per.mean() * 0.0
            )
        else:
            re_loss = re_per.mean()

        z_norm = torch.norm(z, p=2, dim=1)
        margin_t = torch.tensor(self.margin, device=z.device, dtype=z.dtype)
        latent_loss = torch.mean(
            torch.where(
                y.view(-1) == 0,
                z_norm ** 2,                   # normal: pull ‖z‖ → 0
                (z_norm - margin_t) ** 2,       # anomaly: pull ‖z‖ → margin
            )
        )
        return re_loss, latent_loss

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_mae(self, val_loader):
        """Return (avg_loss, avg_re, avg_latent) on normal val samples."""
        self._mae_enc.eval()
        self._mae_dec.eval()
        total_loss = total_re = total_latent = total_seen = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                normal_mask = yb.view(-1) == 0
                if not normal_mask.any():
                    continue
                xb_n = xb[normal_mask]
                yb_n = yb[normal_mask]
                z, xb_hat = self._mae_forward(xb_n)
                re_loss, latent_loss = self._mae_loss(xb_n, z, xb_hat, yb_n)
                loss = re_loss + self.lamda * latent_loss
                bs = xb_n.size(0)
                total_seen += bs
                total_loss += loss.item() * bs
                total_re += re_loss.item() * bs
                total_latent += latent_loss.item() * bs

        self._mae_enc.train()
        self._mae_dec.train()
        if total_seen == 0:
            return float("inf"), float("inf"), float("inf")
        return total_loss / total_seen, total_re / total_seen, total_latent / total_seen

    def _compute_thresholds(self, val_loader):
        """Compute 95th-percentile anomaly thresholds on normal validation data."""
        self._mae_enc.eval()
        self._mae_dec.eval()
        re_all, z_all, total_all = [], [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                normal_mask = yb.view(-1) == 0
                if not normal_mask.any():
                    continue
                xb_n = xb[normal_mask]
                z, xb_hat = self._mae_forward(xb_n)
                re = F.mse_loss(xb_hat, xb_n, reduction="none").mean(dim=1)
                z_norm = torch.norm(z, p=2, dim=1)
                total = re + self.lamda * z_norm ** 2
                re_all.append(re.cpu().numpy())
                z_all.append(z_norm.cpu().numpy())
                total_all.append(total.cpu().numpy())

        self._mae_enc.train()
        self._mae_dec.train()

        re_arr = np.concatenate(re_all)
        z_arr = np.concatenate(z_all)
        total_arr = np.concatenate(total_all)
        return {
            "threshold_re":       (float(re_arr.mean()), float(re_arr.std())),
            "threshold_z":        (float(z_arr.mean()), float(z_arr.std())),
            "threshold_95_latent": float(np.percentile(z_arr, 95)),
            "threshold_95_total":  float(np.percentile(total_arr, 95)),
        }

    # =========================================================================
    # Callbacks
    # =========================================================================

    def set_epoch_end_callback(self, cb):
        self._on_epoch_end_callbacks = cb

    def set_epoch_best_callback(self, cb):
        self._on_epoch_best_callbacks = cb

    # =========================================================================
    # Main training
    # =========================================================================

    @random_state
    def fit(self, full_data, train_data, val_data, discrete_columns=()):
        print(
            "Starting M01AECTGAN training...",
            " data shape:", train_data.shape if hasattr(train_data, "shape") else "?",
            " epochs:", self._epochs,
            " type_train:", self.type_train,
            " mae_start_ratio:", self.mae_start_ratio,
            " pac:", self.pac,
        )

        self._validate_discrete_columns(train_data, discrete_columns)

        # ── Data preparation ──────────────────────────────────────────────
        self._transformer = DataTransformer()
        self._transformer.fit(full_data, discrete_columns)
        train_data_t = self._transformer.transform(train_data)
        print("  Transformed data shape:", train_data_t.shape)

        X_val_t = self._transformer.transform(val_data)
        y_val_zeros = np.zeros(X_val_t.shape[0])
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val_t, dtype=torch.float32),
                torch.tensor(y_val_zeros, dtype=torch.float32),
            ),
            batch_size=self._batch_size, shuffle=False,
        )

        self._data_sampler = DataSampler(
            train_data_t, self._transformer.output_info_list, self._log_frequency
        )
        data_dim = self._transformer.output_dimensions

        # ── Build networks ────────────────────────────────────────────────
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim, data_dim,
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim, pac=self.pac,
        ).to(self._device)

        self._build_mae_networks(data_dim)
        self._extract_feature_indices()
        self._compute_feature_statistics(train_data_t)

        print(
            f"  Feature indices: cont={self._cont_indices.numel()} dims, "
            f"discrete spans={len(self._discrete_slices)}"
        )

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
        optimizer_mae = optim.Adam(
            list(self._mae_enc.parameters()) + list(self._mae_dec.parameters()),
            lr=self.mae_lr,
        )

        mean_z = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std_z = mean_z + 1
        steps_per_epoch = max(len(train_data_t) // self._batch_size, 1)
        mae_start_epoch = int(self.mae_start_ratio * self._epochs)

        # Normal data tensor on CPU (for random MAE batch sampling)
        normal_tensor_cpu = torch.tensor(train_data_t, dtype=torch.float32)

        # ── Training state ────────────────────────────────────────────────
        loss_rows = []
        best_val_loss = float("inf")
        best_enc_state = None
        best_dec_state = None
        best_thresholds = None
        patience_counter = 0

        best_val_auc = -float("inf")
        best_enc_state_auc = None
        best_dec_state_auc = None
        best_thresholds_auc = None

        epoch_iter = tqdm(range(self._epochs), disable=(not self._verbose))

        for i in epoch_iter:
            g_accum = d_accum = 0.0
            mae_accum = re_accum = lat_accum = 0.0

            for _step in range(steps_per_epoch):

                # ── Discriminator step(s) ─────────────────────────────────
                for _ in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean_z, std=std_z)
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
                            train_data_t, self._batch_size, col, opt
                        )
                        fake_cat = fakeact
                        real_cat = torch.from_numpy(real.astype("float32")).to(self._device)
                    else:
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data_t, self._batch_size, col[perm], opt[perm]
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
                fakez = torch.normal(mean=mean_z, std=std_z)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)   # (B, data_dim) — will be reused for MAE

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

                g_accum += loss_g.detach().cpu().item()
                d_accum += loss_d.detach().cpu().item()

                # ── M01AE step ────────────────────────────────────────────
                # Determine whether CTGAN samples are trusted yet
                use_ctgan = (i >= mae_start_epoch) and (self.type_train != "noise")

                # Sample real normal batch (random rows from training data)
                real_idx = torch.randperm(normal_tensor_cpu.size(0))[:self._batch_size]
                real_normal = normal_tensor_cpu[real_idx].to(self._device)

                # Build pseudo-anomaly batch (see _make_pseudo_anomaly for the type_train switch)
                ctgan_sample = fakeact.detach() if use_ctgan else None   # no gradient through G
                x_pseudo = self._make_pseudo_anomaly(real_normal, ctgan_sample, use_ctgan)
                x_normal_mae = real_normal

                # Build combined batch: [normal | pseudo_anomaly]
                x_combined = torch.cat([x_normal_mae, x_pseudo], dim=0)
                y_combined = torch.cat([
                    torch.zeros(x_normal_mae.size(0), device=self._device),
                    torch.ones(x_pseudo.size(0), device=self._device),
                ], dim=0)

                # M01AE forward + two-sided margin loss
                self._mae_enc.train()
                self._mae_dec.train()
                z, x_hat = self._mae_forward(x_combined)
                re_loss, latent_loss = self._mae_loss(x_combined, z, x_hat, y_combined)
                loss_mae = re_loss + self.lamda * latent_loss

                optimizer_mae.zero_grad()
                loss_mae.backward()
                optimizer_mae.step()

                mae_accum += loss_mae.item()
                re_accum += re_loss.item()
                lat_accum += latent_loss.item()

            # ── End of epoch ──────────────────────────────────────────────
            avg_g = g_accum / steps_per_epoch
            avg_d = d_accum / steps_per_epoch
            avg_mae = mae_accum / steps_per_epoch
            avg_re = re_accum / steps_per_epoch
            avg_lat = lat_accum / steps_per_epoch

            val_loss, val_re, val_lat = self._validate_mae(val_loader)
            thresholds = self._compute_thresholds(val_loader)

            ctgan_active = int(i >= mae_start_epoch and self.type_train != "noise")
            val_auc, val_sep = self._evaluate_pseudo_validation_auc(val_loader, bool(ctgan_active))

            print(
                f"Epoch {i:03d}: G={avg_g:.3f} D={avg_d:.3f} | "
                f"MAE={avg_mae:.4f} RE={avg_re:.4f} Z={avg_lat:.4f} | "
                f"Val={val_loss:.4f} ValAUC={val_auc:.4f} "
                f"{'[CTGAN active]' if ctgan_active else '[warmup]'}"
            )

            loss_rows.append({
                "Epoch": i,
                "Generator Loss": avg_g,
                "Discriminator Loss": avg_d,
                "MAE Loss": avg_mae,
                "Reconstruction Loss": avg_re,
                "Latent Loss": avg_lat,
                "Validation Loss": val_loss,
                "Validation RE Loss": val_re,
                "Validation Latent Loss": val_lat,
                "Validation AUC": val_auc,
                "Validation Separation": val_sep,
                "CTGAN Active": ctgan_active,
            })

            # ── Best-by-AUC checkpoint (tracked independently, no early stopping) ──
            if not np.isnan(val_auc) and val_auc > best_val_auc + 1e-8:
                best_val_auc = val_auc
                best_enc_state_auc = copy.deepcopy(self._mae_enc.state_dict())
                best_dec_state_auc = copy.deepcopy(self._mae_dec.state_dict())
                best_thresholds_auc = thresholds

                if self._on_epoch_best_callbacks is not None:
                    self._on_epoch_best_callbacks(
                        model=self, epoch=i, checkpoint_kind="auc",
                        threshold_re=thresholds["threshold_re"],
                        threshold_z=thresholds["threshold_z"],
                        threshold_95=thresholds["threshold_95_latent"],
                        threshold_95_total_loss=thresholds["threshold_95_total"],
                    )
                print(f"  [BEST_AUC] val_auc={val_auc:.6f} (sep={val_sep:.4f}) at epoch {i}")

            # ── Early stopping (best-by-loss) ───────────────────────────────
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_enc_state = copy.deepcopy(self._mae_enc.state_dict())
                best_dec_state = copy.deepcopy(self._mae_dec.state_dict())
                best_thresholds = thresholds
                patience_counter = 0

                if self._on_epoch_best_callbacks is not None:
                    self._on_epoch_best_callbacks(
                        model=self, epoch=i, checkpoint_kind="loss",
                        threshold_re=thresholds["threshold_re"],
                        threshold_z=thresholds["threshold_z"],
                        threshold_95=thresholds["threshold_95_latent"],
                        threshold_95_total_loss=thresholds["threshold_95_total"],
                    )
                print(f"  [BEST] val_loss={val_loss:.6f} at epoch {i}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(
                        f"  Early stopping triggered at epoch {i}. "
                        f"Best val_loss={best_val_loss:.6f}"
                    )
                    break

            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)

        self.loss_values = pd.DataFrame(loss_rows)

        # Restore best M01AE weights (selected by validation loss — the default
        # weights left on self._mae_enc/_mae_dec after fit()). The best-by-AUC
        # weights are kept separately (self._best_mae_state_auc) and were already
        # checkpointed to disk via the epoch-best callback.
        if best_enc_state is not None:
            self._mae_enc.load_state_dict(best_enc_state)
            self._mae_dec.load_state_dict(best_dec_state)
            print(f"Restored best M01AE weights (val_loss={best_val_loss:.6f})")
        self._best_thresholds = best_thresholds
        self._best_thresholds_auc = best_thresholds_auc
        self._best_mae_state_auc = (best_enc_state_auc, best_dec_state_auc)
        print(f"Best val_loss={best_val_loss:.6f}  Best val_AUC={best_val_auc:.6f}")
        print("M01AECTGAN training complete.")

    # =========================================================================
    # CTGAN inference
    # =========================================================================

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Generate n synthetic normal samples using the trained CTGAN generator."""
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condvec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condvec = None

        steps = n // self._batch_size + 1
        data = []
        for _ in range(steps):
            m = torch.zeros(self._batch_size, self._embedding_dim)
            s = m + 1
            fakez = torch.normal(mean=m, std=s).to(self._device)
            condvec = (
                global_condvec.copy() if global_condvec is not None
                else self._data_sampler.sample_original_condvec(self._batch_size)
            )
            if condvec is not None:
                c1 = torch.from_numpy(condvec).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)[:n]
        return self._transformer.inverse_transform(data)

    # =========================================================================
    # Anomaly scoring
    # =========================================================================

    def get_score(self, X, y):
        """Compute anomaly scores for raw (untransformed) data.

        Returns
        -------
        re_errors    : ndarray (N,) — reconstruction error per sample
        z_norm_scores: ndarray (N,) — ‖z‖ per sample
        total_scores : ndarray (N,) — re + lamda * ‖z‖²
        y_true       : ndarray (N,) — ground-truth labels
        """
        self._mae_enc.eval()
        self._mae_dec.eval()

        X_t = self._transformer.transform(
            X if isinstance(X, pd.DataFrame) else X
        )
        X_tensor = torch.tensor(X_t, dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self._batch_size, shuffle=False,
        )

        re_all, z_all, total_all, y_all = [], [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self._device)
                z, xb_hat = self._mae_forward(xb)
                re = F.mse_loss(xb_hat, xb, reduction="none").mean(dim=1)
                z_norm = torch.norm(z, p=2, dim=1)
                total = re + self.lamda * z_norm ** 2
                re_all.append(re.cpu().numpy())
                z_all.append(z_norm.cpu().numpy())
                total_all.append(total.cpu().numpy())
                y_all.append(yb.numpy())

        return (
            np.concatenate(re_all),
            np.concatenate(z_all),
            np.concatenate(total_all),
            np.concatenate(y_all),
        )

    def get_threshold(self, X_val):
        """Compute thresholds from normal validation data (raw, untransformed).

        Returns dict with keys:
          threshold_re, threshold_z, threshold_95_latent, threshold_95_total
        """
        X_t = self._transformer.transform(
            X_val if isinstance(X_val, pd.DataFrame) else X_val
        )
        y_zeros = np.zeros(X_t.shape[0])
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_t, dtype=torch.float32),
                torch.tensor(y_zeros, dtype=torch.float32),
            ),
            batch_size=self._batch_size, shuffle=False,
        )
        return self._compute_thresholds(loader)

    def get_loss_values(self):
        return self.loss_values

    # =========================================================================
    # Device management (required by BaseSynthesizer pickling)
    # =========================================================================

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(device)
        if self._discriminator is not None:
            self._discriminator.to(device)
        if self._mae_enc is not None:
            self._mae_enc.to(device)
        if self._mae_dec is not None:
            self._mae_dec.to(device)
        if self._cont_indices is not None:
            self._cont_indices = self._cont_indices.to(device)
