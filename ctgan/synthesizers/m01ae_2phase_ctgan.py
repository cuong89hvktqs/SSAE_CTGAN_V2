"""M01AE2PhaseCTGAN — Two-phase: CTGAN then M01AE with augmented normal/anomaly.

Phase 1  (ctgan_epochs):
    Standard CTGAN training on normal data.

Phase 2  (mae_epochs):
    1. Generate a fixed CTGAN pool  (ctgan_sample_ratio × N samples).
    2. Train M01AE with early stopping:
         Normal  side  = real_normal[:B/2]  +  ctgan_pool_sample[:B/2]
         Anomaly side  = _add_noise(real_normal[:B/2])
                       + _add_noise(ctgan_pool_sample[:B/2])
    Both sides are balanced (50/50 real vs CTGAN).
    Two-sided margin loss (same as M01AE):
         normal  → pull ‖z‖ → 0    : z_norm²
         anomaly → pull ‖z‖ → m    : (z_norm − margin)²
"""

import copy
import warnings
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from ctgan.synthesizers.maectgan import Discriminator, Generator


class M01AE2PhaseCTGAN(BaseSynthesizer):
    """Two-phase CTGAN + M01AE with augmented normal and anomaly pools.

    Parameters
    ----------
    ctgan_epochs : int
        Phase 1 — CTGAN training epochs.
    mae_epochs : int
        Phase 2 — M01AE training epochs (subject to early stopping).
    ctgan_sample_ratio : float
        Size of the CTGAN pool relative to training data (e.g. 1.0 = same size).
        Pool is generated once at the start of Phase 2 and kept fixed.
    lamda : float
        Weight of latent loss: total = re_loss + lamda * latent_loss.
    margin : float
        Target ‖z‖ for anomaly samples.
    mae_loss_mode : str  {'origin', 'renormal'}
        'renormal' → reconstruction loss on normal samples only.
    mae_lr : float
        M01AE learning rate.
    noise_std : float
        Gaussian noise std (used by 'global' and 'feature' types).
    noise_ratio : float  (0, 1]
        Fraction of continuous features perturbed.
    noise_type : str  {'none','global','feature','feature_outlier'}
        Continuous noise strategy. 'feature_outlier' matches M01AE standalone.
    noise_r : float
        Sign multiplier for feature_outlier: μ + noise_r * noise_k * σ.
    noise_k : float
        Scale factor for feature_outlier.
    cat_noise_type : str  {'none','shuffle','flip','combined'}
        Categorical noise strategy.
    cat_noise_ratio : float  (0, 1]
        Fraction of one-hot spans perturbed.
    patience : int
        Early stopping patience (Phase 2 validation loss).
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
        pac=10,
        cuda=True,
        # ── Two-phase ───────────────────────────────────────────────────────
        ctgan_epochs=150,
        mae_epochs=100,
        ctgan_sample_ratio=1.0,
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
        patience=50,
        # ── Misc ─────────────────────────────────────────────────────────────
        dataset="",
        path="",
    ):
        assert batch_size % 2 == 0, "batch_size must be even."
        if batch_size % pac != 0:
            valid_pac = max(p for p in range(1, pac + 1) if batch_size % p == 0)
            warnings.warn(
                f"M01AE2PhaseCTGAN: batch_size={batch_size} not divisible by pac={pac}. "
                f"Auto-adjusting pac to {valid_pac}.",
                UserWarning,
            )
            pac = valid_pac

        # CTGAN
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
        self.pac = pac

        # Two-phase
        self.ctgan_epochs = int(ctgan_epochs)
        self.mae_epochs = int(mae_epochs)
        self.ctgan_sample_ratio = float(ctgan_sample_ratio)

        # M01AE
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
        self.patience = int(patience)

        # Validation
        if mae_loss_mode not in {"origin", "renormal"}:
            raise ValueError("mae_loss_mode must be 'origin' or 'renormal'.")
        if noise_type not in {"none", "global", "feature", "feature_outlier"}:
            raise ValueError(
                "noise_type must be 'none','global','feature', or 'feature_outlier'."
            )
        if cat_noise_type not in {"none", "shuffle", "flip", "combined"}:
            raise ValueError(
                "cat_noise_type must be 'none','shuffle','flip', or 'combined'."
            )
        if not 0.0 < ctgan_sample_ratio:
            raise ValueError("ctgan_sample_ratio must be > 0.")

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"
        self._device = torch.device(device)

        self.dataset = dataset
        self.path = path

        # Built during fit()
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
        self._mae_enc = None
        self._mae_dec = None
        self._cont_indices = None
        self._discrete_slices = None
        self._feature_means = None
        self._feature_stds = None

        self.ctgan_loss_values = None
        self.mae_loss_values = None
        self._best_thresholds = None
        self._on_epoch_end_callbacks = None
        self._on_epoch_best_callbacks = None

    # =========================================================================
    # CTGAN static helpers
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
                    raise ValueError(
                        f"Unexpected activation: {span_info.activation_fn}"
                    )
                st = ed
        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = st_c = 0
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
    # Feature index extraction + statistics
    # =========================================================================

    def _extract_feature_indices(self):
        cont, discrete, st = [], [], 0
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

    def _compute_feature_statistics(self, train_data_t):
        if self._cont_indices is None or self._cont_indices.numel() == 0:
            self._feature_means = torch.empty(0, device=self._device)
            self._feature_stds = torch.empty(0, device=self._device)
            return
        X = torch.tensor(train_data_t, dtype=torch.float32)
        cont_cpu = self._cont_indices.cpu()
        X_cont = X[:, cont_cpu]
        means = X_cont.mean(dim=0)
        stds = X_cont.std(dim=0).clamp(min=1e-6)
        self._feature_means = means.to(self._device)
        self._feature_stds = stds.to(self._device)
        print(
            f"  Feature stats: {cont_cpu.numel()} cont dims  "
            f"mean=[{means.min():.3f},{means.max():.3f}]  "
            f"std=[{stds.min():.3f},{stds.max():.3f}]"
        )

    # =========================================================================
    # Noise generation (identical to M01AECTGAN)
    # =========================================================================

    def _add_noise_transformed(self, x):
        """Add noise in CTGAN transformed space — matches M01AE._add_noise() logic.

        Continuous dims (tanh in [-1,1]):
          'none'            -> no perturbation
          'global'          -> N(0, noise_std^2), clamp [-1,1]
          'feature'         -> per-feature N(0,(noise_std*sigma_i)^2), clamp [-1,1]
          'feature_outlier' -> set to mu_i + noise_r*noise_k*sigma_i, clamp [-1,1]
        Categorical dims (one-hot spans):
          'none'/'shuffle'/'flip'/'combined' with cat_noise_ratio k-subset.
        """
        x_noisy = x.clone()
        cont_idx = self._cont_indices

        # ── Continuous ──────────────────────────────────────────────────────
        if self.noise_type != "none" and cont_idx.numel() > 0:
            n_cont = cont_idx.numel()
            n_to_noise = max(1, int(n_cont * self.noise_ratio))
            perm = torch.randperm(n_cont, device=x.device)
            feat_pos = perm[:n_to_noise]
            selected = cont_idx[feat_pos]

            if self.noise_type == "global":
                noise = torch.randn(x.size(0), n_to_noise, device=x.device) * self.noise_std
                x_noisy[:, selected] = torch.clamp(
                    x_noisy[:, selected] + noise, -1.0, 1.0
                )
            elif self.noise_type == "feature":
                if self._feature_stds is None or self._feature_stds.numel() == 0:
                    raise RuntimeError("Feature statistics not computed (call fit first).")
                stds = self._feature_stds[feat_pos]
                noise = (
                    torch.randn(x.size(0), n_to_noise, device=x.device)
                    * (stds * self.noise_std).unsqueeze(0)
                )
                x_noisy[:, selected] = torch.clamp(
                    x_noisy[:, selected] + noise, -1.0, 1.0
                )
            elif self.noise_type == "feature_outlier":
                if self._feature_means is None or self._feature_means.numel() == 0:
                    raise RuntimeError("Feature statistics not computed (call fit first).")
                means = self._feature_means[feat_pos]
                stds = self._feature_stds[feat_pos]
                outlier = (means + self.noise_r * self.noise_k * stds).clamp(-1.0, 1.0)
                x_noisy[:, selected] = outlier.unsqueeze(0).expand(x.size(0), -1)

        # ── Categorical ─────────────────────────────────────────────────────
        if self.cat_noise_type != "none" and self._discrete_slices:
            B = x.size(0)
            if B > 1:
                n_spans = len(self._discrete_slices)
                n_cat_noise = max(1, int(n_spans * self.cat_noise_ratio))
                perm_spans = torch.randperm(n_spans, device=x.device).tolist()
                selected_spans = [
                    self._discrete_slices[i] for i in perm_spans[:n_cat_noise]
                ]
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
    # M01AE networks
    # =========================================================================

    def _build_mae_networks(self, input_dim):
        d = input_dim
        lat = round(sqrt(d)) + 1
        self._mae_enc = nn.Sequential(
            nn.Linear(d, round(d * 0.75)), nn.Tanh(),
            nn.Linear(round(d * 0.75), round(d * 0.5)), nn.Tanh(),
            nn.Linear(round(d * 0.5), lat),
        ).to(self._device)
        self._mae_dec = nn.Sequential(
            nn.Linear(lat, round(d * 0.5)), nn.Tanh(),
            nn.Linear(round(d * 0.5), round(d * 0.75)), nn.Tanh(),
            nn.Linear(round(d * 0.75), d),
            nn.Tanh(),   # CTGAN transformed space: continuous in [-1,1]
        ).to(self._device)

    def _mae_forward(self, x):
        z = self._mae_enc(x)
        x_hat = self._mae_dec(z)
        return z, x_hat

    # =========================================================================
    # M01AE two-sided margin loss
    # =========================================================================

    def _mae_loss(self, x, z, x_hat, y):
        """Two-sided margin loss (same as M01AE standalone).

        re_loss    : MSE on normal samples (if mae_loss_mode='renormal')
        latent_loss: normal -> z_norm^2 ; anomaly -> (z_norm - margin)^2
        """
        re_per = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)
        if self.mae_loss_mode == "renormal":
            normal_mask = y.view(-1) == 0
            re_loss = (
                re_per[normal_mask].mean() if normal_mask.any()
                else re_per.mean() * 0.0
            )
        else:
            re_loss = re_per.mean()

        z_norm = torch.norm(z, p=2, dim=1)
        margin_t = torch.tensor(self.margin, device=z.device, dtype=z.dtype)
        latent_loss = torch.mean(
            torch.where(
                y.view(-1) == 0,
                z_norm ** 2,
                (z_norm - margin_t) ** 2,
            )
        )
        return re_loss, latent_loss

    # =========================================================================
    # Validation helpers
    # =========================================================================

    def _validate_mae(self, val_loader):
        self._mae_enc.eval()
        self._mae_dec.eval()
        total_loss = total_re = total_latent = total_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                normal_mask = yb.view(-1) == 0
                if not normal_mask.any():
                    continue
                xb_n, yb_n = xb[normal_mask], yb[normal_mask]
                z, xb_hat = self._mae_forward(xb_n)
                re_l, lat_l = self._mae_loss(xb_n, z, xb_hat, yb_n)
                loss = re_l + self.lamda * lat_l
                bs = xb_n.size(0)
                total_seen += bs
                total_loss += loss.item() * bs
                total_re += re_l.item() * bs
                total_latent += lat_l.item() * bs
        self._mae_enc.train()
        self._mae_dec.train()
        if total_seen == 0:
            return float("inf"), float("inf"), float("inf")
        return (
            total_loss / total_seen,
            total_re / total_seen,
            total_latent / total_seen,
        )

    def _compute_thresholds(self, val_loader):
        self._mae_enc.eval()
        self._mae_dec.eval()
        re_all, z_all, total_all = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self._device)
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
            "threshold_re":        (float(re_arr.mean()), float(re_arr.std())),
            "threshold_z":         (float(z_arr.mean()), float(z_arr.std())),
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
    # CTGAN pool generation
    # =========================================================================

    def _generate_ctgan_pool(self, n_samples):
        """Generate n_samples in CTGAN transformed space (generator frozen)."""
        self._generator.eval()
        pool = []
        with torch.no_grad():
            generated = 0
            while generated < n_samples:
                mean_z = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
                std_z = mean_z + 1
                fakez = torch.normal(mean=mean_z, std=std_z)
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)
                if condvec is not None:
                    c1 = torch.from_numpy(condvec).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                pool.append(fakeact.cpu())
                generated += fakeact.size(0)
        self._generator.train()
        pool_t = torch.cat(pool, dim=0)[:n_samples]
        print(f"  CTGAN pool generated: {pool_t.size(0)} samples")
        return pool_t  # stays on CPU; moved to device in training loop

    # =========================================================================
    # Phase 1: CTGAN training
    # =========================================================================

    def _train_ctgan_phase(self, train_data_t):
        print(f"Phase 1: CTGAN training ({self.ctgan_epochs} epochs) ...")
        steps_per_epoch = max(len(train_data_t) // self._batch_size, 1)
        mean_z = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std_z = mean_z + 1
        loss_rows = []

        epoch_iter = tqdm(range(self.ctgan_epochs), disable=(not self._verbose))
        for i in epoch_iter:
            g_accum = d_accum = 0.0
            for _step in range(steps_per_epoch):
                # Discriminator
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
                        real_cat = torch.from_numpy(real.astype("float32")).to(self._device)
                        fake_cat = fakeact
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
                    self._optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    self._optimizerD.step()

                # Generator
                fakez = torch.normal(mean=mean_z, std=std_z)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1 = None, None
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
                self._optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                self._optimizerG.step()

                g_accum += loss_g.detach().cpu().item()
                d_accum += loss_d.detach().cpu().item()

            avg_g = g_accum / steps_per_epoch
            avg_d = d_accum / steps_per_epoch
            if self._verbose:
                print(f"  [CTGAN] Epoch {i:03d}: G={avg_g:.4f}  D={avg_d:.4f}")

            loss_rows.append({
                "Phase": "CTGAN",
                "Epoch": i,
                "Generator Loss": avg_g,
                "Discriminator Loss": avg_d,
            })

            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)

        self.ctgan_loss_values = pd.DataFrame(loss_rows)
        print("Phase 1 complete.")

    # =========================================================================
    # Phase 2: M01AE training
    # =========================================================================

    def _train_mae_phase(self, train_data_t, val_loader):
        """M01AE training with augmented normal and anomaly.

        Normal  = real_normal[:B/2] + ctgan_pool_sample[:B/2]
        Anomaly = _add_noise(real_normal[:B/2]) + _add_noise(ctgan_sample[:B/2])
        """
        n_train = len(train_data_t)
        n_ctgan = max(self._batch_size, int(n_train * self.ctgan_sample_ratio))

        print(
            f"Phase 2: generating CTGAN pool ({n_ctgan} samples, "
            f"ratio={self.ctgan_sample_ratio}) ..."
        )
        ctgan_pool_cpu = self._generate_ctgan_pool(n_ctgan)   # (n_ctgan, data_dim) on CPU
        real_pool_cpu = torch.tensor(train_data_t, dtype=torch.float32)  # on CPU

        optimizer_mae = optim.Adam(
            list(self._mae_enc.parameters()) + list(self._mae_dec.parameters()),
            lr=self.mae_lr,
        )

        half = self._batch_size // 2
        loss_rows = []
        best_val_loss = float("inf")
        best_enc_state = None
        best_dec_state = None
        best_thresholds = None
        patience_counter = 0

        steps_per_epoch = max(n_train // self._batch_size, 1)

        print(f"Phase 2: M01AE training ({self.mae_epochs} epochs, patience={self.patience}) ...")
        epoch_iter = tqdm(range(self.mae_epochs), disable=(not self._verbose))

        for i in epoch_iter:
            self._mae_enc.train()
            self._mae_dec.train()
            mae_accum = re_accum = lat_accum = 0.0

            for _step in range(steps_per_epoch):
                # ── Sample real normal (half batch) ────────────────────────
                idx_real = torch.randint(0, real_pool_cpu.size(0), (half,))
                x_real = real_pool_cpu[idx_real].to(self._device)   # (half, dim)

                # ── Sample CTGAN pool (half batch) ─────────────────────────
                idx_ctgan = torch.randint(0, ctgan_pool_cpu.size(0), (half,))
                x_ctgan = ctgan_pool_cpu[idx_ctgan].to(self._device)  # (half, dim)

                # ── Normal side: real + ctgan ──────────────────────────────
                x_normal = torch.cat([x_real, x_ctgan], dim=0)          # (B, dim)

                # ── Anomaly side: noise(real) + noise(ctgan) ───────────────
                x_pseudo_real  = self._add_noise_transformed(x_real)    # (half, dim)
                x_pseudo_ctgan = self._add_noise_transformed(x_ctgan)   # (half, dim)
                x_pseudo = torch.cat([x_pseudo_real, x_pseudo_ctgan], dim=0)  # (B, dim)

                # ── Combined batch ─────────────────────────────────────────
                x_combined = torch.cat([x_normal, x_pseudo], dim=0)     # (2B, dim)
                y_combined = torch.cat([
                    torch.zeros(self._batch_size, device=self._device),
                    torch.ones(self._batch_size, device=self._device),
                ], dim=0)                                                # (2B,)

                z, x_hat = self._mae_forward(x_combined)
                re_l, lat_l = self._mae_loss(x_combined, z, x_hat, y_combined)
                loss_mae = re_l + self.lamda * lat_l

                optimizer_mae.zero_grad()
                loss_mae.backward()
                optimizer_mae.step()

                mae_accum += loss_mae.item()
                re_accum  += re_l.item()
                lat_accum += lat_l.item()

            avg_mae = mae_accum / steps_per_epoch
            avg_re  = re_accum  / steps_per_epoch
            avg_lat = lat_accum / steps_per_epoch

            val_loss, val_re, val_lat = self._validate_mae(val_loader)
            thresholds = self._compute_thresholds(val_loader)

            if self._verbose:
                print(
                    f"  [MAE] Epoch {i:03d}: "
                    f"MAE={avg_mae:.4f}  RE={avg_re:.4f}  Z={avg_lat:.4f}  "
                    f"Val={val_loss:.4f}"
                )

            loss_rows.append({
                "Phase": "MAE",
                "Epoch": i,
                "MAE Loss": avg_mae,
                "Reconstruction Loss": avg_re,
                "Latent Loss": avg_lat,
                "Validation Loss": val_loss,
                "Validation RE Loss": val_re,
                "Validation Latent Loss": val_lat,
            })

            # ── Early stopping ────────────────────────────────────────────
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_enc_state = copy.deepcopy(self._mae_enc.state_dict())
                best_dec_state = copy.deepcopy(self._mae_dec.state_dict())
                best_thresholds = thresholds
                patience_counter = 0

                if self._on_epoch_best_callbacks is not None:
                    self._on_epoch_best_callbacks(
                        model=self, epoch=i,
                        threshold_re=thresholds["threshold_re"],
                        threshold_z=thresholds["threshold_z"],
                        threshold_95=thresholds["threshold_95_latent"],
                        threshold_95_total_loss=thresholds["threshold_95_total"],
                    )
                print(f"  [BEST] val_loss={val_loss:.6f} at MAE epoch {i}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(
                        f"  Early stopping at MAE epoch {i}. "
                        f"Best val_loss={best_val_loss:.6f}"
                    )
                    break

            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)

        self.mae_loss_values = pd.DataFrame(loss_rows)

        if best_enc_state is not None:
            self._mae_enc.load_state_dict(best_enc_state)
            self._mae_dec.load_state_dict(best_dec_state)
            print(f"Restored best M01AE weights (val_loss={best_val_loss:.6f})")
        self._best_thresholds = best_thresholds
        print("Phase 2 complete.")

    # =========================================================================
    # Main fit
    # =========================================================================

    @random_state
    def fit(self, full_data, train_data, val_data, discrete_columns=()):
        print(
            "M01AE2PhaseCTGAN fit():"
            f"  data={train_data.shape if hasattr(train_data,'shape') else '?'}"
            f"  ctgan_epochs={self.ctgan_epochs}"
            f"  mae_epochs={self.mae_epochs}"
            f"  ctgan_sample_ratio={self.ctgan_sample_ratio}"
            f"  pac={self.pac}"
        )

        self._validate_discrete_columns(train_data, discrete_columns)

        # ── Data prep ────────────────────────────────────────────────────
        self._transformer = DataTransformer()
        self._transformer.fit(full_data, discrete_columns)
        train_data_t = self._transformer.transform(train_data)
        print(f"  Transformed data shape: {train_data_t.shape}")

        X_val_t = self._transformer.transform(val_data)
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val_t, dtype=torch.float32),
                torch.zeros(X_val_t.shape[0], dtype=torch.float32),
            ),
            batch_size=self._batch_size, shuffle=False,
        )

        self._data_sampler = DataSampler(
            train_data_t, self._transformer.output_info_list, self._log_frequency
        )
        data_dim = self._transformer.output_dimensions

        # ── Networks ─────────────────────────────────────────────────────
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

        self._optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        self._optimizerD = optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr, betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        # ── Phase 1: CTGAN ────────────────────────────────────────────────
        self._train_ctgan_phase(train_data_t)

        # ── Phase 2: M01AE ────────────────────────────────────────────────
        self._train_mae_phase(train_data_t, val_loader)

        print("M01AE2PhaseCTGAN training complete.")

    # =========================================================================
    # CTGAN inference
    # =========================================================================

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
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
        """Compute anomaly scores.

        X : pd.DataFrame or np.ndarray — SDV-preprocessed data
        y : array-like — ground-truth labels (0=normal, 1=anomaly)

        Returns
        -------
        re_errors, z_norm_scores, total_scores, y_true : ndarrays (N,)
        """
        self._mae_enc.eval()
        self._mae_dec.eval()
        X_t = self._transformer.transform(X if isinstance(X, pd.DataFrame) else X)
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
        """Compute thresholds from SDV-preprocessed normal validation data."""
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
        """Return (ctgan_loss_df, mae_loss_df)."""
        return self.ctgan_loss_values, self.mae_loss_values

    # =========================================================================
    # Device management
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
        if self._feature_means is not None and self._feature_means.numel() > 0:
            self._feature_means = self._feature_means.to(device)
            self._feature_stds = self._feature_stds.to(device)
