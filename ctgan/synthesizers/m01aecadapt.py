"""M01AECAdapt — adaptive extension of M01AECTGAN combining two strategies:

Direction 2 — Loss-guided noise_std adaptation:
  Every `noise_adapt_every` epochs (after gate opens + noise_adapt_warmup),
  measure the mean separation between pseudo-anomaly and normal z-norms:
      separation = mean‖z_pseudo‖ − mean‖z_normal‖
  Compare against target zone [noise_target_lo_frac·margin, noise_target_hi_frac·margin]:
      sep < lo → pseudo-anomaly too easy → increase noise_std (+noise_adapt_factor)
      sep > hi → pseudo-anomaly too hard → decrease noise_std (-noise_adapt_factor)
  Adjustment is multiplicative, bounded by [noise_std_min, noise_std_max].
  NOTE: adaptation only starts after the CTGAN gate opens (so the sep signal
  reflects CTGAN+noise diversity, not noise-only fallback).

Direction 3 — CTGAN quality gate:
  The fixed mae_start_ratio warmup is replaced by a data-driven gate.
  Gate opens when EMA-smoothed DISCRIMINATOR loss CoV falls below
  `ctgan_gate_stability` over a rolling window of `ctgan_gate_window` epochs,
  subject to at least `ctgan_gate_min_epochs` having elapsed.
  Using discriminator loss (not generator): D loss reflects the Wasserstein
  distance and converges quickly (~epoch 10-15), while G loss oscillates
  throughout adversarial training and never stabilises.
  Hard fallback: if the gate has not opened by epoch
  round(mae_start_ratio * total_epochs), it is force-opened regardless of CoV.
  Until the gate opens, M01AE always falls back to _add_noise(real_normal).

Direction 2 — Loss-guided noise_std adaptation:
  Smoothed with EMA (alpha=0.3) before comparing against the target zone,
  so that per-epoch z-separation oscillation does not trigger noisy updates.
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.m01aectgan import M01AECTGAN


class M01AECAdapt(M01AECTGAN):
    """Adaptive joint CTGAN + M01AE (loss-guided noise_std + CTGAN quality gate)."""

    def __init__(
        self,
        # ── Standard M01AECTGAN params ──────────────────────────────────
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
        mae_loss_mode="renormal",
        mae_lr=1e-3,
        noise_std=0.1,
        noise_ratio=1.0,
        noise_type="global",
        noise_r=1.0,
        noise_k=3.0,
        cat_noise_type="none",
        cat_noise_ratio=1.0,
        type_train="noise_std",
        mae_start_ratio=0.0,       # unused in CAdapt — gate is the only trigger
        patience=50,
        dataset="",
        path="",
        # ── Direction 2 — Adaptive noise_std ────────────────────────────
        noise_std_min=0.01,
        noise_std_max=2.0,
        noise_adapt_every=5,
        noise_adapt_factor=0.15,
        noise_target_lo_frac=0.5,
        noise_target_hi_frac=0.9,
        noise_adapt_warmup=10,
        # ── Direction 3 — CTGAN quality gate ────────────────────────────
        ctgan_gate_window=10,
        ctgan_gate_stability=0.05,
        ctgan_gate_min_epochs=10,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            cuda=cuda,
            lamda=lamda,
            margin=margin,
            mae_loss_mode=mae_loss_mode,
            mae_lr=mae_lr,
            noise_std=noise_std,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            noise_r=noise_r,
            noise_k=noise_k,
            cat_noise_type=cat_noise_type,
            cat_noise_ratio=cat_noise_ratio,
            type_train=type_train,
            mae_start_ratio=mae_start_ratio,
            patience=patience,
            dataset=dataset,
            path=path,
        )
        # Adaptive params
        self.noise_std_min      = float(noise_std_min)
        self.noise_std_max      = float(noise_std_max)
        self.noise_adapt_every  = int(noise_adapt_every)
        self.noise_adapt_factor = float(noise_adapt_factor)
        self.noise_target_lo_frac = float(noise_target_lo_frac)
        self.noise_target_hi_frac = float(noise_target_hi_frac)
        self.noise_adapt_warmup   = int(noise_adapt_warmup)

        self.ctgan_gate_window     = int(ctgan_gate_window)
        self.ctgan_gate_stability  = float(ctgan_gate_stability)
        self.ctgan_gate_min_epochs = int(ctgan_gate_min_epochs)

        # Runtime state — reset at the start of fit()
        self._ctgan_gate_open   = False
        self._ctgan_gate_epoch  = None
        self._ctgan_gate_forced = False   # True if opened by hard fallback
        self._adapt_log         = []      # [(epoch, noise_std, avg_separation)]

    # =========================================================================
    # Adaptive helpers
    # =========================================================================

    def _maybe_open_gate(self, epoch, disc_loss_history):
        """Open CTGAN gate when EMA discriminator loss is stable (Direction 3).

        Gate opens when CoV of EMA-smoothed disc_loss over rolling window
        falls below ctgan_gate_stability, subject to ctgan_gate_min_epochs.
        If the gate never opens (CTGAN does not converge), type_train stays
        at noise-only for the full run — this is intentional.
        """
        if self._ctgan_gate_open or self.type_train == "noise":
            return

        min_needed = max(self.ctgan_gate_window, self.ctgan_gate_min_epochs)
        if epoch < min_needed - 1 or len(disc_loss_history) < self.ctgan_gate_window:
            return
        window = disc_loss_history[-self.ctgan_gate_window:]
        cov = np.std(window) / (abs(np.mean(window)) + 1e-8)
        if cov <= self.ctgan_gate_stability:
            self._ctgan_gate_open   = True
            self._ctgan_gate_epoch  = epoch
            self._ctgan_gate_forced = False
            print(
                f"  [CTGAN GATE] Opened at epoch {epoch} "
                f"(EMA disc_loss CoV={cov:.4f} <= {self.ctgan_gate_stability})"
            )

    def _maybe_adapt_noise(self, epoch, ema_separation):
        """Adjust noise_std based on EMA-smoothed z-separation signal (Direction 2).

        Only adapts after the CTGAN gate opens (gate_epoch + noise_adapt_warmup).
        Uses EMA-smoothed separation to avoid noisy updates from CTGAN oscillation.
        """
        if not self._ctgan_gate_open or self._ctgan_gate_epoch is None:
            return
        if epoch < self._ctgan_gate_epoch + self.noise_adapt_warmup:
            return
        if (epoch + 1) % self.noise_adapt_every != 0:
            return

        lo = self.noise_target_lo_frac * self.margin
        hi = self.noise_target_hi_frac * self.margin
        old_std = self.noise_std

        if ema_separation < lo:
            self.noise_std = min(
                self.noise_std * (1.0 + self.noise_adapt_factor),
                self.noise_std_max,
            )
        elif ema_separation > hi:
            self.noise_std = max(
                self.noise_std * (1.0 - self.noise_adapt_factor),
                self.noise_std_min,
            )

        self._adapt_log.append((epoch, self.noise_std, ema_separation))
        if abs(self.noise_std - old_std) > 1e-9:
            print(
                f"  [ADAPT noise_std] {old_std:.4f} -> {self.noise_std:.4f}  "
                f"ema_sep={ema_separation:.4f}  target=[{lo:.4f}, {hi:.4f}]"
            )

    # =========================================================================
    # Training loop (override parent fit)
    # =========================================================================

    @random_state
    def fit(self, full_data, train_data, val_data, discrete_columns=()):
        from ctgan.data_sampler import DataSampler
        from ctgan.data_transformer import DataTransformer
        from ctgan.synthesizers.maectgan import Discriminator, Generator

        print(
            "Starting M01AECAdapt training...",
            " data shape:", train_data.shape if hasattr(train_data, "shape") else "?",
            " epochs:", self._epochs,
            " type_train:", self.type_train,
            " noise_std (init):", self.noise_std,
            " ctgan_gate_min_epochs:", self.ctgan_gate_min_epochs,
            " ctgan_gate_window:", self.ctgan_gate_window,
            " noise_adapt_every:", self.noise_adapt_every,
        )

        # Reset adaptive state
        self._ctgan_gate_open  = False
        self._ctgan_gate_epoch = None
        self._adapt_log        = []

        self._validate_discrete_columns(train_data, discrete_columns)

        # ── Data preparation ──────────────────────────────────────────────
        self._transformer = DataTransformer()
        self._transformer.fit(full_data, discrete_columns)
        train_data_t = self._transformer.transform(train_data)
        print("  Transformed data shape:", train_data_t.shape)

        X_val_t    = self._transformer.transform(val_data)
        y_val_zero = np.zeros(X_val_t.shape[0])
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val_t, dtype=torch.float32),
                torch.tensor(y_val_zero, dtype=torch.float32),
            ),
            batch_size=self._batch_size, shuffle=False,
        )

        self._data_sampler = DataSampler(
            train_data_t, self._transformer.output_info_list, self._log_frequency
        )
        data_dim = self._transformer.output_dimensions

        # ── Build networks ─────────────────────────────────────────────────
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

        mean_z           = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std_z            = mean_z + 1
        steps_per_epoch  = max(len(train_data_t) // self._batch_size, 1)
        normal_tensor_cpu = torch.tensor(train_data_t, dtype=torch.float32)

        # ── Training state ─────────────────────────────────────────────────
        loss_rows = []

        best_val_loss   = float("inf")
        best_enc_state  = best_dec_state = best_thresholds = None
        patience_counter = 0

        best_val_auc        = -float("inf")
        best_enc_state_auc  = best_dec_state_auc = best_thresholds_auc = None

        disc_loss_history = []   # EMA-smoothed disc_loss history for CTGAN gate
        _ema_alpha = 0.3         # EMA smoothing factor (lower = smoother)
        _ema_disc  = None        # running EMA of discriminator loss
        _ema_sep   = None        # running EMA of z-separation (for noise adapt)

        # Reset gate state
        self._ctgan_gate_open   = False
        self._ctgan_gate_epoch  = None
        self._ctgan_gate_forced = False

        epoch_iter = tqdm(range(self._epochs), disable=(not self._verbose))

        for i in epoch_iter:
            g_accum = d_accum = 0.0
            mae_accum = re_accum = lat_accum = 0.0
            z_normal_accum = z_pseudo_accum = 0.0

            for _step in range(steps_per_epoch):

                # ── Discriminator step(s) ──────────────────────────────────
                for _ in range(self._discriminator_steps):
                    fakez   = torch.normal(mean=mean_z, std=std_z)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                    fake    = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if condvec is None:
                        real     = self._data_sampler.sample_data(train_data_t, self._batch_size, col, opt)
                        fake_cat = fakeact
                        real_cat = torch.from_numpy(real.astype("float32")).to(self._device)
                    else:
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(train_data_t, self._batch_size, col[perm], opt[perm])
                        c2       = c1[perm]
                        real     = torch.from_numpy(real.astype("float32")).to(self._device)
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)

                    y_fake_d = self._discriminator(fake_cat)
                    y_real   = self._discriminator(real_cat)
                    pen      = self._discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                    loss_d   = -(torch.mean(y_real) - torch.mean(y_fake_d))
                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # ── Generator step ─────────────────────────────────────────
                fakez   = torch.normal(mean=mean_z, std=std_z)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake    = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake        = self._discriminator(torch.cat([fakeact, c1], dim=1))
                    cross_entropy = self._cond_loss(fake, c1, m1)
                else:
                    y_fake        = self._discriminator(fakeact)
                    cross_entropy = 0

                loss_g = -torch.mean(y_fake) + cross_entropy
                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                g_accum += loss_g.detach().cpu().item()
                d_accum += loss_d.detach().cpu().item()

                # ── M01AE step ─────────────────────────────────────────────
                # Direction 3: gate replaces fixed mae_start_epoch
                use_ctgan = self._ctgan_gate_open and (self.type_train != "noise")

                real_idx     = torch.randperm(normal_tensor_cpu.size(0))[:self._batch_size]
                real_normal  = normal_tensor_cpu[real_idx].to(self._device)
                ctgan_sample = fakeact.detach() if use_ctgan else None
                x_pseudo     = self._make_pseudo_anomaly(real_normal, ctgan_sample, use_ctgan)
                x_normal_mae = real_normal

                x_combined = torch.cat([x_normal_mae, x_pseudo], dim=0)
                y_combined = torch.cat([
                    torch.zeros(x_normal_mae.size(0), device=self._device),
                    torch.ones(x_pseudo.size(0),  device=self._device),
                ], dim=0)

                self._mae_enc.train()
                self._mae_dec.train()
                z, x_hat = self._mae_forward(x_combined)
                re_loss, latent_loss = self._mae_loss(x_combined, z, x_hat, y_combined)
                loss_mae = re_loss + self.lamda * latent_loss

                optimizer_mae.zero_grad()
                loss_mae.backward()
                optimizer_mae.step()

                mae_accum += loss_mae.item()
                re_accum  += re_loss.item()
                lat_accum += latent_loss.item()

                # Direction 2: track z-norms for noise adaptation
                with torch.no_grad():
                    split_n = x_normal_mae.size(0)
                    z_normal_accum += torch.norm(z[:split_n].detach(), p=2, dim=1).mean().item()
                    z_pseudo_accum += torch.norm(z[split_n:].detach(), p=2, dim=1).mean().item()

            # ── End of epoch ───────────────────────────────────────────────
            avg_g   = g_accum   / steps_per_epoch
            avg_d   = d_accum   / steps_per_epoch
            avg_mae = mae_accum / steps_per_epoch
            avg_re  = re_accum  / steps_per_epoch
            avg_lat = lat_accum / steps_per_epoch

            avg_z_normal  = z_normal_accum / steps_per_epoch
            avg_z_pseudo  = z_pseudo_accum / steps_per_epoch
            avg_separation = avg_z_pseudo - avg_z_normal

            val_loss, val_re, val_lat = self._validate_mae(val_loader)
            thresholds = self._compute_thresholds(val_loader)

            # Direction 3: update EMA discriminator loss and check gate
            _ema_disc = avg_d if _ema_disc is None else _ema_alpha * avg_d + (1 - _ema_alpha) * _ema_disc
            disc_loss_history.append(_ema_disc)
            self._maybe_open_gate(i, disc_loss_history)

            # Direction 2: noise_std adaptation (EMA-smoothed separation)
            _ema_sep = avg_separation if _ema_sep is None else _ema_alpha * avg_separation + (1 - _ema_alpha) * _ema_sep
            self._maybe_adapt_noise(i, _ema_sep)

            ctgan_active = int(self._ctgan_gate_open and self.type_train != "noise")
            val_auc, val_sep = self._evaluate_pseudo_validation_auc(val_loader, bool(ctgan_active))

            print(
                f"Epoch {i:03d}: G={avg_g:.3f} D={avg_d:.3f} | "
                f"MAE={avg_mae:.4f} RE={avg_re:.4f} Z={avg_lat:.4f} | "
                f"Val={val_loss:.4f} ValAUC={val_auc:.4f} | "
                f"sep={avg_separation:.3f} noise_std={self.noise_std:.4f} "
                f"{'[CTGAN]' if ctgan_active else '[warmup]'}"
            )

            loss_rows.append({
                "Epoch":                   i,
                "Generator Loss":           avg_g,
                "Discriminator Loss":       avg_d,
                "EMA Discriminator Loss":   _ema_disc,
                "MAE Loss":                avg_mae,
                "Reconstruction Loss":     avg_re,
                "Latent Loss":             avg_lat,
                "Validation Loss":         val_loss,
                "Validation RE Loss":      val_re,
                "Validation Latent Loss":  val_lat,
                "Validation AUC":          val_auc,
                "Validation Separation":   val_sep,
                "CTGAN Active":            ctgan_active,
                "CTGAN Gate Forced":       int(self._ctgan_gate_forced),
                # Adaptive columns
                "Current Noise Std":       self.noise_std,
                "Z Normal Mean":           avg_z_normal,
                "Z Pseudo Mean":           avg_z_pseudo,
                "Z Separation":            avg_separation,
                "EMA Z Separation":        _ema_sep,
            })

            # Best-by-AUC checkpoint (no early stopping)
            if not np.isnan(val_auc) and val_auc > best_val_auc + 1e-8:
                best_val_auc       = val_auc
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
                print(f"  [BEST_AUC] val_auc={val_auc:.6f} at epoch {i}")

            # Best-by-loss + early stopping
            if val_loss < best_val_loss - 1e-8:
                best_val_loss  = val_loss
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
                        f"  Early stopping at epoch {i}. "
                        f"Best val_loss={best_val_loss:.6f}"
                    )
                    break

            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)

        self.loss_values = pd.DataFrame(loss_rows)

        if best_enc_state is not None:
            self._mae_enc.load_state_dict(best_enc_state)
            self._mae_dec.load_state_dict(best_dec_state)
            print(f"Restored best M01AE weights (val_loss={best_val_loss:.6f})")
        self._best_thresholds     = best_thresholds
        self._best_thresholds_auc = best_thresholds_auc
        self._best_mae_state_auc  = (best_enc_state_auc, best_dec_state_auc)
        gate_status = f"epoch {self._ctgan_gate_epoch}" if self._ctgan_gate_open else "never opened"
        print(
            f"Best val_loss={best_val_loss:.6f}  Best val_AUC={best_val_auc:.6f}\n"
            f"CTGAN gate: {gate_status}\n"
            f"Final noise_std: {self.noise_std:.4f}  "
            f"range=[{self.noise_std_min}, {self.noise_std_max}]"
        )
        print("M01AECAdapt training complete.")
