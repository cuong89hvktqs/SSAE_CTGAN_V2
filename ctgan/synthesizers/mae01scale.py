"""MAE01Scale — z-norm self-scaling noise variant of M01AECTGAN.

Key idea (Direction B):
    noise_std_effective = noise_scale x max(mean||z_normal||, margin x 0.05)

When gate closed: noise_std stays fixed at initial value.
When gate opens:  noise_std automatically tracks how compressed normal embeddings are.
  - Model early / bad:  mean||z_normal|| large -> noise large (easy pseudo-anomalies)
  - Model later / good: mean||z_normal|| near 0 -> noise small (hard pseudo-anomalies)

This gives implicit curriculum learning with no target zones, no warmup scheduling.

New params vs M01AECTGAN (net +2, removed mae_start_ratio):
    noise_scale    -- multiplier on mean z-norm  (default 1.5)
    gate_stability -- EMA disc_loss CoV threshold to open gate (default 0.05)
    gate_window    -- rolling window AND min-epoch for gate, merged (default 10)
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.m01aecadapt import M01AECAdapt


class MAE01Scale(M01AECAdapt):
    """Z-norm self-scaling noise: noise_std = noise_scale x mean||z_normal||."""

    def __init__(
        self,
        # ── Standard M01AECTGAN params ────────────────────────────────────
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
        noise_std=0.3,          # initial noise (gate closed); overridden after gate opens
        noise_ratio=1.0,
        noise_type="global",
        noise_r=1.0,
        noise_k=3.0,
        cat_noise_type="none",
        cat_noise_ratio=1.0,
        type_train="noise_std",
        mae_start_ratio=0.0,    # API compat only, unused
        patience=50,
        dataset="",
        path="",
        # ── Scale params (3 params total, replaces 10 from M01AECAdapt) ──
        noise_scale=1.5,        # noise_std = noise_scale x mean||z_normal||
        gate_stability=0.05,    # EMA disc_loss CoV threshold
        gate_window=10,         # rolling window + min-epoch (merged into one)
        # ── CTGAN transform ────────────────────────────────────────────────
        max_clusters=10,        # DataTransformer mode-specific normalization:
                                 # each continuous column -> 1 + max_clusters dims.
                                 # Lower for wide datasets (many continuous cols,
                                 # e.g. nb_iot's 115) to avoid huge transformed dim.
        # ── Generated-data snapshot ────────────────────────────────────────
        snapshot_stride=1,      # take every Nth normal-train row per snapshot
                                 # (1 = all rows). Raise for very large datasets
                                 # (e.g. nb_iot) so generated_data_snapshots.csv
                                 # doesn't explode in size.
    ):
        # Call parent with Direction-2 disabled (dummy values that never trigger)
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
            # Direction-2: disabled — z-norm scaling overrides in fit()
            noise_std_min=1e-9,
            noise_std_max=1e9,
            noise_adapt_every=999999,
            noise_adapt_factor=0.0,
            noise_target_lo_frac=0.5,
            noise_target_hi_frac=0.9,
            noise_adapt_warmup=999999,
            # Gate: gate_window used as both window and min-epoch
            ctgan_gate_window=int(gate_window),
            ctgan_gate_stability=float(gate_stability),
            ctgan_gate_min_epochs=int(gate_window),
        )
        self.noise_scale         = float(noise_scale)
        self._initial_noise_std  = float(noise_std)
        self._gate_window_merged = int(gate_window)
        self._gate_stability_val = float(gate_stability)
        self.max_clusters        = int(max_clusters)
        self.snapshot_stride     = max(int(snapshot_stride), 1)
        # dataset/path are accepted by the M01AECTGAN base __init__ but never
        # stored there (only the SDV wrapper keeps them, for checkpoint
        # naming) -- store them here too so _save_generated_snapshot() can
        # find this run's results directory from inside the inner model.
        self.dataset = dataset
        self.path    = path

    # =========================================================================
    # Periodic generated-data snapshot (every SNAPSHOT_EVERY epochs, only once
    # the CTGAN gate has opened) -- for later visualization of how CTGAN's
    # generated distribution and the noise-augmented pseudo-anomalies compare
    # to the real normal data.
    # =========================================================================

    SNAPSHOT_EVERY = 5

    def _save_generated_snapshot(self, epoch, normal_tensor_cpu, chunk=4096):
        """Append per-source snapshots of 4 sources to one growing CSV:
            real_normal   -- real training-normal rows, as-is
            noise_pseudo  -- real_normal + noise (the "noise" pseudo-anomaly)
            ctgan_raw     -- fresh CTGAN-generated rows, unperturbed
            ctgan_noise   -- CTGAN-generated rows + noise (the "std" pseudo-anomaly)

        Saved via DataTransformer.inverse_transform() -- i.e. the SAME
        HyperTransformer-space format as X_test.csv / X_val_normal.csv, so
        visualize_scalemax.py's get_latent_z() can encode these rows
        unchanged, side by side with real test normal/attack samples.

        By default takes ALL normal-train rows (x4 sources) every
        SNAPSHOT_EVERY epochs -- for large datasets (e.g. nb_iot, ~210k
        normal rows) this file grows very large over a full run (tens of
        millions of rows), so self.snapshot_stride (default 1) can subsample
        every Nth row (e.g. 20 -> 1/20 of rows) to keep it manageable.
        """
        normal_tensor_cpu = normal_tensor_cpu[::self.snapshot_stride]
        n_total = normal_tensor_cpu.size(0)
        real_chunks, ctgan_chunks, noise_chunks, ctgan_noise_chunks = [], [], [], []

        with torch.no_grad():
            for start in range(0, n_total, chunk):
                end = min(start + chunk, n_total)
                real_batch  = normal_tensor_cpu[start:end].to(self._device)
                ctgan_batch = self._generate_ctgan_fakeact(end - start)

                real_chunks.append(real_batch.cpu())
                ctgan_chunks.append(ctgan_batch.cpu())
                noise_chunks.append(self._add_noise_transformed(real_batch).cpu())
                ctgan_noise_chunks.append(self._add_noise_transformed(ctgan_batch).cpu())

        sources = {
            "real_normal":  torch.cat(real_chunks,        dim=0).numpy(),
            "noise_pseudo": torch.cat(noise_chunks,       dim=0).numpy(),
            "ctgan_raw":    torch.cat(ctgan_chunks,       dim=0).numpy(),
            "ctgan_noise":  torch.cat(ctgan_noise_chunks, dim=0).numpy(),
        }

        frames = []
        for source, arr in sources.items():
            df = self._transformer.inverse_transform(arr)
            df.insert(0, "source", source)
            df.insert(0, "epoch", epoch)
            frames.append(df)

        combined  = pd.concat(frames, ignore_index=True)
        out_path  = f"{self.path}/generated_data_snapshots.csv"
        first_write = not os.path.exists(out_path)
        combined.to_csv(out_path, mode="a", header=first_write, index=False)
        print(
            f"  [SNAPSHOT] epoch {epoch}: appended {len(combined)} rows "
            f"({n_total} x 4 sources, stride={self.snapshot_stride}) -> {out_path}"
        )

    # =========================================================================
    # Training loop — same as M01AECAdapt but with z-norm scaling instead
    # of separation-based Direction-2 adaptation
    # =========================================================================

    @random_state
    def fit(self, full_data, train_data, val_data, discrete_columns=()):
        from ctgan.data_sampler import DataSampler
        from ctgan.data_transformer import DataTransformer
        from ctgan.synthesizers.maectgan import Discriminator, Generator

        print(
            "Starting MAE01Scale training...",
            " epochs:", self._epochs,
            " type_train:", self.type_train,
            " noise_std (init):", self._initial_noise_std,
            " noise_scale:", self.noise_scale,
            " gate_window:", self._gate_window_merged,
            " gate_stability:", self._gate_stability_val,
            " max_clusters:", self.max_clusters,
        )

        # Reset adaptive state
        self._ctgan_gate_open   = False
        self._ctgan_gate_epoch  = None
        self._ctgan_gate_forced = False
        self._adapt_log         = []
        self.noise_std          = self._initial_noise_std

        self._validate_discrete_columns(train_data, discrete_columns)

        # ── Data preparation ───────────────────────────────────────────────
        self._transformer = DataTransformer(max_clusters=self.max_clusters)
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

        mean_z            = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std_z             = mean_z + 1
        steps_per_epoch   = max(len(train_data_t) // self._batch_size, 1)
        normal_tensor_cpu = torch.tensor(train_data_t, dtype=torch.float32)

        loss_rows        = []
        best_val_loss    = float("inf")
        best_enc_state   = best_dec_state = best_thresholds = None
        patience_counter = 0

        best_val_auc       = -float("inf")
        best_enc_state_auc = best_dec_state_auc = best_thresholds_auc = None

        disc_loss_history = []
        _ema_alpha = 0.3
        _ema_disc  = None

        # ── Optional component profiler (SSAE_PROFILE=1) ───────────────────
        # Opt-in, env-var controlled, zero effect on the default path. Times
        # the per-step blocks (CTGAN D+G updates, pseudo-intrusion generation,
        # MiAE update) and the per-epoch blocks (validation / checkpoint
        # selection, gate + adaptive-scale bookkeeping) with cuda.synchronize()
        # fences, and records peak GPU memory. Rows go to
        # <self.path>/profile_components.csv at the end of fit().
        import os as _os, time as _time
        _profile = _os.environ.get("SSAE_PROFILE", "0") == "1"
        _prof_rows = []
        _cuda = torch.cuda.is_available() and str(self._device).startswith("cuda")
        def _now():
            if _cuda:
                torch.cuda.synchronize()
            return _time.perf_counter()
        if _profile and _cuda:
            torch.cuda.reset_peak_memory_stats()
        _t_fit_start = _time.perf_counter()

        epoch_iter = tqdm(range(self._epochs), disable=(not self._verbose))

        for i in epoch_iter:
            g_accum = d_accum = 0.0
            mae_accum = re_accum = lat_accum = 0.0
            z_normal_accum = z_pseudo_accum = 0.0
            _t_ctgan = _t_pag = _t_mae = 0.0
            _t_epoch0 = _time.perf_counter()

            for _step in range(steps_per_epoch):
                if _profile: _t0 = _now()

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
                if _profile: _t1 = _now(); _t_ctgan += _t1 - _t0

                # ── M01AE step ─────────────────────────────────────────────
                use_ctgan    = self._ctgan_gate_open and (self.type_train != "noise")
                real_idx     = torch.randperm(normal_tensor_cpu.size(0))[:self._batch_size]
                real_normal  = normal_tensor_cpu[real_idx].to(self._device)
                ctgan_sample = fakeact.detach() if use_ctgan else None
                x_pseudo     = self._make_pseudo_anomaly(real_normal, ctgan_sample, use_ctgan)
                x_normal_mae = real_normal
                if _profile: _t2 = _now(); _t_pag += _t2 - _t1

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

                # Track z-norms for scaling signal
                with torch.no_grad():
                    split_n = x_normal_mae.size(0)
                    z_normal_accum += torch.norm(z[:split_n].detach(), p=2, dim=1).mean().item()
                    z_pseudo_accum += torch.norm(z[split_n:].detach(), p=2, dim=1).mean().item()
                if _profile: _t_mae += _now() - _t2

            # ── End of epoch ───────────────────────────────────────────────
            avg_g   = g_accum   / steps_per_epoch
            avg_d   = d_accum   / steps_per_epoch
            avg_mae = mae_accum / steps_per_epoch
            avg_re  = re_accum  / steps_per_epoch
            avg_lat = lat_accum / steps_per_epoch

            avg_z_normal = z_normal_accum / steps_per_epoch
            avg_z_pseudo = z_pseudo_accum / steps_per_epoch

            if _profile: _t3 = _now()
            val_loss, val_re, val_lat = self._validate_mae(val_loader)
            thresholds = self._compute_thresholds(val_loader)
            if _profile: _t4 = _now(); _t_val = _t4 - _t3

            # Gate: EMA disc_loss CoV check
            _ema_disc = avg_d if _ema_disc is None else _ema_alpha * avg_d + (1 - _ema_alpha) * _ema_disc
            disc_loss_history.append(_ema_disc)
            self._maybe_open_gate(i, disc_loss_history)

            # Z-norm scaling: noise_std = noise_scale x max(mean||z_normal||, margin*0.05)
            if self._ctgan_gate_open:
                noise_floor    = self.margin * 0.05
                self.noise_std = self.noise_scale * max(avg_z_normal, noise_floor)
            else:
                self.noise_std = self._initial_noise_std

            if _profile: _t5 = _now(); _t_gate = _t5 - _t4
            ctgan_active = int(self._ctgan_gate_open and self.type_train != "noise")
            val_auc, val_sep = self._evaluate_pseudo_validation_auc(val_loader, bool(ctgan_active))
            if _profile: _t6 = _now(); _t_val += _t6 - _t5

            # Periodic generated-data snapshot -- only once the gate is open
            if self._ctgan_gate_open and (i % self.SNAPSHOT_EVERY == 0):
                self._save_generated_snapshot(i, normal_tensor_cpu)

            print(
                f"Epoch {i:03d}: G={avg_g:.3f} D={avg_d:.3f} | "
                f"MAE={avg_mae:.4f} RE={avg_re:.4f} Z={avg_lat:.4f} | "
                f"Val={val_loss:.4f} ValAUC={val_auc:.4f} | "
                f"z_n={avg_z_normal:.3f} noise={self.noise_std:.4f} "
                f"{'[CTGAN]' if ctgan_active else '[warmup]'}"
            )

            loss_rows.append({
                "Epoch":                  i,
                "Generator Loss":          avg_g,
                "Discriminator Loss":      avg_d,
                "EMA Discriminator Loss":  _ema_disc,
                "MAE Loss":               avg_mae,
                "Reconstruction Loss":    avg_re,
                "Latent Loss":            avg_lat,
                "Validation Loss":        val_loss,
                "Validation RE Loss":     val_re,
                "Validation Latent Loss": val_lat,
                "Validation AUC":         val_auc,
                "Validation Separation":  val_sep,
                "CTGAN Active":           ctgan_active,
                "Current Noise Std":      self.noise_std,
                "Z Normal Mean":          avg_z_normal,
                "Z Pseudo Mean":          avg_z_pseudo,
            })

            # Best-by-AUC checkpoint
            if not np.isnan(val_auc) and val_auc > best_val_auc + 1e-8:
                best_val_auc        = val_auc
                best_enc_state_auc  = copy.deepcopy(self._mae_enc.state_dict())
                best_dec_state_auc  = copy.deepcopy(self._mae_dec.state_dict())
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

            # Best-by-loss + early stopping (checkpoint saving to disk disabled
            # below -- only the best_auc checkpoint is persisted now, to keep
            # per-run disk usage down across many seeds. The in-memory
            # best_val_loss/patience tracking is kept unchanged since early
            # stopping still depends on it.)
            if val_loss < best_val_loss - 1e-8:
                best_val_loss    = val_loss
                best_enc_state   = copy.deepcopy(self._mae_enc.state_dict())
                best_dec_state   = copy.deepcopy(self._mae_dec.state_dict())
                best_thresholds  = thresholds
                patience_counter = 0
                # if self._on_epoch_best_callbacks is not None:
                #     self._on_epoch_best_callbacks(
                #         model=self, epoch=i, checkpoint_kind="loss",
                #         threshold_re=thresholds["threshold_re"],
                #         threshold_z=thresholds["threshold_z"],
                #         threshold_95=thresholds["threshold_95_latent"],
                #         threshold_95_total_loss=thresholds["threshold_95_total"],
                #     )
                print(f"  [BEST] val_loss={val_loss:.6f} at epoch {i}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {i}. Best val_loss={best_val_loss:.6f}")
                    break

            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)
            if _profile:
                _t7 = _now()
                _t_epoch = _t7 - _t_epoch0
                _t_ckpt = _t7 - _t6
                _prof_rows.append({
                    "Epoch": i, "gate_open": int(self._ctgan_gate_open),
                    "steps_per_epoch": steps_per_epoch,
                    "t_ctgan_s": _t_ctgan, "t_pag_s": _t_pag, "t_mae_s": _t_mae,
                    "t_val_s": _t_val, "t_gate_s": _t_gate, "t_ckpt_s": _t_ckpt,
                    "t_epoch_s": _t_epoch,
                    "peak_mem_MB": (torch.cuda.max_memory_allocated() / 2**20) if _cuda else float("nan"),
                })

        self.loss_values = pd.DataFrame(loss_rows)
        if _profile:
            _df = pd.DataFrame(_prof_rows)
            _df.attrs["t_prep_s"] = 0.0
            _out_dir = getattr(self, "path", None) or "."
            _df.to_csv(_os.path.join(_out_dir, "profile_components.csv"), index=False)
            print(f"[profile] wrote {_os.path.join(_out_dir, 'profile_components.csv')}  "
                  f"total fit time {_time.perf_counter() - _t_fit_start:.1f}s")

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
            f"noise_scale={self.noise_scale}  Final noise_std={self.noise_std:.4f}"
        )
        print("MAE01Scale training complete.")
