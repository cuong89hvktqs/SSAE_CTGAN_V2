"""MAE01ScaleMax — MAE01Scale with fixed CoV denominator.

Identical to MAE01Scale except the CTGAN gate uses:
    cov = std(window) / max(|mean(window)|, cov_floor)

instead of:
    cov = std(window) / (|mean(window)| + 1e-8)

Fix: when Discriminator loss converges near 0 (e.g. NSL-KDD), the old
denominator 1e-8 makes CoV astronomically large -> gate never opens.
Clamping to cov_floor (default 0.05) preserves scale-invariance for
large |mean| while gracefully handling near-zero cases.
"""

import numpy as np

from ctgan.synthesizers.mae01scale import MAE01Scale


class MAE01ScaleMax(MAE01Scale):
    """MAE01Scale with max(|mean|, cov_floor) CoV denominator fix."""

    def __init__(
        self,
        # All MAE01Scale params
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
        noise_std=0.3,
        noise_ratio=1.0,
        noise_type="global",
        noise_r=1.0,
        noise_k=3.0,
        cat_noise_type="none",
        cat_noise_ratio=1.0,
        type_train="noise_std",
        mae_start_ratio=0.0,
        patience=50,
        dataset="",
        path="",
        noise_scale=1.5,
        gate_stability=0.05,
        gate_window=10,
        max_clusters=10,
        snapshot_stride=1,
        # ── Fix ───────────────────────────────────────────────────────────
        cov_floor=0.05,         # min denominator for CoV: max(|mean|, cov_floor)
        # ── Near-zero equilibrium branch (OR gate) ────────────────────────
        eps_zero=0.1,           # |mean(EMA disc window)| <= eps_zero
        tol=0.08,               # std(EMA disc window) <= tol
        # ── Fixed-epoch gate override (bypasses the adaptive CoV/near-zero
        # logic entirely) — for comparing the dynamic gate against opening
        # the CTGAN gate at a fixed epoch, e.g. 0/30/50. None (default)
        # keeps the normal adaptive behavior unchanged.
        gate_fixed_epoch=None,
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
            noise_scale=noise_scale,
            gate_stability=gate_stability,
            gate_window=gate_window,
            max_clusters=max_clusters,
            snapshot_stride=snapshot_stride,
        )
        self.cov_floor = float(cov_floor)
        self.eps_zero  = float(eps_zero)
        self.tol       = float(tol)
        self.gate_fixed_epoch = None if gate_fixed_epoch is None else int(gate_fixed_epoch)

    # ── Override: skip the periodic generated_data_snapshots.csv dump ──────
    # (visualization-only artifact from MAE01Scale; disabled here to reduce
    # per-run disk usage across many seeds -- MAE01Scale itself is untouched).

    def _save_generated_snapshot(self, epoch, normal_tensor_cpu, chunk=4096):
        pass

    # ── Only override: OR gate (CoV branch  OR  near-zero equilibrium branch) ──

    def _maybe_open_gate(self, epoch, disc_loss_history):
        if self._ctgan_gate_open or self.type_train == "noise":
            return

        if self.gate_fixed_epoch is not None:
            if epoch >= self.gate_fixed_epoch:
                self._ctgan_gate_open   = True
                self._ctgan_gate_epoch  = epoch
                self._ctgan_gate_forced = True
                print(
                    f"  [CTGAN GATE] Opened at epoch {epoch} via [fixed] "
                    f"(gate_fixed_epoch={self.gate_fixed_epoch})"
                )
            return

        min_needed = max(self.ctgan_gate_window, self.ctgan_gate_min_epochs)
        if epoch < min_needed - 1 or len(disc_loss_history) < self.ctgan_gate_window:
            return

        window = disc_loss_history[-self.ctgan_gate_window:]
        mean_w = float(np.mean(window))
        std_w  = float(np.std(window))

        # (i) CoV branch — disc_loss converged to a non-zero plateau
        cov = std_w / max(abs(mean_w), self.cov_floor)   # cov_floor guards div-by-0
        cov_ok = cov <= self.ctgan_gate_stability

        # (ii) Near-zero equilibrium branch — disc_loss parked around 0 (WGAN converged)
        zero_ok = (abs(mean_w) <= self.eps_zero) and (std_w <= self.tol)

        if cov_ok or zero_ok:
            self._ctgan_gate_open   = True
            self._ctgan_gate_epoch  = epoch
            self._ctgan_gate_forced = False
            branch = "CoV" if cov_ok else "near-zero"
            print(
                f"  [CTGAN GATE] Opened at epoch {epoch} via [{branch}] "
                f"(mean={mean_w:+.4f}, std={std_w:.4f}, CoV={cov:.4f} | "
                f"gs={self.ctgan_gate_stability}, floor={self.cov_floor}, "
                f"eps_zero={self.eps_zero}, tol={self.tol})"
            )
