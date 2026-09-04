"""SDV wrapper for M01AE2PhaseCTGAN (two-phase: CTGAN then M01AE)."""

import os
import cloudpickle

from ctgan.synthesizers.m01ae_2phase_ctgan import M01AE2PhaseCTGAN
from sdv.single_table.maectgan import MAECTGANSynthesizer, _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class M01AE2PhaseCTGANSynthesizer(MAECTGANSynthesizer):
    """SDV wrapper for M01AE2PhaseCTGAN.

    Two-phase:
      Phase 1 (ctgan_epochs) — standard CTGAN training.
      Phase 2 (mae_epochs)   — M01AE with augmented normal + anomaly:
        Normal  = real_normal + CTGAN samples
        Anomaly = _add_noise(real_normal) + _add_noise(CTGAN samples)
    """

    def __init__(
        self,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=None,
        # ── CTGAN ─────────────────────────────────────────────────────────
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
        # ── Two-phase ─────────────────────────────────────────────────────
        ctgan_epochs=150,
        mae_epochs=100,
        ctgan_sample_ratio=1.0,
        # ── M01AE ─────────────────────────────────────────────────────────
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
        # ── Misc ─────────────────────────────────────────────────────────
        dataset="dataset",
        path="results",
    ):
        if locales is None:
            locales = ["en_US"]

        from sdv.single_table.base import BaseSingleTableSynthesizer
        BaseSingleTableSynthesizer.__init__(
            self,
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )

        self.dataset = dataset
        self.path = path
        self.discrete_column = None

        self._model_kwargs = {
            # CTGAN
            "embedding_dim": embedding_dim,
            "generator_dim": generator_dim,
            "discriminator_dim": discriminator_dim,
            "generator_lr": generator_lr,
            "generator_decay": generator_decay,
            "discriminator_lr": discriminator_lr,
            "discriminator_decay": discriminator_decay,
            "batch_size": batch_size,
            "discriminator_steps": discriminator_steps,
            "log_frequency": log_frequency,
            "verbose": verbose,
            "pac": pac,
            "cuda": cuda,
            # Two-phase
            "ctgan_epochs": int(ctgan_epochs),
            "mae_epochs": int(mae_epochs),
            "ctgan_sample_ratio": float(ctgan_sample_ratio),
            # M01AE
            "lamda": float(lamda),
            "margin": float(margin),
            "mae_loss_mode": str(mae_loss_mode),
            "mae_lr": float(mae_lr),
            "noise_std": float(noise_std),
            "noise_ratio": float(noise_ratio),
            "noise_type": str(noise_type),
            "noise_r": float(noise_r),
            "noise_k": float(noise_k),
            "cat_noise_type": str(cat_noise_type),
            "cat_noise_ratio": float(cat_noise_ratio),
            "patience": int(patience),
            # Misc
            "dataset": dataset,
            "path": path,
        }

    # ── Core fit ──────────────────────────────────────────────────────────────

    def _fit(self, full_data, processed_data, val_data):
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = M01AE2PhaseCTGAN(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def get_score(self, X_processed, y):
        return self._model.get_score(X_processed, y)

    def get_threshold(self, X_val_processed):
        return self._model.get_threshold(X_val_processed)

    # ── Loss access ───────────────────────────────────────────────────────────

    def get_loss_values(self):
        """Return (ctgan_loss_df, mae_loss_df)."""
        return self._model.get_loss_values()

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def _save_best_model(
        self, model, epoch,
        threshold_re, threshold_z, threshold_95, threshold_95_total_loss,
    ):
        path = f"{self.path}/checkpoints"
        os.makedirs(path, exist_ok=True)
        path_file      = f"{path}/MAE_{self.dataset}_best.pkl"
        path_epoch     = f"{path}/MAE_{self.dataset}_best_epoch.txt"
        threshold_path = f"{path}/MAE_{self.dataset}_thresholds_best.pkl"

        with open(path_file, "wb") as f:
            cloudpickle.dump(self, f)
        with open(path_epoch, "w") as f:
            f.write(str(epoch))
        with open(threshold_path, "wb") as f:
            cloudpickle.dump(
                (threshold_re, threshold_z, threshold_95, threshold_95_total_loss), f
            )
        print(f"[BEST] Saved best model at MAE epoch {epoch}: {path_file}")
