"""SDV wrapper for MAE01Scale (z-norm self-scaling noise)."""

import os
import cloudpickle

from ctgan.synthesizers.mae01scale import MAE01Scale
from sdv.single_table.m01aecadapt import M01AECAdaptSynthesizer
from sdv.single_table.maectgan import _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns

_DIRECTION2_KEYS = [
    "noise_std_min", "noise_std_max", "noise_adapt_every",
    "noise_adapt_factor", "noise_target_lo_frac", "noise_target_hi_frac",
    "noise_adapt_warmup", "ctgan_gate_min_epochs",
    "ctgan_gate_window", "ctgan_gate_stability",
]


class MAE01ScaleSynthesizer(M01AECAdaptSynthesizer):
    """SDV wrapper for MAE01Scale.

    Extends M01AECTGANSynthesizer with z-norm self-scaling noise and a
    data-driven CTGAN quality gate.  Replaces the 10 adaptive parameters
    of M01AECAdaptSynthesizer with 3:

        noise_scale    -- multiplier: noise_std = noise_scale x mean||z_normal||
        gate_stability -- EMA disc_loss CoV threshold to open gate
        gate_window    -- rolling window AND min-epochs for gate (merged)

    Also exposes max_clusters (default 10, unchanged CTGAN default): the
    max number of Gaussian-mixture modes DataTransformer uses per continuous
    column (each continuous column becomes 1 + max_clusters transformed
    dims). Lower it for wide datasets with many continuous columns (e.g.
    nb_iot's 115) to keep the transformed dimensionality manageable.
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
        epochs=300,
        pac=10,
        cuda=True,
        # ── M01AE ─────────────────────────────────────────────────────────
        lamda=0.1,
        margin=1.0,
        mae_loss_mode="renormal",
        mae_lr=1e-3,
        noise_std=0.3,          # initial noise before gate opens
        noise_ratio=1.0,
        noise_type="global",
        noise_r=1.0,
        noise_k=3.0,
        cat_noise_type="none",
        cat_noise_ratio=1.0,
        # ── Joint ─────────────────────────────────────────────────────────
        type_train="noise_std",
        mae_start_ratio=0.0,    # API compat only, unused
        patience=50,
        # ── Misc ──────────────────────────────────────────────────────────
        dataset="dataset",
        path="results",
        # ── Scale params ──────────────────────────────────────────────────
        noise_scale=1.5,
        gate_stability=0.05,
        gate_window=10,
        # ── CTGAN transform ──────────────────────────────────────────────
        max_clusters=10,
        # ── Generated-data snapshot ───────────────────────────────────────
        snapshot_stride=1,
    ):
        # Call parent with dummy Direction-2 values (they get cleaned up below)
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
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
            # Direction-2 dummy values
            noise_std_min=1e-9,
            noise_std_max=1e9,
            noise_adapt_every=999999,
            noise_adapt_factor=0.0,
            noise_target_lo_frac=0.5,
            noise_target_hi_frac=0.9,
            noise_adapt_warmup=999999,
            ctgan_gate_window=int(gate_window),
            ctgan_gate_stability=float(gate_stability),
            ctgan_gate_min_epochs=int(gate_window),
        )

        # Remove Direction-2 keys — MAE01Scale does not accept them
        for k in _DIRECTION2_KEYS:
            self._model_kwargs.pop(k, None)

        # Add MAE01Scale-specific params
        self._model_kwargs.update({
            "noise_scale":     float(noise_scale),
            "gate_stability":  float(gate_stability),
            "gate_window":     int(gate_window),
            "max_clusters":    int(max_clusters),
            "snapshot_stride": int(snapshot_stride),
        })

    # ── Core fit ──────────────────────────────────────────────────────────────

    def _fit(self, full_data, processed_data, val_data):
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = MAE01Scale(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )
