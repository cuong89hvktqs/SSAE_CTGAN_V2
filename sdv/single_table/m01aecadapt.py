"""SDV wrapper for M01AECAdapt (adaptive noise_std + CTGAN quality gate)."""

import os
import cloudpickle

from ctgan.synthesizers.m01aecadapt import M01AECAdapt
from sdv.single_table.m01aectgan import M01AECTGANSynthesizer
from sdv.single_table.maectgan import _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class M01AECAdaptSynthesizer(M01AECTGANSynthesizer):
    """SDV wrapper for M01AECAdapt.

    Extends M01AECTGANSynthesizer with two adaptive mechanisms:

    Direction 2 — Adaptive noise_std:
        noise_std_min, noise_std_max  — bounds for adaptation
        noise_adapt_every             — adapt every N epochs
        noise_adapt_factor            — multiplicative step (±%)
        noise_target_lo_frac          — lower target as fraction of margin
        noise_target_hi_frac          — upper target as fraction of margin
        noise_adapt_warmup            — epochs before adaptation starts

    Direction 3 — CTGAN quality gate (replaces mae_start_ratio):
        ctgan_gate_window             — rolling window size for gen-loss CoV
        ctgan_gate_stability          — CoV threshold to open gate
        ctgan_gate_min_epochs         — minimum epochs before gate can open
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
        noise_std=0.1,
        noise_ratio=1.0,
        noise_type="global",
        noise_r=1.0,
        noise_k=3.0,
        cat_noise_type="none",
        cat_noise_ratio=1.0,
        # ── Joint ─────────────────────────────────────────────────────────
        type_train="noise_std",
        mae_start_ratio=0.0,       # unused in CAdapt — gate is the only trigger
        patience=50,
        # ── Misc ──────────────────────────────────────────────────────────
        dataset="dataset",
        path="results",
        # ── Direction 2 — Adaptive noise_std ──────────────────────────────
        noise_std_min=0.01,
        noise_std_max=2.0,
        noise_adapt_every=5,
        noise_adapt_factor=0.15,
        noise_target_lo_frac=0.5,
        noise_target_hi_frac=0.9,
        noise_adapt_warmup=10,
        # ── Direction 3 — CTGAN quality gate ──────────────────────────────
        ctgan_gate_window=10,
        ctgan_gate_stability=0.05,
        ctgan_gate_min_epochs=10,
    ):
        # Build parent (sets self._model_kwargs with M01AECTGAN params)
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
        )

        # Extend _model_kwargs with adaptive params
        self._model_kwargs.update({
            "noise_std_min":        float(noise_std_min),
            "noise_std_max":        float(noise_std_max),
            "noise_adapt_every":    int(noise_adapt_every),
            "noise_adapt_factor":   float(noise_adapt_factor),
            "noise_target_lo_frac": float(noise_target_lo_frac),
            "noise_target_hi_frac": float(noise_target_hi_frac),
            "noise_adapt_warmup":   int(noise_adapt_warmup),
            "ctgan_gate_window":    int(ctgan_gate_window),
            "ctgan_gate_stability": float(ctgan_gate_stability),
            "ctgan_gate_min_epochs":int(ctgan_gate_min_epochs),
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

        self._model = M01AECAdapt(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )
