"""SDV wrapper for MAE01ScaleHard (boundary-band hard-negative mining)."""

from ctgan.synthesizers.mae01scale_hard import MAE01ScaleHard
from sdv.single_table.mae01scale_max import MAE01ScaleMaxSynthesizer
from sdv.single_table.maectgan import _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class MAE01ScaleHardSynthesizer(MAE01ScaleMaxSynthesizer):
    """MAE01ScaleMax + soft boundary-band hard-negative mining on CTGAN rows.

    Adds six params vs MAE01ScaleMaxSynthesizer:
        hard_mining      -- enable the mining resampling (default True;
                             False falls back to exact MAE01ScaleMax behavior)
        floor_k          -- floor = floor_k * mean(||z_normal||) (default 1.0)
        ceiling_frac     -- ceiling = ceiling_frac * margin (default 0.7)
        tau_pos          -- band peak position in [floor, ceiling] (default 0.5)
        floor_sharpness  -- >=1, wider zero dead-zone above floor (default 1.5)
        min_gap_frac     -- min (ceiling-floor) as a fraction of margin (default 0.05)
    """

    def __init__(
        self,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=None,
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
        dataset="dataset",
        path="results",
        noise_scale=1.5,
        gate_stability=0.05,
        gate_window=10,
        cov_floor=0.05,
        eps_zero=0.1,
        tol=0.08,
        hard_mining=True,
        floor_k=1.0,
        ceiling_frac=0.7,
        tau_pos=0.5,
        floor_sharpness=1.5,
        min_gap_frac=0.05,
    ):
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
            noise_scale=noise_scale,
            gate_stability=gate_stability,
            gate_window=gate_window,
            cov_floor=cov_floor,
            eps_zero=eps_zero,
            tol=tol,
        )
        self._model_kwargs["hard_mining"]     = bool(hard_mining)
        self._model_kwargs["floor_k"]         = float(floor_k)
        self._model_kwargs["ceiling_frac"]    = float(ceiling_frac)
        self._model_kwargs["tau_pos"]         = float(tau_pos)
        self._model_kwargs["floor_sharpness"] = float(floor_sharpness)
        self._model_kwargs["min_gap_frac"]    = float(min_gap_frac)

    def _fit(self, full_data, processed_data, val_data):
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = MAE01ScaleHard(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )
