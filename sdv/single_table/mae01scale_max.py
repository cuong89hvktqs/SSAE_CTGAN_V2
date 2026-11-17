"""SDV wrapper for MAE01ScaleMax (CoV-floor fix)."""

import os

import cloudpickle

from ctgan.synthesizers.mae01scale_max import MAE01ScaleMax
from sdv.single_table.mae01scale import MAE01ScaleSynthesizer
from sdv.single_table.maectgan import _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class MAE01ScaleMaxSynthesizer(MAE01ScaleSynthesizer):
    """MAE01Scale with max(|mean|, cov_floor) CoV denominator fix.

    Adds four params vs MAE01ScaleSynthesizer:
        cov_floor        -- min denominator for CoV gate check (default 0.05)
        eps_zero         -- |mean(EMA disc window)| threshold, near-zero branch (default 0.1)
        tol              -- std(EMA disc window) threshold, near-zero branch (default 0.08)
        gate_fixed_epoch -- if set, bypasses the adaptive CoV/near-zero gate
                             entirely and opens the CTGAN gate at this fixed
                             epoch instead (e.g. 0/30/50, for ablating the
                             dynamic gate against a fixed schedule). None
                             (default) keeps the normal adaptive behavior.
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
        max_clusters=10,
        snapshot_stride=1,
        cov_floor=0.05,
        eps_zero=0.1,
        tol=0.08,
        gate_fixed_epoch=None,
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
            max_clusters=max_clusters,
            snapshot_stride=snapshot_stride,
        )
        self._model_kwargs["cov_floor"] = float(cov_floor)
        self._model_kwargs["eps_zero"] = float(eps_zero)
        self._model_kwargs["tol"] = float(tol)
        self._model_kwargs["gate_fixed_epoch"] = (
            None if gate_fixed_epoch is None else int(gate_fixed_epoch)
        )

    def _save_best_model(
        self, model, epoch,
        threshold_re, threshold_z, threshold_95, threshold_95_total_loss,
        checkpoint_kind="loss",
    ):
        """Persist the current best checkpoint directly under self.path
        (no "checkpoints" subfolder) -- keeps per-run disk paths shorter,
        scoped to MAE01ScaleMax only (see m01aectgan.py's _save_best_model
        for the shared/original behavior used by other synthesizers).
        """
        if checkpoint_kind not in {"loss", "auc"}:
            raise ValueError("checkpoint_kind must be 'loss' or 'auc'.")
        suffix = "" if checkpoint_kind == "loss" else f"_{checkpoint_kind}"

        path = self.path
        os.makedirs(path, exist_ok=True)
        path_file = f"{path}/MAE_{self.dataset}_best{suffix}.pkl"
        path_epoch = f"{path}/MAE_{self.dataset}_best{suffix}_epoch.txt"
        threshold_path = f"{path}/MAE_{self.dataset}_thresholds_best{suffix}.pkl"

        with open(path_file, "wb") as f:
            cloudpickle.dump(self, f)
        with open(path_epoch, "w") as f:
            f.write(str(epoch))
        with open(threshold_path, "wb") as f:
            cloudpickle.dump(
                (threshold_re, threshold_z, threshold_95, threshold_95_total_loss), f
            )
        print(f"[BEST:{checkpoint_kind}] Saved best model at epoch {epoch}: {path_file}")

    def _fit(self, full_data, processed_data, val_data):
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = MAE01ScaleMax(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )
