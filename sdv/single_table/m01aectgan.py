"""SDV wrapper for M01AECTGAN (joint CTGAN + M01AE two-sided margin)."""

import os
import cloudpickle

from ctgan.synthesizers.m01aectgan import M01AECTGAN
from sdv.single_table.maectgan import MAECTGANSynthesizer, _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class M01AECTGANSynthesizer(MAECTGANSynthesizer):
    """SDV wrapper for M01AECTGAN.

    Joint training: CTGAN generates pseudo-anomaly base; M01AE two-sided margin
    loss separates normal from anomalous latent representations.

    Parameters (M01AE-specific, different from MAECTGANSynthesizer)
    ---------------------------------------------------------------
    mae_lr : float
        Learning rate for M01AE encoder/decoder.  Default 1e-3.
    noise_std : float
        Std of Gaussian noise added to continuous features.  Default 0.1.
    noise_ratio : float
        Fraction of continuous features perturbed per sample.  Default 1.0.
    noise_type : str  {"none", "global"}
        Continuous noise type.  Default "global".
    cat_noise_type : str  {"none", "shuffle", "flip", "combined"}
        Categorical feature noise type.  Default "none".
    cat_noise_ratio : float
        Fraction of categorical spans perturbed.  Default 1.0.
    type_train : str  {"noise", "std", "noise_std"}
        Pseudo-anomaly source:
          "noise"     → _add_noise(real_normal)
          "std"       → _add_noise(ctgan_sample)
          "noise_std" → 50% noise + 50% _add_noise(ctgan_sample)
    mae_start_ratio : float  [0, 1)
        Fraction of epochs before CTGAN samples are used.  Default 0.25.
    patience : int
        Early stopping patience (epochs without val improvement).  Default 50.

    Removed vs MAECTGANSynthesizer
    --------------------------------
    ctgan_noise_start_ratio, ctgan_noise_std, ctgan_noise_feature_ratio,
    ctgan_pseudo_anomaly_mode, ctgan_feature_outlier_r
    (M01AE handles its own noise; CTGAN is only used as diversity source)
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
        mae_start_ratio=0.25,
        patience=50,
        # ── Misc ─────────────────────────────────────────────────────────
        dataset="dataset",
        path="results",
    ):
        if locales is None:
            locales = ["en_US"]

        # Call grandparent (BaseSingleTableSynthesizer) directly to avoid
        # MAECTGANSynthesizer's __init__ polluting _model_kwargs with its own params.
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
            "epochs": epochs,
            "pac": pac,
            "cuda": cuda,
            # M01AE
            "lamda": lamda,
            "margin": margin,
            "mae_loss_mode": str(mae_loss_mode),
            "mae_lr": float(mae_lr),
            "noise_std": float(noise_std),
            "noise_ratio": float(noise_ratio),
            "noise_type": str(noise_type),
            "noise_r": float(noise_r),
            "noise_k": float(noise_k),
            "cat_noise_type": str(cat_noise_type),
            "cat_noise_ratio": float(cat_noise_ratio),
            # Joint
            "type_train": str(type_train),
            "mae_start_ratio": float(mae_start_ratio),
            "patience": int(patience),
            # Misc
            "dataset": dataset,
            "path": path,
        }

    # ── Core fit ──────────────────────────────────────────────────────────────

    def _fit(self, full_data, processed_data, val_data):
        """Fit M01AECTGAN on preprocessed normal training data."""
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = M01AECTGAN(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )

    # ── Scoring (delegate to M01AECTGAN) ─────────────────────────────────────

    def get_score(self, X_processed, y):
        """Compute anomaly scores.

        Parameters
        ----------
        X_processed : pd.DataFrame or np.ndarray
            SDV-preprocessed data (output of _data_processor.transform).
        y : array-like
            Ground-truth labels (0=normal, 1=anomaly).

        Returns
        -------
        re_errors, z_norm_scores, total_scores, y_true
        """
        return self._model.get_score(X_processed, y)

    def get_threshold(self, X_val_processed):
        """Compute thresholds from SDV-preprocessed normal validation data."""
        return self._model.get_threshold(X_val_processed)

    # ── Checkpoint (reuse parent logic) ──────────────────────────────────────

    def _save_best_model(
        self, model, epoch,
        threshold_re, threshold_z, threshold_95, threshold_95_total_loss,
        checkpoint_kind="loss",
    ):
        """Persist the current best checkpoint.

        checkpoint_kind : str  {"loss", "auc"}
            "loss" → best validation loss (file names unchanged for backward
                     compatibility: MAE_<dataset>_best.pkl).
            "auc"  → best validation AUC (pseudo-anomaly built per type_train):
                     MAE_<dataset>_best_auc.pkl.
        """
        if checkpoint_kind not in {"loss", "auc"}:
            raise ValueError("checkpoint_kind must be 'loss' or 'auc'.")
        suffix = "" if checkpoint_kind == "loss" else f"_{checkpoint_kind}"

        path = f"{self.path}/checkpoints"
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
