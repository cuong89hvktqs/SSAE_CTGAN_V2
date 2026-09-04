"""SDV wrapper for MAEAfterCTGAN (two-phase: CTGAN fully first, then MAE)."""

import os
import cloudpickle

from ctgan.synthesizers.maeafterctgan import MAEAfterCTGAN
from sdv.single_table.maectgan import MAECTGANSynthesizer, _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class MAEAfterCTGANSynthesizer(MAECTGANSynthesizer):
    """SDV wrapper for MAEAfterCTGAN.

    Two-phase training: CTGAN is fully trained first (Phase 1), then pseudo-anomaly
    samples are generated once from the converged CTGAN (Phase 2), and finally MAE is
    trained on that fixed pool (Phase 3).

    New / changed parameters vs MAECTGANSynthesizer
    ------------------------------------------------
    mae_epochs : int
        Number of epochs for MAE training (Phase 3).  Default 100.
    pseudo_anomaly_ratio : float
        K% — pseudo-anomaly pool size as a percentage of normal training data.
        100.0 = 1 : 1 ratio, 200.0 = 2 : 1.  Default 100.0.
    filter_mode : str  {"none", "critic", "mae_recon"}
        "none"     — no filtering.
        "critic"   — keep filter_keep_ratio of samples with lowest D score at
                     generation time (Phase 2, applied once).
        "mae_recon"— during MAE training, keep filter_keep_ratio of samples
                     with highest reconstruction error each epoch.
    filter_keep_ratio : float  (0, 1]
        Fraction of pseudo-anomaly samples to keep.  Default 0.7.
    filter_start_ratio : float  [0, 1]
        (mae_recon only) Fraction of mae_epochs to wait before activating the
        reconstruction filter.  Default 0.5.
    ctgan_noise_std : float
        Std of Gaussian noise added to continuous features at generation time.
        0.0 = no noise.  Default 0.0.
    ctgan_noise_feature_ratio : float
        Fraction of continuous features perturbed per sample.  Default 1.0.
    ctgan_pseudo_anomaly_mode : str  {"origin", "feature_outlier"}
        How CTGAN samples are turned into pseudo-anomalies.  Default "origin".
    ctgan_feature_outlier_r : float
        Outlier coefficient for feature_outlier mode.  Default 1.0.

    Removed vs MAECTGANSynthesizer
    -------------------------------
    - mae_start_ratio         (no joint training phase)
    - ctgan_noise_start_ratio (noise applied at generation time, not scheduled)
    - critic_ema_alpha        (no EMA; critic filter applied once at generation)
    """

    def __init__(
        self,
        metadata,
        *,
        mae_epochs: int = 100,
        pseudo_anomaly_ratio: float = 100.0,
        filter_mode: str = "none",
        filter_keep_ratio: float = 0.7,
        filter_start_ratio: float = 0.5,
        ctgan_noise_std: float = 0.0,
        ctgan_noise_feature_ratio: float = 1.0,
        ctgan_pseudo_anomaly_mode: str = "origin",
        ctgan_feature_outlier_r: float = 1.0,
        **kwargs,
    ):
        # Remove keys that don't exist in MAEAfterCTGAN from kwargs before
        # passing to parent (parent sets _model_kwargs from them).
        for obsolete_key in (
            "mae_start_ratio",
            "ctgan_noise_start_ratio",
            "critic_ema_alpha",
        ):
            kwargs.pop(obsolete_key, None)

        super().__init__(metadata=metadata, **kwargs)

        self.mae_epochs = int(mae_epochs)
        self.pseudo_anomaly_ratio = float(pseudo_anomaly_ratio)
        self.filter_mode = str(filter_mode)
        self.filter_keep_ratio = float(filter_keep_ratio)
        self.filter_start_ratio = float(filter_start_ratio)
        self.ctgan_noise_std = float(ctgan_noise_std)
        self.ctgan_noise_feature_ratio = float(ctgan_noise_feature_ratio)
        self.ctgan_pseudo_anomaly_mode = str(ctgan_pseudo_anomaly_mode)
        self.ctgan_feature_outlier_r = float(ctgan_feature_outlier_r)

        # Remove MAECTGANSynthesizer-specific keys set by parent that don't
        # belong in MAEAfterCTGAN's constructor.
        for obsolete_key in (
            "mae_start_ratio",
            "ctgan_noise_start_ratio",
            "critic_ema_alpha",
        ):
            self._model_kwargs.pop(obsolete_key, None)

        self._model_kwargs.update(
            {
                "mae_epochs": int(mae_epochs),
                "pseudo_anomaly_ratio": float(pseudo_anomaly_ratio),
                "filter_mode": str(filter_mode),
                "filter_keep_ratio": float(filter_keep_ratio),
                "filter_start_ratio": float(filter_start_ratio),
                "ctgan_noise_std": float(ctgan_noise_std),
                "ctgan_noise_feature_ratio": float(ctgan_noise_feature_ratio),
                "ctgan_pseudo_anomaly_mode": str(ctgan_pseudo_anomaly_mode),
                "ctgan_feature_outlier_r": float(ctgan_feature_outlier_r),
            }
        )

    def _fit(self, full_data, processed_data, val_data):
        """Fit MAEAfterCTGAN on preprocessed normal training data."""
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = MAEAfterCTGAN(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )
