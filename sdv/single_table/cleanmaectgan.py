"""SDV wrapper for CleanMAECTGAN (pseudo-anomaly filtering for MAE training)."""

import os
import cloudpickle

from ctgan.synthesizers.cleanmaectgan import CleanMAECTGAN
from sdv.single_table.maectgan import MAECTGANSynthesizer, _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class CleanMAECTGANSynthesizer(MAECTGANSynthesizer):
    """SDV wrapper for CleanMAECTGAN.

    Inherits all MAECTGANSynthesizer behaviour and adds parameters that control
    pseudo-anomaly filtering during MAE training.

    New parameters
    --------------
    filter_mode : str  {"none", "critic", "mae_recon"}
        "none"     — no filtering (same as MAECTGAN).
        "critic"   — discard CTGAN samples with high D scores (look normal).
                     Applied before buffer; uses EMA trend threshold.
        "mae_recon"— discard assembled pseudo-anomalies with low MAE
                     reconstruction error (look normal to the MAE).
    filter_keep_ratio : float  (0, 1]
        Fraction of pseudo-anomaly samples to keep each step.  Default 0.7.
    filter_start_ratio : float  [0, 1]
        Fraction of post-MAE-start epochs to wait before filtering activates.
        Default 0.5.
    critic_ema_alpha : float  (0, 1]
        EMA smoothing factor for D-score trend threshold (critic mode only).
        Default 0.3.
    """

    def __init__(
        self,
        metadata,
        *,
        filter_mode: str = "none",
        filter_keep_ratio: float = 0.7,
        filter_start_ratio: float = 0.5,
        critic_ema_alpha: float = 0.3,
        **kwargs,
    ):
        super().__init__(metadata=metadata, **kwargs)

        self.filter_mode = str(filter_mode)
        self.filter_keep_ratio = float(filter_keep_ratio)
        self.filter_start_ratio = float(filter_start_ratio)
        self.critic_ema_alpha = float(critic_ema_alpha)

        self._model_kwargs["filter_mode"] = str(filter_mode)
        self._model_kwargs["filter_keep_ratio"] = float(filter_keep_ratio)
        self._model_kwargs["filter_start_ratio"] = float(filter_start_ratio)
        self._model_kwargs["critic_ema_alpha"] = float(critic_ema_alpha)

    def _fit(self, full_data, processed_data, val_data):
        """Fit CleanMAECTGAN on preprocessed normal training data."""
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = CleanMAECTGAN(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )
