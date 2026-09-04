"""SDV wrapper for CTMAECTGAN (sparse-perturbation variant of MAECTGAN)."""

import os
import cloudpickle

from ctgan.synthesizers.ctmaectgan import CTMAECTGAN
from sdv.single_table.maectgan import MAECTGANSynthesizer, _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class CTMAECTGANSynthesizer(MAECTGANSynthesizer):
    """SDV wrapper for CTMAECTGAN.

    Inherits all MAECTGANSynthesizer behavior and adds three parameters
    that control the sparse perturbation strategy.

    New parameters
    --------------
    ctgan_noise_type : str  {"dense", "sparse", "directional"}
        Noise on CTGAN-generated samples. "dense" = original isotropic.
        "sparse" = symmetric noise on k features per row, rest unchanged.
        "directional" = sparse noise, always outward from normal mean.
    real_perturb_mode : str  {"dense", "sparse", "directional"}
        Perturbation of real normal data.
        "dense"       — original mix+scale (good for UNSW, CIC-IoT).
        "sparse"      — sparse noise on k features.
        "directional" — sparse outward noise (good for CIC-IDS near-normal attacks).
    ctgan_sparse_k_ratio : float  (0, 1]
        Fraction of continuous features perturbed when either mode is "sparse".
    ctgan_sparse_noise_std : float  >= 0
        Std of noise applied to the k selected features in sparse mode.

    All other parameters are identical to MAECTGANSynthesizer.
    """

    def __init__(
        self,
        metadata,
        *,
        ctgan_noise_type: str = "dense",
        real_perturb_mode: str = "dense",
        ctgan_sparse_k_ratio: float = 0.1,
        ctgan_sparse_noise_std: float = 0.3,
        score_filter_ratio: float = 0.0,
        score_filter_start_ratio: float = 0.5,
        score_alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(metadata=metadata, **kwargs)

        self.ctgan_noise_type = ctgan_noise_type
        self.real_perturb_mode = real_perturb_mode
        self.ctgan_sparse_k_ratio = float(ctgan_sparse_k_ratio)
        self.ctgan_sparse_noise_std = float(ctgan_sparse_noise_std)
        self.score_filter_ratio = float(score_filter_ratio)
        self.score_filter_start_ratio = float(score_filter_start_ratio)
        self.score_alpha = float(score_alpha)

        # Extend the model_kwargs so CTMAECTGAN.__init__ receives them.
        self._model_kwargs["ctgan_noise_type"] = ctgan_noise_type
        self._model_kwargs["real_perturb_mode"] = real_perturb_mode
        self._model_kwargs["ctgan_sparse_k_ratio"] = float(ctgan_sparse_k_ratio)
        self._model_kwargs["ctgan_sparse_noise_std"] = float(ctgan_sparse_noise_std)
        self._model_kwargs["score_filter_ratio"] = float(score_filter_ratio)
        self._model_kwargs["score_filter_start_ratio"] = float(score_filter_start_ratio)
        self._model_kwargs["score_alpha"] = float(score_alpha)

    # ─────────────────────────────────────────────────────────────────────
    # Override _fit to instantiate CTMAECTGAN instead of MAECTGAN
    # ─────────────────────────────────────────────────────────────────────

    def _fit(self, full_data, processed_data, val_data):
        """Fit CTMAECTGAN on preprocessed normal training data."""
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = CTMAECTGAN(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns
        )

    # _save_best_model, _save_checkpoint, get_score_mae, get_threshold_mae,
    # get_latent_z_mae, _sample, get_loss_values — all inherited from parent.
