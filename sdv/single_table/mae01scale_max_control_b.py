"""SDV wrapper for Control B (isotropic-noise pseudo-intrusion ablation).

Identical to MAE01ScaleMaxSynthesizer._fit() except it instantiates
MAE01ScaleMaxControlB instead of MAE01ScaleMax -- everything else (metadata
handling, discrete-column detection, checkpoint callback) is inherited
unchanged.
"""

from ctgan.synthesizers.mae01scale_max_control_b import MAE01ScaleMaxControlB
from sdv.single_table.mae01scale_max import MAE01ScaleMaxSynthesizer
from sdv.single_table.maectgan import _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns


class MAE01ScaleMaxControlBSynthesizer(MAE01ScaleMaxSynthesizer):

    def _fit(self, full_data, processed_data, val_data):
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(), full_data, transformers
        )

        self._model = MAE01ScaleMaxControlB(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data, processed_data, val_data,
            discrete_columns=discrete_columns,
        )
