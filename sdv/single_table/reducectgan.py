"""SDV wrapper for reduced-space MAE-CTGAN."""

import os

import cloudpickle
import pandas as pd

from ctgan.synthesizers import REDUCEMAECTGAN
from sdv.single_table.maectgan import (
    MAECTGANSynthesizer,
    _validate_no_category_dtype,
)
from sdv.single_table.utils import detect_discrete_columns


class REDUCEMAECTGANSynthesizer(MAECTGANSynthesizer):
    """MAE-CTGAN wrapper that trains MAE on MinMax-scaled source features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("KHOI TAO REDUCEMAECTGANSynthesizer TRONG SDV")

    def _fit(self, full_data, processed_data, val_data):
        """Fit CTGAN in expanded space and MAE in reduced MinMax space."""
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(),
            full_data,
            transformers,
        )
        self._model = REDUCEMAECTGAN(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(
            full_data,
            processed_data,
            val_data,
            discrete_columns=discrete_columns,
        )

    def _source_data(self, data):
        """Transform raw table data into the source numeric space if needed."""
        if not isinstance(data, pd.DataFrame):
            return data

        frame = data.copy()
        frame.columns = frame.columns.map(str)

        hyper_transformer = getattr(self._data_processor, "_hyper_transformer", None)
        input_columns = getattr(hyper_transformer, "_input_columns", None)
        if input_columns:
            input_columns = [str(column) for column in input_columns]
            if all(column in frame.columns for column in input_columns):
                return self._data_processor.transform(frame.loc[:, input_columns])

        return frame

    def transform_to_mae_space(self, data, already_source=False):
        """Return MinMax-scaled MAE input features."""
        source_data = data if already_source else self._source_data(data)
        return self._model.transform_source_to_mae_space(source_data)

    def get_score_mae(
        self,
        X,
        y,
        already_scaled=False,
        already_source=False,
    ):
        if already_scaled:
            return self._model.get_score_mae(X, y, already_scaled=True)

        source_data = X if already_source else self._source_data(X)
        return self._model.get_score_mae(source_data, y)

    def get_threshold_mae(
        self,
        X,
        already_scaled=False,
        already_source=False,
    ):
        if already_scaled:
            return self._model.get_threshold_mae(X, already_scaled=True)

        source_data = X if already_source else self._source_data(X)
        return self._model.get_threshold_mae(source_data)

    def get_latent_z_mae(
        self,
        X,
        batch_size=None,
        already_scaled=False,
        already_source=False,
    ):
        if already_scaled:
            return self._model.get_latent_z_mae(
                X,
                batch_size=batch_size,
                already_scaled=True,
            )

        source_data = X if already_source else self._source_data(X)
        return self._model.get_latent_z_mae(source_data, batch_size=batch_size)

    def _save_checkpoint(self, model, epoch):
        if epoch in [20, 40, 60, 80] or (epoch >= 100 and epoch % 10 == 0):
            path = f"{self.path}/checkpoints"
            os.makedirs(path, exist_ok=True)
            path_file = f"{path}/MAE_{self.dataset}_epoch_{epoch}.pkl"

            with open(path_file, "wb") as f:
                cloudpickle.dump(self, f)

            print(f"[CHECKPOINT] Saved at epoch {epoch}")

    def _save_best_model(
        self,
        model,
        epoch,
        threshold_re,
        threshold_z_0,
        threshold_95,
        threshold_95_total_loss,
    ):
        path = f"{self.path}/checkpoints"
        os.makedirs(path, exist_ok=True)
        path_file = f"{path}/MAE_{self.dataset}_best.pkl"
        path_file_save_epoch = f"{path}/MAE_{self.dataset}_best_epoch.txt"
        threshold_path = f"{path}/MAE_{self.dataset}_thresholds_best.pkl"
        with open(path_file, "wb") as f:
            cloudpickle.dump(self, f)
        with open(path_file_save_epoch, "w") as f:
            f.write(str(epoch))
        with open(threshold_path, "wb") as f:
            cloudpickle.dump(
                (
                    threshold_re,
                    threshold_z_0,
                    threshold_95,
                    threshold_95_total_loss,
                ),
                f,
            )
        print(f"[CHECKPOINT] Saved best model at epoch {epoch}: {path_file}")
