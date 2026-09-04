"""Wrapper around CTGAN model."""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from sdmetrics import visualization
from ctgan.synthesizers import OUTLINEHAECTGAN
from sdv.errors import InvalidDataTypeError, NotFittedError
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns
from sdv.utils.mixins import MissingModuleMixin
import cloudpickle
import os

try:
    from ctgan import CTGAN, TVAE
    from ctgan.synthesizers._utils import get_enable_gpu_value

    import_error = None
except ModuleNotFoundError as e:
    CTGAN = None
    TVAE = None
    import_error = e


def _validate_no_category_dtype(data):
    """Check that given data has no 'category' dtype columns.

    Args:
        data (pd.DataFrame):
            Data to check.

    Raises:
        - ``InvalidDataTypeError`` if any columns in the data have 'category' dtype.
    """
    category_cols = [
        col for col, dtype in data.dtypes.items() if isinstance(dtype, pd.CategoricalDtype)
    ]
    if category_cols:
        categoricals = "', '".join(category_cols)
        error_msg = (
            f"Columns ['{categoricals}'] are stored as a 'category' type, which is not "
            "supported. Please cast these columns to an 'object' to continue."
        )
        raise InvalidDataTypeError(error_msg)


class LossValuesMixin:
    """Mixin for accessing loss values from synthesizers."""

    def get_loss_values(self):
        """Get the loss values from the model.

        Returns:
            pd.DataFrame:
                Dataframe containing the loss values per epoch.

        Raises:
            - ``NotFittedError`` if synthesizer has not been fitted.
        """
        if not self._fitted:
            err_msg = 'Loss values are not available yet. Please fit your synthesizer first.'
            raise NotFittedError(err_msg)

        return self._model.loss_values.copy()

    def get_loss_values_plot(self, title='CTGAN loss function'):
        """Create a loss values plot from the model.

        Args:
            title (pd.DataFrame):
                Title string for the graph created.

        Returns:
            plotly.graph_objects._figure.Figure:
                1D marginal distribution plot (i.e. a histogram) of the columns.

        Raises:
            - ``NotFittedError`` if synthesizer has not been fitted.
        """
        if not self._fitted:
            err_msg = 'Loss values are not available yet. Please fit your synthesizer first.'
            raise NotFittedError(err_msg)

        # Tidy up the loss values data
        loss_df = self._model.loss_values.copy()

        # Create a pretty chart using Plotly Express
        fig = px.line(
            loss_df,
            x='Epoch',
            y=['Generator Loss', 'Discriminator Loss'],
            color_discrete_map={
                'Generator Loss': visualization.PlotConfig.DATACEBO_DARK,
                'Discriminator Loss': visualization.PlotConfig.DATACEBO_GREEN,
            },
        )
        fig.update_layout(
            template='plotly_white',
            legend_title_text='',
            legend_orientation='v',
            plot_bgcolor=visualization.PlotConfig.BACKGROUND_COLOR,
            font={'size': visualization.PlotConfig.FONT_SIZE},
        )
        fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss')
        return fig


class OUTLINEHAECTGANSynthesizer(LossValuesMixin, MissingModuleMixin, BaseSingleTableSynthesizer):
    """Model wrapping ``CTGAN`` model.

    Args:
        metadata (sdv.metadata.Metadata):
            Single table metadata representing the data that this synthesizer will be used for.
            * sdv.metadata.SingleTableMetadata can be used but will be deprecated.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        enable_gpu (bool):
            Whether to attempt to use GPU for computation.
            Defaults to ``True``.
        cuda (bool or str):
            **Deprecated**
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {'categorical': None, 'boolean': None}

    def __init__(
        self,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=['en_US'],
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
        enable_gpu=True,
        cuda=None,
        lamda=0.1, margin=1.0, dataset="dataset", path="results",
        type_train="all", loss_method="original", focal_gamma=2.0,
        buffer_multiplier=8, use_buffer=True, type_val="val_one_class",
        k1=1.0, k2=3.0, use_boundary_anomaly=False, boundary_beta=0.1, outlier_mode="tail_abs",
        boundary_dim_prob=0.2, boundary_band_width=0.1,
        save_ae_train_samples=False, ae_train_sample_max_rows=20000,
        ae_train_sample_save_path=None
    ):
        if CTGAN is None:
            self.raise_module_not_found_error(import_error)
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )
        print(f"INITIALIZING OUTLINEHAECTGANSynthesizer with loss_method={loss_method}")
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.enable_gpu = get_enable_gpu_value(enable_gpu, cuda)
        
        self.lamda = lamda
        self.margin = margin
        self.dataset = dataset
        self.path = path
        self.type_train = type_train
        self.loss_method = loss_method
        self.focal_gamma = float(focal_gamma)
        self.buffer_multiplier = max(1, int(buffer_multiplier))
        self.use_buffer = bool(use_buffer)
        self.type_val = type_val
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.use_boundary_anomaly = bool(use_boundary_anomaly)
        self.boundary_beta = float(boundary_beta)
        self.outlier_mode = outlier_mode
        self.boundary_dim_prob = float(boundary_dim_prob)
        self.boundary_band_width = max(0.0, float(boundary_band_width))
        self.save_ae_train_samples = bool(save_ae_train_samples)
        self.ae_train_sample_max_rows = max(0, int(ae_train_sample_max_rows))
        self.ae_train_sample_save_path = ae_train_sample_save_path
        
        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'enable_gpu': self.enable_gpu,
            'lamda': lamda,
            'margin': margin,
            'type_train': type_train,
            'loss_method': loss_method,
            'focal_gamma': float(focal_gamma),
            'buffer_multiplier': max(1, int(buffer_multiplier)),
            'use_buffer': bool(use_buffer),
            'type_val': type_val,
            'k1': float(k1),
            'k2': float(k2),
            'use_boundary_anomaly': bool(use_boundary_anomaly),
            'boundary_beta': float(boundary_beta),
            'outlier_mode': outlier_mode,
            'boundary_dim_prob': float(boundary_dim_prob),
            'boundary_band_width': max(0.0, float(boundary_band_width)),
            'save_ae_train_samples': bool(save_ae_train_samples),
            'ae_train_sample_max_rows': max(0, int(ae_train_sample_max_rows)),
            'ae_train_sample_save_path': ae_train_sample_save_path,
        }

    def _estimate_num_columns(self, data):
        """Estimate the number of columns that the data will generate.

        Estimates that continuous columns generate 11 columns and categorical ones
        create n where n is the number of unique categories.

        Args:
            data (pandas.DataFrame):
                Data to estimate the number of columns from.

        Returns:
            int:
                Number of estimate columns.
        """
        sdtypes = self._data_processor.get_sdtypes()
        transformers = self.get_transformers()
        num_generated_columns = {}
        for column in data.columns:
            if column not in sdtypes:
                continue

            if sdtypes[column] in {'numerical', 'datetime'}:
                num_generated_columns[column] = 11

            elif sdtypes[column] in {'categorical', 'boolean'}:
                if transformers.get(column) is None:
                    num_categories = data[column].fillna(np.nan).nunique(dropna=False)
                    num_generated_columns[column] = num_categories
                else:
                    num_generated_columns[column] = 11

        return num_generated_columns

    def _print_warning(self, data):
        """Print a warning if the number of columns generated is over 1000."""
        dict_generated_columns = self._estimate_num_columns(data)
        if sum(dict_generated_columns.values()) > 1000:
            header = {'Original Column Name  ': 'Est # of Columns (CTGAN)'}
            dict_generated_columns = {**header, **dict_generated_columns}
            longest_column_name = len(max(dict_generated_columns, key=len))
            cap = '<' + str(longest_column_name)
            lines_to_print = []
            for column, num_generated_columns in dict_generated_columns.items():
                lines_to_print.append(f'{column:{cap}} {num_generated_columns}')

            generated_columns_str = '\n'.join(lines_to_print)
            print(  # noqa: T201
                'PerformanceAlert: Using the CTGANSynthesizer on this data is not recommended. '
                'To model this data, CTGAN will generate a large number of columns.'
                '\n\n'
                f'{generated_columns_str}'
                '\n\n'
                'We recommend preprocessing discrete columns that can have many values, '
                "using 'update_transformers'. Or you may drop columns that are not necessary "
                'to model. (Exit this script using ctrl-C)'
            )

    def _preprocess(self, data):
        self._data_processor.fit(data)
        self._print_warning(data)

        return self._data_processor.transform(data)

    def _fit(self, full_data, processed_data, val_data, val_labels=None):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(self.metadata, processed_data, transformers)
        
        self._model =OUTLINEHAECTGAN(**self._model_kwargs)
        self._model.set_epoch_best_callback(self._save_best_model)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Attempting to run cuBLAS.*')
            self._model.fit(full_data,processed_data, val_data,val_labels=val_labels,discrete_columns=discrete_columns)
        
        self._fitted = True

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError("CTGANSynthesizer doesn't support conditional sampling.")

    def get_score_ha(self, X_test, y_test, batch_size=125):
        """Get anomaly scores from the OUTLINEHA model."""
        if not hasattr(self, '_model') or self._model is None:
            raise NotFittedError("Model has not been fitted yet")

        return self._model.get_score_ha(X_test, y_test)

    def get_threshold_ha(self, X_val):
        """Calculate thresholds from a normal-only validation set."""
        if not hasattr(self, '_model') or self._model is None:
            raise NotFittedError("Model has not been fitted yet")

        return self._model.get_threshold_ha(X_val)

    def get_latent_z_ha(self, X):
        if not hasattr(self, '_model') or self._model is None:
            raise NotFittedError("Model has not been fitted yet")

        return self._model.get_latent_z_ha(X)

    def _save_best_model(self, model, epoch, threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss):
        path = f"{self.path}/checkpoints"
        os.makedirs(path, exist_ok=True)
        path_file = f"{path}/OUTLINEHA_{self.dataset}_best.pkl"
        path_file_save_epoch = f"{path}/OUTLINEHA_{self.dataset}_best_epoch.txt"
        with open(path_file, "wb") as f:
            cloudpickle.dump(self, f)
        with open(path_file_save_epoch, "w") as f:
            f.write(str(epoch))
        with open(f"{path}/OUTLINEHA_{self.dataset}_thresholds_best.pkl", "wb") as f:
            cloudpickle.dump((threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss), f)
        print(f"[CHECKPOINT] Saved best OUTLINEHA model at epoch {epoch}")
