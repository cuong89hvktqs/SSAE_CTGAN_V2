"""Wrapper around CTGAN model."""
import numpy as np
import pandas as pd
import plotly.express as px
#from ctgan.synthesizers import CTGAN
from ctgan.synthesizers import MAECTGAN 
from ctgan.synthesizers import TVAE

from sdmetrics import visualization

from sdv.errors import InvalidDataTypeError, NotFittedError
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns
import cloudpickle
import os

def _validate_no_category_dtype(data):
    """Check that given data has no 'category' dtype columns.

    Args:
        data (pd.DataFrame):
            Data to check.

    Raises:
        - ``InvalidDataTypeError`` if any columns in the data have 'category' dtype.
    """
    category_cols = [
        col for col, dtype in data.dtypes.items() if pd.api.types.is_categorical_dtype(dtype)
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

        Raises:
            - ``NotFittedError`` if synthesizer has not been fitted.

        Returns:
            pd.DataFrame:
                Dataframe containing the loss values per epoch.
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

        Raises:
            - ``NotFittedError`` if synthesizer has not been fitted.

        Returns:
            plotly.graph_objects._figure.Figure:
                1D marginal distribution plot (i.e. a histogram) of the columns.
        """
        if not self._fitted:
            err_msg = 'Loss values are not available yet. Please fit your synthesizer first.'
            raise NotFittedError(err_msg)

        # Tidy up the loss values data
        loss_df = self._model.loss_values.copy()

        # Create a pretty chart using Plotly Express
        fig = px.line(
            loss_df, x='Epoch',
            y=['Generator Loss', 'Discriminator Loss'],
            color_discrete_map={
                'Generator Loss': visualization.PlotConfig.DATACEBO_DARK,
                'Discriminator Loss': visualization.PlotConfig.DATACEBO_GREEN
            },
        )
        fig.update_layout(
            template='plotly_white',
            legend_title_text='',
            legend_orientation='v',
            plot_bgcolor=visualization.PlotConfig.BACKGROUND_COLOR,
            font={'size': visualization.PlotConfig.FONT_SIZE}
        )
        fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss')
        return fig


class MAECTGANSynthesizer(LossValuesMixin, BaseSingleTableSynthesizer):
    """Model wrapping ``CTGAN`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
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
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {
        'categorical': None,
        'boolean': None
    }

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True,
                 locales=['en_US'], embedding_dim=128, generator_dim=(256, 256),
                 discriminator_dim=(256, 256), generator_lr=2e-4, generator_decay=1e-6,
                 discriminator_lr=2e-4, discriminator_decay=1e-6, batch_size=500,
                 discriminator_steps=1, log_frequency=True, verbose=False, epochs=300,
                 pac=10, cuda=True,lamda=0.1, margin=5.0,dataset="dataset",path="results",type_train="all"):

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )
        #print("KHOI TAO CTGAN")
        print("KHOI TAO MAECTGANSynthesizer TRONG SDV")
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
        self.cuda = cuda
        #self.lamda=lamda
        #self.lamda_detect=lamda_detect
        #self.beta_g=beta_g
        #self.noise=noise
        self.discrete_column = None
        self.lamda=lamda
        self.margin=margin
        self.dataset=dataset
        self.path=path
        self.type_train=type_train

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
            'cuda': cuda,
            'lamda':lamda,
            'margin':margin,
            'type_train':type_train
            #'lamda':lamda
            #'lamda_detect':lamda_detect
            #'beta_g':beta_g
            #'noise':noise
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
            print(  # noqa: T001
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
        """
        Preprocess the data before giving it to the CTGAN model.

        This step is necessary because CTGAN does not support categorical values.
        We fit the data processor to the data. The data processor is an instance of
        the NumericalFormatter class, which is in charge of transforming the data
        into numerical data.

        After fitting the data processor, we print a warning if the number of columns
        generated is over 1000. This is because CTGAN will generate a large number of
        columns if there are many categories in the data.

        Finally, we transform the data using the data processor and return it.
        """

        self.validate(data)
        self._data_processor.fit(data)
        self._print_warning(data)

        return self._data_processor.transform(data)

    def _fit(self,full_data, processed_data, val_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
            y_train (pandas.Series or None):
                Optional target values for supervised training.
        """
        _validate_no_category_dtype(processed_data)
        _validate_no_category_dtype(full_data)
        self._fitted = True

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(),
            full_data,
            transformers
        )
        self._model = MAECTGAN(**self._model_kwargs)
        self.discrete_column = discrete_columns
        self._model.set_epoch_best_callback(self._save_best_model)
        self._model.fit(full_data, processed_data,val_data,discrete_columns=discrete_columns)
        #self._model.fit_org(processed_data, discrete_columns=discrete_columns)


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
    def get_score_mae(self, X_processed,y):
        """Tính anomaly score từ D2.
        Score càng thấp → càng bất thường
        """
        #X_processed = self._data_processor.transform(X)
        
        
        return self._model.get_score_mae(X_processed, y)

    def get_threshold_mae(self, X_processed):
        """Tính threshold từ tập validation."""
        #X_processed = self._data_processor.transform(X)
        return self._model.get_threshold_mae(X_processed)

    def get_latent_z_mae(self, X):
        return self._model.get_latent_z_mae(X)
    
    def _save_checkpoint(self, model, epoch):
        
        if epoch in [20, 40, 60, 80] or (epoch >= 100 and epoch % 10 == 0):
            path = f"{self.path}/checkpoints"
            os.makedirs(path, exist_ok=True)
            path_file = f"{path}/MAE_{self.dataset}_epoch_{epoch}.pkl"

            with open(path_file, "wb") as f:
                cloudpickle.dump(self, f)

            print(f"[CHECKPOINT] Saved at epoch {epoch}") 
        """
        if (epoch >= 1000 and epoch % 10 == 0):
            path = f"{self.path}/checkpoints"
            os.makedirs(path, exist_ok=True)
            path_file = f"{path}/MAE_{self.dataset}_epoch_{epoch}.pkl"

            with open(path_file, "wb") as f:
                cloudpickle.dump(self, f)

            print(f"[CHECKPOINT] Saved at epoch {epoch}") 
        """
    def _save_best_model(self, model,epoch,threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss):
        path = f"{self.path}/checkpoints"
        os.makedirs(path, exist_ok=True)
        path_file = f"{path}/MAE_{self.dataset}_best.pkl"
        path_file_save_epoch = f"{path}/MAE_{self.dataset}_best_epoch.txt"
        with open(path_file, 'wb') as f:
            cloudpickle.dump(self, f)
        with open(path_file_save_epoch, 'w') as f:
            f.write(str(epoch))
        with open(f"{path}/MAE_{self.dataset}_thresholds_best.pkl", 'wb') as f:
            cloudpickle.dump((threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss), f)
        print(f"[CHECKPOINT] Saved best model at epoch {epoch}")
