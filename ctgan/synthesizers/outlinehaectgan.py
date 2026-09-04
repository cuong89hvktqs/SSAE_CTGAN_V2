"""CTGAN module."""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional,Tanh, MSELoss,L1Loss,Sigmoid
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.errors import InvalidDataError
from ctgan.synthesizers._utils import _format_score, _set_device, validate_and_set_device
from ctgan.synthesizers.base import BaseSynthesizer, random_state
import torch.nn.functional as F
from scipy.stats import truncnorm
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from math import sqrt
import time

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data

class HAE(Module):
    """Hybrid AutoEncoder used for anomaly scoring.
    This module computes reconstruction and latent-based anomaly components.
    """

    def __init__(self, input_dim, lamda=0.1, margin=1.0, loss_method="original", focal_gamma=2.0):
        """Initialize HAE encoder/decoder and loss settings.
        Supports original, asymmetric, dual, contrastive, focal and re_normal_only loss variants.
        """
        super(HAE, self).__init__()
        self.enc = Sequential(
            Linear(input_dim, round(input_dim * 0.75)),
            Tanh(),
            Linear(round(input_dim * 0.75), round(input_dim * 0.5)),
            Tanh(),
            Linear(round(input_dim * 0.5), round(sqrt(input_dim)) + 1),
        )
        self.dec = Sequential(
            Linear(round(sqrt(input_dim)) + 1, round(input_dim * 0.5)),
            Tanh(),
            Linear(round(input_dim * 0.5), round(input_dim * 0.75)),
            Tanh(),
            Linear(round(input_dim * 0.75), input_dim),
            Sigmoid()
        )
        self.loss_function = MSELoss()
        self.latent_distance = MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.lamda=lamda
        self.margin=margin
        self.loss_method = loss_method
        self.focal_gamma = focal_gamma
        self.normal_center = None

    def forward(self, input_):
        """Run a forward pass through HAE.
        Returns latent vectors and reconstructed inputs.
        """
        encoded = self.enc(input_)
        decoded = self.dec(encoded)
        return encoded, decoded
    def _latent_score(self, encoded, y=None, update_center=False):
        """Compute latent score for each sample.
        In contrastive mode, score is measured against the normal centroid.
        """
        if self.loss_method == "contrastive":
            if self.normal_center is None or update_center:
                if y is not None and torch.any(y.view(-1) == 0):
                    normal_z = encoded[y.view(-1) == 0]
                else:
                    normal_z = encoded
                self.normal_center = normal_z.detach().mean(dim=0)
            return torch.norm(encoded - self.normal_center.unsqueeze(0), p=2, dim=1)

        return torch.norm(encoded, p=2, dim=1)

    def _latent_training_loss(self, latent_score, y):
        """Compute latent loss from latent scores and labels.
        Normal samples are pulled to 0 while anomalies are pulled to margin m.
        """
        y = y.view(-1)
        margin = torch.tensor(
            self.margin,
            device=latent_score.device,
            dtype=latent_score.dtype
        )

        # Old focal loss:
        # if self.loss_method == "focal":
        #     gap = torch.relu(self.margin - latent_score)
        #     hard_weight = torch.pow(gap / (self.margin + 1e-8), self.focal_gamma)
        #     return torch.mean(torch.where(
        #         y == 0,
        #         latent_score ** 2,
        #         hard_weight * (gap ** 2)
        #     ))
        #
        # Old dual loss:
        # if self.loss_method == "dual":
        #     return torch.mean(torch.where(
        #         y == 0,
        #         latent_score ** 2,
        #         torch.relu(self.margin - latent_score)
        #     ))
        #
        # Old margin loss:
        # return torch.mean(torch.where(
        #     y == 0,
        #     latent_score ** 2,
        #     torch.relu(self.margin - latent_score) ** 2
        # ))
        #
        # New target loss:
        # - normal samples are pulled to 0
        # - anomaly samples are pulled to margin m
        return torch.mean(torch.where(
            y == 0,
            latent_score ** 2,
            (latent_score - margin) ** 2
        ))

    def _score_components(self, x, encoded, xb_hat, y=None, update_center=False):
        """Compute reconstruction, latent and total anomaly components.
        Total score is defined as reconstruction + lambda * latent^2.
        """
        re_per_sample = F.mse_loss(xb_hat, x, reduction="none").mean(dim=1)
        latent_score = self._latent_score(encoded, y=y, update_center=update_center)
        total_score = re_per_sample + self.lamda * (latent_score ** 2)
        return re_per_sample, latent_score, total_score

    def calculate_loss(self, x, encoded, xb_hat, y):
        """Build training loss terms for HAE.
        Asymmetric mode increases reconstruction weight for anomaly-labeled samples.
        """
        re_per_sample, latent_score, _ = self._score_components(
            x, encoded, xb_hat, y=y, update_center=(self.loss_method == "contrastive")
        )
        if self.loss_method == "asymmetric":
            weights = torch.where(y.view(-1) == 0, 1.0, 2.0).to(re_per_sample.device)
            re_loss = (re_per_sample * weights).mean()
        elif self.loss_method == "re_normal_only":
            normal_mask = (y.view(-1) == 0)
            if torch.any(normal_mask):
                re_loss = re_per_sample[normal_mask].mean()
            else:
                re_loss = re_per_sample.mean() * 0.0
        elif self.loss_method == "focal":
            gap = torch.relu(self.margin - latent_score)
            hard_weight = 1.0 + torch.pow(gap / (self.margin + 1e-8), self.focal_gamma)
            weights = torch.where(y.view(-1) == 0, torch.ones_like(hard_weight), hard_weight)
            re_loss = (re_per_sample * weights).mean()
        else:
            re_loss = re_per_sample.mean()

        latent_loss = self._latent_training_loss(latent_score, y)
        return re_loss, latent_loss
    def train_hae(self, epoch,train_data_loader):
        """Train HAE for one epoch loop over a data loader.
        Returns final loss values and collected latent vectors for analysis.
        """
        self.train()
        list_latent=[]
        for input, label in train_data_loader:
            input, label = input.to(self.device), label.to(self.device)
            latent_z, xb_hat = self.forward(input)
            loss_re, loss_z = self.calculate_loss(
                input, latent_z, xb_hat, label
            )
            loss = loss_re + self.lamda * loss_z
            self.optimizer.zero_grad()                                                                              
            loss.backward()
            self.optimizer.step()
            list_latent.append(latent_z.detach().cpu().numpy())
        list_latent = np.concatenate(list_latent, axis=0)
        return loss, loss_re, loss_z, list_latent
    def validate(self, val_data_loader, epoch):
        """Evaluate HAE on validation data.
        Returns validation loss and thresholds used for anomaly inference.
        """
        del epoch
        self.eval()
        val_losses = []
        val_re_losses = []
        latent_score_all = []
        y_val_true = []

        with torch.no_grad():
            for xb, yb in val_data_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                z, xb_hat = self.forward(xb)
                re_loss, latent_score, total_score = self._score_components(
                    xb, z, xb_hat, y=yb, update_center=(self.loss_method == "contrastive")
                )

                val_losses.append(total_score.cpu().numpy())
                val_re_losses.append(re_loss.cpu().numpy())
                latent_score_all.append(latent_score.cpu().numpy())
                y_val_true.append(yb.cpu().numpy())

        # =============================
        # Aggregate
        # =============================
        val_losses = np.concatenate(val_losses)
        val_re_losses = np.concatenate(val_re_losses)
        latent_score_all = np.concatenate(latent_score_all)
        y_val_true = np.concatenate(y_val_true)

        avg_loss = np.mean(val_losses)

        # =============================
        # Thresholds (NORMAL only)
        # =============================
        # Reconstruction threshold
        re_normal = val_re_losses[y_val_true == 0]
        mu_re = np.mean(re_normal)
        std_re = np.std(re_normal)
        threshold_re = (mu_re, std_re)

        latent_normal = latent_score_all[y_val_true == 0]
        mu_z = np.mean(latent_normal)
        std_z = np.std(latent_normal)
        threshold_z = (mu_z, std_z)
        threshold_95 = np.percentile(latent_normal, 95)
        threshold_95_total_loss = np.percentile(val_losses[y_val_true == 0], 95)
        val_auc_total = None
        threshold_auc_total = threshold_95_total_loss
        y_val_bin = (y_val_true != 0).astype(int)
        if np.unique(y_val_bin).size >= 2:
            val_auc_total = float(roc_auc_score(y_val_bin, val_losses))
            fpr, tpr, thresholds = roc_curve(y_val_bin, val_losses)
            j_stat = tpr - fpr
            finite_mask = np.isfinite(thresholds)
            if np.any(finite_mask):
                valid_idx = np.where(finite_mask)[0]
                best_idx = valid_idx[np.argmax(j_stat[valid_idx])]
                threshold_auc_total = float(thresholds[best_idx])

        return (
            avg_loss,
            threshold_re,
            threshold_z,
            threshold_95,
            threshold_95_total_loss,
            val_auc_total,
            threshold_auc_total,
        )
    def compute_average_losses(self, data_loader):
        """Compute average total/reconstruction/latent losses.
        Used for tracking validation behavior over epochs.
        """
        self.eval()
        re_losses = []
        latent_scores = []
        total_losses = []

        with torch.no_grad():
            for xb, yb in data_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                z, xb_hat = self.forward(xb)
                re_loss, latent_score, total_loss = self._score_components(
                    xb, z, xb_hat, y=yb, update_center=False
                )

                re_losses.append(re_loss.cpu().numpy())
                latent_scores.append(latent_score.cpu().numpy())
                total_losses.append(total_loss.cpu().numpy())

        # =============================
        # Aggregate
        # =============================
        re_losses = np.concatenate(re_losses)
        latent_scores = np.concatenate(latent_scores)
        total_losses = np.concatenate(total_losses)

        return  np.mean(total_losses),np.mean(re_losses), np.mean(latent_scores)
    
    def get_score_ha(
        self,
        X_test,
        y_test,
        batch_size=125,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Compute anomaly scores for a test set.
        Returns reconstruction errors, latent scores and ground-truth labels.
        """
        self.eval()
        device = torch.device(device)
        self.to(device)

        # =============================
        # Convert input
        # =============================
        if isinstance(X_test, np.ndarray):
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
        else:
            X_tensor = X_test.float()

        if isinstance(y_test, np.ndarray):
            y_tensor = torch.tensor(y_test, dtype=torch.float32)
        elif hasattr(y_test, "values"):
            y_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        else:
            y_tensor = torch.tensor(y_test, dtype=torch.float32)

        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)

        test_loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=False
        )

        if device.type == "cuda":
            for i, (xb, yb) in enumerate(test_loader):
                if i == 1: break
                _ = self.forward(xb)
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # =============================
        # Buffers
        # =============================
        re_errors = []
        z_norm_scores = []
        y_true = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                z, xb_hat = self.forward(xb)

                re, latent_score, _ = self._score_components(xb, z, xb_hat, y=yb)
                re_errors.append(re.cpu().numpy())
                z_norm_scores.append(latent_score.cpu().numpy())
                y_true.append(yb.cpu().numpy())

            re_errors = np.concatenate(re_errors)
            z_norm_scores = np.concatenate(z_norm_scores)
            y_true = np.concatenate(y_true)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            print(f"[Time] get_score_ha: {(t1 - t0) * 1000:.2f} ms")
            return re_errors, z_norm_scores, y_true

    def plot_latent_first_component_hist_from_latents(self, latents, out_dir, epoch, bins=50):
        """Plot histogram of the first latent component.
        Helps inspect latent distribution drift across epochs.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)
        if latents is None or latents.shape[0] == 0:
            return None

        first_comp = np.asarray(latents)[:, 0]
        fname_base = f"epoch{epoch}"

        # Check unique values and adjust bins if needed
        unique_vals = np.unique(first_comp)
        if len(unique_vals) < 2:
            # Not enough data to plot a histogram
            return None
        safe_bins = min(bins, len(unique_vals))
        if safe_bins < 1:
            safe_bins = 1
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        try:
            plt.hist(first_comp, bins=safe_bins, color="C2", alpha=0.85)
        except Exception as e:
            print(f"[Warning] Could not plot histogram: {e}")
            plt.close()
            return None
        plt.title(f"Latent dim[0] histogram - epoch {epoch}")
        plt.xlabel("latent[0]")
        plt.ylabel("count")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{fname_base}_latent_dim0_hist.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

class OUTLINEHAECTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
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
        cuda (bool):
            **Deprecated** Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
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
        lamda=0.1, margin=1.0, type_train="all", loss_method="original",
        focal_gamma=2.0, buffer_multiplier=8, use_buffer=True, type_val="val_one_class",
        k1=1.0, k2=3.0, use_boundary_anomaly=False, boundary_beta=0.1, outlier_mode="tail_abs",
        boundary_dim_prob=0.2, boundary_band_width=0.1,
        save_ae_train_samples=False, ae_train_sample_max_rows=20000,
        ae_train_sample_save_path=None
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self._device = validate_and_set_device(enable_gpu, cuda)
        self._enable_gpu = cuda if cuda is not None else enable_gpu
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None
        
        self.lamda=lamda
        self.margin=margin
        self.type_train=type_train
        self.loss_method=loss_method
        self.focal_gamma = float(focal_gamma)
        self.buffer_multiplier = max(1, int(buffer_multiplier))
        self.use_buffer = bool(use_buffer)
        self.type_val = type_val
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.use_boundary_anomaly = bool(use_boundary_anomaly)
        self.boundary_beta = float(boundary_beta)
        self.outlier_mode = outlier_mode
        if self.outlier_mode not in {"tail_abs", "truncnorm", "boundary_band"}:
            raise ValueError("outlier_mode must be one of: tail_abs, truncnorm, boundary_band.")
        self.boundary_dim_prob = float(boundary_dim_prob)
        if not 0.0 <= self.boundary_dim_prob <= 1.0:
            raise ValueError("boundary_dim_prob must be in [0, 1].")
        self.boundary_band_width = max(0.0, float(boundary_band_width))
        self.save_ae_train_samples = bool(save_ae_train_samples)
        self.ae_train_sample_max_rows = max(0, int(ae_train_sample_max_rows))
        self.ae_train_sample_save_path = ae_train_sample_save_path
        
        self._hae=None
        
        self._on_epoch_end_callbacks = None
        self._on_epoch_best_callbacks = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def _validate_null_data(self, train_data, discrete_columns):
        """Check whether null values exist in continuous ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            continuous_cols = list(set(train_data.columns) - set(discrete_columns))
            any_nulls = train_data[continuous_cols].isna().any().any()
        else:
            continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
            any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()

        if any_nulls:
            raise InvalidDataError(
                'CTGAN does not support null values in the continuous training data. '
                'Please remove all null values from your continuous training data.'
            )

    @staticmethod
    def _sample_latent_outside_sigma(mean, std, sigma_multiplier):
        """Sample latent values outside [mean - m*std, mean + m*std]."""
        sigma_multiplier = max(0.0, float(sigma_multiplier))
        direction = torch.where(
            torch.rand_like(mean) < 0.5,
            torch.full_like(mean, -1.0),
            torch.ones_like(mean),
        )
        tail = torch.abs(torch.normal(mean=torch.zeros_like(mean), std=std))
        return mean + direction * ((std * sigma_multiplier) + tail)

    @staticmethod
    def _sample_latent_truncnorm(mean, std, sigma_multiplier):
        """Sample latent values from both truncated normal tails."""
        sigma_multiplier = max(0.0, float(sigma_multiplier))
        upper_samples = truncnorm.rvs(
            sigma_multiplier,
            np.inf,
            loc=0.0,
            scale=1.0,
            size=tuple(mean.shape),
        )
        lower_samples = truncnorm.rvs(
            -np.inf,
            -sigma_multiplier,
            loc=0.0,
            scale=1.0,
            size=tuple(mean.shape),
        )
        choose_upper = np.random.random(size=tuple(mean.shape)) < 0.5
        samples = np.where(choose_upper, upper_samples, lower_samples)
        samples = torch.as_tensor(samples, dtype=mean.dtype, device=mean.device)
        return mean + std * samples

    @staticmethod
    def _sample_latent_boundary_band(
        mean,
        std,
        sigma_multiplier,
        boundary_dim_prob=0.2,
        band_width=0.1,
    ):
        """Sample near-boundary latent values only for a subset of dimensions."""
        sigma_multiplier = max(0.0, float(sigma_multiplier))
        boundary_dim_prob = min(1.0, max(0.0, float(boundary_dim_prob)))
        band_width = max(0.0, float(band_width))

        z = torch.normal(mean=mean, std=std)
        if boundary_dim_prob == 0.0:
            return z

        mask = torch.rand_like(z) < boundary_dim_prob
        sign = torch.where(
            torch.rand_like(z) < 0.5,
            torch.full_like(z, -1.0),
            torch.ones_like(z),
        )
        radius = sigma_multiplier + torch.rand_like(z) * band_width
        z_boundary = mean + sign * std * radius
        return torch.where(mask, z_boundary, z)

    def _sample_latent_outlier(self, mean, std, sigma_multiplier):
        """Dispatch latent outlier sampling by configured mode."""
        """
        #tail_abs: lấy giá trị ngẫu nhiên từ hai tail của phân phối chuẩn (bên trái < mean - m*std, bên phải > mean + m*std).
        #truncnorm: lấy giá trị ngẫu nhiên từ hai tail của phân phối chuẩn nhưng sử dụng phân phối chuẩn cắt 
        (truncated normal) để đảm bảo không có giá trị nào nằm trong khoảng [mean - m*std, mean + m*std].
        #boundary_band: lấy giá trị ngẫu nhiên gần biên của phân phối chuẩn, nhưng chỉ áp dụng cho một phần các chiều 
        # (theo xác suất boundary_dim_prob). Các giá trị này sẽ nằm trong một dải (band) xung quanh biên, với độ rộng được
        # xác định bởi band_width.
        """
        if self.outlier_mode == "tail_abs":
            return self._sample_latent_outside_sigma(mean, std, sigma_multiplier)
        if self.outlier_mode == "truncnorm":
            return self._sample_latent_truncnorm(mean, std, sigma_multiplier)
        if self.outlier_mode == "boundary_band":
            return self._sample_latent_boundary_band(
                mean,
                std,
                sigma_multiplier,
                boundary_dim_prob=self.boundary_dim_prob,
                band_width=self.boundary_band_width,
            )
        raise ValueError("outlier_mode must be one of: tail_abs, truncnorm, boundary_band.")

    @staticmethod
    def _append_ae_train_sample_store(store, new_samples, max_rows):
        """Append generated AE-train samples into a CPU reservoir."""
        max_rows = int(max_rows)
        if max_rows <= 0 or new_samples is None or new_samples.size(0) == 0:
            return store

        new_samples = new_samples.detach().cpu()
        if new_samples.size(0) > max_rows:
            idx = torch.randperm(new_samples.size(0))[:max_rows]
            new_samples = new_samples[idx]

        if store is None:
            store = new_samples
        else:
            store = torch.cat([store, new_samples], dim=0)

        if store.size(0) > max_rows:
            idx = torch.randperm(store.size(0))[:max_rows]
            store = store[idx]

        return store.detach()

    def _save_ae_train_sample_store(self, store):
        """Save generated samples used by AE training for later t-SNE analysis."""
        if not self.save_ae_train_samples:
            return None
        if self.ae_train_sample_save_path is None:
            return None
        if store is None or store.size(0) == 0:
            print("[SAVE] No AE train generated samples to save.")
            return None

        out_dir = os.path.dirname(os.path.abspath(self.ae_train_sample_save_path))
        os.makedirs(out_dir, exist_ok=True)
        arr = store.detach().cpu().numpy().astype("float32")
        recovered = self._transformer.inverse_transform(arr)
        if isinstance(recovered, pd.DataFrame):
            save_data = recovered
        else:
            columns = [f"x_{idx}" for idx in range(recovered.shape[1])]
            save_data = pd.DataFrame(recovered, columns=columns)
        save_data.to_csv(self.ae_train_sample_save_path, index=False)
        print(
            "[SAVE] AE train generated samples:",
            self.ae_train_sample_save_path,
            "saved_shape:",
            save_data.shape,
            "internal_shape:",
            arr.shape,
        )
        return self.ae_train_sample_save_path
    
    @random_state
    def fit(self, full_data,train_data, val_data, val_labels=None, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)


        print(
            "Starting OUTLINEHAECTGAN training..., data shape:", train_data.shape,
            " discrete_columns:", discrete_columns, " epochs:", self._epochs,
            "lamda:", self.lamda, " margin:", self.margin, " k1:", self.k1, " k2:", self.k2,
            " use_buffer:", self.use_buffer,
            " buffer_multiplier:", self.buffer_multiplier,
            " outlier_mode:", self.outlier_mode,
            " boundary_dim_or_perturb_prob:", self.boundary_dim_prob,
            " boundary_band_width:", self.boundary_band_width,
            " use_boundary_anomaly:", self.use_boundary_anomaly,
            " boundary_beta:", self.boundary_beta,
            " save_ae_train_samples:", self.save_ae_train_samples,
            " ae_train_sample_max_rows:", self.ae_train_sample_max_rows,
        )
        fake_buff_list_NOR = None # List to store generated fake samples for buffering (normal)
        fake_buff_list_ANO = None # List to store generated fake samples for buffering (anomaly)
        val_synth_store = None
        ae_train_generated_store = None
        if self.type_val not in {"val_one_class", "val_multi_class"}:
            raise ValueError("type_val must be 'val_one_class' or 'val_multi_class'.")
        
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)
        X_val_train_processed = self._transformer.transform(val_data)
        if val_labels is None:
            y_val_train = np.zeros(X_val_train_processed.shape[0], dtype=np.float32)
        else:
            if hasattr(val_labels, "values"):
                y_val_train = val_labels.values.astype(np.float32)
            else:
                y_val_train = np.asarray(val_labels, dtype=np.float32)
            y_val_train = (y_val_train != 0).astype(np.float32)
            if y_val_train.shape[0] != X_val_train_processed.shape[0]:
                raise ValueError("val_labels length must match val_data length.")
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_train_processed, dtype=torch.float32), torch.tensor(y_val_train, dtype=torch.float32))
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)
        
        
        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        
        self._hae=HAE(
            input_dim=data_dim,
            lamda=self.lamda,
            margin=self.margin,
            loss_method=self.loss_method,
            focal_gamma=self.focal_gamma,
        ).to(self._device)
        
        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss','HAE Loss','Reconstruction Loss','Latent Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen}) | Discrim. ({dis})'
            epoch_iterator.set_description(
                description.format(gen=_format_score(0), dis=_format_score(0))
            )

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        best_val_avg_total_loss = float("inf")
        best_val_auc_total = -float("inf")
        
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                
                real_data=None
                
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)
                    real_data=real
                    
                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                
                fakez_buffer_nor = self._sample_latent_outlier(
                    mean, std, sigma_multiplier=self.k1
                )
                fakez_buffer_ano = self._sample_latent_outlier(
                    mean, std, sigma_multiplier=self.k2
                )

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                    fakez_buffer_nor = torch.cat([fakez_buffer_nor, c1], dim=1)
                    fakez_buffer_ano = torch.cat([fakez_buffer_ano, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                
                fake_buffer_nor = self._generator(fakez_buffer_nor)
                fakeact_buffer_nor = self._apply_activate(fake_buffer_nor)

                fake_buffer_ano = self._generator(fakez_buffer_ano)
                fakeact_buffer_ano = self._apply_activate(fake_buffer_ano)
                
                if self.use_buffer:
                    fake_buff_list_NOR = self.to_buffer(
                        fake_buff_list_NOR,
                        fakeact_buffer_nor,
                        self._batch_size,
                        self.buffer_multiplier
                    )
                    fake_buff_list_ANO = self.to_buffer(
                        fake_buff_list_ANO,
                        fakeact_buffer_ano,
                        self._batch_size,
                        self.buffer_multiplier
                    )
                real_mix=self.make_unsw_mix_anomaly(real_data)
                real_scale=self.make_unsw_scale_anomaly(real_data)
                real_perturbed=0.5*real_mix + 0.5*real_scale
                
                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                #train the HAE
                if not self.use_buffer:
                    fakeact_form_buffer_NOR = fakeact_buffer_nor
                elif fake_buff_list_NOR is None:
                    fakeact_form_buffer_NOR = fakeact_buffer_nor
                else:
                    fakeact_form_buffer_NOR=self.from_buffer(fake_buff_list_NOR, self._batch_size)
                
                if not self.use_buffer:
                    fakeact_form_buffer_ANO = fakeact_buffer_ano
                elif fake_buff_list_ANO is None:
                    fakeact_form_buffer_ANO=fakeact_buffer_ano
                else:
                    fakeact_form_buffer_ANO=self.from_buffer(fake_buff_list_ANO, self._batch_size)

                if self.use_boundary_anomaly:
                    fakeact_form_buffer_ANO = self.make_ctgan_boundary_anomaly(
                        real_data,
                        fakeact_form_buffer_ANO,
                        beta=self.boundary_beta,
                        perturb_prob=self.boundary_dim_prob,
                    )
                    fakeact_form_buffer_NOR = self.make_ctgan_boundary_anomaly(
                        real_data,
                        fakeact_form_buffer_NOR,
                        beta=self.boundary_beta,
                        perturb_prob=self.boundary_dim_prob,
                    )
                # Train HAE at every discriminator step.
                
                
                
                if(self.type_train=="all"):
                    fake_data=torch.cat([fakeact_form_buffer_ANO, real_perturbed,fakeact_form_buffer_NOR], dim=0)
                elif(self.type_train=="noise"):
                    fake_data=real_perturbed
                elif(self.type_train=="3std"):
                    fake_data=fakeact_form_buffer_ANO
                elif(self.type_train=="std"):
                    fake_data=fakeact_form_buffer_NOR
                elif(self.type_train=="std_3std"):
                    fake_data=torch.cat([fakeact_form_buffer_ANO, fakeact_form_buffer_NOR], dim=0)
                elif(self.type_train=="noise_std"):
                    fake_data=torch.cat([real_perturbed,fakeact_form_buffer_NOR], dim=0)
                elif(self.type_train=="noise_3std"):
                    fake_data=torch.cat([real_perturbed,fakeact_form_buffer_ANO], dim=0)
                else:
                    fake_data=None
                    print("type_train is not valid")
                y_fake_hae = torch.ones(fake_data.size(0), 1, device=self._device)
                
                
                #perm = torch.randperm(fake_data.size(0), device=self._device)[:num_samples]
                fake_data_sample = fake_data
                y_fake_hae_sample = y_fake_hae
                if self.save_ae_train_samples:
                    ae_train_generated_store = self._append_ae_train_sample_store(
                        ae_train_generated_store,
                        fake_data_sample,
                        self.ae_train_sample_max_rows,
                    )
                if self.type_val == "val_multi_class":
                    # Store the synthetic-abnormal data stream used by AE so it can
                    # be combined with normal-only validation for AUC-based selection.
                    if self.use_buffer:
                        val_synth_store = self.to_buffer(
                            val_synth_store, fake_data_sample, self._batch_size, self.buffer_multiplier
                        )
                    else:
                        val_synth_store = fake_data_sample.detach()
                
                #real_data_hae = torch.cat([real_data, fakeact_form_buffer_NOR], dim=0)
                #y_real_hae = torch.zeros(real_data_hae.size(0), 1, device=self._device)
                y_real = torch.zeros(real_data.size(0), 1, device=self._device)
                  
                
                hae_dataset=torch.cat([fake_data_sample, real_data], dim=0)
                hae_dataset=torch.cat([fake_data_sample, real_data], dim=0)
                y_hae=torch.cat([y_fake_hae_sample, y_real], dim=0)
                hae_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(hae_dataset, y_hae),
                    batch_size=self._batch_size,
                    shuffle=True
                )
                hae_loss, re_loss, latent_loss,list_latent = self._hae.train_hae(epoch=i, train_data_loader=hae_dataloader)
                
            current_val_loader = val_data_loader
            if (
                self.type_val == "val_multi_class"
                and val_synth_store is not None
                and val_synth_store.size(0) > 0
            ):
                synth_for_val = val_synth_store.detach().cpu().numpy().astype("float32")
                n_val_normal = X_val_train_processed.shape[0]
                if synth_for_val.shape[0] > n_val_normal:
                    idx = np.random.choice(synth_for_val.shape[0], size=n_val_normal, replace=False)
                    synth_for_val = synth_for_val[idx]

                x_val_mix = np.concatenate([X_val_train_processed, synth_for_val], axis=0)
                y_val_mix = np.concatenate(
                    [
                        np.zeros(X_val_train_processed.shape[0], dtype=np.float32),
                        np.ones(synth_for_val.shape[0], dtype=np.float32),
                    ],
                    axis=0,
                )
                val_mix_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(x_val_mix, dtype=torch.float32),
                    torch.tensor(y_val_mix, dtype=torch.float32),
                )
                current_val_loader = torch.utils.data.DataLoader(
                    val_mix_dataset, batch_size=self._batch_size, shuffle=False
                )

            (
                avg_val_total_loss,
                threshold_re,
                threshold_z_0,
                threshold_95,
                threshold_95_total_loss,
                val_auc_total,
                threshold_auc_total,
            ) = self._hae.validate(
                current_val_loader,
                epoch=i
            )

            if self.type_val == "val_multi_class" and val_auc_total is not None:
                is_better = val_auc_total > best_val_auc_total
                if is_better:
                    best_val_auc_total = val_auc_total
                    if self._on_epoch_best_callbacks is not None:
                        self._on_epoch_best_callbacks(
                            model=self,
                            epoch=i,
                            threshold_re=threshold_re,
                            threshold_z_0=threshold_z_0,
                            threshold_95=threshold_95,
                            threshold_95_total_loss=threshold_auc_total
                        )
                    print(f"[VAL-MULTI] epoch={i} auc_total={val_auc_total:.6f} (best)")
            else:
                if avg_val_total_loss < best_val_avg_total_loss:
                    best_val_avg_total_loss = avg_val_total_loss
                    if self._on_epoch_best_callbacks is not None:
                        self._on_epoch_best_callbacks(
                            model=self,
                            epoch=i,
                            threshold_re=threshold_re,
                            threshold_z_0=threshold_z_0,
                            threshold_95=threshold_95,
                            threshold_95_total_loss=threshold_95_total_loss
                        )
                    print(
                        f"[VAL-ONE] epoch={i} avg_total_loss={avg_val_total_loss:.6f} "
                        f"p95_total={threshold_95_total_loss:.6f} (best)"
                    )
                           
            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            self._hae.plot_latent_first_component_hist_from_latents(
                np.asarray(list_latent),
                f"hae_latent_hist_{epochs}",  # Ensure out_dir is a string, not ndarray
                i,
            )
            
            epoch_loss_df = pd.DataFrame({
            'Epoch': [i],
            'Generator Loss': [generator_loss],
            'Discriminator Loss': [discriminator_loss],
            'HAE Loss': [hae_loss.item() if torch.is_tensor(hae_loss) else hae_loss],
            'Reconstruction Loss': [re_loss.item() if torch.is_tensor(re_loss) else re_loss],
            'Latent Loss': [latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss]
            })
            print(f"Epoch {i}: Generator Loss = {generator_loss}, Discriminator Loss = {discriminator_loss}, HAE Loss = {hae_loss}, Reconstruction Loss = {re_loss}, Latent Loss = {latent_loss}", "data hae shape: ",hae_dataset.shape)
            
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(
                        gen=_format_score(generator_loss),
                        dis=_format_score(discriminator_loss),
                    )
                )
            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)
        
        self._save_ae_train_sample_store(ae_train_generated_store)

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        enable_gpu = getattr(self, '_enable_gpu', True)
        self._device = _set_device(enable_gpu, device)
        if self._generator is not None:
            self._generator.to(self._device)
        
        if self._hae is not None:
            self._hae.to(self._device)
            
    def get_score_ha(self, X,y):
        """Compute anomaly scores through the HAE branch.
        Input data is transformed before scoring in latent/reconstruction space.
        """
        X_processed = self._transformer.transform(X)
        score_res, latent_loss, y_true = self._hae.get_score_ha(X_processed, y,batch_size=self._batch_size,device=self._device)
        return score_res, latent_loss, y_true

    def get_latent_z_ha(self, X, batch_size=None):
        """Extract latent vectors from the HAE encoder.
        Useful for diagnostics and downstream analysis.
        """
        if batch_size is None:
            batch_size = self._batch_size
        X_processed = self._transformer.transform(X)
        if isinstance(X_processed, np.ndarray):
            X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        else:
            X_tensor = X_processed.float()
        X_tensor = X_tensor.to(self._device)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
        z_all = []
        self._hae.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device)
                z, _ = self._hae.forward(xb)
                z_all.append(z.cpu().numpy())
        return np.concatenate(z_all)
    def get_threshold_ha(self, X_val_train):
        """Estimate anomaly thresholds from validation-normal data.
        Returns reconstruction/latent/percentile thresholds for inference.
        """
        X_val_train_processed = self._transformer.transform(X_val_train)
        y_val_train = np.zeros(X_val_train_processed.shape[0])
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_train_processed, dtype=torch.float32), torch.tensor(y_val_train, dtype=torch.float32))
        val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
        
        (
            avg_loss,
            threshold_re,
            threshold_z_0,
            threshold_95,
            threshold_95_total_loss,
            _val_auc_total,
            _threshold_auc_total,
        ) = self._hae.validate(val_data_loader,epoch=0)
        return avg_loss, threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss
    
    @staticmethod
    def to_buffer(buffer, new_fake, buf_per_iter, buffer_multiplier=8):
        """Append generated samples into replay buffer with size control.
        The buffer keeps a rolling set of synthetic anomalies for HAE training.
        """
        new_fake = new_fake.clone().detach()
        batch_size, data_dim = new_fake.shape  # new_fake: (B, D)

        perm = torch.randperm(batch_size)
        idx = perm[:buf_per_iter]
        new_buffer = new_fake.detach()[idx]  # (buf_per_iter, D)

        if buffer is None:
            buffer = new_buffer  # (buf_per_iter, D)
        else:
            buffer = buffer.clone().detach().to(new_fake.device)

            buffer = torch.cat((buffer, new_buffer), dim=0)

            max_buffer_size = max(1, int(buffer_multiplier)) * buf_per_iter
            keep_size = min(max_buffer_size, buffer.size(0))

            perm = torch.randperm(buffer.size(0), device=buffer.device)
            idx = perm[:keep_size]
            buffer = buffer[idx].detach()

        return buffer



    @staticmethod
    def from_buffer(buffer, num_of_imgs):
        """Randomly sample a mini-batch from replay buffer.
        If the buffer is smaller than requested, all available items are returned.
        """
        buffer = buffer.clone().detach()
        perm = torch.randperm(buffer.size(0))
        idx = perm[:min(num_of_imgs, buffer.size(0))]
        return buffer[idx]
    def get_continuous_indices(self, device=None):
        """Return transformed indices that belong to continuous tanh spans."""
        if self._transformer is None:
            raise RuntimeError("DataTransformer must be fitted before locating continuous indices.")

        continuous_idx = []
        st = 0

        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                ed = st + span_info.dim

                if span_info.activation_fn == "tanh":
                    continuous_idx.extend(range(st, ed))

                st = ed

        return torch.tensor(continuous_idx, device=device or self._device, dtype=torch.long)
    # Previous boundary anomaly implementation, kept for comparison:
    # - If fake_batch size differed from real_batch, fake rows were sampled at random.
    # - Direction was computed by index: fake_batch[:, cont_idx] - real_batch[:, cont_idx].
    # - This could pair a real row with an unrelated fake row.
    # - The new version matches each real row to its nearest fake row in standardized
    #   continuous space, then perturbs real data with the original unstandardized direction.
    def make_ctgan_boundary_anomaly(self, real_batch, fake_batch, beta=0.1, perturb_prob=0.2):
        """Create boundary pseudo-anomalies near real rows using nearest fake directions."""
        if fake_batch is None:
            return real_batch.clone()

        x = real_batch.clone()
        fake_batch = fake_batch.detach().to(real_batch.device)
        if fake_batch.size(0) == 0:
            return x
        cont_idx = self.get_continuous_indices(device=real_batch.device)
        if cont_idx.numel() == 0:
            return x

        real_cont = real_batch[:, cont_idx]
        fake_cont = fake_batch[:, cont_idx]

        # Standardization is used only for nearest-neighbor matching, so features
        # with small variance do not dominate the distance calculation.
        feature_std = real_cont.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-3)
        real_scaled = real_cont / feature_std
        fake_scaled = fake_cont / feature_std

        dist = torch.cdist(real_scaled, fake_scaled, p=2)
        nearest_idx = dist.argmin(dim=1)

        matched_fake = fake_cont[nearest_idx]
        direction = matched_fake - real_cont

        perturb_prob = min(max(float(perturb_prob), 0.0), 1.0)
        mask = (torch.rand_like(direction) < perturb_prob).float()
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any() and mask.size(1) > 0:
            rand_cols = torch.randint(
                0,
                mask.size(1),
                (int(empty_rows.sum().item()),),
                device=mask.device,
            )
            mask[empty_rows, rand_cols] = 1.0

        direction = direction * mask
        alpha = torch.rand(real_cont.size(0), 1, device=real_cont.device) * float(beta)

        x[:, cont_idx] = real_cont + alpha * direction
        x[:, cont_idx] = torch.clamp(x[:, cont_idx], -1.0, 1.0)

        return x
    #chỉ phá continuous features, không làm hỏng phần categorical/one-hot
    def make_unsw_scale_anomaly(self,real_batch, noise=0.1):
        """Create scaled anomalies by multiplicative noise on real samples.
        This strategy perturbs feature magnitudes while preserving rough structure.
        """
        return real_batch * (1 + noise * torch.randn_like(real_batch))

    def make_unsw_mix_anomaly(self, real_batch):
        """Create mixed anomalies by feature shuffling inside a batch.
        This strategy breaks cross-feature relationships to simulate abnormality.
        """
        B, D = real_batch.shape
        x = real_batch.clone()
        mask = torch.rand(D, device=x.device) < 0.3
        for j in range(D):
            if mask[j]:
                x[:, j] = x[torch.randperm(B, device=x.device), j]
        return x
    
    def set_epoch_end_callback(self, callback):
        """Register callback executed at the end of each epoch.
        Callback signature is callback(model=self, epoch=int).
        """
        self._on_epoch_end_callbacks = callback
    def set_epoch_best_callback(self, callback):
        """Register callback executed when a new best epoch is found.
        Callback receives model, epoch and threshold metadata.
        """
        self._on_epoch_best_callbacks = callback

