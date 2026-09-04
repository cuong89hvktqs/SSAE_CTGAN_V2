"""Reduced-space MAE-CTGAN.

This variant keeps the CTGAN training path unchanged, but trains the MAE in a
reduced source space. Generated CTGAN batches are first inverse-transformed back
from CTGAN's expanded representation, then scaled with a MinMaxScaler fitted on
normal training rows.
"""

import warnings
from math import sqrt

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.maectgan import (
    Discriminator,
    Generator,
    MAE,
    MAECTGAN,
)


class REDUCEMAECTGAN(MAECTGAN):
    """MAE-CTGAN where MAE is trained after reducing CTGAN samples.

    CTGAN still learns on its own transformed representation. Only the MAE
    branch is changed:

    CTGAN expanded sample -> inverse_transform -> source feature space
    -> MinMaxScaler fitted on normal train rows -> MAE.
    """

    def __init__(self, *args, mae_scaler=None, mae_input_columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mae_scaler = mae_scaler
        self._mae_input_columns = (
            [str(column) for column in mae_input_columns]
            if mae_input_columns is not None
            else None
        )
        self._mae_fill_values = None
        self._mae_category_maps = {}

    def fit_mae_scaler(self, source_data):
        """Fit the MinMaxScaler on normal source-space training rows."""
        source_frame = self._source_frame(source_data, fit_columns=True)
        self._mae_fill_values = source_frame.median(numeric_only=True).fillna(0.0)
        source_frame = source_frame.fillna(self._mae_fill_values)
        self._mae_scaler = MinMaxScaler()
        self._mae_scaler.fit(source_frame)
        self._mae_input_columns = [str(column) for column in source_frame.columns]
        return self._mae_scaler

    def _source_frame(self, source_data, fit_columns=False):
        """Return numeric source-space data in the scaler column order."""
        if isinstance(source_data, pd.DataFrame):
            source_frame = source_data.copy()
        else:
            columns = self._mae_input_columns
            source_frame = pd.DataFrame(source_data, columns=columns)

        source_frame.columns = source_frame.columns.map(str)
        if fit_columns or self._mae_input_columns is None:
            self._mae_input_columns = [str(column) for column in source_frame.columns]
        else:
            missing = [
                column for column in self._mae_input_columns
                if column not in source_frame.columns
            ]
            if missing:
                raise ValueError(f"Missing MAE source columns: {missing}")
            source_frame = source_frame.loc[:, self._mae_input_columns]

        for column in source_frame.columns:
            series = source_frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            has_nonnumeric_values = numeric.isna() & series.notna()

            if fit_columns and has_nonnumeric_values.any():
                categories = pd.Series(series.dropna().astype(str).unique())
                self._mae_category_maps[column] = {
                    value: idx for idx, value in enumerate(categories)
                }

            if column in self._mae_category_maps:
                mapped = series.astype(str).map(self._mae_category_maps[column])
                source_frame[column] = mapped.where(series.notna(), np.nan)
                unknown_mask = series.notna() & mapped.isna()
                source_frame.loc[unknown_mask, column] = -1.0
            else:
                source_frame[column] = numeric

        source_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self._mae_fill_values is not None:
            source_frame = source_frame.fillna(self._mae_fill_values)
        else:
            source_frame = source_frame.fillna(0.0)

        return source_frame.astype(np.float32)

    def transform_source_to_mae_space(self, source_data):
        """Scale source-space data with the normal-train MinMaxScaler."""
        if self._mae_scaler is None:
            raise RuntimeError("MAE MinMaxScaler has not been fitted.")

        source_frame = self._source_frame(source_data)
        scaled = self._mae_scaler.transform(source_frame)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
        return pd.DataFrame(scaled.astype(np.float32), columns=self._mae_input_columns)

    def _ctgan_batch_to_mae_tensor(self, batch):
        """Convert a CTGAN transformed tensor batch to scaled MAE tensor input."""
        if self._transformer is None:
            raise RuntimeError("CTGAN DataTransformer has not been fitted.")

        batch_np = batch.detach().cpu().numpy()
        source_batch = self._transformer.inverse_transform(batch_np)
        scaled_batch = self.transform_source_to_mae_space(source_batch)
        return torch.tensor(
            scaled_batch.to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self._device,
        )

    def _scaled_loader(self, scaled_data, labels=None, shuffle=False):
        if isinstance(scaled_data, pd.DataFrame):
            values = scaled_data.to_numpy(dtype=np.float32)
        else:
            values = np.asarray(scaled_data, dtype=np.float32)

        if labels is None:
            labels = np.zeros(values.shape[0], dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        dataset = TensorDataset(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle)

    def _ctgan_loader(self, ctgan_data, labels=None):
        if isinstance(ctgan_data, pd.DataFrame):
            values = ctgan_data.to_numpy(dtype=np.float32)
        else:
            values = np.asarray(ctgan_data, dtype=np.float32)

        if labels is None:
            labels = np.zeros(values.shape[0], dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        dataset = TensorDataset(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=False)

    @random_state
    def fit(self, full_data, train_data, val_data, discrete_columns=(), epochs=None):
        print(
            "Starting REDUCEMAECTGAN training..., data shape:",
            train_data.shape,
            " discrete_columns:",
            discrete_columns,
            " epochs:",
            self._epochs,
            "lamda:",
            self.lamda,
            " margin:",
            self.margin,
        )
        fake_buff_list_NOR = None
        fake_buff_list_ANO = None
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will "
                    "be removed in a future version. Please pass `epochs` to the "
                    "constructor instead"
                ),
                DeprecationWarning,
            )

        if self._mae_scaler is None:
            self.fit_mae_scaler(train_data)

        self._transformer = DataTransformer()
        self._transformer.fit(full_data, discrete_columns)

        train_data = self._transformer.transform(train_data)
        print("CTGAN data after transform shape:", train_data.shape)
        print(
            "Reduced MAE input shape:",
            (len(train_data), len(self._mae_input_columns)),
        )

        val_ctgan_processed = self._transformer.transform(val_data)
        val_scaled = self.transform_source_to_mae_space(val_data)
        y_val_train = np.zeros(val_scaled.shape[0], dtype=np.float32)
        val_data_loader = self._scaled_loader(val_scaled, y_val_train, shuffle=False)
        val_ctgan_loader = self._ctgan_loader(val_ctgan_processed, y_val_train)
        self._set_validation_continuous_stats(
            torch.tensor(
                np.asarray(val_ctgan_processed, dtype=np.float32),
                dtype=torch.float32,
                device=self._device,
            )
        )

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency,
        )

        data_dim = self._transformer.output_dimensions
        mae_input_dim = len(self._mae_input_columns)
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)
        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)
        self._mae = MAE(
            input_dim=mae_input_dim,
            lamda=self.lamda,
            margin=self.margin,
            mae_loss_mode=self.mae_loss_mode,
        ).to(self._device)
        self._mae.device = self._device

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizerD = optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        self.loss_values = pd.DataFrame(columns=[
            "Epoch",
            "Generator Loss",
            "Discriminator Loss",
            "MAE Loss",
            "Reconstruction Loss",
            "Latent Loss",
            "Validation Loss",
            "Validation Reconstruction Loss",
            "Validation Latent Loss",
            "Pseudo Val AUC",
            "Pseudo Separation",
        ])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = "Gen. ({gen:.2f}) | Disc. ({dis:.2f})"
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        best_pseudo_val_auc = -float("inf")
        best_pseudo_val_separation = -float("inf")
        mae_start_epoch = int(np.ceil(epochs * self.mae_start_ratio))
        ctgan_noise_start_epoch = int(np.ceil(epochs * self.ctgan_noise_start_ratio))
        print(
            "[SCHEDULE] MAE pseudo-anomaly training starts after",
            mae_start_epoch,
            "of",
            epochs,
            "epochs; generated-sample Gaussian noise starts after",
            ctgan_noise_start_epoch,
            "epochs with std",
            self.ctgan_noise_std,
            "on feature ratio",
            self.ctgan_noise_feature_ratio,
            "; pseudo-anomaly mode",
            self.ctgan_pseudo_anomaly_mode,
            "with feature outlier r",
            self.ctgan_feature_outlier_r,
            "; MAE loss mode",
            self.mae_loss_mode,
            ".",
        )
        if mae_start_epoch >= epochs:
            print("[WARN] mae_start_ratio leaves no epochs for MAE training in this run.")

        for i in epoch_iterator:
            mae_active = i >= mae_start_epoch
            ctgan_noise_active = mae_active and i >= ctgan_noise_start_epoch
            gaussian_noise_active = (
                ctgan_noise_active and self.ctgan_pseudo_anomaly_mode == "origin"
            )
            mae_loss = np.nan
            re_loss = np.nan
            latent_loss = np.nan
            mae_dataset = None

            for _ in range(steps_per_epoch):
                real_data = None
                loss_d = torch.tensor(np.nan, device=self._device)
                for _ in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    fakez_3 = torch.normal(mean=mean, std=std * sqrt(3))
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)

                    if c1 is not None:
                        fakez = torch.cat([fakez, c1], dim=1)
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                        fakez_3 = torch.cat([fakez_3, c1], dim=1)
                    fake_3 = self._generator(fakez_3)
                    fakeact_3 = self._apply_activate(fake_3)

                    if condvec is None:
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    real = torch.from_numpy(real.astype("float32")).to(self._device)
                    real_data = real
                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        fake_cat = fakeact
                        real_cat = real

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)
                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                fakez_3 = torch.normal(mean=mean, std=std * sqrt(3))
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, _, _ = None, None, None, None
                else:
                    c1, m1, _, _ = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                    fakez_3 = torch.cat([fakez_3, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_3 = self._generator(fakez_3)
                fakeact_3 = self._apply_activate(fake_3)

                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)

                cross_entropy = 0 if condvec is None else self._cond_loss(fake, c1, m1)
                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                if not mae_active:
                    continue

                if self.ctgan_pseudo_anomaly_mode == "feature_outlier":
                    fakeact_for_mae_nor = self.make_ctgan_feature_outlier_anomaly(
                        real_data,
                        fakeact,
                        r=self.ctgan_feature_outlier_r,
                    )
                    fakeact_for_mae_ano = self.make_ctgan_feature_outlier_anomaly(
                        real_data,
                        fakeact_3,
                        r=self.ctgan_feature_outlier_r,
                    )
                else:
                    fakeact_for_mae_nor = fakeact
                    fakeact_for_mae_ano = fakeact_3
                    if gaussian_noise_active:
                        fakeact_for_mae_nor = self.add_gaussian_noise_to_ctgan_samples(
                            fakeact_for_mae_nor,
                            noise_std=self.ctgan_noise_std,
                            feature_ratio=self.ctgan_noise_feature_ratio,
                        )

                fake_buff_list_NOR = self.to_buffer(
                    fake_buff_list_NOR, fakeact_for_mae_nor, self._batch_size
                )
                fake_buff_list_ANO = self.to_buffer(
                    fake_buff_list_ANO, fakeact_for_mae_ano, self._batch_size
                )
                real_perturbed = self.make_real_perturbed_anomaly(real_data)

                if fake_buff_list_NOR is None:
                    fakeact_form_buffer_NOR = fakeact_for_mae_nor
                else:
                    fakeact_form_buffer_NOR = self.from_buffer(
                        fake_buff_list_NOR, self._batch_size
                    )

                if fake_buff_list_ANO is None:
                    fakeact_form_buffer_ANO = fakeact_for_mae_ano
                else:
                    fakeact_form_buffer_ANO = self.from_buffer(
                        fake_buff_list_ANO, self._batch_size
                    )

                if self.type_train == "all":
                    fake_data = torch.cat(
                        [fakeact_form_buffer_ANO, real_perturbed, fakeact_form_buffer_NOR],
                        dim=0,
                    )
                elif self.type_train == "noise":
                    fake_data = real_perturbed
                elif self.type_train == "3std":
                    fake_data = fakeact_form_buffer_ANO
                elif self.type_train == "std":
                    fake_data = fakeact_form_buffer_NOR
                elif self.type_train == "std_3std":
                    fake_data = torch.cat(
                        [fakeact_form_buffer_ANO, fakeact_form_buffer_NOR],
                        dim=0,
                    )
                elif self.type_train == "noise_std":
                    fake_data = torch.cat([real_perturbed, fakeact_form_buffer_NOR], dim=0)
                elif self.type_train == "noise_3std":
                    fake_data = torch.cat([real_perturbed, fakeact_form_buffer_ANO], dim=0)
                else:
                    raise ValueError(f"type_train is not valid: {self.type_train}")

                fake_data_mae = self._ctgan_batch_to_mae_tensor(fake_data)
                real_data_mae = self._ctgan_batch_to_mae_tensor(real_data)
                y_fake_mae = torch.ones(fake_data_mae.size(0), 1, device=self._device)
                y_real = torch.zeros(real_data_mae.size(0), 1, device=self._device)
                mae_dataset = torch.cat([fake_data_mae, real_data_mae], dim=0)
                y_mae = torch.cat([y_fake_mae, y_real], dim=0)
                mae_dataloader = DataLoader(
                    TensorDataset(mae_dataset, y_mae),
                    batch_size=self._batch_size,
                    shuffle=True,
                )
                mae_loss, re_loss, latent_loss, _ = self._mae.train_mae(
                    epoch=i,
                    train_data_loader=mae_dataloader,
                )

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            avg_val_total_loss = np.nan
            avg_val_re_loss = np.nan
            avg_val_latent_loss = np.nan
            pseudo_val_auc = np.nan
            pseudo_val_separation = np.nan

            if mae_active:
                (
                    avg_val_total_loss,
                    threshold_re,
                    threshold_z_0,
                    threshold_95,
                    threshold_95_total_loss,
                ) = self._mae.validate(val_data_loader, epoch=i)
                avg_val_re_loss = getattr(
                    self._mae,
                    "last_validation_reconstruction_loss",
                    np.nan,
                )
                avg_val_latent_loss = getattr(
                    self._mae,
                    "last_validation_latent_loss",
                    np.nan,
                )
                pseudo_val_auc, pseudo_val_separation = (
                    self.evaluate_pseudo_validation_auc(val_ctgan_loader)
                )
                is_best_checkpoint = (
                    pseudo_val_auc > best_pseudo_val_auc
                    or (
                        np.isclose(pseudo_val_auc, best_pseudo_val_auc)
                        and pseudo_val_separation > best_pseudo_val_separation
                    )
                )
                if is_best_checkpoint:
                    best_pseudo_val_auc = pseudo_val_auc
                    best_pseudo_val_separation = pseudo_val_separation
                    if self._on_epoch_best_callbacks is not None:
                        self._on_epoch_best_callbacks(
                            model=self,
                            epoch=i,
                            threshold_re=threshold_re,
                            threshold_z_0=threshold_z_0,
                            threshold_95=threshold_95,
                            threshold_95_total_loss=threshold_95_total_loss,
                        )
                    print(
                        f"[BEST] epoch={i} pseudo_val_auc={pseudo_val_auc:.6f} "
                        f"pseudo_separation={pseudo_val_separation:.6f} "
                        f"val_total_loss={avg_val_total_loss:.6f}"
                    )
                mae_data_shape = mae_dataset.shape
            else:
                mae_data_shape = "warmup: MAE not trained"

            epoch_loss_df = pd.DataFrame({
                "Epoch": [i],
                "Generator Loss": [generator_loss],
                "Discriminator Loss": [discriminator_loss],
                "MAE Loss": [mae_loss.item() if torch.is_tensor(mae_loss) else mae_loss],
                "Reconstruction Loss": [
                    re_loss.item() if torch.is_tensor(re_loss) else re_loss
                ],
                "Latent Loss": [
                    latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
                ],
                "Validation Loss": [avg_val_total_loss],
                "Validation Reconstruction Loss": [avg_val_re_loss],
                "Validation Latent Loss": [avg_val_latent_loss],
                "Pseudo Val AUC": [pseudo_val_auc],
                "Pseudo Separation": [pseudo_val_separation],
            })
            print(
                f"Epoch {i}: Generator Loss = {generator_loss}, "
                f"Discriminator Loss = {discriminator_loss}, MAE Loss = {mae_loss}, "
                f"Reconstruction Loss = {re_loss}, Latent Loss = {latent_loss}, "
                f"CTGAN Gaussian noise active = {gaussian_noise_active}, "
                f"Pseudo Val AUC = {pseudo_val_auc}, "
                f"Pseudo Separation = {pseudo_val_separation}",
                "data mae shape:",
                mae_data_shape,
            )

            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)

    def evaluate_pseudo_validation_auc(self, val_ctgan_data_loader):
        """Evaluate normal-vs-pseudo separation in reduced MAE space."""
        self._mae.eval()
        normal_scores = []
        pseudo_scores = []

        with torch.no_grad():
            for xb_ctgan, _ in val_ctgan_data_loader:
                xb_ctgan = xb_ctgan.to(self._device)
                xb_pseudo_ctgan = self.make_validation_pseudo_anomaly(xb_ctgan)
                xb_mae = self._ctgan_batch_to_mae_tensor(xb_ctgan)
                xb_pseudo_mae = self._ctgan_batch_to_mae_tensor(xb_pseudo_ctgan)

                z_normal, _ = self._mae.forward(xb_mae)
                z_pseudo, _ = self._mae.forward(xb_pseudo_mae)

                normal_scores.append(torch.norm(z_normal, p=2, dim=1).cpu().numpy())
                pseudo_scores.append(torch.norm(z_pseudo, p=2, dim=1).cpu().numpy())

        if not normal_scores or not pseudo_scores:
            return 0.0, -float("inf")

        normal_scores = np.concatenate(normal_scores)
        pseudo_scores = np.concatenate(pseudo_scores)
        y_pseudo = np.concatenate([
            np.zeros_like(normal_scores, dtype=np.int64),
            np.ones_like(pseudo_scores, dtype=np.int64),
        ])
        score_pseudo = np.concatenate([normal_scores, pseudo_scores])
        pseudo_auc = roc_auc_score(y_pseudo, score_pseudo)
        pseudo_separation = float(np.mean(pseudo_scores) - np.mean(normal_scores))
        return float(pseudo_auc), pseudo_separation

    def get_score_mae(self, X, y, already_scaled=False):
        if already_scaled:
            X_mae = np.asarray(X, dtype=np.float32)
        else:
            X_mae = self.transform_source_to_mae_space(X).to_numpy(dtype=np.float32)
        return self._mae.get_score_mae(
            X_mae,
            y,
            batch_size=self._batch_size,
            device=self._device,
        )

    def get_threshold_mae(self, X_val_train, already_scaled=False):
        if already_scaled:
            X_mae = np.asarray(X_val_train, dtype=np.float32)
        else:
            X_mae = self.transform_source_to_mae_space(X_val_train).to_numpy(dtype=np.float32)
        y_val_train = np.zeros(X_mae.shape[0], dtype=np.float32)
        val_data_loader = self._scaled_loader(X_mae, y_val_train, shuffle=False)
        return self._mae.validate(val_data_loader, epoch=0)

    def get_latent_z_mae(self, X, batch_size=None, already_scaled=False):
        if batch_size is None:
            batch_size = self._batch_size
        if already_scaled:
            X_mae = np.asarray(X, dtype=np.float32)
        else:
            X_mae = self.transform_source_to_mae_space(X).to_numpy(dtype=np.float32)

        X_tensor = torch.tensor(X_mae, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
        z_all = []
        self._mae.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device)
                z, _ = self._mae.forward(xb)
                z_all.append(z.cpu().numpy())
        return np.concatenate(z_all)
