"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, Tanh, MSELoss,L1Loss,Sigmoid
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
import torch.nn.functional as F
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
    
from torch.utils.data import DataLoader, TensorDataset
class Discriminator(Module):
    """MÃ¡ÂºÂ¡ng phÃƒÂ¢n biÃ¡Â»â€¡t (Discriminator) cho CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        """KhÃ¡Â»Å¸i tÃ¡ÂºÂ¡o mÃ¡ÂºÂ¡ng phÃƒÂ¢n biÃ¡Â»â€¡t.

        Tham sÃ¡Â»â€˜:
            input_dim (int): KÃƒÂ­ch thÃ†Â°Ã¡Â»â€ºc Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o.
            discriminator_dim (list): KÃƒÂ­ch thÃ†Â°Ã¡Â»â€ºc cÃƒÂ¡c lÃ¡Â»â€ºp cÃ¡Â»Â§a mÃ¡ÂºÂ¡ng phÃƒÂ¢n biÃ¡Â»â€¡t.
            pac (int): SÃ¡Â»â€˜ lÃƒÂ¡t (pack) Ã„â€˜Ã¡Â»Æ’ nhÃƒÂ³m dÃ¡Â»Â¯ liÃ¡Â»â€¡u.

        TrÃ¡ÂºÂ£ vÃ¡Â»Â:
            None
        """
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            # LÃ¡Â»â€ºp tuyÃ¡ÂºÂ¿n tÃƒÂ­nh Ã„â€˜Ã¡ÂºÂ§u tiÃƒÂªn nhÃ¡ÂºÂ­n Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o Ã„â€˜ÃƒÂ£ Ã„â€˜Ã†Â°Ã¡Â»Â£c pack.
            # CÃƒÂ¡c lÃ¡Â»â€ºp tiÃ¡ÂºÂ¿p theo nhÃ¡ÂºÂ­n Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a lÃ¡Â»â€ºp trÃ†Â°Ã¡Â»â€ºc Ã„â€˜ÃƒÂ³.
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        # LÃ¡Â»â€ºp cuÃ¡Â»â€˜i cÃƒÂ¹ng lÃƒÂ  lÃ¡Â»â€ºp tuyÃ¡ÂºÂ¿n tÃƒÂ­nh vÃ¡Â»â€ºi mÃ¡Â»â„¢t Ã„â€˜Ã¡ÂºÂ§u ra.
        # Ã„ÂÃƒÂ¢y lÃƒÂ  xÃƒÂ¡c suÃ¡ÂºÂ¥t Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o lÃƒÂ  dÃ¡Â»Â¯ liÃ¡Â»â€¡u thÃ¡ÂºÂ­t.
        seq += [Linear(dim, 1)] #1 Ã„â€˜Ã¡ÂºÂ§u ra Ã„â€˜Ã¡ÂºÂ¡i diÃ¡Â»â€¡n cho logit/score: sample nÃƒÂ y lÃƒÂ  real hay fake (chÃ†Â°a sigmoid, vÃƒÂ¬ WGAN loss dÃƒÂ¹ng raw score).
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """
        TÃƒÂ­nh toÃƒÂ¡n gradient penalty.

        Ã„ÂÃƒÂ¢y lÃƒÂ  mÃ¡Â»â„¢t thÃƒÂ nh phÃ¡ÂºÂ§n phÃ¡ÂºÂ¡t Ã„â€˜Ã†Â°Ã¡Â»Â£c cÃ¡Â»â„¢ng vÃƒÂ o loss cÃ¡Â»Â§a mÃ¡ÂºÂ¡ng phÃƒÂ¢n biÃ¡Â»â€¡t.
        NÃƒÂ³ tÃ¡Â»Â· lÃ¡Â»â€¡ vÃ¡Â»â€ºi bÃƒÂ¬nh phÃ†Â°Ã†Â¡ng Ã„â€˜Ã¡Â»â„¢ lÃ¡Â»â€¡ch giÃ¡Â»Â¯a chuÃ¡ÂºÂ©n cÃ¡Â»Â§a gradient vÃƒÂ  1.

        Gradient Ã„â€˜Ã†Â°Ã¡Â»Â£c tÃƒÂ­nh lÃƒÂ  gradient cÃ¡Â»Â§a Ã„â€˜Ã¡ÂºÂ§u ra mÃ¡ÂºÂ¡ng phÃƒÂ¢n biÃ¡Â»â€¡t theo Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o.
        Ã„ÂÃ¡ÂºÂ§u vÃƒÂ o lÃƒÂ  tÃ¡Â»â€¢ hÃ¡Â»Â£p tuyÃ¡ÂºÂ¿n tÃƒÂ­nh giÃ¡Â»Â¯a mÃ¡ÂºÂ«u dÃ¡Â»Â¯ liÃ¡Â»â€¡u thÃ¡ÂºÂ­t vÃƒÂ  giÃ¡ÂºÂ£, tÃ¡Â»â€¢ hÃ¡Â»Â£p nÃƒÂ y Ã„â€˜Ã†Â°Ã¡Â»Â£c lÃ¡ÂºÂ¥y ngÃ¡ÂºÂ«u nhiÃƒÂªn tÃ¡Â»Â« [0, 1].
        """
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        # Compute the linear combination of real and fake data.
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        # Compute the discriminator's output given the linear combination.
        disc_interpolates = self(interpolates)

        # Compute the gradient of the discriminator's output with respect
        # to the linear combination.
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Compute the norm of the gradient.
        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1

        # Compute the gradient penalty.
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """ÃƒÂp dÃ¡Â»Â¥ng mÃ¡ÂºÂ¡ng phÃƒÂ¢n biÃ¡Â»â€¡t lÃƒÂªn `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """LÃ¡Â»â€ºp residual cho CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """ÃƒÂp dÃ¡Â»Â¥ng lÃ¡Â»â€ºp residual lÃƒÂªn `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """MÃ¡ÂºÂ¡ng sinh dÃ¡Â»Â¯ liÃ¡Â»â€¡u (Generator) cho CTGAN."""

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
        """ÃƒÂp dÃ¡Â»Â¥ng mÃ¡ÂºÂ¡ng sinh dÃ¡Â»Â¯ liÃ¡Â»â€¡u lÃƒÂªn `input_`."""
        data = self.seq(input_)
        return data

class REC(Module):
    """AutoEncoder used by RECTGAN with reconstruction-only anomaly training."""

    def __init__(self, input_dim, re_abnormal_target=1.0):
        super(REC, self).__init__()
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
            Sigmoid(),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.re_abnormal_target = float(re_abnormal_target)

    def forward(self, input_):
        encoded = self.enc(input_)
        decoded = self.dec(encoded)
        return encoded, decoded

    @staticmethod
    def reconstruction_error(x, xb_hat):
        return F.mse_loss(xb_hat, x, reduction="none").mean(dim=1)

    def abnormal_reconstruction_target_loss(self, x, xb_hat):
        # sqrt((x - x_hat)^2) is the element-wise absolute reconstruction error.
        abs_error = torch.abs(xb_hat - x)
        target = torch.tensor(
            self.re_abnormal_target,
            device=abs_error.device,
            dtype=abs_error.dtype,
        )
        return ((target - abs_error) ** 2).mean(dim=1)

    def calculate_loss(self, x, encoded, xb_hat, y):
        del encoded
        y = y.view(-1)
        normal_mask = y == 0
        abnormal_mask = y != 0

        normal_re_per_sample = self.reconstruction_error(x, xb_hat)
        abnormal_re_per_sample = self.abnormal_reconstruction_target_loss(x, xb_hat)

        if torch.any(normal_mask):
            normal_re_loss = normal_re_per_sample[normal_mask].mean()
        else:
            normal_re_loss = normal_re_per_sample.mean() * 0.0

        if torch.any(abnormal_mask):
            abnormal_re_loss = abnormal_re_per_sample[abnormal_mask].mean()
        else:
            abnormal_re_loss = abnormal_re_per_sample.mean() * 0.0

        total_loss = normal_re_loss + abnormal_re_loss
        return total_loss, normal_re_loss, abnormal_re_loss

    def train_rec(self, epoch, train_data_loader):
        del epoch
        self.train()
        list_latent = []
        total_loss = normal_re_loss = abnormal_re_loss = torch.tensor(
            0.0, device=self.device
        )
        for input_, label in train_data_loader:
            input_, label = input_.to(self.device), label.to(self.device)
            latent_z, xb_hat = self.forward(input_)
            total_loss, normal_re_loss, abnormal_re_loss = self.calculate_loss(
                input_, latent_z, xb_hat, label
            )
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            list_latent.append(latent_z.detach().cpu().numpy())

        if list_latent:
            list_latent = np.concatenate(list_latent, axis=0)
        else:
            list_latent = np.empty((0, 0))
        return total_loss, normal_re_loss, abnormal_re_loss, list_latent

    def validate(self, val_data_loader, epoch):
        del epoch
        self.eval()
        re_losses = []
        y_val_true = []

        with torch.no_grad():
            for xb, yb in val_data_loader:
                xb = xb.to(self.device)
                _, xb_hat = self.forward(xb)
                re_loss = self.reconstruction_error(xb, xb_hat)
                re_losses.append(re_loss.cpu().numpy())
                y_val_true.append(yb.cpu().numpy())

        re_losses = np.concatenate(re_losses)
        y_val_true = np.concatenate(y_val_true)
        re_normal = re_losses[y_val_true == 0]
        if re_normal.size == 0:
            re_normal = re_losses

        avg_re_loss = float(np.mean(re_normal))
        threshold_re = (float(np.mean(re_normal)), float(np.std(re_normal)))
        threshold_re95 = float(np.percentile(re_normal, 95))
        return avg_re_loss, threshold_re, threshold_re95

    def compute_average_losses(self, data_loader):
        self.eval()
        re_losses = []

        with torch.no_grad():
            for xb, _ in data_loader:
                xb = xb.to(self.device)
                _, xb_hat = self.forward(xb)
                re_losses.append(self.reconstruction_error(xb, xb_hat).cpu().numpy())

        re_losses = np.concatenate(re_losses)
        mean_re = float(np.mean(re_losses))
        return mean_re, mean_re, 0.0

    def get_score_rec(
        self,
        X_test,
        y_test,
        batch_size=125,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.eval()
        device = torch.device(device)
        self.to(device)

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
            shuffle=False,
        )

        if device.type == "cuda":
            for i, (xb, _) in enumerate(test_loader):
                if i == 1:
                    break
                _ = self.forward(xb.to(device))
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        re_errors = []
        y_true = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                _, xb_hat = self.forward(xb)
                re = self.reconstruction_error(xb, xb_hat)
                re_errors.append(re.cpu().numpy())
                y_true.append(yb.cpu().numpy())

        re_errors = np.concatenate(re_errors)
        y_true = np.concatenate(y_true)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        print(f"[Time] get_score_rec: {(t1 - t0) * 1000:.2f} ms")
        return re_errors, y_true


class RECTGAN(BaseSynthesizer):
    """MÃƒÂ´ hÃƒÂ¬nh CTGAN tÃ¡Â»â€¢ng hÃ¡Â»Â£p bÃ¡ÂºÂ£ng cÃƒÂ³ Ã„â€˜iÃ¡Â»Âu kiÃ¡Â»â€¡n.

    Ã„ÂÃƒÂ¢y lÃƒÂ  lÃ¡Â»â€ºp lÃƒÂµi cÃ¡Â»Â§a dÃ¡Â»Â± ÃƒÂ¡n CTGAN, nÃ†Â¡i cÃƒÂ¡c thÃƒÂ nh phÃ¡ÂºÂ§n khÃƒÂ¡c nhau Ã„â€˜Ã†Â°Ã¡Â»Â£c phÃ¡Â»â€˜i hÃ¡Â»Â£p vÃ¡Â»â€ºi nhau.
    Ã„ÂÃ¡Â»Æ’ biÃ¡ÂºÂ¿t thÃƒÂªm chi tiÃ¡ÂºÂ¿t vÃ¡Â»Â quy trÃƒÂ¬nh, xem bÃƒÂ i bÃƒÂ¡o [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503).

    Tham sÃ¡Â»â€˜:
            "rec_latent_hist",  # Ensure out_dir is a string, not ndarray
        generator_dim (tuple hoÃ¡ÂºÂ·c list): KÃƒÂ­ch thÃ†Â°Ã¡Â»â€ºc Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a tÃ¡Â»Â«ng lÃ¡Â»â€ºp Residual. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh (256, 256).
        discriminator_dim (tuple hoÃ¡ÂºÂ·c list): KÃƒÂ­ch thÃ†Â°Ã¡Â»â€ºc Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a tÃ¡Â»Â«ng lÃ¡Â»â€ºp Discriminator. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh (256, 256).
        generator_lr (float): Learning rate cho generator. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh 2e-4.
        generator_decay (float): TrÃ¡Â»Âng sÃ¡Â»â€˜ giÃ¡ÂºÂ£m cho Adam Optimizer cÃ¡Â»Â§a generator. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh 1e-6.
        discriminator_lr (float): Learning rate cho discriminator. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh 2e-4.
        discriminator_decay (float): TrÃ¡Â»Âng sÃ¡Â»â€˜ giÃ¡ÂºÂ£m cho Adam Optimizer cÃ¡Â»Â§a discriminator. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh 1e-6.
        batch_size (int): SÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng mÃ¡ÂºÂ«u dÃ¡Â»Â¯ liÃ¡Â»â€¡u xÃ¡Â»Â­ lÃƒÂ½ mÃ¡Â»â€”i bÃ†Â°Ã¡Â»â€ºc.
        discriminator_steps (int): SÃ¡Â»â€˜ lÃ¡ÂºÂ§n cÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t discriminator cho mÃ¡Â»â€”i lÃ¡ÂºÂ§n cÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t generator.
        log_frequency (bool): CÃƒÂ³ sÃ¡Â»Â­ dÃ¡Â»Â¥ng tÃ¡ÂºÂ§n suÃ¡ÂºÂ¥t log cÃ¡Â»Â§a cÃƒÂ¡c mÃ¡Â»Â©c phÃƒÂ¢n loÃ¡ÂºÂ¡i khi lÃ¡ÂºÂ¥y mÃ¡ÂºÂ«u Ã„â€˜iÃ¡Â»Âu kiÃ¡Â»â€¡n khÃƒÂ´ng. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh True.
        verbose (bool): CÃƒÂ³ in tiÃ¡ÂºÂ¿n trÃƒÂ¬nh ra mÃƒÂ n hÃƒÂ¬nh khÃƒÂ´ng. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh False.
        epochs (int): SÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng epoch huÃ¡ÂºÂ¥n luyÃ¡Â»â€¡n. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh 300.
        pac (int): SÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng mÃ¡ÂºÂ«u nhÃƒÂ³m lÃ¡ÂºÂ¡i khi ÃƒÂ¡p dÃ¡Â»Â¥ng discriminator. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh 10.
        cuda (bool): CÃƒÂ³ sÃ¡Â»Â­ dÃ¡Â»Â¥ng GPU khÃƒÂ´ng. NÃ¡ÂºÂ¿u False hoÃ¡ÂºÂ·c khÃƒÂ´ng cÃƒÂ³ CUDA thÃƒÂ¬ dÃƒÂ¹ng CPU. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh True.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 re_abnormal_target=1.0, type_train="all", rec_start_ratio=0.0,
                 ctgan_noise_start_ratio=0.75, ctgan_noise_std=0.1,
                 ctgan_noise_feature_ratio=1.0, ctgan_pseudo_anomaly_mode="origin",
                 ctgan_feature_outlier_r=1.0):
        """
        HÃƒÂ m khÃ¡Â»Å¸i tÃ¡ÂºÂ¡o cho lÃ¡Â»â€ºp CTGAN. NhÃ¡ÂºÂ­n cÃƒÂ¡c siÃƒÂªu tham sÃ¡Â»â€˜ vÃƒÂ  khÃ¡Â»Å¸i tÃ¡ÂºÂ¡o Ã„â€˜Ã¡Â»â€˜i tÃ†Â°Ã¡Â»Â£ng theo cÃƒÂ¡c tham sÃ¡Â»â€˜ nÃƒÂ y.

        :param embedding_dim: KÃƒÂ­ch thÃ†Â°Ã¡Â»â€ºc mÃ¡ÂºÂ«u ngÃ¡ÂºÂ«u nhiÃƒÂªn Ã„â€˜Ã†Â°a vÃƒÂ o Generator.
        :param generator_dim: KÃƒÂ­ch thÃ†Â°Ã¡Â»â€ºc Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a tÃ¡Â»Â«ng lÃ¡Â»â€ºp Residual.
        :param discriminator_dim: KÃƒÂ­ch thÃ†Â°Ã¡Â»â€ºc Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a tÃ¡Â»Â«ng lÃ¡Â»â€ºp Discriminator.
        :param generator_lr: Learning rate cho generator.
        :param generator_decay: TrÃ¡Â»Âng sÃ¡Â»â€˜ giÃ¡ÂºÂ£m cho Adam Optimizer cÃ¡Â»Â§a generator.
        :param discriminator_lr: Learning rate cho discriminator.
        :param discriminator_decay: TrÃ¡Â»Âng sÃ¡Â»â€˜ giÃ¡ÂºÂ£m cho Adam Optimizer cÃ¡Â»Â§a discriminator.
        :param batch_size: SÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng mÃ¡ÂºÂ«u dÃ¡Â»Â¯ liÃ¡Â»â€¡u xÃ¡Â»Â­ lÃƒÂ½ mÃ¡Â»â€”i bÃ†Â°Ã¡Â»â€ºc.
        :param discriminator_steps: SÃ¡Â»â€˜ lÃ¡ÂºÂ§n cÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t discriminator cho mÃ¡Â»â€”i lÃ¡ÂºÂ§n cÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t generator.
        :param log_frequency: CÃƒÂ³ sÃ¡Â»Â­ dÃ¡Â»Â¥ng tÃ¡ÂºÂ§n suÃ¡ÂºÂ¥t log cÃ¡Â»Â§a cÃƒÂ¡c mÃ¡Â»Â©c phÃƒÂ¢n loÃ¡ÂºÂ¡i khi lÃ¡ÂºÂ¥y mÃ¡ÂºÂ«u Ã„â€˜iÃ¡Â»Âu kiÃ¡Â»â€¡n khÃƒÂ´ng.
        :param verbose: CÃƒÂ³ in tiÃ¡ÂºÂ¿n trÃƒÂ¬nh ra mÃƒÂ n hÃƒÂ¬nh khÃƒÂ´ng.
        :param epochs: SÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng epoch huÃ¡ÂºÂ¥n luyÃ¡Â»â€¡n.
        :param pac: SÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng mÃ¡ÂºÂ«u nhÃƒÂ³m lÃ¡ÂºÂ¡i khi ÃƒÂ¡p dÃ¡Â»Â¥ng discriminator.
        :param cuda: CÃƒÂ³ sÃ¡Â»Â­ dÃ¡Â»Â¥ng GPU khÃƒÂ´ng.
        """

        assert batch_size % 2 == 0, "Batch size must be divisible by 2."

        # Set the hyperparameters
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
        self.re_abnormal_target = float(re_abnormal_target)
        self.type_train=type_train
        self.rec_start_ratio = float(rec_start_ratio)
        self.ctgan_noise_start_ratio = float(ctgan_noise_start_ratio)
        self.ctgan_noise_std = float(ctgan_noise_std)
        self.ctgan_noise_feature_ratio = float(ctgan_noise_feature_ratio)
        self.ctgan_pseudo_anomaly_mode = (
            "feature_outlier"
            if ctgan_pseudo_anomaly_mode == "feature_outliner"
            else str(ctgan_pseudo_anomaly_mode)
        )
        self.ctgan_feature_outlier_r = float(ctgan_feature_outlier_r)
        if not 0.0 <= self.rec_start_ratio <= 1.0:
            raise ValueError("rec_start_ratio must be in [0, 1].")
        if not 0.0 <= self.ctgan_noise_start_ratio <= 1.0:
            raise ValueError("ctgan_noise_start_ratio must be in [0, 1].")
        if self.ctgan_noise_std < 0.0:
            raise ValueError("ctgan_noise_std must be greater than or equal to 0.")
        if not 0.0 <= self.ctgan_noise_feature_ratio <= 1.0:
            raise ValueError("ctgan_noise_feature_ratio must be in [0, 1].")
        if self.ctgan_pseudo_anomaly_mode not in {"origin", "feature_outlier"}:
            raise ValueError(
                "ctgan_pseudo_anomaly_mode must be 'origin' or 'feature_outlier'."
            )
        if self.ctgan_feature_outlier_r < 0.0:
            raise ValueError("ctgan_feature_outlier_r must be greater than or equal to 0.")
        if self.re_abnormal_target < 0.0:
            raise ValueError("re_abnormal_target must be greater than or equal to 0.")

        # Set the device to be used
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        # Set the objects to None
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
        self.loss_values = None
        self._rec=None
        
        self._on_epoch_end_callbacks = None
        self._on_epoch_best_callbacks = None
        self._validation_cont_mean = None
        self._validation_cont_std = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """XÃ¡Â»Â­ lÃƒÂ½ sÃ¡Â»Â± khÃƒÂ´ng Ã¡Â»â€¢n Ã„â€˜Ã¡Â»â€¹nh cÃ¡Â»Â§a gumbel_softmax cho cÃƒÂ¡c phiÃƒÂªn bÃ¡ÂºÂ£n torch cÃ…Â©.

        Tham khÃ¡ÂºÂ£o thÃƒÂªm vÃ¡Â»Â vÃ¡ÂºÂ¥n Ã„â€˜Ã¡Â»Â nÃƒÂ y:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Tham sÃ¡Â»â€˜:
            logits [Ã¢â‚¬Â¦, num_features]: Log xÃƒÂ¡c suÃ¡ÂºÂ¥t chÃ†Â°a chuÃ¡ÂºÂ©n hÃƒÂ³a
            tau: NhiÃ¡Â»â€¡t Ã„â€˜Ã¡Â»â„¢ (temperature) khÃƒÂ´ng ÃƒÂ¢m
            hard (bool): NÃ¡ÂºÂ¿u True, trÃ¡ÂºÂ£ vÃ¡Â»Â vector one-hot, nhÃ†Â°ng vÃ¡ÂºÂ«n tÃƒÂ­nh gradient nhÃ†Â° soft sample
            dim (int): ChiÃ¡Â»Âu Ã„â€˜Ã¡Â»Æ’ tÃƒÂ­nh softmax. MÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh -1.

        TrÃ¡ÂºÂ£ vÃ¡Â»Â:
            Tensor lÃ¡ÂºÂ¥y mÃ¡ÂºÂ«u cÃƒÂ¹ng shape vÃ¡Â»â€ºi logits tÃ¡Â»Â« phÃƒÂ¢n phÃ¡Â»â€˜i Gumbel-Softmax.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """ÃƒÂp dÃ¡Â»Â¥ng hÃƒÂ m kÃƒÂ­ch hoÃ¡ÂºÂ¡t phÃƒÂ¹ hÃ¡Â»Â£p lÃƒÂªn Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a generator."""
        # HÃƒÂ m nÃƒÂ y nhÃ¡ÂºÂ­n Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a generator vÃƒÂ  ÃƒÂ¡p dÃ¡Â»Â¥ng hÃƒÂ m kÃƒÂ­ch hoÃ¡ÂºÂ¡t phÃƒÂ¹ hÃ¡Â»Â£p.
        # Ã„ÂÃ¡ÂºÂ§u ra cÃ¡Â»Â§a generator lÃƒÂ  tensor cÃƒÂ³ shape (batch_size, total_dim), Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a hÃƒÂ m nÃƒÂ y cÃ…Â©ng vÃ¡ÂºÂ­y.

        # LÃ¡ÂºÂ·p qua tÃ¡Â»Â«ng cÃ¡Â»â„¢t trong Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a generator vÃƒÂ  ÃƒÂ¡p dÃ¡Â»Â¥ng hÃƒÂ m kÃƒÂ­ch hoÃ¡ÂºÂ¡t Ã„â€˜Ã†Â°Ã¡Â»Â£c chÃ¡Â»â€° Ã„â€˜Ã¡Â»â€¹nh trong transformer.
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    # For tanh activation function, we just apply torch.tanh
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    # For softmax activation function, we use our custom
                    # _gumbel_softmax function to ensure that we get a valid
                    # probability distribution.
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    # If the activation function is not recognized, raise an error
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        # We concatenate all the transformed columns into a single tensor
        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """TÃƒÂ­nh loss cross entropy trÃƒÂªn cÃƒÂ¡c cÃ¡Â»â„¢t phÃƒÂ¢n loÃ¡ÂºÂ¡i cÃ¡Â»â€˜ Ã„â€˜Ã¡Â»â€¹nh.

        HÃƒÂ m nÃƒÂ y nhÃ¡ÂºÂ­n Ã„â€˜Ã¡ÂºÂ§u ra cÃ¡Â»Â§a generator, vector Ã„â€˜iÃ¡Â»Âu kiÃ¡Â»â€¡n vÃƒÂ  vector mask. Sau Ã„â€˜ÃƒÂ³ tÃƒÂ­nh loss cross entropy trÃƒÂªn tÃ¡Â»Â«ng cÃ¡Â»â„¢t phÃƒÂ¢n loÃ¡ÂºÂ¡i vÃƒÂ  trÃ¡ÂºÂ£ vÃ¡Â»Â tÃ¡Â»â€¢ng loss.

        Ã„ÂÃ¡ÂºÂ§u tiÃƒÂªn khÃ¡Â»Å¸i tÃ¡ÂºÂ¡o hai con trÃ¡Â»Â st vÃƒÂ  st_c Ã„â€˜Ã¡Â»Æ’ theo dÃƒÂµi vÃ¡Â»â€¹ trÃƒÂ­ bÃ¡ÂºÂ¯t Ã„â€˜Ã¡ÂºÂ§u cÃ¡Â»Â§a cÃ¡Â»â„¢t hiÃ¡Â»â€¡n tÃ¡ÂºÂ¡i trong Ã„â€˜Ã¡ÂºÂ§u ra generator vÃƒÂ  vector Ã„â€˜iÃ¡Â»Âu kiÃ¡Â»â€¡n.

        Sau Ã„â€˜ÃƒÂ³ lÃ¡ÂºÂ·p qua tÃ¡Â»Â«ng cÃ¡Â»â„¢t trong Ã„â€˜Ã¡ÂºÂ§u ra generator. NÃ¡ÂºÂ¿u cÃ¡Â»â„¢t khÃƒÂ´ng phÃ¡ÂºÂ£i phÃƒÂ¢n loÃ¡ÂºÂ¡i, chÃ¡Â»â€° cÃ¡ÂºÂ§n tÃ„Æ’ng con trÃ¡Â»Â theo sÃ¡Â»â€˜ chiÃ¡Â»Âu cÃ¡Â»Â§a cÃ¡Â»â„¢t Ã„â€˜ÃƒÂ³.

        NÃ¡ÂºÂ¿u lÃƒÂ  cÃ¡Â»â„¢t phÃƒÂ¢n loÃ¡ÂºÂ¡i, tÃƒÂ­nh loss cross entropy giÃ¡Â»Â¯a Ã„â€˜Ã¡ÂºÂ§u ra generator vÃƒÂ  vector Ã„â€˜iÃ¡Â»Âu kiÃ¡Â»â€¡n, cÃ¡Â»â„¢ng vÃƒÂ o tÃ¡Â»â€¢ng loss.

        CuÃ¡Â»â€˜i cÃƒÂ¹ng trÃ¡ÂºÂ£ vÃ¡Â»Â tÃ¡Â»â€¢ng loss chia cho sÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng mÃ¡ÂºÂ«u dÃ¡Â»Â¯ liÃ¡Â»â€¡u.
        """

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
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """KiÃ¡Â»Æ’m tra cÃƒÂ¡c cÃ¡Â»â„¢t phÃƒÂ¢n loÃ¡ÂºÂ¡i cÃƒÂ³ tÃ¡Â»â€œn tÃ¡ÂºÂ¡i trong dÃ¡Â»Â¯ liÃ¡Â»â€¡u huÃ¡ÂºÂ¥n luyÃ¡Â»â€¡n khÃƒÂ´ng.

        Tham sÃ¡Â»â€˜:
            train_data (numpy.ndarray hoÃ¡ÂºÂ·c pandas.DataFrame): DÃ¡Â»Â¯ liÃ¡Â»â€¡u huÃ¡ÂºÂ¥n luyÃ¡Â»â€¡n, phÃ¡ÂºÂ£i lÃƒÂ  mÃ¡ÂºÂ£ng 2 chiÃ¡Â»Âu hoÃ¡ÂºÂ·c DataFrame.
            discrete_columns (list): Danh sÃƒÂ¡ch cÃƒÂ¡c cÃ¡Â»â„¢t phÃƒÂ¢n loÃ¡ÂºÂ¡i dÃƒÂ¹ng Ã„â€˜Ã¡Â»Æ’ tÃ¡ÂºÂ¡o vector Ã„â€˜iÃ¡Â»Âu kiÃ¡Â»â€¡n. NÃ¡ÂºÂ¿u lÃƒÂ  numpy thÃƒÂ¬ lÃƒÂ  chÃ¡Â»â€° sÃ¡Â»â€˜, nÃ¡ÂºÂ¿u lÃƒÂ  DataFrame thÃƒÂ¬ lÃƒÂ  tÃƒÂªn cÃ¡Â»â„¢t.

        NÃ¡ÂºÂ¿u train_data lÃƒÂ  DataFrame, kiÃ¡Â»Æ’m tra tÃ¡ÂºÂ¥t cÃ¡ÂºÂ£ cÃƒÂ¡c cÃ¡Â»â„¢t trong discrete_columns cÃƒÂ³ trong DataFrame khÃƒÂ´ng. NÃ¡ÂºÂ¿u khÃƒÂ´ng, bÃƒÂ¡o lÃ¡Â»â€”i ValueError.

        NÃ¡ÂºÂ¿u train_data lÃƒÂ  numpy array, kiÃ¡Â»Æ’m tra cÃƒÂ¡c chÃ¡Â»â€° sÃ¡Â»â€˜ cÃƒÂ³ hÃ¡Â»Â£p lÃ¡Â»â€¡ khÃƒÂ´ng. NÃ¡ÂºÂ¿u cÃƒÂ³ chÃ¡Â»â€° sÃ¡Â»â€˜ khÃƒÂ´ng hÃ¡Â»Â£p lÃ¡Â»â€¡, bÃƒÂ¡o lÃ¡Â»â€”i ValueError.

        NÃ¡ÂºÂ¿u train_data khÃƒÂ´ng phÃ¡ÂºÂ£i DataFrame hoÃ¡ÂºÂ·c numpy array, bÃƒÂ¡o lÃ¡Â»â€”i TypeError.
        """
        if isinstance(train_data, pd.DataFrame):
            # If train_data is a pandas DataFrame, we check if all the columns
            # in discrete_columns are present in the DataFrame
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            # If train_data is a numpy array, we check if all the columns in
            # discrete_columns are valid indices into the array
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            # If train_data is not either a pandas DataFrame or a numpy array,
            # we raise a TypeError
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            # If there are any invalid columns, we raise a ValueError
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, full_data,train_data, val_data, discrete_columns=(), epochs=None):

        # Validate discrete_columns
        
        print(
            "Starting RECTGAN training..., data shape:",
            train_data.shape,
            " discrete_columns:",
            discrete_columns,
            " epochs:",
            self._epochs,
            " R:",
            self.re_abnormal_target,
        )
        fake_buff_list_NOR = None # List to store generated fake samples for buffering (normal)
        fake_buff_list_ANO = None # List to store generated fake samples for buffering (anomaly)
        self._validate_discrete_columns(train_data, discrete_columns)

        # Determine the number of epochs to train
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )
        print(f'Starting training REC-CTGAN for {epochs} epochs..., data shape: {train_data.shape}    ')
        # Fit the DataTransformer to the training data # tÃ¡ÂºÂ¡o bÃ¡Â»â„¢ transform, lÃ†Â°u thÃƒÂ´ng tin cÃƒÂ¡c cÃ¡Â»â„¢t sau khi transform
        self._transformer = DataTransformer()
        self._transformer.fit(full_data, discrete_columns)

        # Transform the training data # transform data thÃƒÂ nh dÃ¡ÂºÂ¡ng mode-specific normalization
        train_data = self._transformer.transform(train_data)
        print("data after transform shape:",train_data.shape)
        
        X_val_train_processed = self._transformer.transform(val_data)
        y_val_train = np.zeros(X_val_train_processed.shape[0])  # GiÃ¡ÂºÂ£ sÃ¡Â»Â­ nhÃƒÂ£n lÃƒÂ  0 cho tÃ¡ÂºÂ¥t cÃ¡ÂºÂ£ cÃƒÂ¡c mÃ¡ÂºÂ«u trong tÃ¡ÂºÂ­p huÃ¡ÂºÂ¥n luyÃ¡Â»â€¡n bÃƒÂ¬nh thÃ†Â°Ã¡Â»Âng
        val_tensor = torch.tensor(X_val_train_processed, dtype=torch.float32)
        self._set_validation_continuous_stats(val_tensor)
        val_dataset = torch.utils.data.TensorDataset(val_tensor, torch.tensor(y_val_train, dtype=torch.float32))
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)
        
        # Fit the DataSampler to the transformed training data
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency
        )

        # Determine the dimensionality of the transformed data
        data_dim = self._transformer.output_dimensions

        # Fit the Generator to the transformed training data
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        # Fit the Discriminator to the transformed training data
        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)
        self._rec=REC(
            input_dim=data_dim,
            re_abnormal_target=self.re_abnormal_target,
        ).to(self._device)
        self._rec.device = self._device

        # Initialize the optimizers for the Generator and the Discriminator
        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )
        # Initialize the mean and std tensors for the Generator
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        # Initialize the loss values DataFrame
        self.loss_values = pd.DataFrame(columns=[
            'Epoch',
            'Generator Loss',
            'Discriminator Loss',
            'REC Loss',
            'Normal Reconstruction Loss',
            'Abnormal Reconstruction Loss',
            'Validation Reconstruction Loss',
            'Pseudo Val AUC',
            'Pseudo Re Separation',
        ])

        # Train the Generator and the Discriminator
        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Disc. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        # Select the best checkpoint by pseudo-validation separation.
        best_pseudo_val_auc = -float("inf")
        best_pseudo_val_separation = -float("inf")
        best_val_re_loss = float("inf")
        rec_start_epoch = int(np.ceil(epochs * self.rec_start_ratio))
        ctgan_noise_start_epoch = int(np.ceil(epochs * self.ctgan_noise_start_ratio))
        print(
            "[SCHEDULE] REC pseudo-anomaly training starts after",
            rec_start_epoch,
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
            "; R",
            self.re_abnormal_target,
            ".",
        )
        if rec_start_epoch >= epochs:
            print("[WARN] rec_start_ratio leaves no epochs for REC training in this run.")
        
        for i in epoch_iterator:
            rec_active = i >= rec_start_epoch
            ctgan_noise_active = rec_active and i >= ctgan_noise_start_epoch
            gaussian_noise_active = (
                ctgan_noise_active and self.ctgan_pseudo_anomaly_mode == "origin"
            )
            rec_loss = np.nan
            normal_re_loss = np.nan
            abnormal_re_loss = np.nan
            list_latent = None
            rec_dataset = None
            for id_ in range(steps_per_epoch):

                # Train the Discriminator for a certain number of steps
                real_data=None
                for n in range(self._discriminator_steps):
                    # Sample a random 1 tensor
                    fakez = torch.normal(mean=mean, std=std)
                    fakez_3 = torch.normal(mean=mean, std=std * sqrt(3))
                    # Sample a condition vector
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None # cond vector, mask, discrete_column_id, category_id_in_col
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device) # cond vector
                        m1 = torch.from_numpy(m1).to(self._device) # mask vector

                    # Generate a synthetic sample
                    if c1 is not None:
                        fakez = torch.cat([fakez, c1], dim=1)
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake) # apply activation function chuyÃ¡Â»Æ’n Ã„â€˜Ã¡ÂºÂ§u
                    
                    
                    if c1 is not None:
                        fakez_3 = torch.cat([fakez_3, c1], dim=1)
                    fake_3 = self._generator(fakez_3)
                    fakeact_3 = self._apply_activate(fake_3)

                    # Sample real data
                    if condvec is None:
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt)
                    else:
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    # Compute the loss for the Discriminator
                    real = torch.from_numpy(real.astype('float32')).to(self._device)
                    real_data=real
                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        fake_cat = fakeact
                        real_cat = real

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # Train the Generator
                fakez = torch.normal(mean=mean, std=std)
                fakez_3 = torch.normal(mean=mean, std=std * sqrt(3))
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
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

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                if not rec_active:
                    continue

                if self.ctgan_pseudo_anomaly_mode == "feature_outlier":
                    fakeact_for_rec_nor = self.make_ctgan_feature_outlier_anomaly(
                        real_data,
                        fakeact,
                        r=self.ctgan_feature_outlier_r,
                    )
                    fakeact_for_rec_ano = self.make_ctgan_feature_outlier_anomaly(
                        real_data,
                        fakeact_3,
                        r=self.ctgan_feature_outlier_r,
                    )
                else:
                    fakeact_for_rec_nor = fakeact
                    fakeact_for_rec_ano = fakeact_3
                    if gaussian_noise_active:
                        fakeact_for_rec_nor = self.add_gaussian_noise_to_ctgan_samples(
                            fakeact_for_rec_nor,
                            noise_std=self.ctgan_noise_std,
                            feature_ratio=self.ctgan_noise_feature_ratio,
                        )
                        # fakeact_for_rec_ano = self.add_gaussian_noise_to_ctgan_samples(
                        #     fakeact_for_rec_ano,
                        #     noise_std=self.ctgan_noise_std,
                        #     feature_ratio=self.ctgan_noise_feature_ratio,
                        # )

                fake_buff_list_NOR=self.to_buffer(fake_buff_list_NOR,fakeact_for_rec_nor,self._batch_size)
                fake_buff_list_ANO=self.to_buffer(fake_buff_list_ANO,fakeact_for_rec_ano,self._batch_size)
                
                real_perturbed = self.make_real_perturbed_anomaly(real_data)
                
                # lÃ†Â°u dÃ¡Â»Â¯ dl sinh vÃƒÂ o bÃ¡Â»â„¢ Ã„â€˜Ã¡Â»â€¡m
                 # LÃ¡ÂºÂ¥y dÃ¡Â»Â¯ liÃ¡Â»â€¡u fake tÃ¡Â»Â« buffer nÃ¡ÂºÂ¿u cÃƒÂ³, nÃ¡ÂºÂ¿u khÃƒÂ´ng thÃƒÂ¬ dÃƒÂ¹ng fake mÃ¡Â»â€ºi tÃ¡ÂºÂ¡o
                if fake_buff_list_NOR is None:
                    fakeact_form_buffer_NOR = fakeact_for_rec_nor
                else:
                    fakeact_form_buffer_NOR=self.from_buffer(fake_buff_list_NOR, self._batch_size)
                
                if fake_buff_list_ANO is None:
                    fakeact_form_buffer_ANO=fakeact_for_rec_ano # buffer cho anomaly critic
                else:
                    fakeact_form_buffer_ANO=self.from_buffer(fake_buff_list_ANO, self._batch_size)
                # train REC every discriminator step
                
                
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
                
                ##fake_data=torch.cat([fakeact_form_buffer_ANO, real_perturbed,fakeact_form_buffer_ANO], dim=0)
                #fake_data=torch.cat([real_perturbed,fakeact_form_buffer_NOR], dim=0)
                y_fake_rec = torch.ones(fake_data.size(0), 1, device=self._device)
                
                
                #num_samples = self._batch_size  # sÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng mÃ¡ÂºÂ«u bÃ¡ÂºÂ¡n muÃ¡Â»â€˜n lÃ¡ÂºÂ¥y
                #perm = torch.randperm(fake_data.size(0), device=self._device)[:num_samples]
                fake_data_sample = fake_data
                y_fake_rec_sample = y_fake_rec
                
                y_real = torch.zeros(real_data.size(0), 1, device=self._device)
                rec_dataset=torch.cat([fake_data_sample, real_data], dim=0)
                y_rec=torch.cat([y_fake_rec_sample, y_real], dim=0)
                rec_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(rec_dataset, y_rec),
                    batch_size=self._batch_size,
                    shuffle=True
                )
                rec_loss, normal_re_loss, abnormal_re_loss, list_latent = self._rec.train_rec(
                    epoch=i,
                    train_data_loader=rec_dataloader,
                )
            
            # Compute the generator loss and the discriminator loss
            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            avg_val_re_loss = np.nan
            pseudo_val_auc = np.nan
            pseudo_val_threshold = np.nan
            pseudo_val_separation = np.nan
            if rec_active:
                # Evaluate and select a best checkpoint only after REC has been updated.
                avg_val_re_loss, threshold_re, threshold_re95 = self._rec.validate(
                    val_data_loader,
                    epoch=i
                )
                pseudo_val_auc, pseudo_val_threshold, pseudo_val_separation = self.evaluate_pseudo_validation_auc(
                    val_data_loader
                )
                is_best_auc_checkpoint = (
                    pseudo_val_auc > best_pseudo_val_auc
                    or (
                        np.isclose(pseudo_val_auc, best_pseudo_val_auc)
                        and pseudo_val_separation > best_pseudo_val_separation
                    )
                )
                is_best_re95_checkpoint = avg_val_re_loss < best_val_re_loss

                if is_best_auc_checkpoint:
                    best_pseudo_val_auc = pseudo_val_auc
                    best_pseudo_val_separation = pseudo_val_separation
                    if self._on_epoch_best_callbacks is not None:
                        self._on_epoch_best_callbacks(
                            model=self,
                            epoch=i,
                            checkpoint_kind="auc",
                            threshold_re=threshold_re,
                            threshold_re95=threshold_re95,
                            threshold_auc=pseudo_val_threshold,
                            pseudo_val_auc=pseudo_val_auc,
                            pseudo_val_separation=pseudo_val_separation,
                            avg_val_re_loss=avg_val_re_loss,
                        )
                    print(
                        f"[BEST_AUC] epoch={i} pseudo_val_auc={pseudo_val_auc:.6f} "
                        f"pseudo_separation={pseudo_val_separation:.6f} "
                        f"val_re_loss={avg_val_re_loss:.6f} "
                        f"threshold_re95={threshold_re95:.6f} "
                        f"threshold_auc={pseudo_val_threshold:.6f}"
                    )

                if is_best_re95_checkpoint:
                    best_val_re_loss = avg_val_re_loss
                    if self._on_epoch_best_callbacks is not None:
                        self._on_epoch_best_callbacks(
                            model=self,
                            epoch=i,
                            checkpoint_kind="re95",
                            threshold_re=threshold_re,
                            threshold_re95=threshold_re95,
                            threshold_auc=pseudo_val_threshold,
                            pseudo_val_auc=pseudo_val_auc,
                            pseudo_val_separation=pseudo_val_separation,
                            avg_val_re_loss=avg_val_re_loss,
                        )
                    print(
                        f"[BEST_RE95] epoch={i} val_re_loss={avg_val_re_loss:.6f} "
                        f"pseudo_val_auc={pseudo_val_auc:.6f} "
                        f"threshold_re95={threshold_re95:.6f} "
                        f"threshold_auc={pseudo_val_threshold:.6f}"
                    )
                # self._rec.plot_latent_first_component_hist_from_latents(
                #     np.asarray(list_latent),
                #     f"rec_latent_hist_{epochs}",  # Ensure out_dir is a string, not ndarray
                #     i,
                # )
                rec_data_shape = rec_dataset.shape
            else:
                rec_data_shape = "warmup: REC not trained"

            # Append the loss values to the loss values DataFrame
            epoch_loss_df = pd.DataFrame({
            'Epoch': [i],
            'Generator Loss': [generator_loss],
            'Discriminator Loss': [discriminator_loss],
            'REC Loss': [rec_loss.item() if torch.is_tensor(rec_loss) else rec_loss],
            'Normal Reconstruction Loss': [normal_re_loss.item() if torch.is_tensor(normal_re_loss) else normal_re_loss],
            'Abnormal Reconstruction Loss': [abnormal_re_loss.item() if torch.is_tensor(abnormal_re_loss) else abnormal_re_loss],
            'Validation Reconstruction Loss': [avg_val_re_loss],
            'Pseudo Val AUC': [pseudo_val_auc],
            'Pseudo Re Separation': [pseudo_val_separation],
            })
            print(
                f"Epoch {i}: Generator Loss = {generator_loss}, Discriminator Loss = {discriminator_loss}, "
                f"REC Loss = {rec_loss}, Normal RE Loss = {normal_re_loss}, "
                f"Abnormal RE Target Loss = {abnormal_re_loss}, "
                f"CTGAN Gaussian noise active = {gaussian_noise_active}, "
                f"Pseudo Val AUC = {pseudo_val_auc}, Pseudo RE Separation = {pseudo_val_separation}",
                "data rec shape: ",
                rec_data_shape,
            )
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            # Print the loss values
            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
            if self._on_epoch_end_callbacks is not None:
                self._on_epoch_end_callbacks(model=self, epoch=i)

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """LÃ¡ÂºÂ¥y mÃ¡ÂºÂ«u dÃ¡Â»Â¯ liÃ¡Â»â€¡u giÃ¡Â»â€˜ng vÃ¡Â»â€ºi dÃ¡Â»Â¯ liÃ¡Â»â€¡u huÃ¡ÂºÂ¥n luyÃ¡Â»â€¡n.

        ChÃ¡Â»Ân condition_column vÃƒÂ  condition_value sÃ¡ÂºÂ½ tÃ„Æ’ng xÃƒÂ¡c suÃ¡ÂºÂ¥t xuÃ¡ÂºÂ¥t hiÃ¡Â»â€¡n giÃƒÂ¡ trÃ¡Â»â€¹ Ã„â€˜ÃƒÂ³ trong cÃ¡Â»â„¢t phÃƒÂ¢n loÃ¡ÂºÂ¡i tÃ†Â°Ã†Â¡ng Ã¡Â»Â©ng.

        Tham sÃ¡Â»â€˜:
            n (int): SÃ¡Â»â€˜ lÃ†Â°Ã¡Â»Â£ng dÃƒÂ²ng cÃ¡ÂºÂ§n lÃ¡ÂºÂ¥y mÃ¡ÂºÂ«u.
            condition_column (string): TÃƒÂªn cÃ¡Â»â„¢t phÃƒÂ¢n loÃ¡ÂºÂ¡i.
            condition_value (string): GiÃƒÂ¡ trÃ¡Â»â€¹ phÃƒÂ¢n loÃ¡ÂºÂ¡i mong muÃ¡Â»â€˜n tÃ„Æ’ng xÃƒÂ¡c suÃ¡ÂºÂ¥t xuÃ¡ÂºÂ¥t hiÃ¡Â»â€¡n.

        TrÃ¡ÂºÂ£ vÃ¡Â»Â:
            numpy.ndarray hoÃ¡ÂºÂ·c pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
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
        """ThiÃ¡ÂºÂ¿t lÃ¡ÂºÂ­p thiÃ¡ÂºÂ¿t bÃ¡Â»â€¹ sÃ¡Â»Â­ dÃ¡Â»Â¥ng ('GPU' hoÃ¡ÂºÂ·c 'CPU')."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
    def get_score_rec(self, X,y):
        """TÃƒÂ­nh anomaly score tÃ¡Â»Â« D2.
        Score cÃƒÂ ng thÃ¡ÂºÂ¥p Ã¢â€ â€™ cÃƒÂ ng bÃ¡ÂºÂ¥t thÃ†Â°Ã¡Â»Âng
        """
        X_processed = self._transformer.transform(X)
        score_res, y_true = self._rec.get_score_rec(
            X_processed,
            y,
            batch_size=self._batch_size,
            device=self._device,
        )
        return score_res, y_true

    def get_latent_z_mae(self, X, batch_size=None):
        """Return REC encoder latent vectors (z) for debugging."""
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
        self._rec.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device)
                z, _ = self._rec.forward(xb)
                z_all.append(z.cpu().numpy())
        return np.concatenate(z_all)
    def get_threshold_rec(self, X_val_train):
        """TÃƒÂ­nh threshold tÃ¡Â»Â« tÃ¡ÂºÂ­p validation."""
        X_val_train_processed = self._transformer.transform(X_val_train)
        y_val_train = np.zeros(X_val_train_processed.shape[0])  # GiÃ¡ÂºÂ£ sÃ¡Â»Â­ nhÃƒÂ£n lÃƒÂ  0 cho tÃ¡ÂºÂ¥t cÃ¡ÂºÂ£ cÃƒÂ¡c mÃ¡ÂºÂ«u trong tÃ¡ÂºÂ­p huÃ¡ÂºÂ¥n luyÃ¡Â»â€¡n
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_train_processed, dtype=torch.float32), torch.tensor(y_val_train, dtype=torch.float32))
        val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
        
        avg_loss, threshold_re, threshold_re95 = self._rec.validate(val_data_loader,epoch=0)
        return avg_loss, threshold_re, threshold_re95
    
    @staticmethod
    def to_buffer(buffer, new_fake, buf_per_iter):
        """
        CÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t replay buffer bÃ¡ÂºÂ±ng cÃƒÂ¡ch:
        - LÃ¡ÂºÂ¥y `buf_per_iter` mÃ¡ÂºÂ«u tÃ¡Â»Â« batch fake mÃ¡Â»â€ºi (`new_fake`)
        - NÃ¡Â»â€˜i vÃƒÂ o buffer hiÃ¡Â»â€¡n cÃƒÂ³
        - GiÃ¡Â»Â¯ lÃ¡ÂºÂ¡i tÃ¡Â»â€˜i Ã„â€˜a `8 * buf_per_iter` mÃ¡ÂºÂ«u trong buffer
        """
        # Ã„ÂÃ¡ÂºÂ§u vÃƒÂ o new_fake: (batch_size, D)
        # Ã„ÂÃ¡ÂºÂ§u ra buffer:   (<= 8 * buf_per_iter, D)
        new_fake = new_fake.clone().detach()  # CLONE + DETACH an toÃƒÂ n
        batch_size, data_dim = new_fake.shape  # new_fake: (B, D)

        # LÃ¡ÂºÂ¥y ngÃ¡ÂºÂ«u nhiÃƒÂªn buf_per_iter mÃ¡ÂºÂ«u tÃ¡Â»Â« new_fake
        perm = torch.randperm(batch_size)
        idx = perm[:buf_per_iter]
        new_buffer = new_fake.detach()[idx]  # (buf_per_iter, D)

        if buffer is None:
            # NÃ¡ÂºÂ¿u chÃ†Â°a cÃƒÂ³ buffer, khÃ¡Â»Å¸i tÃ¡ÂºÂ¡o luÃƒÂ´n
            buffer = new_buffer  # (buf_per_iter, D)
        else:
            buffer = buffer.clone().detach().to(new_fake.device)

            # NÃ¡Â»â€˜i fake mÃ¡Â»â€ºi vÃƒÂ o buffer cÃ…Â©
            buffer = torch.cat((buffer, new_buffer), dim=0)

            # Shuffle vÃƒÂ  giÃ¡Â»Â¯ lÃ¡ÂºÂ¡i tÃ¡Â»â€˜i Ã„â€˜a 8 * buf_per_iter mÃ¡ÂºÂ«u
            max_buffer_size = 8 * buf_per_iter
            keep_size = min(max_buffer_size, buffer.size(0))

            perm = torch.randperm(buffer.size(0), device=buffer.device)
            idx = perm[:keep_size]
            buffer = buffer[idx].detach()

        return buffer



    @staticmethod
    def from_buffer(buffer, num_of_imgs):
        buffer = buffer.clone().detach()
        perm = torch.randperm(buffer.size(0))
        idx = perm[:min(num_of_imgs, buffer.size(0))]
        return buffer[idx]

    def get_continuous_indices(self, device=None):
        """Return transformed indices that use continuous tanh spans."""
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

    def get_discrete_span_slices(self):
        """Return transformed slices for true discrete/categorical softmax columns."""
        if self._transformer is None:
            raise RuntimeError("DataTransformer must be fitted before locating discrete spans.")

        discrete_slices = []
        st = 0
        for column_info in self._transformer.output_info_list:
            is_discrete_column = (
                len(column_info) == 1 and column_info[0].activation_fn == "softmax"
            )
            for span_info in column_info:
                ed = st + span_info.dim
                if is_discrete_column:
                    discrete_slices.append((st, ed))
                st = ed

        return discrete_slices

    def copy_discrete_spans(self, target_batch, source_batch):
        """Copy true discrete/categorical spans from source into target."""
        x = target_batch.clone()
        for st, ed in self.get_discrete_span_slices():
            x[:, st:ed] = source_batch[:, st:ed]
        return x

    def make_categorical_span_anomaly(self, real_batch, span_prob=0.3):
        """Shuffle a subset of categorical one-hot spans across rows."""
        x = real_batch.clone()
        discrete_slices = self.get_discrete_span_slices()
        B = x.size(0)
        if B <= 1 or not discrete_slices or span_prob <= 0.0:
            return x

        span_prob = min(1.0, float(span_prob))
        changed = False
        for st, ed in discrete_slices:
            if torch.rand((), device=x.device) >= span_prob:
                continue
            perm = torch.randperm(B, device=x.device)
            if torch.equal(perm, torch.arange(B, device=x.device)):
                perm = torch.roll(perm, shifts=1)
            x[:, st:ed] = x[perm, st:ed]
            changed = True

        if not changed:
            span_id = torch.randint(len(discrete_slices), (1,), device=x.device).item()
            st, ed = discrete_slices[span_id]
            perm = torch.randperm(B, device=x.device)
            if torch.equal(perm, torch.arange(B, device=x.device)):
                perm = torch.roll(perm, shifts=1)
            x[:, st:ed] = x[perm, st:ed]

        return x

    def make_real_perturbed_anomaly(self, real_batch):
        """Create pseudo anomalies by perturbing continuous and categorical spans."""
        real_mix = self.make_unsw_mix_anomaly(real_batch)
        real_scale = self.make_unsw_scale_anomaly(real_batch)
        real_categorical = self.make_categorical_span_anomaly(real_batch)
        real_perturbed = 0.5 * real_mix + 0.5 * real_scale
        return self.copy_discrete_spans(real_perturbed, real_categorical)

    def _set_validation_continuous_stats(self, validation_data):
        """Cache normal validation statistics used to build pseudo anomalies."""
        cont_idx = self.get_continuous_indices(device=validation_data.device)
        if cont_idx.numel() == 0 or validation_data.size(0) == 0:
            self._validation_cont_mean = None
            self._validation_cont_std = None
            return

        validation_cont = validation_data[:, cont_idx]
        self._validation_cont_mean = validation_cont.mean(dim=0).detach()
        self._validation_cont_std = validation_cont.std(dim=0, unbiased=False).detach()

    def _get_validation_continuous_stats(self, batch, cont_idx):
        """Return cached validation stats, falling back to the current batch."""
        mean = getattr(self, "_validation_cont_mean", None)
        std = getattr(self, "_validation_cont_std", None)

        if mean is None or std is None or mean.numel() != cont_idx.numel():
            batch_cont = batch[:, cont_idx]
            mean = batch_cont.mean(dim=0)
            std = batch_cont.std(dim=0, unbiased=False)
        else:
            mean = mean.to(batch.device)
            std = std.to(batch.device)

        return mean, std

    def make_validation_std_anomaly(self, real_batch, scale):
        """Push normal validation rows outward by validation continuous std."""
        x = real_batch.clone()
        cont_idx = self.get_continuous_indices(device=x.device)
        if cont_idx.numel() == 0:
            return self.make_categorical_span_anomaly(real_batch)

        mean, feature_std = self._get_validation_continuous_stats(x, cont_idx)
        feature_std = torch.clamp(feature_std, min=1e-6)
        cont_values = x[:, cont_idx]
        direction = torch.where(
            cont_values >= mean.unsqueeze(0),
            torch.ones_like(cont_values),
            -torch.ones_like(cont_values),
        )
        x[:, cont_idx] = torch.clamp(
            cont_values + direction * float(scale) * feature_std.unsqueeze(0),
            -1.0,
            1.0,
        )
        return x

    def _validation_pseudo_modes(self):
        """Map type_train to validation pseudo-anomaly modes."""
        mode_map = {
            "noise": ("noise",),
            "std": ("std",),
            "3std": ("3std",),
            "std_3std": ("std", "3std"),
            "noise_std": ("noise", "std"),
            "noise_3std": ("noise", "3std"),
            "all": ("noise", "std", "3std"),
        }
        if self.type_train not in mode_map:
            raise ValueError(f"type_train is not valid: {self.type_train}")
        return mode_map[self.type_train]

    def make_validation_pseudo_anomaly(self, normal_batch):
        """Create one pseudo-abnormal validation row per normal row."""
        modes = self._validation_pseudo_modes()
        pseudo_batches = []
        for mode_idx, mode in enumerate(modes):
            mode_batch = normal_batch[mode_idx::len(modes)]
            if mode_batch.size(0) == 0:
                continue

            if mode == "noise":
                pseudo_batches.append(self.make_real_perturbed_anomaly(mode_batch))
            elif mode == "std":
                pseudo_batches.append(self.make_validation_std_anomaly(mode_batch, 1.0))
            elif mode == "3std":
                pseudo_batches.append(self.make_validation_std_anomaly(mode_batch, sqrt(3)))

        if not pseudo_batches:
            return normal_batch.new_empty((0, normal_batch.size(1)))

        return torch.cat(pseudo_batches, dim=0)

    def evaluate_pseudo_validation_auc(self, val_data_loader):
        """Evaluate separation between normal validation rows and pseudo anomalies."""
        self._rec.eval()
        normal_scores = []
        pseudo_scores = []

        with torch.no_grad():
            for xb, _ in val_data_loader:
                xb = xb.to(self._device)
                xb_pseudo = self.make_validation_pseudo_anomaly(xb)

                _, xb_hat = self._rec.forward(xb)
                _, xb_pseudo_hat = self._rec.forward(xb_pseudo)

                normal_scores.append(
                    self._rec.reconstruction_error(xb, xb_hat).cpu().numpy()
                )
                pseudo_scores.append(
                    self._rec.reconstruction_error(xb_pseudo, xb_pseudo_hat).cpu().numpy()
                )

        if not normal_scores or not pseudo_scores:
            return 0.0, 0.0, -float("inf")

        normal_scores = np.concatenate(normal_scores)
        pseudo_scores = np.concatenate(pseudo_scores)
        y_pseudo = np.concatenate(
            [
                np.zeros_like(normal_scores, dtype=np.int64),
                np.ones_like(pseudo_scores, dtype=np.int64),
            ]
        )
        score_pseudo = np.concatenate([normal_scores, pseudo_scores])
        pseudo_auc = roc_auc_score(y_pseudo, score_pseudo)
        fpr, tpr, thresholds = roc_curve(y_pseudo, score_pseudo)
        finite_mask = np.isfinite(thresholds)
        if np.any(finite_mask):
            j_scores = tpr[finite_mask] - fpr[finite_mask]
            threshold_auc = float(thresholds[finite_mask][np.argmax(j_scores)])
        else:
            threshold_auc = float(np.percentile(normal_scores, 95))
        pseudo_separation = float(np.mean(pseudo_scores) - np.mean(normal_scores))
        return float(pseudo_auc), threshold_auc, pseudo_separation

    def add_gaussian_noise_to_ctgan_samples(
        self, fake_batch, noise_std=None, feature_ratio=None
    ):
        """Perturb a sampled fraction of generated continuous features."""
        x = fake_batch.clone().detach()
        cont_idx = self.get_continuous_indices(device=x.device)
        noise_std = self.ctgan_noise_std if noise_std is None else float(noise_std)
        feature_ratio = (
            self.ctgan_noise_feature_ratio
            if feature_ratio is None
            else float(feature_ratio)
        )
        if cont_idx.numel() == 0 or feature_ratio == 0.0 or noise_std == 0.0:
            return x

        num_features = max(1, int(np.ceil(cont_idx.numel() * feature_ratio)))
        selected = cont_idx[
            torch.randperm(cont_idx.numel(), device=x.device)[:num_features]
        ]
        gaussian_noise = noise_std * torch.randn_like(x[:, selected])
        x[:, selected] = torch.clamp(x[:, selected] + gaussian_noise, -1.0, 1.0)
        return x

    def make_ctgan_feature_outlier_anomaly(self, real_batch, generated_batch, r=None):
        """Use real non-continuous spans with CTGAN feature-statistic outliers."""
        x = real_batch.clone().detach()
        cont_idx = self.get_continuous_indices(device=x.device)
        if cont_idx.numel() == 0:
            return x

        generated_cont = generated_batch.detach().to(x.device)[:, cont_idx]
        feature_mean = generated_cont.mean(dim=0)
        feature_std = generated_cont.std(dim=0, unbiased=False)
        coefficient = self.ctgan_feature_outlier_r if r is None else float(r)
        outlier_values = feature_mean + coefficient * feature_std
        x[:, cont_idx] = torch.clamp(outlier_values.unsqueeze(0), -1.0, 1.0)
        return x

    def make_unsw_scale_anomaly(self, real_batch, noise=0.1):
        """Perturb only transformed continuous spans with multiplicative Gaussian noise."""
        x = real_batch.clone()
        cont_idx = self.get_continuous_indices(device=x.device)
        if cont_idx.numel() > 0:
            x[:, cont_idx] = x[:, cont_idx] * (
                1 + float(noise) * torch.randn_like(x[:, cont_idx])
            )
            x[:, cont_idx] = torch.clamp(x[:, cont_idx], -1.0, 1.0)
        return x

    def make_unsw_mix_anomaly(self,real_batch):
        """Shuffle a subset of continuous transformed features across rows."""
        B, _ = real_batch.shape
        x = real_batch.clone()
        cont_idx = self.get_continuous_indices(device=x.device)
        mask = torch.rand(cont_idx.numel(), device=x.device) < 0.3
        for j in cont_idx[mask].tolist():
            if B > 1:
                x[:, j] = x[torch.randperm(B), j]
        return x
    def set_epoch_end_callback(self, callback):
        """
        callback(model=self, epoch=int)
        """
        self._on_epoch_end_callbacks = callback
    def set_epoch_best_callback(self, callback):
        """
        callback(model=self, epoch=int)
        """
        self._on_epoch_best_callbacks = callback

