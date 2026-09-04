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
    """Máº¡ng phÃ¢n biá»‡t (Discriminator) cho CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        """Khá»Ÿi táº¡o máº¡ng phÃ¢n biá»‡t.

        Tham sá»‘:
            input_dim (int): KÃ­ch thÆ°á»›c Ä‘áº§u vÃ o.
            discriminator_dim (list): KÃ­ch thÆ°á»›c cÃ¡c lá»›p cá»§a máº¡ng phÃ¢n biá»‡t.
            pac (int): Sá»‘ lÃ¡t (pack) Ä‘á»ƒ nhÃ³m dá»¯ liá»‡u.

        Tráº£ vá»:
            None
        """
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            # Lá»›p tuyáº¿n tÃ­nh Ä‘áº§u tiÃªn nháº­n Ä‘áº§u vÃ o Ä‘Ã£ Ä‘Æ°á»£c pack.
            # CÃ¡c lá»›p tiáº¿p theo nháº­n Ä‘áº§u ra cá»§a lá»›p trÆ°á»›c Ä‘Ã³.
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        # Lá»›p cuá»‘i cÃ¹ng lÃ  lá»›p tuyáº¿n tÃ­nh vá»›i má»™t Ä‘áº§u ra.
        # ÄÃ¢y lÃ  xÃ¡c suáº¥t Ä‘áº§u vÃ o lÃ  dá»¯ liá»‡u tháº­t.
        seq += [Linear(dim, 1)] #1 Ä‘áº§u ra Ä‘áº¡i diá»‡n cho logit/score: sample nÃ y lÃ  real hay fake (chÆ°a sigmoid, vÃ¬ WGAN loss dÃ¹ng raw score).
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """
        TÃ­nh toÃ¡n gradient penalty.

        ÄÃ¢y lÃ  má»™t thÃ nh pháº§n pháº¡t Ä‘Æ°á»£c cá»™ng vÃ o loss cá»§a máº¡ng phÃ¢n biá»‡t.
        NÃ³ tá»· lá»‡ vá»›i bÃ¬nh phÆ°Æ¡ng Ä‘á»™ lá»‡ch giá»¯a chuáº©n cá»§a gradient vÃ  1.

        Gradient Ä‘Æ°á»£c tÃ­nh lÃ  gradient cá»§a Ä‘áº§u ra máº¡ng phÃ¢n biá»‡t theo Ä‘áº§u vÃ o.
        Äáº§u vÃ o lÃ  tá»• há»£p tuyáº¿n tÃ­nh giá»¯a máº«u dá»¯ liá»‡u tháº­t vÃ  giáº£, tá»• há»£p nÃ y Ä‘Æ°á»£c láº¥y ngáº«u nhiÃªn tá»« [0, 1].
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
        """Ãp dá»¥ng máº¡ng phÃ¢n biá»‡t lÃªn `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Lá»›p residual cho CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Ãp dá»¥ng lá»›p residual lÃªn `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Máº¡ng sinh dá»¯ liá»‡u (Generator) cho CTGAN."""

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
        """Ãp dá»¥ng máº¡ng sinh dá»¯ liá»‡u lÃªn `input_`."""
        data = self.seq(input_)
        return data

class MAE(Module):
    """Máº¡ng AutoEncoder há»— trá»£ cho CTGAN."""

    def __init__(self, input_dim, lamda=0.1, margin=1.0, mae_loss_mode="origin"):
        super(MAE, self).__init__()
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
        self.loss_function = MSELoss() #máº·c Ä‘á»‹nh tÃ­nh mean
        self.latent_distance = MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.lamda=lamda
        self.margin=margin
        self.mae_loss_mode = str(mae_loss_mode)
        if self.mae_loss_mode not in {"origin", "renormal"}:
            raise ValueError("mae_loss_mode must be 'origin' or 'renormal'.")

    def forward(self, input_):
        """Ãp dá»¥ng máº¡ng AutoEncoder lÃªn `input_`."""
        encoded = self.enc(input_)
        decoded = self.dec(encoded)
        return encoded, decoded
    def calculate_loss(self, x,encoded,xb_hat, y):
        if self.mae_loss_mode == "renormal":
            re_per_sample = F.mse_loss(xb_hat, x, reduction="none").mean(dim=1)
            normal_mask = y.view(-1) == 0
            if torch.any(normal_mask):
                re_loss = re_per_sample[normal_mask].mean()
            else:
                re_loss = re_per_sample.mean() * 0.0
        else:
            re_loss = self.loss_function(xb_hat, x)
       
        margin = torch.tensor(
            self.margin,
            device=encoded.device,
            dtype=encoded.dtype
        )
        # -----------------------------
        # 2. Latent norm
        # -----------------------------
        z_norm = torch.norm(encoded, p=2, dim=1)#huáº©n L2 (Euclidean norm) ,norm theo tá»«ng sample, khÃ´ng pháº£i toÃ n batch.

       
        # - normal samples are pulled to 0
        # - anomaly samples are pulled to margin m
        latent_loss = torch.mean(torch.where(
            y.view(-1) == 0,
            z_norm ** 2,
            (z_norm - margin) ** 2
        ))
 # Old margin loss:
        # latent_loss = torch.mean(torch.where(
        #     y.view(-1) == 0,
        #     z_norm ** 2,
        #     torch.relu(margin - z_norm) ** 2
        # ))
        #
        # New target loss:
        
        return re_loss, latent_loss
    def train_mae(self, epoch,train_data_loader):
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
    """
    def validate(self, val_data_loader,epoch):
        self.eval()
        val_losses, val_re_losses, val_latent_losses = [], [],[]
        y_val_true = []
        with torch.no_grad():
            for input, label in val_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                latent_z, xb_hat = self.forward(input)
                val_re_loss=MSELoss(reduction="none")(xb_hat, input).mean(dim=1)
                target_latent = (label != 0).float()
                if latent_z.dim() > 1 and latent_z.shape[1] > 1:
                    target_latent = target_latent.unsqueeze(1).expand_as(latent_z)
                val_latent_loss=MSELoss(reduction="none")(latent_z,target_latent).mean(dim=1)
                val_loss= val_re_loss + self.lamda * val_latent_loss
                
                val_losses.append(val_loss.cpu().numpy().tolist())
                val_re_losses.append(val_re_loss.cpu().numpy().tolist())
                val_latent_losses.append(val_latent_loss.cpu().numpy().tolist())
                y_val_true.append(label.cpu().numpy())
                
            val_losses = np.concatenate(val_losses)
            val_re_losses = np.concatenate(val_re_losses)
            val_latent_losses = np.concatenate(val_latent_losses)
            y_val_true = np.concatenate(y_val_true)
            
            avg_loss = np.mean(val_losses)
            
            mu_re = np.mean(val_re_losses)
            std_re = np.std(val_re_losses)
            threshold_re = (mu_re,std_re) if val_re_losses is not None else None
            
            val_latent_losses_0= val_latent_losses[y_val_true == 0]
            mu_z_0 = np.mean(val_latent_losses_0)
            std_z_0 = np.std(val_latent_losses_0)
            threshold_z_0 = (mu_z_0,std_z_0) if val_latent_losses_0 is not None else None

            return avg_loss, threshold_re, threshold_z_0
        """
    def validate(self, val_data_loader, epoch):
        self.eval()
        val_losses = []
        val_re_losses = []
        val_latent_losses = []
        z_norm_all = []
        y_val_true = []

        with torch.no_grad():
            for xb, yb in val_data_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                z, xb_hat = self.forward(xb)

                # -----------------------------
                # 1. Reconstruction loss
                # -----------------------------
                re_loss = F.mse_loss(xb_hat, xb, reduction="none").mean(dim=1)

                # -----------------------------
                # 2. Latent norm
                # -----------------------------
                z_norm = torch.norm(z, p=2, dim=1)#huáº©n L2 (Euclidean norm) ,norm theo tá»«ng sample, khÃ´ng pháº£i toÃ n batch.
                margin = torch.tensor(
                    self.margin,
                    device=z_norm.device,
                    dtype=z_norm.dtype
                )

                # -----------------------------
                # 3. Target-based latent loss
                # -----------------------------
                # Old margin loss:
                # latent_loss = torch.where(
                #     yb.view(-1) == 0,
                #     z_norm ** 2,
                #     torch.relu(margin - z_norm) ** 2
                # )
                #
                # New target loss:
                # - normal samples are pulled to 0
                # - anomaly samples are pulled to margin m
                latent_loss = torch.where(
                    yb.view(-1) == 0,
                    z_norm ** 2,
                    (z_norm - margin) ** 2
                )

                # -----------------------------
                # 4. Total loss
                # -----------------------------
                total_loss = re_loss + self.lamda * latent_loss

                # -----------------------------
                # 5. LÆ°u káº¿t quáº£
                # -----------------------------
                val_losses.append(total_loss.cpu().numpy())
                val_re_losses.append(re_loss.cpu().numpy())
                val_latent_losses.append(latent_loss.cpu().numpy())
                z_norm_all.append(z_norm.cpu().numpy())
                y_val_true.append(yb.cpu().numpy())

        # =============================
        # Aggregate
        # =============================
        val_losses = np.concatenate(val_losses)
        val_re_losses = np.concatenate(val_re_losses)
        val_latent_losses = np.concatenate(val_latent_losses)
        z_norm_all = np.concatenate(z_norm_all)
        y_val_true = np.concatenate(y_val_true)

        avg_loss = np.mean(val_losses)
        self.last_validation_loss = float(avg_loss)
        self.last_validation_reconstruction_loss = float(np.mean(val_re_losses))
        self.last_validation_latent_loss = float(np.mean(val_latent_losses))

        # =============================
        # Thresholds (NORMAL only)
        # =============================
        # Reconstruction threshold
        re_normal = val_re_losses[y_val_true == 0]
        mu_re = np.mean(re_normal)
        std_re = np.std(re_normal)
        threshold_re = (mu_re, std_re)

        # Latent norm threshold
        z_norm_normal = z_norm_all[y_val_true == 0]
        mu_z = np.mean(z_norm_normal)
        std_z = np.std(z_norm_normal)
        threshold_z = (mu_z, std_z)
        threshold_95 = np.percentile(z_norm_normal, 95)
        threshold_95_total_loss = np.percentile(val_losses, 95)
        

        return avg_loss, threshold_re, threshold_z, threshold_95, threshold_95_total_loss  
    def compute_average_losses(self, data_loader):
        """
        Compute average reconstruction, latent and total loss
        over a given dataloader.
        """
        self.eval()
        re_losses = []
        latent_losses = []
        total_losses = []

        with torch.no_grad():
            for xb, yb in data_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                z, xb_hat = self.forward(xb)

                # -----------------------------
                # Reconstruction loss (per sample)
                # -----------------------------
                re_loss = F.mse_loss(xb_hat, xb, reduction="none").mean(dim=1)

                # -----------------------------
                # Latent norm
                # -----------------------------
                z_norm = torch.norm(z, p=2, dim=1)
                margin = torch.tensor(
                    self.margin,
                    device=z_norm.device,
                    dtype=z_norm.dtype
                )

                # -----------------------------
                # Target-based latent loss
                # -----------------------------
                # Old margin loss:
                # latent_loss = torch.where(
                #     yb.view(-1) == 0,
                #     z_norm ** 2,
                #     torch.relu(margin - z_norm) ** 2
                # )
                #
                # New target loss:
                # - normal samples are pulled to 0
                # - anomaly samples are pulled to margin m
                latent_loss = torch.where(
                    yb.view(-1) == 0,
                    z_norm ** 2,
                    (z_norm - margin) ** 2
                )

                # -----------------------------
                # Total loss
                # -----------------------------
                total_loss = re_loss + self.lamda * latent_loss

                re_losses.append(re_loss.cpu().numpy())
                latent_losses.append(latent_loss.cpu().numpy())
                total_losses.append(total_loss.cpu().numpy())

        # =============================
        # Aggregate
        # =============================
        re_losses = np.concatenate(re_losses)
        latent_losses = np.concatenate(latent_losses)
        total_losses = np.concatenate(total_losses)

        return  np.mean(total_losses),np.mean(re_losses), np.mean(latent_losses)
    
    def get_score_mae(
        self,
        X_test,
        y_test,
        batch_size=125,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
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

        # warmup (khÃ´ng tÃ­nh vÃ o timing)
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

                # -------------------------
                # Reconstruction error
                # -------------------------
                
                #re = MSELoss()(xb_hat, xb, reduction="none").mean(dim=1)
                re=F.mse_loss(xb_hat, xb, reduction="none").mean(dim=1)

                # -------------------------
                # Latent anomaly score
                # -------------------------
                z_norm = torch.norm(z, p=2, dim=1)

                # -------------------------
                # Save
                # -------------------------
                re_errors.append(re.cpu().numpy())
                z_norm_scores.append(z_norm.cpu().numpy())
                y_true.append(yb.cpu().numpy())

            re_errors = np.concatenate(re_errors)
            z_norm_scores = np.concatenate(z_norm_scores)
            y_true = np.concatenate(y_true)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            print(f"[Time] get_score_mae: {(t1 - t0) * 1000:.2f} ms")
            return re_errors, z_norm_scores, y_true

    def plot_latent_first_component_hist_from_latents(self, latents, out_dir, epoch, bins=50):
        """
        Plot and save a histogram of the first component (index 0) of the latent vectors.
        Also save the raw values as a CSV for further analysis.

        - latents: numpy array shape (N, D)
        - out_dir: directory to save outputs
        - epoch, client_id: used for filenames
        - bins: number of histogram bins
        Returns path to saved PNG or None if nothing saved.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)
        if latents is None or latents.shape[0] == 0:
            return None

        # Äáº£m báº£o chá»‰ láº¥y thÃ nh pháº§n Ä‘áº§u tiÃªn (1 chiá»u)
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
            # Chá»‰ truyá»n máº£ng 1 chiá»u vÃ o plt.hist Ä‘á»ƒ trÃ¡nh cáº£nh bÃ¡o mÃ u
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


class MAECTGAN(BaseSynthesizer):
    """MÃ´ hÃ¬nh CTGAN tá»•ng há»£p báº£ng cÃ³ Ä‘iá»u kiá»‡n.

    ÄÃ¢y lÃ  lá»›p lÃµi cá»§a dá»± Ã¡n CTGAN, nÆ¡i cÃ¡c thÃ nh pháº§n khÃ¡c nhau Ä‘Æ°á»£c phá»‘i há»£p vá»›i nhau.
    Äá»ƒ biáº¿t thÃªm chi tiáº¿t vá» quy trÃ¬nh, xem bÃ i bÃ¡o [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503).

    Tham sá»‘:
            "mae_latent_hist",  # Ensure out_dir is a string, not ndarray
        generator_dim (tuple hoáº·c list): KÃ­ch thÆ°á»›c Ä‘áº§u ra cá»§a tá»«ng lá»›p Residual. Máº·c Ä‘á»‹nh (256, 256).
        discriminator_dim (tuple hoáº·c list): KÃ­ch thÆ°á»›c Ä‘áº§u ra cá»§a tá»«ng lá»›p Discriminator. Máº·c Ä‘á»‹nh (256, 256).
        generator_lr (float): Learning rate cho generator. Máº·c Ä‘á»‹nh 2e-4.
        generator_decay (float): Trá»ng sá»‘ giáº£m cho Adam Optimizer cá»§a generator. Máº·c Ä‘á»‹nh 1e-6.
        discriminator_lr (float): Learning rate cho discriminator. Máº·c Ä‘á»‹nh 2e-4.
        discriminator_decay (float): Trá»ng sá»‘ giáº£m cho Adam Optimizer cá»§a discriminator. Máº·c Ä‘á»‹nh 1e-6.
        batch_size (int): Sá»‘ lÆ°á»£ng máº«u dá»¯ liá»‡u xá»­ lÃ½ má»—i bÆ°á»›c.
        discriminator_steps (int): Sá»‘ láº§n cáº­p nháº­t discriminator cho má»—i láº§n cáº­p nháº­t generator.
        log_frequency (bool): CÃ³ sá»­ dá»¥ng táº§n suáº¥t log cá»§a cÃ¡c má»©c phÃ¢n loáº¡i khi láº¥y máº«u Ä‘iá»u kiá»‡n khÃ´ng. Máº·c Ä‘á»‹nh True.
        verbose (bool): CÃ³ in tiáº¿n trÃ¬nh ra mÃ n hÃ¬nh khÃ´ng. Máº·c Ä‘á»‹nh False.
        epochs (int): Sá»‘ lÆ°á»£ng epoch huáº¥n luyá»‡n. Máº·c Ä‘á»‹nh 300.
        pac (int): Sá»‘ lÆ°á»£ng máº«u nhÃ³m láº¡i khi Ã¡p dá»¥ng discriminator. Máº·c Ä‘á»‹nh 10.
        cuda (bool): CÃ³ sá»­ dá»¥ng GPU khÃ´ng. Náº¿u False hoáº·c khÃ´ng cÃ³ CUDA thÃ¬ dÃ¹ng CPU. Máº·c Ä‘á»‹nh True.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, lamda=0.1,
                 margin=1.0, mae_loss_mode="origin", type_train="all", mae_start_ratio=0.0,
                 ctgan_noise_start_ratio=0.75, ctgan_noise_std=0.1,
                 ctgan_noise_feature_ratio=1.0, ctgan_pseudo_anomaly_mode="origin",
                 ctgan_feature_outlier_r=1.0):
        """
        HÃ m khá»Ÿi táº¡o cho lá»›p CTGAN. Nháº­n cÃ¡c siÃªu tham sá»‘ vÃ  khá»Ÿi táº¡o Ä‘á»‘i tÆ°á»£ng theo cÃ¡c tham sá»‘ nÃ y.

        :param embedding_dim: KÃ­ch thÆ°á»›c máº«u ngáº«u nhiÃªn Ä‘Æ°a vÃ o Generator.
        :param generator_dim: KÃ­ch thÆ°á»›c Ä‘áº§u ra cá»§a tá»«ng lá»›p Residual.
        :param discriminator_dim: KÃ­ch thÆ°á»›c Ä‘áº§u ra cá»§a tá»«ng lá»›p Discriminator.
        :param generator_lr: Learning rate cho generator.
        :param generator_decay: Trá»ng sá»‘ giáº£m cho Adam Optimizer cá»§a generator.
        :param discriminator_lr: Learning rate cho discriminator.
        :param discriminator_decay: Trá»ng sá»‘ giáº£m cho Adam Optimizer cá»§a discriminator.
        :param batch_size: Sá»‘ lÆ°á»£ng máº«u dá»¯ liá»‡u xá»­ lÃ½ má»—i bÆ°á»›c.
        :param discriminator_steps: Sá»‘ láº§n cáº­p nháº­t discriminator cho má»—i láº§n cáº­p nháº­t generator.
        :param log_frequency: CÃ³ sá»­ dá»¥ng táº§n suáº¥t log cá»§a cÃ¡c má»©c phÃ¢n loáº¡i khi láº¥y máº«u Ä‘iá»u kiá»‡n khÃ´ng.
        :param verbose: CÃ³ in tiáº¿n trÃ¬nh ra mÃ n hÃ¬nh khÃ´ng.
        :param epochs: Sá»‘ lÆ°á»£ng epoch huáº¥n luyá»‡n.
        :param pac: Sá»‘ lÆ°á»£ng máº«u nhÃ³m láº¡i khi Ã¡p dá»¥ng discriminator.
        :param cuda: CÃ³ sá»­ dá»¥ng GPU khÃ´ng.
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
        self.lamda=lamda
        self.margin=margin
        self.mae_loss_mode = str(mae_loss_mode)
        self.type_train=type_train
        self.mae_start_ratio = float(mae_start_ratio)
        self.ctgan_noise_start_ratio = float(ctgan_noise_start_ratio)
        self.ctgan_noise_std = float(ctgan_noise_std)
        self.ctgan_noise_feature_ratio = float(ctgan_noise_feature_ratio)
        self.ctgan_pseudo_anomaly_mode = (
            "feature_outlier"
            if ctgan_pseudo_anomaly_mode == "feature_outliner"
            else str(ctgan_pseudo_anomaly_mode)
        )
        self.ctgan_feature_outlier_r = float(ctgan_feature_outlier_r)
        if not 0.0 <= self.mae_start_ratio <= 1.0:
            raise ValueError("mae_start_ratio must be in [0, 1].")
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
        if self.mae_loss_mode not in {"origin", "renormal"}:
            raise ValueError("mae_loss_mode must be 'origin' or 'renormal'.")

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
        self._mae=None
        
        self._on_epoch_end_callbacks = None
        self._on_epoch_best_callbacks = None
        self._validation_cont_mean = None
        self._validation_cont_std = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Xá»­ lÃ½ sá»± khÃ´ng á»•n Ä‘á»‹nh cá»§a gumbel_softmax cho cÃ¡c phiÃªn báº£n torch cÅ©.

        Tham kháº£o thÃªm vá» váº¥n Ä‘á» nÃ y:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Tham sá»‘:
            logits [â€¦, num_features]: Log xÃ¡c suáº¥t chÆ°a chuáº©n hÃ³a
            tau: Nhiá»‡t Ä‘á»™ (temperature) khÃ´ng Ã¢m
            hard (bool): Náº¿u True, tráº£ vá» vector one-hot, nhÆ°ng váº«n tÃ­nh gradient nhÆ° soft sample
            dim (int): Chiá»u Ä‘á»ƒ tÃ­nh softmax. Máº·c Ä‘á»‹nh -1.

        Tráº£ vá»:
            Tensor láº¥y máº«u cÃ¹ng shape vá»›i logits tá»« phÃ¢n phá»‘i Gumbel-Softmax.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Ãp dá»¥ng hÃ m kÃ­ch hoáº¡t phÃ¹ há»£p lÃªn Ä‘áº§u ra cá»§a generator."""
        # HÃ m nÃ y nháº­n Ä‘áº§u ra cá»§a generator vÃ  Ã¡p dá»¥ng hÃ m kÃ­ch hoáº¡t phÃ¹ há»£p.
        # Äáº§u ra cá»§a generator lÃ  tensor cÃ³ shape (batch_size, total_dim), Ä‘áº§u ra cá»§a hÃ m nÃ y cÅ©ng váº­y.

        # Láº·p qua tá»«ng cá»™t trong Ä‘áº§u ra cá»§a generator vÃ  Ã¡p dá»¥ng hÃ m kÃ­ch hoáº¡t Ä‘Æ°á»£c chá»‰ Ä‘á»‹nh trong transformer.
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
        """TÃ­nh loss cross entropy trÃªn cÃ¡c cá»™t phÃ¢n loáº¡i cá»‘ Ä‘á»‹nh.

        HÃ m nÃ y nháº­n Ä‘áº§u ra cá»§a generator, vector Ä‘iá»u kiá»‡n vÃ  vector mask. Sau Ä‘Ã³ tÃ­nh loss cross entropy trÃªn tá»«ng cá»™t phÃ¢n loáº¡i vÃ  tráº£ vá» tá»•ng loss.

        Äáº§u tiÃªn khá»Ÿi táº¡o hai con trá» st vÃ  st_c Ä‘á»ƒ theo dÃµi vá»‹ trÃ­ báº¯t Ä‘áº§u cá»§a cá»™t hiá»‡n táº¡i trong Ä‘áº§u ra generator vÃ  vector Ä‘iá»u kiá»‡n.

        Sau Ä‘Ã³ láº·p qua tá»«ng cá»™t trong Ä‘áº§u ra generator. Náº¿u cá»™t khÃ´ng pháº£i phÃ¢n loáº¡i, chá»‰ cáº§n tÄƒng con trá» theo sá»‘ chiá»u cá»§a cá»™t Ä‘Ã³.

        Náº¿u lÃ  cá»™t phÃ¢n loáº¡i, tÃ­nh loss cross entropy giá»¯a Ä‘áº§u ra generator vÃ  vector Ä‘iá»u kiá»‡n, cá»™ng vÃ o tá»•ng loss.

        Cuá»‘i cÃ¹ng tráº£ vá» tá»•ng loss chia cho sá»‘ lÆ°á»£ng máº«u dá»¯ liá»‡u.
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
        """Kiá»ƒm tra cÃ¡c cá»™t phÃ¢n loáº¡i cÃ³ tá»“n táº¡i trong dá»¯ liá»‡u huáº¥n luyá»‡n khÃ´ng.

        Tham sá»‘:
            train_data (numpy.ndarray hoáº·c pandas.DataFrame): Dá»¯ liá»‡u huáº¥n luyá»‡n, pháº£i lÃ  máº£ng 2 chiá»u hoáº·c DataFrame.
            discrete_columns (list): Danh sÃ¡ch cÃ¡c cá»™t phÃ¢n loáº¡i dÃ¹ng Ä‘á»ƒ táº¡o vector Ä‘iá»u kiá»‡n. Náº¿u lÃ  numpy thÃ¬ lÃ  chá»‰ sá»‘, náº¿u lÃ  DataFrame thÃ¬ lÃ  tÃªn cá»™t.

        Náº¿u train_data lÃ  DataFrame, kiá»ƒm tra táº¥t cáº£ cÃ¡c cá»™t trong discrete_columns cÃ³ trong DataFrame khÃ´ng. Náº¿u khÃ´ng, bÃ¡o lá»—i ValueError.

        Náº¿u train_data lÃ  numpy array, kiá»ƒm tra cÃ¡c chá»‰ sá»‘ cÃ³ há»£p lá»‡ khÃ´ng. Náº¿u cÃ³ chá»‰ sá»‘ khÃ´ng há»£p lá»‡, bÃ¡o lá»—i ValueError.

        Náº¿u train_data khÃ´ng pháº£i DataFrame hoáº·c numpy array, bÃ¡o lá»—i TypeError.
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
        
        print("Starting MAECTGAN training..., data shape:", train_data.shape, " discrete_columns:", discrete_columns," epochs:", self._epochs, "lamda:", self.lamda, " margin:", self.margin)
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
        print(f'Starting training mae-CTGAN for {epochs} epochs..., data shape: {train_data.shape}    ')
        # Fit the DataTransformer to the training data # táº¡o bá»™ transform, lÆ°u thÃ´ng tin cÃ¡c cá»™t sau khi transform
        self._transformer = DataTransformer()
        self._transformer.fit(full_data, discrete_columns)

        # Transform the training data # transform data thÃ nh dáº¡ng mode-specific normalization
        train_data = self._transformer.transform(train_data)
        print("data after transform shape:",train_data.shape)
        
        X_val_train_processed = self._transformer.transform(val_data)
        y_val_train = np.zeros(X_val_train_processed.shape[0])  # Giáº£ sá»­ nhÃ£n lÃ  0 cho táº¥t cáº£ cÃ¡c máº«u trong táº­p huáº¥n luyá»‡n bÃ¬nh thÆ°á»ng
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
        self._mae=MAE(
            input_dim=data_dim,
            lamda=self.lamda,
            margin=self.margin,
            mae_loss_mode=self.mae_loss_mode,
        ).to(self._device)

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
            'MAE Loss',
            'Reconstruction Loss',
            'Latent Loss',
            'Validation Loss',
            'Validation Reconstruction Loss',
            'Validation Latent Loss',
            'Pseudo Val AUC',
            'Pseudo Separation',
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
            list_latent = None
            mae_dataset = None
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
                    fakeact = self._apply_activate(fake) # apply activation function chuyá»ƒn Ä‘áº§u
                    
                    
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
                        # fakeact_for_mae_ano = self.add_gaussian_noise_to_ctgan_samples(
                        #     fakeact_for_mae_ano,
                        #     noise_std=self.ctgan_noise_std,
                        #     feature_ratio=self.ctgan_noise_feature_ratio,
                        # )

                fake_buff_list_NOR=self.to_buffer(fake_buff_list_NOR,fakeact_for_mae_nor,self._batch_size)
                fake_buff_list_ANO=self.to_buffer(fake_buff_list_ANO,fakeact_for_mae_ano,self._batch_size)
                
                real_perturbed = self.make_real_perturbed_anomaly(real_data)
                
                # lÆ°u dá»¯ dl sinh vÃ o bá»™ Ä‘á»‡m
                 # Láº¥y dá»¯ liá»‡u fake tá»« buffer náº¿u cÃ³, náº¿u khÃ´ng thÃ¬ dÃ¹ng fake má»›i táº¡o
                if fake_buff_list_NOR is None:
                    fakeact_form_buffer_NOR = fakeact_for_mae_nor
                else:
                    fakeact_form_buffer_NOR=self.from_buffer(fake_buff_list_NOR, self._batch_size)
                
                if fake_buff_list_ANO is None:
                    fakeact_form_buffer_ANO=fakeact_for_mae_ano # buffer cho anomaly critic
                else:
                    fakeact_form_buffer_ANO=self.from_buffer(fake_buff_list_ANO, self._batch_size)
                #train MAE every discriminator step
                
                
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

                if fake_data is not None:
                    fake_data = self._post_assemble_pseudo_anomalies(fake_data, i, epochs)

                ##fake_data=torch.cat([fakeact_form_buffer_ANO, real_perturbed,fakeact_form_buffer_ANO], dim=0)
                #fake_data=torch.cat([real_perturbed,fakeact_form_buffer_NOR], dim=0)
                y_fake_mae = torch.ones(fake_data.size(0), 1, device=self._device)
                
                
                #num_samples = self._batch_size  # sá»‘ lÆ°á»£ng máº«u báº¡n muá»‘n láº¥y
                #perm = torch.randperm(fake_data.size(0), device=self._device)[:num_samples]
                fake_data_sample = fake_data
                y_fake_mae_sample = y_fake_mae
                
                #real_data_mae = torch.cat([real_data, fakeact_form_buffer_NOR], dim=0)
                #y_real_mae = torch.zeros(real_data_mae.size(0), 1, device=self._device)
                y_real = torch.zeros(real_data.size(0), 1, device=self._device)
                mae_dataset=torch.cat([fake_data_sample, real_data], dim=0)
                y_mae=torch.cat([y_fake_mae_sample, y_real], dim=0)
                mae_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(mae_dataset, y_mae),
                    batch_size=self._batch_size,
                    shuffle=True
                )
                mae_loss, re_loss, latent_loss,list_latent = self._mae.train_mae(epoch=i, train_data_loader=mae_dataloader)
            
            # Compute the generator loss and the discriminator loss
            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            avg_val_total_loss = np.nan
            avg_val_re_loss = np.nan
            avg_val_latent_loss = np.nan
            pseudo_val_auc = np.nan
            pseudo_val_separation = np.nan
            if mae_active:
                # Evaluate and select a best checkpoint only after MAE has been updated.
                avg_val_total_loss, threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss = self._mae.validate(
                    val_data_loader,
                    epoch=i
                )
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
                pseudo_val_auc, pseudo_val_separation = self.evaluate_pseudo_validation_auc(
                    val_data_loader
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
                            threshold_95_total_loss=threshold_95_total_loss
                        )
                    print(
                        f"[BEST] epoch={i} pseudo_val_auc={pseudo_val_auc:.6f} "
                        f"pseudo_separation={pseudo_val_separation:.6f} "
                        f"val_total_loss={avg_val_total_loss:.6f}"
                    )
                # self._mae.plot_latent_first_component_hist_from_latents(
                #     np.asarray(list_latent),
                #     f"mae_latent_hist_{epochs}",  # Ensure out_dir is a string, not ndarray
                #     i,
                # )
                mae_data_shape = mae_dataset.shape
            else:
                mae_data_shape = "warmup: MAE not trained"

            # Append the loss values to the loss values DataFrame
            epoch_loss_df = pd.DataFrame({
            'Epoch': [i],
            'Generator Loss': [generator_loss],
            'Discriminator Loss': [discriminator_loss],
            'MAE Loss': [mae_loss.item() if torch.is_tensor(mae_loss) else mae_loss],
            'Reconstruction Loss': [re_loss.item() if torch.is_tensor(re_loss) else re_loss],
            'Latent Loss': [latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss],
            'Validation Loss': [avg_val_total_loss],
            'Validation Reconstruction Loss': [avg_val_re_loss],
            'Validation Latent Loss': [avg_val_latent_loss],
            'Pseudo Val AUC': [pseudo_val_auc],
            'Pseudo Separation': [pseudo_val_separation],
            })
            print(
                f"Epoch {i}: Generator Loss = {generator_loss}, Discriminator Loss = {discriminator_loss}, "
                f"MAE Loss = {mae_loss}, Reconstruction Loss = {re_loss}, Latent Loss = {latent_loss}, "
                f"CTGAN Gaussian noise active = {gaussian_noise_active}, "
                f"Pseudo Val AUC = {pseudo_val_auc}, Pseudo Separation = {pseudo_val_separation}",
                "data mae shape: ",
                mae_data_shape,
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
        """Láº¥y máº«u dá»¯ liá»‡u giá»‘ng vá»›i dá»¯ liá»‡u huáº¥n luyá»‡n.

        Chá»n condition_column vÃ  condition_value sáº½ tÄƒng xÃ¡c suáº¥t xuáº¥t hiá»‡n giÃ¡ trá»‹ Ä‘Ã³ trong cá»™t phÃ¢n loáº¡i tÆ°Æ¡ng á»©ng.

        Tham sá»‘:
            n (int): Sá»‘ lÆ°á»£ng dÃ²ng cáº§n láº¥y máº«u.
            condition_column (string): TÃªn cá»™t phÃ¢n loáº¡i.
            condition_value (string): GiÃ¡ trá»‹ phÃ¢n loáº¡i mong muá»‘n tÄƒng xÃ¡c suáº¥t xuáº¥t hiá»‡n.

        Tráº£ vá»:
            numpy.ndarray hoáº·c pandas.DataFrame
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
        """Thiáº¿t láº­p thiáº¿t bá»‹ sá»­ dá»¥ng ('GPU' hoáº·c 'CPU')."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
    def get_score_mae(self, X,y):
        """TÃ­nh anomaly score tá»« D2.
        Score cÃ ng tháº¥p â†’ cÃ ng báº¥t thÆ°á»ng
        """
        X_processed = self._transformer.transform(X)
        score_res, latent_loss, y_true = self._mae.get_score_mae(X_processed, y,batch_size=self._batch_size,device=self._device)
        return score_res, latent_loss, y_true

    def get_latent_z_mae(self, X, batch_size=None):
        """Return MAE latent vectors (z) for input X."""
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
        self._mae.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device)
                z, _ = self._mae.forward(xb)
                z_all.append(z.cpu().numpy())
        return np.concatenate(z_all)
    def get_threshold_mae(self, X_val_train):
        """TÃ­nh threshold tá»« táº­p validation."""
        X_val_train_processed = self._transformer.transform(X_val_train)
        y_val_train = np.zeros(X_val_train_processed.shape[0])  # Giáº£ sá»­ nhÃ£n lÃ  0 cho táº¥t cáº£ cÃ¡c máº«u trong táº­p huáº¥n luyá»‡n
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_train_processed, dtype=torch.float32), torch.tensor(y_val_train, dtype=torch.float32))
        val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
        
        avg_loss, threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss = self._mae.validate(val_data_loader,epoch=0)
        return avg_loss, threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss
    
    @staticmethod
    def to_buffer(buffer, new_fake, buf_per_iter):
        """
        Cáº­p nháº­t replay buffer báº±ng cÃ¡ch:
        - Láº¥y `buf_per_iter` máº«u tá»« batch fake má»›i (`new_fake`)
        - Ná»‘i vÃ o buffer hiá»‡n cÃ³
        - Giá»¯ láº¡i tá»‘i Ä‘a `8 * buf_per_iter` máº«u trong buffer
        """
        # Äáº§u vÃ o new_fake: (batch_size, D)
        # Äáº§u ra buffer:   (<= 8 * buf_per_iter, D)
        new_fake = new_fake.clone().detach()  # CLONE + DETACH an toÃ n
        batch_size, data_dim = new_fake.shape  # new_fake: (B, D)

        # Láº¥y ngáº«u nhiÃªn buf_per_iter máº«u tá»« new_fake
        perm = torch.randperm(batch_size)
        idx = perm[:buf_per_iter]
        new_buffer = new_fake.detach()[idx]  # (buf_per_iter, D)

        if buffer is None:
            # Náº¿u chÆ°a cÃ³ buffer, khá»Ÿi táº¡o luÃ´n
            buffer = new_buffer  # (buf_per_iter, D)
        else:
            buffer = buffer.clone().detach().to(new_fake.device)

            # Ná»‘i fake má»›i vÃ o buffer cÅ©
            buffer = torch.cat((buffer, new_buffer), dim=0)

            # Shuffle vÃ  giá»¯ láº¡i tá»‘i Ä‘a 8 * buf_per_iter máº«u
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

    def _post_assemble_pseudo_anomalies(self, fake_data, epoch, total_epochs):
        """Hook called after the pseudo-anomaly batch is assembled each training step.

        Override in subclasses to filter or modify pseudo-anomalies.
        Default implementation returns fake_data unchanged.
        """
        return fake_data

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
        self._mae.eval()
        normal_scores = []
        pseudo_scores = []

        with torch.no_grad():
            for xb, _ in val_data_loader:
                xb = xb.to(self._device)
                xb_pseudo = self.make_validation_pseudo_anomaly(xb)

                z_normal, _ = self._mae.forward(xb)
                z_pseudo, _ = self._mae.forward(xb_pseudo)

                normal_scores.append(torch.norm(z_normal, p=2, dim=1).cpu().numpy())
                pseudo_scores.append(torch.norm(z_pseudo, p=2, dim=1).cpu().numpy())

        if not normal_scores or not pseudo_scores:
            return 0.0, -float("inf")

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
        pseudo_separation = float(np.mean(pseudo_scores) - np.mean(normal_scores))
        return float(pseudo_auc), pseudo_separation

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
