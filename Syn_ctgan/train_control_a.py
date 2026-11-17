"""Control A (Reviewer 4, Comment 2b): MiAE with ONLY the normal-compactness
term, NO pseudo-intrusions at all -- a DeepSVDD-style regularizer bolted onto
a plain autoencoder, trained in the exact same CTGAN-transformed input space
as SSAE so the comparison isolates one question: is the pseudo-intrusion /
radial-margin mechanism actually necessary, or does compactness-only
regularization already explain SSAE's gain over the plain AE?

Loss (every epoch, normal-train only): re_loss + lamda * mean(||z_normal||^2)
-- literally the y=0 branch of MAE01ScaleMax's Eq. 15 (see
ctgan/synthesizers/m01aectgan.py:_mae_loss), with the y=1 (pseudo-anomaly)
branch never constructed. No CTGAN generator/discriminator training, no PAG,
no activation gate: the DataTransformer is fitted (not adversarially trained)
only to obtain the identical numeric representation SSAE operates in.

Encoder/decoder architecture is copied verbatim from
MAE01ScaleMaxSynthesizer._build_mae_networks so depth/width are not a
confound: d -> 0.75d -> 0.5d -> floor(sqrt(d))+1, decoder mirrored with Tanh
output.

No pseudo-validation AUC exists here (no anomalies are ever generated), so
model selection uses best validation loss instead of best validation AUC.
Output schema matches every other method's best_auc/test_mode_total.csv +
best_auc/per_attack_seeds.csv (the "best_auc" directory name is kept only
for compatibility with the existing aggregation scripts, e.g.
build_seeds_summary_excel.py's *_perattack() readers).

Usage (mirrors train_mae01scale_max.py):
    python train_control_a.py --d unsw --seed 42 --time 08 --e 150 --b 128 \
        --lae 0.1 --per 1.0
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import argparse
import copy
import csv
import random
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sdv.metadata import SingleTableMetadata
from sdv.single_table.utils import detect_discrete_columns
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split

from ctgan.data_transformer import DataTransformer

from val_perattack_mae01scale_max import ATTACK_LOADERS, FAR_LEVELS, evaluate_ids_metrics

from process_data import (
    load_data_cic_iot,
    load_data_nsl_kdd,
    load_data_unsw,
    load_data_wsn_ds,
)

import sdv, ctgan
print(sdv.__file__)
print(ctgan.__file__)

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description="Train Control A (compactness-only MiAE, no pseudo-intrusions)")
parser.add_argument("--d",    type=str,   default="unsw")
parser.add_argument("--e",    type=int,   default=150)
parser.add_argument("--b",    type=int,   default=128)
parser.add_argument("--per",  type=float, default=1.0)
parser.add_argument("--seed", type=int,   default=42)
parser.add_argument("--time", type=int,   default=8)
parser.add_argument("--lae",  type=float, default=0.1, help="lambda weight on the compactness term")
parser.add_argument("--mae_lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--max_clusters", type=int, default=10)
args = parser.parse_args()

if not 0.0 < args.per <= 1.0:
    parser.error("--per must be in (0, 1].")
if args.patience < 1:
    parser.error("--patience must be >= 1.")

dataset_raw = args.d.strip().lower().replace("-", "_")
dataset = dataset_raw

epochs      = args.e
batch_size  = args.b
lamda       = args.lae
percentage  = args.per
mae_lr      = args.mae_lr
patience    = args.patience
max_clusters = args.max_clusters
seed        = args.seed
time_tag    = f"{int(args.time):02d}"
run_folder  = f"t_{time_tag}_ControlA_se_{seed}"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed)

print(
    "ControlA | dataset:", dataset, " epochs:", epochs, " batch:", batch_size,
    " lamda:", lamda, " patience:", patience,
)

DATASET_LOADERS = {
    "unsw":    load_data_unsw,
    "nsl_kdd": load_data_nsl_kdd,
    "cic_iot": load_data_cic_iot,
    "wsn_ds":  lambda: load_data_wsn_ds(seed=seed, test_size=0.2),
}
if dataset not in DATASET_LOADERS:
    raise ValueError(f"Unsupported dataset '{args.d}'. Supported: {', '.join(DATASET_LOADERS)}")

df_full, y_full, df_test, y_test = DATASET_LOADERS[dataset]()
print("Full shape:", df_full.shape, " Test shape:", df_test.shape)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(BASE_DIR, f"res/{run_folder}/per_{percentage}/{dataset}_{epochs}e_l{lamda}")
os.makedirs(results_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Encoder / decoder -- copied verbatim from
# ctgan/synthesizers/m01aectgan.py:_build_mae_networks, so architecture is
# not a confound relative to SSAE.
# =============================================================================

def build_mae_networks(input_dim, dev):
    d = input_dim
    lat = round(sqrt(d)) + 1
    enc = nn.Sequential(
        nn.Linear(d, round(d * 0.75)),
        nn.Tanh(),
        nn.Linear(round(d * 0.75), round(d * 0.5)),
        nn.Tanh(),
        nn.Linear(round(d * 0.5), lat),
    ).to(dev)
    dec = nn.Sequential(
        nn.Linear(lat, round(d * 0.5)),
        nn.Tanh(),
        nn.Linear(round(d * 0.5), round(d * 0.75)),
        nn.Tanh(),
        nn.Linear(round(d * 0.75), d),
        nn.Tanh(),
    ).to(dev)
    return enc, dec


def mae_loss(x, z, x_hat, lamda):
    """y=0 branch only (Eq. 15 of the manuscript): reconstruction + compactness."""
    re_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
    compactness = torch.mean(torch.norm(z, p=2, dim=1) ** 2)
    return re_loss, compactness


def main():
    n_test = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)

    normal_mask = y_train == 0
    X_train_normal_full = X_train_raw.loc[normal_mask].copy()
    y_train_normal_full = y_train.loc[normal_mask].copy()

    sampled_idx = X_train_normal_full.sample(frac=percentage, random_state=seed).index
    X_normal_sub = X_train_normal_full.loc[sampled_idx].reset_index(drop=True)

    X_train_normal, X_val_raw = train_test_split(
        X_normal_sub, test_size=0.2, random_state=seed
    )
    X_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    X_test_raw.columns = X_test_raw.columns.map(str)

    # Same metadata/discrete-column detection as the official pipeline, so
    # Control A sees the IDENTICAL representation SSAE is trained on.
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_train_normal)
    if dataset == "nb_iot":
        metadata.update_columns(X_train_normal.columns.tolist(), sdtype="numerical")
    discrete_columns = detect_discrete_columns(metadata, X_train_normal, {})
    print(f"Discrete columns: {len(discrete_columns)} / {X_train_normal.shape[1]}")

    transformer = DataTransformer(max_clusters=max_clusters)
    transformer.fit(X_train_normal, discrete_columns=discrete_columns)

    X_train_t = torch.tensor(transformer.transform(X_train_normal), dtype=torch.float32)
    X_val_t   = torch.tensor(transformer.transform(X_val_raw),      dtype=torch.float32)
    X_test_t  = torch.tensor(transformer.transform(X_test_raw),     dtype=torch.float32)

    input_dim = X_train_t.shape[1]
    print("Transformed dim:", input_dim)

    enc, dec = build_mae_networks(input_dim, device)
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=mae_lr)

    X_train_t = X_train_t.to(device)
    X_val_t = X_val_t.to(device)

    n_train = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_enc_state = copy.deepcopy(enc.state_dict())
    best_dec_state = copy.deepcopy(dec.state_dict())
    epochs_no_improve = 0

    loss_rows = []
    for ep in range(epochs):
        enc.train(); dec.train()
        perm = torch.randperm(n_train, device=device)
        re_accum = comp_accum = 0.0
        steps = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx]
            z = enc(xb)
            x_hat = dec(z)
            re_loss, compactness = mae_loss(xb, z, x_hat, lamda)
            loss = re_loss + lamda * compactness

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            re_accum += re_loss.item()
            comp_accum += compactness.item()
            steps += 1

        avg_re = re_accum / steps
        avg_comp = comp_accum / steps

        enc.eval(); dec.eval()
        with torch.no_grad():
            z_val = enc(X_val_t)
            x_val_hat = dec(z_val)
            val_re, val_comp = mae_loss(X_val_t, z_val, x_val_hat, lamda)
            val_loss = (val_re + lamda * val_comp).item()

        print(f"Epoch {ep:03d}: RE={avg_re:.4f} Compact={avg_comp:.4f} | ValLoss={val_loss:.4f}")
        loss_rows.append({"Epoch": ep, "Reconstruction Loss": avg_re,
                           "Compactness": avg_comp, "Validation Loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_enc_state = copy.deepcopy(enc.state_dict())
            best_dec_state = copy.deepcopy(dec.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep} (patience={patience}).")
                break

    pd.DataFrame(loss_rows).to_csv(
        os.path.join(results_dir, f"control_a_loss_{epochs}e.csv"), index=False
    )

    enc.load_state_dict(best_enc_state)
    dec.load_state_dict(best_dec_state)
    enc.eval(); dec.eval()

    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "input_dim": input_dim},
               os.path.join(results_dir, "control_a_model.pt"))
    with open(os.path.join(results_dir, "control_a_transformer.pkl"), "wb") as f:
        import pickle
        pickle.dump(transformer, f)

    def score_fn(x_t):
        with torch.no_grad():
            z = enc(x_t.to(device))
            x_hat = dec(z)
            re_per = torch.nn.functional.mse_loss(x_hat, x_t.to(device), reduction="none").mean(dim=1)
            z_norm2 = torch.norm(z, p=2, dim=1) ** 2
            total = re_per + lamda * z_norm2
        return total.cpu().numpy()

    val_total = score_fn(X_val_t)
    test_total = score_fn(X_test_t)
    y_true_bin = (np.asarray(y_test) != 0).astype(int)

    save_dir = os.path.join(results_dir, "best_auc")  # name kept for schema compatibility (see docstring)
    os.makedirs(save_dir, exist_ok=True)

    val_thresholds = {}
    save_path = os.path.join(save_dir, "test_mode_total.csv")
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for percentile, far_label in ((95, "TPR@FAR=5%"), (99, "TPR@FAR=1%")):
            threshold = float(np.percentile(val_total, percentile))
            val_thresholds[percentile] = threshold
            y_pred = (test_total > threshold).astype(int)
            auc = roc_auc_score(y_true_bin, test_total) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
            mcc = matthews_corrcoef(y_true_bin, y_pred) * 100
            tpr = recall_score(y_true_bin, y_pred, zero_division=0) * 100
            w.writerow([f"Threshold_{percentile}", f"{threshold:.6f}"])
            w.writerow([far_label, f"{tpr:.6f}"])
            w.writerow([f"AUC_{percentile}", f"{auc:.6f}"])
            w.writerow([f"MCC_{percentile}", f"{mcc:.6f}"])
            print(f"  [{far_label} @ p{percentile}] AUC_{percentile}={auc:.2f}% MCC_{percentile}={mcc:.2f}% {far_label}={tpr:.2f}% thr={threshold:.6f}")
    print(f"Overall eval done -> {save_path}")

    # Per-attack breakdown (reuse the SAME loaders/metric function as every other method)
    loader = ATTACK_LOADERS.get(dataset)
    if loader is None:
        print(f"[WARN] No attack-label loader for dataset '{dataset}' -- skip per_attack_seeds.csv.")
        return
    attack_labels, normal_tag = loader(seed)
    if len(attack_labels) != len(test_total):
        print(f"[WARN] attack labels ({len(attack_labels)}) != test scores ({len(test_total)}) -- skip per_attack_seeds.csv.")
        return

    normal_mask_t = (attack_labels == normal_tag).to_numpy()
    n_normal = int(normal_mask_t.sum())
    attacks = sorted(a for a in attack_labels.unique() if a != normal_tag)
    scores_normal = test_total[normal_mask_t]
    normal_thresholds = {far: np.percentile(scores_normal, 100 * (1 - far)) for far in FAR_LEVELS}

    rows = []
    for attack in attacks:
        atk_mask = (attack_labels == attack).to_numpy()
        sub_scores = np.concatenate([scores_normal, test_total[atk_mask]])
        sub_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(int(atk_mask.sum()), dtype=int)])
        metrics = evaluate_ids_metrics(sub_y, sub_scores, normal_thresholds, val_thresholds)
        rows.append({"Dataset": dataset, "Seed": seed, "Attack": attack,
                     "N": int(atk_mask.sum()), "Score": "total", **metrics})

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(save_dir, "per_attack_seeds.csv"), index=False)
        print(f"Per-attack eval done -> {save_dir}/per_attack_seeds.csv")

    print("=== Done ===")


if __name__ == "__main__":
    main()
