"""Training script for MAE01ScaleMax — CoV-floor fix for CTGAN gate.

Same as MAE01Scale but gate uses max(|mean(window)|, cov_floor) denominator
instead of (|mean(window)| + 1e-8), fixing the near-zero D_loss problem
(e.g. NSL-KDD where D_loss converges to ~0).

Extra param vs MAE01Scale:
    --cov_floor   min denominator for CoV gate check (default 0.05)
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import argparse
import csv
import pickle
import random

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table.mae01scale_max import MAE01ScaleMaxSynthesizer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from process_data import (
    load_data_bot_iot,
    load_data_cic_ids,
    load_data_cic_iot,
    load_data_cidds_001,
    load_data_insdn,
    load_data_nb_iot,
    load_data_nsl_kdd,
    load_data_ton_iot,
    load_data_unsw,
    load_data_wsn_ds,
)

import sdv, ctgan
print(sdv.__file__)
print(ctgan.__file__)

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description="Train MAE01ScaleMax")

# Dataset / basics
parser.add_argument("--d",             type=str,   default="unsw")
parser.add_argument("--e",             type=int,   default=150)
parser.add_argument("--b",             type=int,   default=128)
parser.add_argument("--per",           type=float, default=1.0)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--time",          type=int,   default=1)
parser.add_argument("--n_max_samples", type=int,   default=500_000)

# M01AE
parser.add_argument("--lae",           type=float, default=0.1)
parser.add_argument("--mae",           type=float, default=2.0)
parser.add_argument("--mae_loss_mode", type=str,   default="renormal",
                    choices=["origin", "renormal"])
parser.add_argument("--mae_lr",        type=float, default=1e-3)

# Noise
parser.add_argument("--noise_std",   type=float, default=0.3)
parser.add_argument("--noise_ratio", type=float, default=0.5)
parser.add_argument("--noise_type",  type=str,   default="global",
                    choices=["none", "global", "feature", "feature_outlier"])
parser.add_argument("--noise_r",     type=float, default=1.0)
parser.add_argument("--noise_k",     type=float, default=3.0)
parser.add_argument("--cat_noise_type",  type=str, default="combined",
                    choices=["none", "shuffle", "flip", "combined"])
parser.add_argument("--cat_noise_ratio", type=float, default=0.1)

# Joint training
parser.add_argument("--type_train", type=str, default="noise_std",
                    choices=["noise", "std", "noise_std"])
parser.add_argument("--patience",   type=int, default=50)

# Scale params
parser.add_argument("--noise_scale",    type=float, default=1.5)
parser.add_argument("--gate_stability", type=float, default=0.05,
                    help="CoV threshold to open CTGAN gate")
parser.add_argument("--gate_window",    type=int,   default=10,
                    help="rolling window + min-epoch (merged)")
parser.add_argument("--max_clusters",   type=int,   default=None,
                    help="DataTransformer max Gaussian-mixture modes per "
                         "continuous column (each -> 1 + max_clusters dims). "
                         "Default: 10 (ctgan.DataTransformer default), same "
                         "for every dataset.")
parser.add_argument("--snapshot_stride", type=int,  default=None,
                    help="take every Nth normal-train row when saving "
                         "generated_data_snapshots.csv (1 = all rows). "
                         "Default: 1, except nb_iot which defaults to 20 "
                         "(its ~210K normal rows would otherwise make the "
                         "snapshot file huge).")

# CoV-floor fix
parser.add_argument("--cov_floor", type=float, default=0.05,
                    help="min denominator: cov = std / max(|mean|, cov_floor)")

# Near-zero equilibrium branch (OR gate)
parser.add_argument("--eps_zero", type=float, default=0.1,
                    help="near-zero branch: |mean(EMA disc window)| <= eps_zero")
parser.add_argument("--gate_fixed_epoch", type=int, default=None,
                    help="if set, bypasses the adaptive CoV/near-zero gate entirely and "
                         "opens the CTGAN gate at this fixed epoch instead (e.g. 0/30/50), "
                         "for ablating the dynamic gate against a fixed schedule")
parser.add_argument("--tol", type=float, default=0.08,
                    help="near-zero branch: std(EMA disc window) <= tol")

args = parser.parse_args()

# ── Validation ────────────────────────────────────────────────────────────────
if not 0.0 < args.per <= 1.0:
    parser.error("--per must be in (0, 1].")
if args.noise_std < 0:
    parser.error("--noise_std must be >= 0.")
if args.noise_scale <= 0:
    parser.error("--noise_scale must be > 0.")
if args.gate_stability <= 0:
    parser.error("--gate_stability must be > 0.")
if args.gate_window < 2:
    parser.error("--gate_window must be >= 2.")
if args.cov_floor <= 0:
    parser.error("--cov_floor must be > 0.")
if args.eps_zero <= 0:
    parser.error("--eps_zero must be > 0.")
if args.tol <= 0:
    parser.error("--tol must be > 0.")
if args.gate_fixed_epoch is not None and args.gate_fixed_epoch < 0:
    parser.error("--gate_fixed_epoch must be >= 0.")
if args.patience < 1:
    parser.error("--patience must be >= 1.")
if args.max_clusters is not None and args.max_clusters < 1:
    parser.error("--max_clusters must be >= 1.")
if args.snapshot_stride is not None and args.snapshot_stride < 1:
    parser.error("--snapshot_stride must be >= 1.")

dataset_raw = args.d.strip().lower().replace("-", "_")
if dataset_raw == "cidds001":
    dataset_raw = "cidds_001"
dataset = dataset_raw

max_clusters = args.max_clusters if args.max_clusters is not None else 10  # ctgan.DataTransformer default, same for every dataset

if args.snapshot_stride is not None:
    snapshot_stride = args.snapshot_stride
elif dataset == "nb_iot":
    snapshot_stride = 20  # ~210K normal rows -- keep generated_data_snapshots.csv manageable
else:
    snapshot_stride = 1   # take every normal-train row for every other dataset

epochs          = args.e
batch_size      = args.b
lamda           = args.lae
margin          = args.mae
mae_loss_mode   = args.mae_loss_mode
mae_lr          = args.mae_lr
percentage      = args.per
noise_std       = args.noise_std
noise_ratio     = args.noise_ratio
noise_type      = args.noise_type
noise_r         = args.noise_r
noise_k         = args.noise_k
cat_noise_type  = args.cat_noise_type
cat_noise_ratio = args.cat_noise_ratio
type_train      = args.type_train
patience        = args.patience
seed            = args.seed
n_max_samples   = args.n_max_samples
run_time        = args.time
time_tag        = f"{int(run_time):02d}"
run_folder      = f"mae01scale_max_se_{seed}_t_{time_tag}"

noise_scale    = args.noise_scale
gate_stability = args.gate_stability
gate_window    = args.gate_window
cov_floor      = args.cov_floor
eps_zero       = args.eps_zero
tol            = args.tol
gate_fixed_epoch = args.gate_fixed_epoch


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


set_seed(seed)

print(
    "MAE01ScaleMax | dataset:", dataset,
    " epochs:", epochs, " batch:", batch_size,
    " lae:", lamda, " margin:", margin,
    " noise_std (init):", noise_std,
    " noise_scale:", noise_scale,
    " gate_window:", gate_window, " gate_stability:", gate_stability,
    " cov_floor:", cov_floor,
    " eps_zero:", eps_zero, " tol:", tol, " patience:", patience,
    " max_clusters:", max_clusters, " snapshot_stride:", snapshot_stride,
    " gate_fixed_epoch:", gate_fixed_epoch,
)

DATASET_LOADERS = {
    "unsw":      load_data_unsw,
    "cic_ids":   load_data_cic_ids,
    "nsl_kdd":   load_data_nsl_kdd,
    "cic_iot":   load_data_cic_iot,
    "cidds_001": lambda: load_data_cidds_001(seed=seed, test_size=0.4),
    "wsn_ds":    lambda: load_data_wsn_ds(seed=seed, test_size=0.2),
    "bot_iot":   lambda: load_data_bot_iot(seed=seed, n_max_samples=n_max_samples, test_size=0.2),
    "ton_iot":   lambda: load_data_ton_iot(seed=seed, n_max_samples=n_max_samples, test_size=0.4),
    "insdn":     lambda: load_data_insdn(seed=seed, test_size=0.2),
    "nb_iot":    lambda: load_data_nb_iot(seed=seed, test_size=0.2),
}
if dataset not in DATASET_LOADERS:
    raise ValueError(f"Unsupported dataset '{args.d}'. Supported: {', '.join(DATASET_LOADERS)}")

df_full, y_full, df_test, y_test = DATASET_LOADERS[dataset]()
print("Full shape:", df_full.shape, " Test shape:", df_test.shape)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_noise_tag = noise_type
if noise_type in {"feature", "feature_outlier"}:
    _noise_tag = f"{noise_type}_r{noise_r}_k{noise_k}"

_gatefix_tag = "" if gate_fixed_epoch is None else f"_gf{gate_fixed_epoch}"
results_dir = os.path.join(
    BASE_DIR,
    f"res/{run_folder}/per_{percentage}/"
    f"{dataset}_{epochs}e_m{margin}_l{lamda}"
    f"_{mae_loss_mode}_scalemax-{type_train}_gw{gate_window}_gs{gate_stability}"
    f"_cf{cov_floor}_ns{noise_scale}_nstd{noise_std}_{_noise_tag}_conra{noise_ratio}"
    f"__cat_{cat_noise_type}_catra{cat_noise_ratio}{_gatefix_tag}"
)
os.makedirs(results_dir, exist_ok=True)


def save_threshold(threshold, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Threshold saved to {path}")


# =============================================================================
# Training
# =============================================================================

def train_mae01scale_max(epochs):
    n_test      = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    y_train     = y_full.iloc[:-n_test].reset_index(drop=True)

    normal_mask           = y_train == 0
    X_train_normal_full   = X_train_raw.loc[normal_mask].copy()
    y_train_normal_full   = y_train.loc[normal_mask].copy()

    sampled_idx  = X_train_normal_full.sample(frac=percentage, random_state=seed).index
    X_normal_sub = X_train_normal_full.loc[sampled_idx].reset_index(drop=True)
    y_normal_sub = y_train_normal_full.loc[sampled_idx].reset_index(drop=True)

    X_train_normal, X_val_raw, y_train_normal, y_val = train_test_split(
        X_normal_sub, y_normal_sub, test_size=0.2, random_state=seed
    )
    X_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    X_test_raw.columns = X_test_raw.columns.map(str)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_train_normal)

    if dataset == "nb_iot":
        # All 115 nb_iot features are pre-extracted flow statistics (weights,
        # means, variances, ...) -- there is no genuinely categorical column.
        # detect_from_dataframe() sometimes mis-flags a subset of them as
        # categorical when a low-cardinality sample is drawn (varies by seed/
        # n_max_samples), which one-hot-blows the transformed dimensionality
        # up to ~1000+. Force every column to numerical unconditionally so
        # the fix does not depend on which columns happened to be flagged.
        cols_before = metadata.to_dict()["columns"]
        n_cat = sum(1 for v in cols_before.values() if v.get("sdtype") != "numerical")
        metadata.update_columns(X_train_normal.columns.tolist(), sdtype="numerical")
        print(f"[nb_iot] Forced all {len(X_train_normal.columns)} columns to "
              f"sdtype='numerical' ({n_cat} were auto-detected as categorical).")

    meta_file = os.path.join(results_dir, f"{dataset}_mae01scale_max_metadata.json")
    if os.path.exists(meta_file):
        os.remove(meta_file)
    metadata.save_to_json(meta_file)

    synthesizer = MAE01ScaleMaxSynthesizer(
        metadata=metadata,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        cuda=True,
        epochs=epochs,
        batch_size=batch_size,
        lamda=lamda,
        margin=margin,
        mae_loss_mode=mae_loss_mode,
        mae_lr=mae_lr,
        noise_std=noise_std,
        noise_ratio=noise_ratio,
        noise_type=noise_type,
        noise_r=noise_r,
        noise_k=noise_k,
        cat_noise_type=cat_noise_type,
        cat_noise_ratio=cat_noise_ratio,
        type_train=type_train,
        mae_start_ratio=0.0,
        patience=patience,
        dataset=dataset,
        path=results_dir,
        noise_scale=noise_scale,
        gate_stability=gate_stability,
        gate_window=gate_window,
        max_clusters=max_clusters,
        snapshot_stride=snapshot_stride,
        cov_floor=cov_floor,
        eps_zero=eps_zero,
        tol=tol,
        gate_fixed_epoch=gate_fixed_epoch,
    )

    X_train_proc = synthesizer.preprocess(X_train_normal)
    X_val  = synthesizer._data_processor.transform(X_val_raw)
    X_test = synthesizer._data_processor.transform(X_test_raw)

    X_test.to_csv(os.path.join(results_dir, f"{dataset}_X_test.csv"), index=False)
    y_test.to_frame(name="label").to_csv(
        os.path.join(results_dir, f"{dataset}_y_test.csv"), index=False
    )
    X_val.to_csv(os.path.join(results_dir, f"{dataset}_X_val_normal.csv"), index=False)
    y_val.to_frame(name="label").to_csv(
        os.path.join(results_dir, f"{dataset}_y_val_normal.csv"), index=False
    )

    print(f"Train normal: {len(X_train_proc)}  Val: {len(X_val)}  Test: {len(X_test)}")

    synthesizer._fit(X_train_proc, X_train_proc, X_val)

    thresholds = synthesizer.get_threshold(X_val)
    print("Thresholds:", thresholds)
    save_threshold(thresholds, results_dir, f"{dataset}_mae01scale_max_thresholds.pkl")

    model_path = os.path.join(results_dir, f"MAE01ScaleMax_{dataset}_{epochs}e.pkl")
    synthesizer.save(model_path)

    # ── Plots ─────────────────────────────────────────────────────────────
    loss_df  = synthesizer._model.get_loss_values()
    loss_csv = os.path.join(results_dir, f"mae01scale_max_loss_{epochs}e.csv")
    loss_df.to_csv(loss_csv, index=False)

    _gate_epoch = synthesizer._model._ctgan_gate_epoch

    def _vline(ax, epoch, label_prefix):
        if epoch is not None:
            ax.axvline(epoch, color="orange", linestyle="--",
                       label=f"{label_prefix} (epoch {epoch})")

    # CTGAN loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_df["Epoch"], loss_df["Generator Loss"],     label="Generator Loss")
    ax.plot(loss_df["Epoch"], loss_df["Discriminator Loss"], label="Discriminator Loss")
    ax.plot(loss_df["Epoch"], loss_df["EMA Discriminator Loss"],
            label="EMA Disc Loss", linestyle="--", alpha=0.7)
    _vline(ax, _gate_epoch, "Gate open")
    ax.set(xlabel="Epoch", ylabel="Loss",
           title=f"MAE01ScaleMax — CTGAN loss\n{dataset}")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"loss_ctgan_{dataset}_{epochs}e.png"))
    plt.close(fig)

    # M01AE loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_df["Epoch"], loss_df["MAE Loss"],            label="Total loss")
    ax.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction loss")
    ax.plot(loss_df["Epoch"], lamda * loss_df["Latent Loss"], label="Margin loss")
    _vline(ax, _gate_epoch, "Gate open")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"loss_mae_{dataset}_{epochs}e.png"))
    plt.close(fig)

    # Validation loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_df["Epoch"], loss_df["Validation Loss"], label="Val loss")
    _vline(ax, _gate_epoch, "Gate open")
    ax.set(xlabel="Epoch", ylabel="Loss",
           title=f"MAE01ScaleMax — Validation loss\n{dataset}")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"loss_val_{dataset}_{epochs}e.png"))
    plt.close(fig)

    # Z-norm scaling plot
    if "Current Noise Std" in loss_df.columns and "Z Normal Mean" in loss_df.columns:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()

        ax1.plot(loss_df["Epoch"], loss_df["Current Noise Std"],
                 color="tab:blue", label=f"noise_std (scale={noise_scale})")
        ax1.axhline(noise_std, color="tab:blue", linestyle=":", alpha=0.5,
                    label=f"init noise_std={noise_std}")

        ax2.plot(loss_df["Epoch"], loss_df["Z Normal Mean"],
                 color="tab:green", alpha=0.8, label="mean ||z_normal||")
        ax2.plot(loss_df["Epoch"], loss_df["Z Pseudo Mean"],
                 color="tab:red", alpha=0.5, linestyle="--", label="mean ||z_pseudo||")

        _vline(ax1, _gate_epoch, "Gate")
        ax1.set(xlabel="Epoch", ylabel="noise_std",
                title=f"MAE01ScaleMax — Z-norm self-scaling\n{dataset}")
        ax2.set_ylabel("mean ||z||")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"scale_noise_{dataset}_{epochs}e.png"))
        plt.close(fig)

    print("Training complete.")
    return synthesizer, thresholds, X_test


# =============================================================================
# Evaluation
# =============================================================================

def predict_and_evaluate(scores, y_true, threshold, save_dir, filename, score_mode="dist"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    thr_val = (float(threshold[0]) + 3.0 * float(threshold[1])) \
        if score_mode == "std" else float(threshold)

    y_true_bin = (np.asarray(y_true) != 0).astype(int)
    y_pred     = (np.asarray(scores) > thr_val).astype(int)

    auc       = roc_auc_score(y_true_bin, scores) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
    acc       = accuracy_score(y_true_bin, y_pred) * 100
    f1        = f1_score(y_true_bin, y_pred, zero_division=0) * 100
    precision = precision_score(y_true_bin, y_pred, zero_division=0) * 100
    recall    = recall_score(y_true_bin, y_pred, zero_division=0) * 100
    mcc       = matthews_corrcoef(y_true_bin, y_pred) * 100
    cm        = confusion_matrix(y_true_bin, y_pred)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Threshold", f"{thr_val:.6f}"])
        w.writerow(["AUC",       f"{auc:.6f}"])
        w.writerow(["F1",        f"{f1:.6f}"])
        w.writerow(["MCC",       f"{mcc:.6f}"])
        w.writerow(["Precision", f"{precision:.6f}"])
        w.writerow(["Recall",    f"{recall:.6f}"])
        w.writerow(["ACC",       f"{acc:.6f}"])
        w.writerow([])
        w.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
        w.writerow(["True 0", cm[0, 0], cm[0, 1]])
        w.writerow(["True 1", cm[1, 0], cm[1, 1]])

    print(
        f"  [{filename}] AUC={auc:.2f}% F1={f1:.2f}% MCC={mcc:.2f}% "
        f"Prec={precision:.2f}% Rec={recall:.2f}% thr={thr_val:.6f}"
    )
    return {"AUC": auc, "F1": f1, "MCC": mcc, "Precision": precision,
            "Recall": recall, "ACC": acc}


def run_evaluation(synthesizer, X_test, y_test, thresholds, save_dir):
    re_errors, z_norm, total_scores, y_true = synthesizer.get_score(X_test, y_test)

    predict_and_evaluate(re_errors,    y_true, thresholds["threshold_re"],
                         save_dir, "test_mode_re.csv",       score_mode="std")
    predict_and_evaluate(z_norm,       y_true, thresholds["threshold_z"],
                         save_dir, "test_mode_latent.csv",   score_mode="std")
    predict_and_evaluate(z_norm,       y_true, thresholds["threshold_95_latent"],
                         save_dir, "test_mode_latent95.csv", score_mode="dist")
    predict_and_evaluate(total_scores, y_true, thresholds["threshold_95_total"],
                         save_dir, "test_mode_total95.csv",  score_mode="dist")


def post_train_checkpoint_evaluation(results_dir, dataset, val_type):
    """Evaluate one best checkpoint; results saved to checkpoints/best_<val_type>/."""
    if val_type not in {"loss", "auc"}:
        raise ValueError("val_type must be 'loss' or 'auc'.")
    suffix = "" if val_type == "loss" else f"_{val_type}"

    print(f"\n--- Post-train eval: best_{val_type} checkpoint ---")
    x_test_path    = os.path.join(results_dir, f"{dataset}_X_test.csv")
    y_test_path    = os.path.join(results_dir, f"{dataset}_y_test.csv")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    model_path     = os.path.join(checkpoint_dir, f"MAE_{dataset}_best{suffix}.pkl")
    thr_path       = os.path.join(checkpoint_dir, f"MAE_{dataset}_thresholds_best{suffix}.pkl")
    save_dir       = os.path.join(checkpoint_dir, f"best_{val_type}")

    for p in [x_test_path, y_test_path, model_path, thr_path]:
        if not os.path.exists(p):
            print(f"[WARN] Missing: {p} -- skip.")
            return

    X_test_csv = pd.read_csv(x_test_path)
    y_test_csv = pd.read_csv(y_test_path)["label"]

    with open(model_path, "rb") as f:
        best_synth = cloudpickle.load(f)
    with open(thr_path, "rb") as f:
        thr_re, thr_z, thr_95, thr_95_total = cloudpickle.load(f)

    best_thresholds = {
        "threshold_re":        thr_re,
        "threshold_z":         thr_z,
        "threshold_95_latent": thr_95,
        "threshold_95_total":  thr_95_total,
    }
    run_evaluation(best_synth, X_test_csv, y_test_csv, best_thresholds, save_dir)
    print(f"Checkpoint eval done -> {save_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=== MAE01ScaleMax Training ===")
    synthesizer, thresholds, X_test = train_mae01scale_max(epochs=epochs)
    run_evaluation(synthesizer, X_test, y_test, thresholds, results_dir)
    post_train_checkpoint_evaluation(results_dir, dataset, "loss")
    post_train_checkpoint_evaluation(results_dir, dataset, "auc")
    print("=== Done ===")
