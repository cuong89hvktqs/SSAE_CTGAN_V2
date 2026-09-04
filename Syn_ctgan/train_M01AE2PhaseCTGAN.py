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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

import ctgan
import sdv
from sdv.metadata import SingleTableMetadata
from sdv.single_table.m01ae_2phase_ctgan import M01AE2PhaseCTGANSynthesizer
from process_data import (
    load_data_unsw,
    load_data_cic_ids,
    load_data_nsl_kdd,
    load_data_cic_iot,
    load_data_cidds_001,
    load_data_wsn_ds,
    load_data_bot_iot,
    load_data_ton_iot,
)

print(sdv.__file__)
print(ctgan.__file__)

# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Train M01AE2PhaseCTGAN (Phase1=CTGAN, Phase2=M01AE augmented normal+anomaly)"
)

# ── Dataset ───────────────────────────────────────────────────────────────────
parser.add_argument("--d",   type=str,   default="unsw",  help="Dataset name")
parser.add_argument("--per", type=float, default=1.0,     help="Fraction of normal training data (0<per<=1)")
parser.add_argument("--seed",type=int,   default=42,      help="Random seed")
parser.add_argument("--time",type=int,   default=1,       help="Run index (used for output folder naming)")
parser.add_argument("--n_max_samples", type=int, default=500_000,
                    help="Max rows for large datasets (bot_iot, ton_iot)")

# ── Two-phase epochs ──────────────────────────────────────────────────────────
parser.add_argument("--ctgan_epochs", type=int, default=150,
                    help="Phase 1: CTGAN training epochs")
parser.add_argument("--mae_epochs",   type=int, default=100,
                    help="Phase 2: M01AE training epochs (upper bound; early stopping may cut short)")
parser.add_argument("--ctgan_sample_ratio", type=float, default=1.0,
                    help="Size of CTGAN pool relative to training data (1.0 = same size)")

# ── CTGAN architecture ────────────────────────────────────────────────────────
parser.add_argument("--b",   type=int,   default=128,     help="Batch size")

# ── M01AE ─────────────────────────────────────────────────────────────────────
parser.add_argument("--lae",  type=float, default=0.1,    help="Lambda — latent loss weight")
parser.add_argument("--mae",  type=float, default=1.0,    help="Margin for two-sided loss")
parser.add_argument("--mae_loss_mode", type=str, choices=["origin", "renormal"],
                    default="renormal",
                    help="RE loss mode: 'renormal'=normal samples only (recommended)")
parser.add_argument("--mae_lr", type=float, default=1e-3, help="M01AE learning rate")
parser.add_argument("--patience", type=int, default=50,
                    help="Early stopping patience for Phase 2 (M01AE val loss)")

# ── Noise (continuous) ────────────────────────────────────────────────────────
parser.add_argument("--noise_type", type=str, default="global",
                    choices=["none", "global", "feature", "feature_outlier"],
                    help=(
                        "Continuous noise: "
                        "'global'=N(0,noise_std^2), "
                        "'feature'=per-feature scaled, "
                        "'feature_outlier'=mu+noise_r*noise_k*sigma (same as M01AE standalone)"
                    ))
parser.add_argument("--noise_std",   type=float, default=0.1,
                    help="Gaussian noise std ('global' / 'feature')")
parser.add_argument("--noise_ratio", type=float, default=1.0,
                    help="Fraction of continuous features perturbed per sample")
parser.add_argument("--noise_r",     type=float, default=1.0,
                    help="Outlier sign multiplier for 'feature_outlier': mu + noise_r*noise_k*sigma")
parser.add_argument("--noise_k",     type=float, default=3.0,
                    help="Outlier scale for 'feature_outlier': mu + noise_r*noise_k*sigma")

# ── Noise (categorical) ───────────────────────────────────────────────────────
parser.add_argument("--cat_noise_type", type=str, default="none",
                    choices=["none", "shuffle", "flip", "combined"],
                    help="Categorical feature noise type")
parser.add_argument("--cat_noise_ratio", type=float, default=1.0,
                    help="Fraction of one-hot spans perturbed per sample")

args = parser.parse_args()

# =============================================================================
# Validation
# =============================================================================

if not 0.0 < args.per <= 1.0:
    parser.error("--per must be in (0, 1].")
if args.ctgan_epochs < 1:
    parser.error("--ctgan_epochs must be >= 1.")
if args.mae_epochs < 1:
    parser.error("--mae_epochs must be >= 1.")
if args.ctgan_sample_ratio <= 0:
    parser.error("--ctgan_sample_ratio must be > 0.")
if not 0.0 < args.noise_ratio <= 1.0:
    parser.error("--noise_ratio must be in (0, 1].")
if not 0.0 < args.cat_noise_ratio <= 1.0:
    parser.error("--cat_noise_ratio must be in (0, 1].")
if args.patience < 1:
    parser.error("--patience must be >= 1.")

# =============================================================================
# Config
# =============================================================================

dataset_raw = args.d.strip().lower().replace("-", "_")
if dataset_raw == "cidds001":
    dataset_raw = "cidds_001"
dataset = dataset_raw

ctgan_epochs       = args.ctgan_epochs
mae_epochs         = args.mae_epochs
ctgan_sample_ratio = args.ctgan_sample_ratio
batch_size         = args.b
lamda              = args.lae
margin             = args.mae
mae_loss_mode      = args.mae_loss_mode
mae_lr             = args.mae_lr
patience           = args.patience
noise_type         = args.noise_type
noise_std          = args.noise_std
noise_ratio        = args.noise_ratio
noise_r            = args.noise_r
noise_k            = args.noise_k
cat_noise_type     = args.cat_noise_type
cat_noise_ratio    = args.cat_noise_ratio
percentage         = args.per
seed               = args.seed
n_max_samples      = args.n_max_samples
time_tag           = f"{int(args.time):02d}"

_noise_tag = noise_type
if noise_type in {"feature", "feature_outlier"}:
    _noise_tag = f"{noise_type}_r{noise_r}_k{noise_k}"

run_folder = f"m01ae_2phase_s_{seed}_t_{time_tag}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(
    BASE_DIR,
    f"res/{run_folder}/per_{percentage}/"
    f"{dataset}_{ctgan_epochs}c_{mae_epochs}m"
    f"_ratio{ctgan_sample_ratio}"
    f"_m{margin}_l{lamda}_{mae_loss_mode}"
    f"_ns{noise_std}_{_noise_tag}_conra{noise_ratio}"
    f"_cat{cat_noise_type}_cr{cat_noise_ratio}"
)
os.makedirs(results_dir, exist_ok=True)


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
    f"Dataset={dataset}  ctgan_epochs={ctgan_epochs}  mae_epochs={mae_epochs}"
    f"  ctgan_sample_ratio={ctgan_sample_ratio}  batch={batch_size}"
    f"  lambda={lamda}  margin={margin}  mae_loss_mode={mae_loss_mode}"
    f"  mae_lr={mae_lr}  patience={patience}"
    f"  noise_type={noise_type}  noise_std={noise_std}"
    f"  noise_r={noise_r}  noise_k={noise_k}"
    f"  noise_ratio={noise_ratio}"
    f"  cat_noise_type={cat_noise_type}  cat_noise_ratio={cat_noise_ratio}"
    f"  per={percentage}  seed={seed}  time={time_tag}"
)

# =============================================================================
# Dataset loaders
# =============================================================================

DATASET_LOADERS = {
    "unsw":      load_data_unsw,
    "cic_ids":   load_data_cic_ids,
    "nsl_kdd":   load_data_nsl_kdd,
    "cic_iot":   load_data_cic_iot,
    "cidds_001": lambda: load_data_cidds_001(seed=seed, test_size=0.4),
    "wsn_ds":    lambda: load_data_wsn_ds(seed=seed, test_size=0.2),
    "bot_iot":   lambda: load_data_bot_iot(seed=seed, n_max_samples=n_max_samples, test_size=0.2),
    "ton_iot":   lambda: load_data_ton_iot(seed=seed, n_max_samples=n_max_samples, test_size=0.4),
}
if dataset not in DATASET_LOADERS:
    raise ValueError(
        f"Unsupported dataset '{args.d}'. Supported: {', '.join(DATASET_LOADERS)}"
    )

df_full, y_full, df_test, y_test = DATASET_LOADERS[dataset]()
print(f"Full shape: {df_full.shape}   Test shape: {df_test.shape}")

# =============================================================================
# Helpers
# =============================================================================

def save_obj(obj, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


def predict_and_evaluate(scores, y_true, threshold, save_dir, filename,
                         score_mode="dist"):
    """Evaluate and write CSV.

    score_mode:
      'dist'  — threshold is a scalar; flag if score > threshold
      'std'   — threshold is (mu, sigma); flag if score > mu + 3*sigma
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if score_mode == "std":
        mu, sigma = threshold
        thr_val = mu + 3.0 * sigma
    else:
        thr_val = float(threshold)

    y_true_bin = (np.asarray(y_true) != 0).astype(int)
    y_pred     = (np.asarray(scores) > thr_val).astype(int)

    auc  = roc_auc_score(y_true_bin, scores) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
    acc  = accuracy_score(y_true_bin, y_pred) * 100
    f1   = f1_score(y_true_bin, y_pred, zero_division=0) * 100
    prec = precision_score(y_true_bin, y_pred, zero_division=0) * 100
    rec  = recall_score(y_true_bin, y_pred, zero_division=0) * 100
    mcc  = matthews_corrcoef(y_true_bin, y_pred) * 100
    cm   = confusion_matrix(y_true_bin, y_pred)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Threshold", f"{thr_val:.6f}"])
        w.writerow(["AUC",       f"{auc:.6f}"])
        w.writerow(["F1",        f"{f1:.6f}"])
        w.writerow(["MCC",       f"{mcc:.6f}"])
        w.writerow(["Precision", f"{prec:.6f}"])
        w.writerow(["Recall",    f"{rec:.6f}"])
        w.writerow(["ACC",       f"{acc:.6f}"])
        w.writerow([])
        w.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
        w.writerow(["True 0", cm[0, 0], cm[0, 1]])
        w.writerow(["True 1", cm[1, 0], cm[1, 1]])

    print(
        f"  [{filename}]  AUC={auc:.2f}%  F1={f1:.2f}%  MCC={mcc:.2f}%"
        f"  Prec={prec:.2f}%  Rec={rec:.2f}%  thr={thr_val:.6f}"
    )
    return {"AUC": auc, "F1": f1, "MCC": mcc, "Precision": prec, "Recall": rec, "ACC": acc}


def run_evaluation(synthesizer, X_test, y_test, thresholds, save_dir):
    re_errors, z_norm, total_scores, y_true = synthesizer.get_score(X_test, y_test)

    predict_and_evaluate(re_errors, y_true, thresholds["threshold_re"],
                         save_dir, "test_mode_re.csv", score_mode="std")
    predict_and_evaluate(z_norm, y_true, thresholds["threshold_z"],
                         save_dir, "test_mode_latent.csv", score_mode="std")
    predict_and_evaluate(z_norm, y_true, thresholds["threshold_95_latent"],
                         save_dir, "test_mode_latent95.csv", score_mode="dist")
    predict_and_evaluate(total_scores, y_true, thresholds["threshold_95_total"],
                         save_dir, "test_mode_total95.csv", score_mode="dist")

# =============================================================================
# Training
# =============================================================================

def train(results_dir):
    n_test    = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)

    # Keep only normal for training
    normal_mask = y_train == 0
    X_normal_full = X_train_raw.loc[normal_mask].copy()
    y_normal_full = y_train.loc[normal_mask].copy()

    sampled_idx = X_normal_full.sample(frac=percentage, random_state=seed).index
    X_normal_sub = X_normal_full.loc[sampled_idx].reset_index(drop=True)
    y_normal_sub = y_normal_full.loc[sampled_idx].reset_index(drop=True)

    X_train_normal, X_val_raw, y_train_normal, y_val = train_test_split(
        X_normal_sub, y_normal_sub, test_size=0.2, random_state=seed
    )
    X_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    X_test_raw.columns = X_test_raw.columns.map(str)

    print(
        f"Train normal: {len(X_train_normal)}   "
        f"Val normal: {len(X_val_raw)}   "
        f"Test: {len(X_test_raw)}"
    )

    # ── Metadata ──────────────────────────────────────────────────────────
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_train_normal)
    meta_file = os.path.join(results_dir, f"{dataset}_metadata.json")
    if os.path.exists(meta_file):
        os.remove(meta_file)
    metadata.save_to_json(meta_file)

    # ── Synthesizer ───────────────────────────────────────────────────────
    synthesizer = M01AE2PhaseCTGANSynthesizer(
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
        batch_size=batch_size,
        # Two-phase
        ctgan_epochs=ctgan_epochs,
        mae_epochs=mae_epochs,
        ctgan_sample_ratio=ctgan_sample_ratio,
        # M01AE
        lamda=lamda,
        margin=margin,
        mae_loss_mode=mae_loss_mode,
        mae_lr=mae_lr,
        patience=patience,
        # Noise
        noise_type=noise_type,
        noise_std=noise_std,
        noise_ratio=noise_ratio,
        noise_r=noise_r,
        noise_k=noise_k,
        cat_noise_type=cat_noise_type,
        cat_noise_ratio=cat_noise_ratio,
        # Misc
        dataset=dataset,
        path=results_dir,
    )

    # ── Preprocess ────────────────────────────────────────────────────────
    X_train_proc = synthesizer.preprocess(X_train_normal)
    X_val        = synthesizer._data_processor.transform(X_val_raw)
    X_test       = synthesizer._data_processor.transform(X_test_raw)

    # Save splits for post-eval
    X_test.to_csv(os.path.join(results_dir, f"{dataset}_X_test.csv"), index=False)
    y_test.to_frame(name="label").to_csv(
        os.path.join(results_dir, f"{dataset}_y_test.csv"), index=False
    )
    X_val.to_csv(os.path.join(results_dir, f"{dataset}_X_val_normal.csv"), index=False)
    y_val.to_frame(name="label").to_csv(
        os.path.join(results_dir, f"{dataset}_y_val_normal.csv"), index=False
    )

    # ── Fit ───────────────────────────────────────────────────────────────
    synthesizer._fit(X_train_proc, X_train_proc, X_val)

    # ── Thresholds ────────────────────────────────────────────────────────
    thresholds = synthesizer.get_threshold(X_val)
    print("Thresholds:", {k: (round(v[0], 6), round(v[1], 6)) if isinstance(v, tuple)
                          else round(v, 6) for k, v in thresholds.items()})
    save_obj(thresholds, results_dir, f"{dataset}_thresholds.pkl")

    # ── Save model ────────────────────────────────────────────────────────
    model_path = os.path.join(
        results_dir,
        f"M01AE2Phase_{dataset}_{ctgan_epochs}c_{mae_epochs}m.pkl"
    )
    synthesizer.save(model_path)

    # ── Loss plots ────────────────────────────────────────────────────────
    ctgan_df, mae_df = synthesizer.get_loss_values()

    # Save CSVs
    ctgan_df.to_csv(os.path.join(results_dir, "loss_ctgan_phase.csv"), index=False)
    mae_df.to_csv(os.path.join(results_dir, "loss_mae_phase.csv"), index=False)

    # Phase 1: CTGAN loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ctgan_df["Epoch"], ctgan_df["Generator Loss"], label="Generator Loss")
    ax.plot(ctgan_df["Epoch"], ctgan_df["Discriminator Loss"], label="Discriminator Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Phase 1 — CTGAN loss\n{dataset}  ctgan_epochs={ctgan_epochs}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "loss_phase1_ctgan.png"))
    plt.close(fig)

    # Phase 2: MAE training loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mae_df["Epoch"], mae_df["MAE Loss"], label="MAE total")
    ax.plot(mae_df["Epoch"], mae_df["Reconstruction Loss"], label="Reconstruction")
    ax.plot(mae_df["Epoch"], lamda * mae_df["Latent Loss"], label="lambda*Latent")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Phase 2 — M01AE training loss\n{dataset}  mae_epochs={mae_epochs}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "loss_phase2_mae_train.png"))
    plt.close(fig)

    # Phase 2: MAE validation loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mae_df["Epoch"], mae_df["Validation Loss"], label="Val Loss", color="steelblue")
    ax.plot(mae_df["Epoch"], mae_df["Validation RE Loss"], label="Val RE", color="orange",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Phase 2 — M01AE validation loss\n{dataset}  patience={patience}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "loss_phase2_mae_val.png"))
    plt.close(fig)

    print(f"Loss plots saved to {results_dir}")
    return synthesizer, thresholds, X_test


def post_train_checkpoint_evaluation(results_dir, dataset):
    """Evaluate the best checkpoint saved by early stopping."""
    x_test_path = os.path.join(results_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(results_dir, f"{dataset}_y_test.csv")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    best_model_path = os.path.join(checkpoint_dir, f"MAE_{dataset}_best.pkl")
    best_thr_path   = os.path.join(checkpoint_dir, f"MAE_{dataset}_thresholds_best.pkl")

    for p in [x_test_path, y_test_path, best_model_path, best_thr_path]:
        if not os.path.exists(p):
            print(f"[WARN] Missing: {p} — skip checkpoint eval.")
            return

    X_test_csv = pd.read_csv(x_test_path)
    y_test_csv = pd.read_csv(y_test_path)["label"]
    best_synth = M01AE2PhaseCTGANSynthesizer.load(best_model_path)
    with open(best_thr_path, "rb") as f:
        thr_re, thr_z, thr_95, thr_95_total = pickle.load(f)

    best_thresholds = {
        "threshold_re":        thr_re,
        "threshold_z":         thr_z,
        "threshold_95_latent": thr_95,
        "threshold_95_total":  thr_95_total,
    }
    print("=== Checkpoint (best val loss) evaluation ===")
    run_evaluation(best_synth, X_test_csv, y_test_csv, best_thresholds, checkpoint_dir)
    print("Checkpoint evaluation done.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=== M01AE2PhaseCTGAN Training ===")
    synthesizer, thresholds, X_test = train(results_dir)

    print("=== Final model evaluation ===")
    run_evaluation(synthesizer, X_test, y_test, thresholds, results_dir)

    post_train_checkpoint_evaluation(results_dir, dataset)
    print("=== Done ===")

# =============================================================================
# GIAI THICH THAM SO
# =============================================================================
#
# Two-phase design:
#   Phase 1 (ctgan_epochs): CTGAN training on normal data (standard).
#   Phase 2 (mae_epochs):
#     Pool = ctgan_sample_ratio x N CTGAN samples (generated ONCE, fixed).
#     Each step:
#       Normal  = real_normal[:B/2]      + ctgan_pool[:B/2]
#       Anomaly = _add_noise(real[:B/2]) + _add_noise(ctgan[:B/2])
#     Two-sided margin loss: normal->||z||->0, anomaly->||z||->margin.
#     Early stopping on validation loss (patience epochs).
#
# Key params:
#   --ctgan_epochs     Phase 1 length (default 150).
#   --mae_epochs       Phase 2 upper bound (default 100, may stop earlier).
#   --ctgan_sample_ratio  Pool size / N_train (default 1.0 = same size).
#   --patience         Early stopping patience (default 50).
#
# Noise params (same as M01AE standalone):
#   --noise_type feature_outlier  mu + noise_r*noise_k*sigma  [recommended]
#   --noise_r 5  --noise_k 0.1
#   --noise_ratio 1.0   fraction of continuous features perturbed
#   --cat_noise_type combined/shuffle/flip/none
#   --cat_noise_ratio   fraction of one-hot spans perturbed
#
# Output per run:
#   results/<run_folder>/per_<per>/2phase_<dataset>_.../
#   |-- loss_phase1_ctgan.png       Generator / Discriminator loss
#   |-- loss_phase2_mae_train.png   MAE total / RE / Latent
#   |-- loss_phase2_mae_val.png     Validation loss
#   |-- loss_ctgan_phase.csv
#   |-- loss_mae_phase.csv
#   |-- test_mode_re.csv            RE score  (mu+3sigma threshold)
#   |-- test_mode_latent.csv        ||z|| score (mu+3sigma)
#   |-- test_mode_latent95.csv      ||z|| score (95th pct)
#   |-- test_mode_total95.csv       total score (95th pct)
#   |-- M01AE2Phase_<dataset>_*.pkl
#   |-- <dataset>_thresholds.pkl
#   `-- checkpoints/
#       |-- MAE_<dataset>_best.pkl
#       `-- MAE_<dataset>_thresholds_best.pkl
#
# Example commands (Ubuntu):
#
# UNSW — best from M01AE standalone (noise_ratio=1.0, combined, cat_ratio=0.1):
#   python Syn_ctgan/train_M01AE2PhaseCTGAN.py \
#     --d unsw --ctgan_epochs 150 --mae_epochs 100 --b 64 \
#     --noise_type feature_outlier --noise_r 5 --noise_k 0.1 \
#     --noise_ratio 1.0 --cat_noise_type combined --cat_noise_ratio 0.1 \
#     --lae 0.1 --mae 1.0 --mae_loss_mode renormal \
#     --ctgan_sample_ratio 1.0 --patience 50 --seed 42 --time 1
#
# CIC-IDS — best (noise_ratio=0.2, shuffle, cat_ratio=1.0):
#   python Syn_ctgan/train_M01AE2PhaseCTGAN.py \
#     --d cic_ids --ctgan_epochs 150 --mae_epochs 100 --b 64 \
#     --noise_type feature_outlier --noise_r 5 --noise_k 0.1 \
#     --noise_ratio 0.2 --cat_noise_type shuffle --cat_noise_ratio 1.0 \
#     --lae 0.1 --mae 1.0 --mae_loss_mode renormal \
#     --ctgan_sample_ratio 1.0 --patience 50 --seed 42 --time 1
