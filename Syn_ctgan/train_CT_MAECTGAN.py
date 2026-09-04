import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table.ctmaectgan import CTMAECTGANSynthesizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
)
from sdv.errors import InvalidDataTypeError
import matplotlib.pyplot as plt
import pickle
import torch
import csv
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
from sklearn.model_selection import train_test_split
import sdv
print(sdv.__file__)
import random
import ctgan
print(ctgan.__file__)

import argparse

parser = argparse.ArgumentParser(description="Train CT-MAECTGAN (sparse perturbation variant)")

# ── Dataset / training basics ────────────────────────────────────────────────
parser.add_argument('--d',    type=str,   default='unsw',    help='Dataset name')
parser.add_argument('--e',    type=int,   default=200,       help='Epochs')
parser.add_argument('--b',    type=int,   default=128,       help='Batch size')
parser.add_argument('--lae',  type=float, default=0.1,       help='Lambda (AE loss weight)')
parser.add_argument('--mae',  type=float, default=1.0,       help='Margin')
parser.add_argument(
    '--mae_loss_mode', type=str, choices=['origin', 'renormal'], default='renormal',
    help='MAE loss mode: origin=all samples; renormal=normal only reconstruction.',
)
parser.add_argument('--per',  type=float, default=1.0,       help='Fraction of normal data to use')
parser.add_argument(
    '--type_train', type=str,
    choices=["all", "noise", "std", "3std", "std_3std", "noise_std", "noise_3std"],
    default="noise_std",
    help='Pseudo-anomaly composition for MAE training set.',
)

# ── MAE schedule ─────────────────────────────────────────────────────────────
parser.add_argument(
    '--mae_start_ratio', type=float, default=0.25,
    help='Fraction of epochs before MAE pseudo-anomaly training starts (0.0-1.0).',
)

# ── CTGAN noise (dense mode, kept for backward compat) ───────────────────────
parser.add_argument(
    '--ctgan_noise_start_ratio', type=float, default=0.75,
    help='Fraction of epochs before Gaussian noise is added to CTGAN samples (0.0-1.0).',
)
parser.add_argument(
    '--ctgan_noise_std', type=float, default=0.05,
    help='Std of dense Gaussian noise on CTGAN continuous features.',
)
parser.add_argument(
    '--ctgan_noise_feature_ratio', type=float, default=0.5,
    help='Fraction of continuous features perturbed in dense noise mode (0.0-1.0).',
)
parser.add_argument(
    '--ctgan_pseudo_anomaly_mode', type=str,
    choices=['origin', 'feature_outlier'], default='origin',
    help='How CTGAN samples become pseudo-anomalies.',
)
parser.add_argument(
    '--ctgan_feature_outlier_r', type=float, default=1.0,
    help='Outlier coefficient for feature_outlier mode.',
)

# ── Sparse perturbation (new in CT-MAECTGAN) ─────────────────────────────────
parser.add_argument(
    '--ctgan_noise_type', type=str, choices=['dense', 'sparse', 'directional'], default='dense',
    help=(
        'Noise on CTGAN-generated samples. '
        '"dense" = original isotropic noise on ctgan_noise_feature_ratio fraction. '
        '"sparse" = noise on k features per row, rest unchanged.'
    ),
)
parser.add_argument(
    '--real_perturb_mode', type=str, choices=['dense', 'sparse', 'directional'], default='dense',
    help=(
        'Perturbation of real normal data for MAE training set. '
        '"dense"       = original mix (cross-row shuffle) + scale (all features). '
        '               Use for UNSW, CIC-IoT (globally different attacks). '
        '"sparse"      = symmetric noise on k features per row, rest unchanged. '
        '"directional" = sparse noise, always outward from normal mean. '
        '               Best for CIC-IDS near-normal attacks (Slowloris, Bot, FTP-Patator).'
    ),
)
parser.add_argument(
    '--ctgan_sparse_k_ratio', type=float, default=0.1,
    help=(
        'Fraction of continuous features perturbed per row in sparse mode. '
        'k = max(1, ceil(n_cont * ratio)). '
        'Recommended: 0.05 for CIC-IDS (near-normal attacks), '
        '0.10-0.15 for UNSW/CIC-IoT.'
    ),
)
parser.add_argument(
    '--ctgan_sparse_noise_std', type=float, default=0.3,
    help=(
        'Std of noise applied to the k selected features in sparse mode. '
        'Can be larger than dense std because only k features are perturbed.'
    ),
)

# ── Hướng A: Pseudo-anomaly score filtering ───────────────────────────────────
parser.add_argument(
    '--score_filter_ratio', type=float, default=0.0,
    help=(
        '[Hướng A] Fraction of pseudo-anomalies to keep per step (0=disabled). '
        'E.g. 0.5 keeps the 50%% with highest reconstruction error (hardest), '
        'discarding easy samples that look too normal to the current MAE. '
        'Filtering starts after score_filter_start_ratio of MAE training.'
    ),
)
parser.add_argument(
    '--score_filter_start_ratio', type=float, default=0.5,
    help=(
        '[Hướng A] Fraction of MAE-active epochs before filtering kicks in. '
        '0.5 = start filtering halfway through MAE training (default).'
    ),
)

# ── Hướng B: Combined anomaly score ───────────────────────────────────────────
parser.add_argument(
    '--score_alpha', type=float, default=1.0,
    help=(
        '[Hướng B] Weight for reconstruction error in combined anomaly score: '
        'score = alpha*norm(re) + (1-alpha)*norm(latent_norm). '
        '1.0 = pure reconstruction error (default, original behavior). '
        '0.5 = equal combination. '
        '0.0 = pure latent norm.'
    ),
)

# ── Run metadata ─────────────────────────────────────────────────────────────
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--time', type=int, default=1,  help='Run index (e.g. 1 -> time_01)')
parser.add_argument(
    '--n_max_samples', type=int, default=500_000,
    help='Max rows to sample from large datasets (bot_iot, ton_iot). Default 500000.'
)

args = parser.parse_args()

# ── Validation ───────────────────────────────────────────────────────────────
if not 0.0 <= args.mae_start_ratio < 1.0:
    parser.error('--mae_start_ratio must be in [0, 1).')
if not 0.0 <= args.ctgan_noise_start_ratio <= 1.0:
    parser.error('--ctgan_noise_start_ratio must be in [0, 1].')
if args.ctgan_noise_std < 0.0:
    parser.error('--ctgan_noise_std must be >= 0.')
if not 0.0 <= args.ctgan_noise_feature_ratio <= 1.0:
    parser.error('--ctgan_noise_feature_ratio must be in [0, 1].')
if args.ctgan_feature_outlier_r < 0.0:
    parser.error('--ctgan_feature_outlier_r must be >= 0.')
if not 0.0 < args.ctgan_sparse_k_ratio <= 1.0:
    parser.error('--ctgan_sparse_k_ratio must be in (0, 1].')
if args.ctgan_sparse_noise_std < 0.0:
    parser.error('--ctgan_sparse_noise_std must be >= 0.')
if not 0.0 <= args.score_filter_ratio <= 1.0:
    parser.error('--score_filter_ratio must be in [0, 1].')
if not 0.0 <= args.score_filter_start_ratio <= 1.0:
    parser.error('--score_filter_start_ratio must be in [0, 1].')
if not 0.0 <= args.score_alpha <= 1.0:
    parser.error('--score_alpha must be in [0, 1].')
if args.ctgan_noise_type == 'dense' and args.real_perturb_mode == 'dense':
    print("[INFO] Both ctgan_noise_type and real_perturb_mode are 'dense' — "
          "behavior is identical to original MAECTGAN.")
if not 0.0 < args.per <= 1.0:
    parser.error('--per must be in (0, 1].')


def normalize_dataset_name(name):
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "cidds001":
        return "cidds_001"
    return normalized


dataset = normalize_dataset_name(args.d)
epochs = args.e
batch_size = args.b
lamda = args.lae
margin = args.mae
mae_loss_mode = args.mae_loss_mode
percentage = args.per
type_train = args.type_train
mae_start_ratio = args.mae_start_ratio
ctgan_noise_start_ratio = args.ctgan_noise_start_ratio
ctgan_noise_std = args.ctgan_noise_std
ctgan_noise_feature_ratio = args.ctgan_noise_feature_ratio
ctgan_pseudo_anomaly_mode = (
    'feature_outlier'
    if args.ctgan_pseudo_anomaly_mode == 'feature_outliner'
    else args.ctgan_pseudo_anomaly_mode
)
ctgan_feature_outlier_r = args.ctgan_feature_outlier_r
ctgan_noise_type = args.ctgan_noise_type
real_perturb_mode = args.real_perturb_mode
ctgan_sparse_k_ratio = args.ctgan_sparse_k_ratio
ctgan_sparse_noise_std = args.ctgan_sparse_noise_std
score_filter_ratio = args.score_filter_ratio
score_filter_start_ratio = args.score_filter_start_ratio
score_alpha = args.score_alpha
seed = args.seed
n_max_samples = args.n_max_samples
run_time = args.time
time_tag = f"{int(run_time):02d}"
run_folder = f"ctmae_{type_train}_seed_{seed}_time_{time_tag}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed)

print(
    "Dataset:", dataset,
    " Epochs:", epochs,
    " Batch size:", batch_size,
    " Lambda:", lamda,
    " Margin:", margin,
    " MAE loss mode:", mae_loss_mode,
    " Percentage:", percentage,
    " Type train:", type_train,
    " MAE start ratio:", mae_start_ratio,
    " CTGAN noise type:", ctgan_noise_type,
    " Real perturb mode:", real_perturb_mode,
    " Sparse k ratio:", ctgan_sparse_k_ratio,
    " Sparse noise std:", ctgan_sparse_noise_std,
    " Dense noise start ratio:", ctgan_noise_start_ratio,
    " Dense noise std:", ctgan_noise_std,
    " Dense noise feature ratio:", ctgan_noise_feature_ratio,
    " Pseudo anomaly mode:", ctgan_pseudo_anomaly_mode,
    " Feature outlier r:", ctgan_feature_outlier_r,
    " Run time:", time_tag,
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
}
if dataset not in DATASET_LOADERS:
    supported = ", ".join(DATASET_LOADERS)
    raise ValueError(f"Unsupported dataset '{args.d}'. Supported: {supported}")

df_full, y_full, df_test, y_test = DATASET_LOADERS[dataset]()

print("Full shape:", df_full.shape, " Test shape:", df_test.shape)
for label, count in zip(*np.unique(y_full, return_counts=True)):
    print(f"  Train label {label}: {count} samples")
for label, count in zip(*np.unique(y_test, return_counts=True)):
    print(f"  Test  label {label}: {count} samples")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Results directory encodes all key hyperparameters for traceability.
_sparse_tag = f"sparse_k{ctgan_sparse_k_ratio}_std{ctgan_sparse_noise_std}"
_dense_tag  = f"dense_nstart{ctgan_noise_start_ratio}_nfeat{ctgan_noise_feature_ratio}_nstd{ctgan_noise_std}"
_cn_tag = _sparse_tag if ctgan_noise_type in {"sparse", "directional"} else _dense_tag
if ctgan_noise_type == "directional":
    _cn_tag = "dir_" + _sparse_tag
_rp_tag = real_perturb_mode  # "dense" | "sparse" | "directional"

results_dir = os.path.join(
    BASE_DIR,
    f"results/{run_folder}/per_{percentage}/"
    f"ctmae_{dataset}_{epochs}_m_{margin}_l_{lamda}"
    f"_maeloss_{mae_loss_mode}"
    f"_mstart_{mae_start_ratio}"
    f"_cn_{_cn_tag}"
    f"_rp_{_rp_tag}"
    f"_pmode_{ctgan_pseudo_anomaly_mode}"
)
os.makedirs(results_dir, exist_ok=True)


def save_threshold(threshold, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Threshold saved to {path}")


def load_threshold(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_ctmae_ctgan(epochs):
    print("Training CT-MAECTGAN, epochs =", epochs)

    n_test = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)
    y_test_local = y_full.iloc[-n_test:].reset_index(drop=True)

    normal_mask = y_train == 0
    X_train_normal_full = X_train_raw.loc[normal_mask].copy()
    y_train_normal_full = y_train.loc[normal_mask].copy()

    sampled_idx = X_train_normal_full.sample(frac=percentage, random_state=seed).index
    X_normal_sub = X_train_normal_full.loc[sampled_idx].reset_index(drop=True)
    y_normal_sub = y_train_normal_full.loc[sampled_idx].reset_index(drop=True)

    X_train_normal, X_val_raw, y_train_normal, y_val = train_test_split(
        X_normal_sub, y_normal_sub, test_size=0.2, random_state=seed
    )
    X_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    X_test_raw.columns = X_test_raw.columns.map(str)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_train_normal)
    metadata_file = os.path.join(results_dir, f"{dataset}_CTMAE_metadata_epochs_{epochs}.json")
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    metadata.save_to_json(metadata_file)

    synthesizer = CTMAECTGANSynthesizer(
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
        pac=8,
        cuda=True,
        epochs=epochs,
        batch_size=batch_size,
        lamda=lamda,
        margin=margin,
        mae_loss_mode=mae_loss_mode,
        dataset=dataset,
        path=results_dir,
        type_train=type_train,
        mae_start_ratio=mae_start_ratio,
        ctgan_noise_start_ratio=ctgan_noise_start_ratio,
        ctgan_noise_std=ctgan_noise_std,
        ctgan_noise_feature_ratio=ctgan_noise_feature_ratio,
        ctgan_pseudo_anomaly_mode=ctgan_pseudo_anomaly_mode,
        ctgan_feature_outlier_r=ctgan_feature_outlier_r,
        # ── sparse / real perturbation params ─────────────────────────────
        ctgan_noise_type=ctgan_noise_type,
        real_perturb_mode=real_perturb_mode,
        ctgan_sparse_k_ratio=ctgan_sparse_k_ratio,
        ctgan_sparse_noise_std=ctgan_sparse_noise_std,
        # ── Hướng A: pseudo-anomaly score filtering ────────────────────────
        score_filter_ratio=score_filter_ratio,
        score_filter_start_ratio=score_filter_start_ratio,
        # ── Hướng B: combined anomaly score ────────────────────────────────
        score_alpha=score_alpha,
    )

    X_train_normal_proc = synthesizer.preprocess(X_train_normal)
    X_val = synthesizer._data_processor.transform(X_val_raw)
    X_test = synthesizer._data_processor.transform(X_test_raw)

    # Save test set for post-train evaluation
    X_test.to_csv(os.path.join(results_dir, f"{dataset}_X_test.csv"), index=False)
    y_test_local.to_frame(name="label").to_csv(
        os.path.join(results_dir, f"{dataset}_y_test.csv"), index=False
    )
    X_val.to_csv(os.path.join(results_dir, f"{dataset}_X_val_normal.csv"), index=False)
    y_val.to_frame(name="label").to_csv(
        os.path.join(results_dir, f"{dataset}_y_val_normal.csv"), index=False
    )

    print(
        f"Train normal: {len(X_train_normal_proc)}  "
        f"Val normal: {len(X_val)}  "
        f"Test: {len(X_test)}"
    )

    synthesizer._fit(X_train_normal_proc, X_train_normal_proc, X_val)

    # ── Threshold from validation ─────────────────────────────────────────
    avg_loss, threshold_re, threshold_z_0, threshold_95, threshold_95_total = (
        synthesizer.get_threshold_mae(X_val)
    )
    print("Threshold_re:", threshold_re)
    print("Threshold_z:", threshold_z_0)
    print("Threshold_95:", threshold_95)
    print("Threshold_95_total:", threshold_95_total)

    # ── Hướng B: combined score threshold from validation ─────────────────
    threshold_combined = _compute_combined_threshold(synthesizer, X_val, y_val, score_alpha)
    if threshold_combined is not None:
        print(f"Threshold_combined (alpha={score_alpha}): {threshold_combined:.6f}")

    threshold_bundle = (threshold_re, threshold_z_0, threshold_95, threshold_95_total,
                        threshold_combined, score_alpha)
    save_threshold(
        threshold_bundle, save_dir=results_dir,
        name=f"{dataset}_ctmae_threshold_epochs_{epochs}.pkl"
    )

    # ── Save final model ──────────────────────────────────────────────────
    file_model = os.path.join(results_dir, f"CTMAE_{dataset}_epochs_{epochs}.pkl")
    synthesizer.save(file_model)

    # ── Loss curves ───────────────────────────────────────────────────────
    loss_df = synthesizer.get_loss_values()
    log_file = os.path.join(results_dir, f"ctmae_loss_log_epochs_{epochs}.csv")
    loss_df.to_csv(log_file, index=False)
    print(f"Loss log saved to {log_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["Epoch"], loss_df["MAE Loss"], label="MAE total Loss")
    plt.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction Loss")
    plt.plot(loss_df["Epoch"], lamda * loss_df["Latent Loss"], label="lambda*Latent Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CT-MAECTGAN Training Loss\ndataset={dataset} epochs={epochs}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_mae_{dataset}_epochs_{epochs}.png"))
    plt.close()

    val_loss_df = loss_df.dropna(subset=["Validation Loss"])
    if not val_loss_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(val_loss_df["Epoch"], val_loss_df["Validation Loss"], label="Val total")
        if "Validation Reconstruction Loss" in val_loss_df.columns:
            plt.plot(val_loss_df["Epoch"], val_loss_df["Validation Reconstruction Loss"],
                     label="Val Reconstruction")
        if "Validation Latent Loss" in val_loss_df.columns:
            plt.plot(val_loss_df["Epoch"], lamda * val_loss_df["Validation Latent Loss"],
                     label="lambda*Val Latent")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"CT-MAECTGAN Validation Loss\ndataset={dataset} epochs={epochs}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(results_dir, f"loss_ctmae_val_{dataset}_epochs_{epochs}.png")
        )
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["Epoch"], loss_df["Generator Loss"], label="Generator Loss")
    plt.plot(loss_df["Epoch"], loss_df["Discriminator Loss"], label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CT-MAECTGAN CTGAN Loss\ndataset={dataset} epochs={epochs}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_ctgan_{dataset}_epochs_{epochs}.png"))
    plt.close()

    print("Training completed.")
    return synthesizer


def predict_and_evaluate(
    Z_norm_test, y_true, threshold, save_dir, filename, k_std=3.0, mode="std"
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    print(f"Mean z_norm normal:  {Z_norm_test[y_true == 0].mean():.6f}")
    print(f"Mean z_norm anomaly: {Z_norm_test[y_true == 1].mean():.6f}")

    if mode == "dist":
        threshold_test = threshold
    else:
        mu, std = threshold
        threshold_test = mu + k_std * std

    y_pred = (Z_norm_test > threshold_test).astype(int)
    y_true = (y_true != 0).astype(int)

    unique_classes = np.unique(y_true)
    auc = roc_auc_score(y_true, Z_norm_test) * 100 if len(unique_classes) >= 2 else 0.0
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)

    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Threshold", f"{threshold_test:.6f}"])
        writer.writerow(["AUC", f"{auc:.6f}"])
        writer.writerow(["F1", f"{f1:.6f}"])
        writer.writerow(["MCC", f"{mcc:.6f}"])
        writer.writerow(["Precision", f"{precision:.6f}"])
        writer.writerow(["Recall", f"{recall:.6f}"])
        writer.writerow(["ACC", f"{acc:.6f}"])
        writer.writerow([])
        writer.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
        writer.writerow(["True 0", cm[0, 0], cm[0, 1]])
        writer.writerow(["True 1", cm[1, 0], cm[1, 1]])

    print(f"Saved evaluation to {save_path}")
    return {
        "re_errors": Z_norm_test,
        "y_pred": y_pred,
        "metrics": {"AUC": auc, "ACC": acc, "F1": f1,
                    "Precision": precision, "Recall": recall, "MCC": mcc},
        "confusion_matrix": cm,
    }


def _compute_total_loss(re_loss, z_norm):
    return re_loss + lamda * (z_norm ** 2)


def _minmax_norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def _compute_combined_threshold(synthesizer, X_val, y_val, alpha):
    """Hướng B: compute 95th-percentile combined-score threshold from val normal data.

    Returns None when alpha==1.0 (pure re score, no combination needed).
    """
    if alpha >= 1.0:
        return None
    val_re, val_z, _ = synthesizer.get_score_mae(X_val, y_val.values if hasattr(y_val, 'values') else y_val)
    val_combined = alpha * _minmax_norm(val_re) + (1.0 - alpha) * _minmax_norm(val_z)
    return float(np.percentile(val_combined, 95))


def _unpack_thresholds(thresholds):
    """Unpack threshold bundle — supports both old 4-tuple and new 6-tuple format."""
    if len(thresholds) >= 6:
        thr_re, thr_z, thr_95, thr_95_total, thr_combined, t_alpha = thresholds[:6]
    else:
        thr_re, thr_z, thr_95, thr_95_total = thresholds[:4]
        thr_combined, t_alpha = None, 1.0
    return thr_re, thr_z, thr_95, thr_95_total, thr_combined, t_alpha


def _evaluate_model_with_thresholds(synthesizer, X_test, y_test, thresholds, save_dir, prefix):
    thr_re, _thr_z, thr_95, thr_95_total, thr_combined, t_alpha = _unpack_thresholds(thresholds)

    # get_score_mae returns (re_or_combined, z_norm, y_true)
    # When score_alpha < 1.0 the model's get_score_mae already returns combined score
    # as first element; we still need raw re for total_loss, so call super path via
    # a fresh pass with alpha=1.0 when combined is active.
    raw_re, z_norm, y_true = synthesizer.get_score_mae(X_test, y_test)

    # If the model returns combined score as first element (score_alpha < 1.0),
    # raw_re is already the combined score — use it for mode_re evaluation.
    predict_and_evaluate(raw_re, y_true, thr_re,
                         save_dir=save_dir, filename=f"{prefix}_mode_re.csv", mode="re")
    result_95 = predict_and_evaluate(z_norm, y_true, thr_95,
                                     save_dir=save_dir, filename=f"{prefix}_mode_95.csv",
                                     mode="dist")
    total_loss = _compute_total_loss(raw_re, z_norm)
    result_total95 = predict_and_evaluate(total_loss, y_true, thr_95_total,
                                          save_dir=save_dir,
                                          filename=f"{prefix}_mode_total95.csv",
                                          mode="dist")

    # Hướng B: explicit combined evaluation when threshold is available
    result_combined = None
    if thr_combined is not None and t_alpha < 1.0:
        combined = t_alpha * _minmax_norm(raw_re) + (1.0 - t_alpha) * _minmax_norm(z_norm)
        result_combined = predict_and_evaluate(
            combined, y_true, thr_combined,
            save_dir=save_dir,
            filename=f"{prefix}_mode_combined_a{t_alpha:.2f}.csv",
            mode="dist",
        )

    return {"result_95": result_95, "result_total95": result_total95,
            "result_combined": result_combined}


def post_train_checkpoint_evaluation(results_dir, dataset, epochs):
    """Evaluate the validation-selected best checkpoint on held-out test data."""
    del epochs
    print("Post-train evaluation on held-out test set...")
    x_test_path = os.path.join(results_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(results_dir, f"{dataset}_y_test.csv")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    best_model_path = os.path.join(checkpoint_dir, f"MAE_{dataset}_best.pkl")
    best_threshold_path = os.path.join(checkpoint_dir, f"MAE_{dataset}_thresholds_best.pkl")

    for path in [x_test_path, y_test_path, best_model_path, best_threshold_path]:
        if not os.path.exists(path):
            print(f"[WARN] Missing file, skip post-train eval: {path}")
            return

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    best_synth = CTMAECTGANSynthesizer.load(best_model_path)
    with open(best_threshold_path, "rb") as f:
        best_thresholds = pickle.load(f)

    _evaluate_model_with_thresholds(
        synthesizer=best_synth,
        X_test=X_test,
        y_test=y_test,
        thresholds=best_thresholds,
        save_dir=checkpoint_dir,
        prefix=f"test_{dataset}_best",
    )
    print("Post-train evaluation done.")


if __name__ == "__main__":
    print("Starting CT-MAECTGAN training...")
    synthesizer = train_ctmae_ctgan(epochs=epochs)
    post_train_checkpoint_evaluation(
        results_dir=results_dir, dataset=dataset, epochs=epochs
    )
    print("Done.")

# ─────────────────────────────────────────────────────────────────────────────
# Giải thích tham số (CLI arguments reference)
# ─────────────────────────────────────────────────────────────────────────────
#
# ── Dữ liệu / cơ bản ─────────────────────────────────────────────────────────
#
# --d  (dataset)
#   Tên dataset cần train.
#   Giá trị hợp lệ: unsw | cic_ids | cic_iot | nsl_kdd | cidds_001
#   Mặc định: unsw
#
# --e  (epochs)
#   Số epoch huấn luyện.
#   Nhiều epoch hơn giúp CTGAN học phân phối tốt hơn, nhưng tốn thời gian.
#   Khuyến nghị: 100–200.  Mặc định: 200
#
# --b  (batch_size)
#   Kích thước mini-batch cho cả CTGAN và MAE.
#   Mặc định: 128
#
# --per  (percentage)
#   Tỷ lệ dữ liệu normal dùng để train (0 < per ≤ 1).
#   per=0.5 → dùng 50% ngẫu nhiên của tập train normal.
#   Mặc định: 1.0
#
# --seed
#   Random seed để tái tạo kết quả.
#   Mặc định: 42
#
# --time
#   Chỉ số lần chạy (ví dụ: 1 → time_01 trong tên folder kết quả).
#   Dùng để chạy nhiều lần với cùng tham số mà không ghi đè nhau.
#   Mặc định: 1
#
# ── MAE / Autoencoder ────────────────────────────────────────────────────────
#
# --lae  (lambda)
#   Trọng số của AE reconstruction loss trong tổng loss:
#     total_loss = gan_loss + lae * reconstruction_loss + margin_loss
#   Giá trị nhỏ (0.1–0.3) → MAE tập trung vào phân biệt hơn là tái tạo.
#   Mặc định: 0.1
#
# --mae  (margin)
#   Ngưỡng margin trong hàm loss:
#     margin_loss = max(0, margin - ||z_anomaly||)
#   Normal → latent norm ≈ 0;  Anomaly → latent norm ≥ margin.
#   Tăng margin → khoảng cách phân tách lớn hơn, cần data đủ đa dạng.
#   Khuyến nghị: 1.0 cho UNSW/CIC-IoT; 2.0 cho CIC-IDS.  Mặc định: 1.0
#
# --mae_loss_mode  {origin | renormal}
#   Cách tính reconstruction loss:
#   "origin"   → tính trên cả normal và pseudo-anomaly.
#   "renormal" → chỉ tính reconstruction trên normal; pseudo-anomaly chỉ
#                đóng góp vào margin loss. Giúp MAE không học cách tái tạo
#                anomaly, từ đó anomaly score cao hơn.
#   Mặc định: renormal
#
# --mae_start_ratio
#   Tỷ lệ epoch đầu chỉ train CTGAN, chưa train MAE.
#   mae_start_ratio=0.25 và e=200 → 50 epoch đầu train CTGAN thuần.
#   Mục đích: để CTGAN học phân phối tốt trước, pseudo-anomaly mới có chất lượng.
#   Mặc định: 0.25
#
# --type_train  {all | noise | std | 3std | std_3std | noise_std | noise_3std}
#   Thành phần của tập pseudo-anomaly dùng để train MAE:
#   "noise"     → chỉ dùng real normal data bị perturb (an toàn nhất, ít label noise).
#   "std"       → chỉ dùng CTGAN samples nằm trong khoảng 1-std của phân phối.
#   "3std"      → chỉ dùng CTGAN samples nằm trong khoảng 3-std.
#   "noise_std" → kết hợp real perturbed + CTGAN std samples.
#   "noise_3std"→ kết hợp real perturbed + CTGAN 3std samples.
#   "std_3std"  → kết hợp CTGAN std + 3std samples.
#   "all"       → dùng tất cả.
#   Lưu ý: với CIC-IDS, "noise" giảm label noise từ CTGAN samples trông giống normal.
#   Mặc định: noise_std
#
# ── CTGAN noise (chế độ dense — tương thích ngược) ───────────────────────────
#
# --ctgan_noise_start_ratio
#   Tỷ lệ epoch đầu KHÔNG thêm noise vào CTGAN samples.
#   ctgan_noise_start_ratio=0.75 và e=200 → 150 epoch đầu không có noise.
#   Mục đích: CTGAN học ổn định trước, sau đó mới thêm nhiễu để tạo pseudo-anomaly.
#   Mặc định: 0.75
#
# --ctgan_noise_std
#   Std của Gaussian noise thêm vào feature liên tục (chỉ dùng khi ctgan_noise_type=dense).
#   Mặc định: 0.05
#
# --ctgan_noise_feature_ratio
#   Tỷ lệ feature liên tục bị thêm noise trong chế độ dense (0 < ratio ≤ 1).
#   Mặc định: 0.5
#
# --ctgan_pseudo_anomaly_mode  {origin | feature_outlier}
#   Cách CTGAN samples được biến thành pseudo-anomaly:
#   "origin"         → dùng trực tiếp CTGAN samples (sau khi thêm noise nếu có).
#   "feature_outlier"→ áp thêm bội số outlier lên các feature, tạo anomaly cực đoan hơn.
#   Mặc định: origin
#
# --ctgan_feature_outlier_r
#   Hệ số phóng đại cho chế độ feature_outlier.
#   Mặc định: 1.0
#
# ── Sparse / Directional perturbation (mới trong CT-MAECTGAN) ────────────────
#
# --ctgan_noise_type  {dense | sparse | directional}
#   Kiểu nhiễu thêm vào CTGAN-generated samples:
#   "dense"       → nhiễu Gaussian isotropic lên ctgan_noise_feature_ratio% features.
#                   Giống hành vi gốc MAE-CTGAN.
#   "sparse"      → nhiễu Gaussian đối xứng (±) lên k features ngẫu nhiên mỗi dòng,
#                   các feature còn lại giữ nguyên.
#   "directional" → giống sparse nhưng hướng nhiễu luôn đẩy feature RA NGOÀI
#                   trung tâm phân phối normal (không bao giờ đẩy vào trong).
#                   Phù hợp nhất cho CIC-IDS (Slowloris, Bot, FTP-Patator).
#   Mặc định: dense
#
# --real_perturb_mode  {dense | sparse | directional}
#   Kiểu perturbation áp lên real normal data để tạo pseudo-anomaly cho MAE:
#   "dense"       → mix (cross-row feature shuffle) + scale (nhân nhiễu lên TẤT CẢ
#                   features liên tục). Tốt cho UNSW, CIC-IoT (attack khác global).
#   "sparse"      → nhiễu đối xứng lên k features, phần còn lại không đổi.
#   "directional" → nhiễu luôn đẩy ra ngoài trung tâm normal.
#                   Khuyến nghị cho CIC-IDS.
#   Mặc định: dense
#
# --ctgan_sparse_k_ratio
#   Tỷ lệ features liên tục bị perturb mỗi dòng trong chế độ sparse/directional.
#   k = max(1, ceil(n_continuous_features * k_ratio))
#   Giá trị nhỏ → anomaly tập trung vào ít feature → giống attack thực (Slowloris).
#   Khuyến nghị: 0.03–0.05 cho CIC-IDS; 0.10–0.15 cho UNSW/CIC-IoT.
#   Mặc định: 0.1
#
# --ctgan_sparse_noise_std
#   Std của noise trên k features được chọn trong chế độ sparse/directional.
#   Có thể dùng giá trị lớn hơn dense std vì chỉ tác động lên ít feature.
#   Khuyến nghị: 0.3–0.5.  Mặc định: 0.3
#
# ─────────────────────────────────────────────────────────────────────────────
# Quick-start examples
# ─────────────────────────────────────────────────────────────────────────────
#
# UNSW / CIC-IoT — giữ real_perturb_mode=dense (mix+scale), chỉ sparse CTGAN noise:
# python train_CT_MAECTGAN.py --d unsw --e 200 --b 128 \
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal \
#   --per 1.0 --type_train noise_std --mae_start_ratio 0.25 \
#   --ctgan_noise_type sparse --ctgan_sparse_k_ratio 0.15 --ctgan_sparse_noise_std 0.3 \
#   --real_perturb_mode dense \
#   --ctgan_pseudo_anomaly_mode origin --ctgan_noise_start_ratio 0.75 \
#   --seed 42 --time 1
#
# CIC-IDS — directional perturbation (near-normal attacks, recommended):
# python train_CT_MAECTGAN.py --d cic_ids --e 200 --b 128 \
#   --lae 0.3 --mae 2.0 --mae_loss_mode renormal \
#   --per 1.0 --type_train noise --mae_start_ratio 0.25 \
#   --ctgan_noise_type directional --ctgan_sparse_k_ratio 0.03 --ctgan_sparse_noise_std 0.5 \
#   --real_perturb_mode directional \
#   --ctgan_pseudo_anomaly_mode origin --ctgan_noise_start_ratio 0.75 \
#   --seed 42 --time 1
#
# Dense fallback (identical to original MAE-CTGAN):
# python train_CT_MAECTGAN.py --d unsw --e 200 --b 128 \
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal \
#   --per 1.0 --type_train noise_std --mae_start_ratio 0.25 \
#   --ctgan_noise_type dense --real_perturb_mode dense \
#   --ctgan_noise_start_ratio 0.75 --ctgan_noise_std 0.05 --ctgan_noise_feature_ratio 0.5 \
#   --seed 42 --time 1
