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
from sdv.single_table.maeafterctgan import MAEAfterCTGANSynthesizer
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

parser = argparse.ArgumentParser(
    description="Train MAEAfterCTGAN (two-phase: CTGAN fully first, then MAE)"
)

# ── Dataset / training basics ────────────────────────────────────────────────
parser.add_argument('--d',   type=str,   default='unsw',   help='Dataset name')
parser.add_argument('--e',   type=int,   default=200,      help='CTGAN epochs (Phase 1)')
parser.add_argument('--b',   type=int,   default=128,      help='Batch size')
parser.add_argument('--lae', type=float, default=0.1,      help='Lambda (AE loss weight)')
parser.add_argument('--mae', type=float, default=1.0,      help='Margin')
parser.add_argument(
    '--mae_loss_mode', type=str, choices=['origin', 'renormal'], default='renormal',
    help='MAE loss mode: origin=all samples; renormal=normal-only reconstruction.',
)
parser.add_argument('--per', type=float, default=1.0,
                    help='Fraction of normal training data to use (0 < per <= 1)')
parser.add_argument(
    '--type_train', type=str,
    choices=["all", "noise", "std", "3std", "std_3std", "noise_std", "noise_3std"],
    default="noise_std",
    help='Pseudo-anomaly composition for MAE training set.',
)

# ── MAE schedule (Phase 3) ───────────────────────────────────────────────────
parser.add_argument(
    '--mae_epochs', type=int, default=100,
    help='Number of epochs for MAE training (Phase 3).  Default: 100.',
)

# ── Pseudo-anomaly pool ───────────────────────────────────────────────────────
parser.add_argument(
    '--pseudo_anomaly_ratio', type=float, default=100.0,
    help=(
        'K%% — pseudo-anomaly pool size as a percentage of normal training data. '
        '100.0 = 1:1 ratio, 200.0 = 2:1 ratio.  Default: 100.0'
    ),
)

# ── CTGAN noise (applied at generation time) ──────────────────────────────────
parser.add_argument(
    '--ctgan_noise_std', type=float, default=0.05,
    help='Std of Gaussian noise added to CTGAN samples at generation time.  Default: 0.05',
)
parser.add_argument(
    '--ctgan_noise_feature_ratio', type=float, default=0.5,
    help='Fraction of continuous features perturbed per noise step.  Default: 0.5',
)
parser.add_argument(
    '--ctgan_pseudo_anomaly_mode', type=str,
    choices=['origin', 'feature_outlier'], default='origin',
    help='How CTGAN samples become pseudo-anomalies.  Default: origin',
)
parser.add_argument(
    '--ctgan_feature_outlier_r', type=float, default=1.0,
    help='Outlier coefficient for feature_outlier mode.  Default: 1.0',
)

# ── Pseudo-anomaly filter ─────────────────────────────────────────────────────
parser.add_argument(
    '--filter_mode', type=str, default='none',
    choices=['none', 'critic', 'mae_recon'],
    help=(
        'Filtering strategy for pseudo-anomaly samples. '
        '"none"     = no filtering (baseline). '
        '"critic"   = at generation time, keep filter_keep_ratio of samples '
        '             with lowest D score (most anomalous). Applied once. '
        '"mae_recon"= during MAE training, keep filter_keep_ratio of samples '
        '             with highest reconstruction error each epoch. '
        'Default: none'
    ),
)
parser.add_argument(
    '--filter_keep_ratio', type=float, default=0.7,
    help=(
        'Fraction of pseudo-anomaly samples to KEEP each step. '
        '0.7 = keep 70%%, discard 30%% that look most normal. '
        '1.0 = effectively disables filtering.  Default: 0.7'
    ),
)
parser.add_argument(
    '--filter_start_ratio', type=float, default=0.5,
    help=(
        'Fraction of mae_epochs to wait before activating the MAE reconstruction filter. '
        '0.5 = start filtering at epoch mae_epochs/2.  '
        'Only used when filter_mode=mae_recon.  Default: 0.5'
    ),
)

# ── Run metadata ─────────────────────────────────────────────────────────────
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--time', type=int, default=1,  help='Run index (e.g. 1 -> time_01)')
parser.add_argument(
    '--n_max_samples', type=int, default=500_000,
    help='Max rows for large datasets (bot_iot, ton_iot).  Default: 500000.'
)

args = parser.parse_args()

# ── Validation ───────────────────────────────────────────────────────────────
if args.mae_epochs < 1:
    parser.error('--mae_epochs must be >= 1.')
if args.pseudo_anomaly_ratio <= 0:
    parser.error('--pseudo_anomaly_ratio must be > 0.')
if args.ctgan_noise_std < 0.0:
    parser.error('--ctgan_noise_std must be >= 0.')
if not 0.0 <= args.ctgan_noise_feature_ratio <= 1.0:
    parser.error('--ctgan_noise_feature_ratio must be in [0, 1].')
if not 0.0 < args.filter_keep_ratio <= 1.0:
    parser.error('--filter_keep_ratio must be in (0, 1].')
if not 0.0 <= args.filter_start_ratio <= 1.0:
    parser.error('--filter_start_ratio must be in [0, 1].')
if not 0.0 < args.per <= 1.0:
    parser.error('--per must be in (0, 1].')

dataset_raw = args.d.strip().lower().replace("-", "_")
if dataset_raw == "cidds001":
    dataset_raw = "cidds_001"
dataset = dataset_raw

epochs = args.e
batch_size = args.b
lamda = args.lae
margin = args.mae
mae_loss_mode = args.mae_loss_mode
percentage = args.per
type_train = args.type_train
mae_epochs = args.mae_epochs
pseudo_anomaly_ratio = args.pseudo_anomaly_ratio
ctgan_noise_std = args.ctgan_noise_std
ctgan_noise_feature_ratio = args.ctgan_noise_feature_ratio
ctgan_pseudo_anomaly_mode = (
    'feature_outlier'
    if args.ctgan_pseudo_anomaly_mode == 'feature_outliner'
    else args.ctgan_pseudo_anomaly_mode
)
ctgan_feature_outlier_r = args.ctgan_feature_outlier_r
filter_mode = args.filter_mode
filter_keep_ratio = args.filter_keep_ratio
filter_start_ratio = args.filter_start_ratio
seed = args.seed
n_max_samples = args.n_max_samples
run_time = args.time
time_tag = f"{int(run_time):02d}"
run_folder = f"mae_after_ctgan_{type_train}_seed_{seed}_time_{time_tag}"


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
    "Dataset:", dataset,
    " CTGAN Epochs:", epochs,
    " MAE Epochs:", mae_epochs,
    " Batch size:", batch_size,
    " Lambda:", lamda,
    " Margin:", margin,
    " MAE loss mode:", mae_loss_mode,
    " Percentage:", percentage,
    " Type train:", type_train,
    " pseudo_anomaly_ratio:", pseudo_anomaly_ratio,
    " filter_mode:", filter_mode,
    " filter_keep_ratio:", filter_keep_ratio,
    " filter_start_ratio:", filter_start_ratio,
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

_filter_tag = f"mode_{filter_mode}_keep{filter_keep_ratio}_start{filter_start_ratio}"
results_dir = os.path.join(
    BASE_DIR,
    f"results/{run_folder}/per_{percentage}/"
    f"mae_after_{dataset}_{epochs}ctgan_{mae_epochs}mae_m_{margin}_l_{lamda}"
    f"_maeloss_{mae_loss_mode}"
    f"_pseudo{pseudo_anomaly_ratio}"
    f"_filter_{_filter_tag}"
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


def train_maeafterctgan(epochs):
    print("Training MAEAfterCTGAN, CTGAN epochs =", epochs, " MAE epochs =", mae_epochs)

    n_test = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)

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
    metadata_file = os.path.join(
        results_dir, f"{dataset}_mae_after_metadata_epochs_{epochs}.json"
    )
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    metadata.save_to_json(metadata_file)

    synthesizer = MAEAfterCTGANSynthesizer(
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
        dataset=dataset,
        path=results_dir,
        type_train=type_train,
        # ── Phase 3 ───────────────────────────────────────────────────────
        mae_epochs=mae_epochs,
        pseudo_anomaly_ratio=pseudo_anomaly_ratio,
        # ── CTGAN noise (generation time) ────────────────────────────────
        ctgan_noise_std=ctgan_noise_std,
        ctgan_noise_feature_ratio=ctgan_noise_feature_ratio,
        ctgan_pseudo_anomaly_mode=ctgan_pseudo_anomaly_mode,
        ctgan_feature_outlier_r=ctgan_feature_outlier_r,
        # ── filter ────────────────────────────────────────────────────────
        filter_mode=filter_mode,
        filter_keep_ratio=filter_keep_ratio,
        filter_start_ratio=filter_start_ratio,
    )

    X_train_normal_proc = synthesizer.preprocess(X_train_normal)
    X_val = synthesizer._data_processor.transform(X_val_raw)
    X_test = synthesizer._data_processor.transform(X_test_raw)

    X_test.to_csv(os.path.join(results_dir, f"{dataset}_X_test.csv"), index=False)
    y_test.to_frame(name="label").to_csv(
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

    threshold_bundle = (threshold_re, threshold_z_0, threshold_95, threshold_95_total)
    save_threshold(
        threshold_bundle,
        save_dir=results_dir,
        name=f"{dataset}_mae_after_threshold_epochs_{epochs}.pkl",
    )

    file_model = os.path.join(
        results_dir, f"MAE_After_{dataset}_epochs_{epochs}.pkl"
    )
    synthesizer.save(file_model)

    # ── Loss curves ───────────────────────────────────────────────────────
    # Phase 1: CTGAN loss
    ctgan_loss_df = synthesizer._model.ctgan_loss_values
    if ctgan_loss_df is not None and not ctgan_loss_df.empty:
        ctgan_log_file = os.path.join(
            results_dir, f"mae_after_ctgan_loss_log_epochs_{epochs}.csv"
        )
        ctgan_loss_df.to_csv(ctgan_log_file, index=False)
        print(f"CTGAN loss log saved to {ctgan_log_file}")

        plt.figure(figsize=(10, 6))
        plt.plot(ctgan_loss_df["Epoch"], ctgan_loss_df["Generator Loss"], label="Generator Loss")
        plt.plot(ctgan_loss_df["Epoch"], ctgan_loss_df["Discriminator Loss"], label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"MAEAfterCTGAN — CTGAN Phase\ndataset={dataset} ctgan_epochs={epochs}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(results_dir, f"loss_ctgan_phase_{dataset}_epochs_{epochs}.png")
        )
        plt.close()

    # Phase 3: MAE loss
    loss_df = synthesizer.get_loss_values()
    log_file = os.path.join(results_dir, f"mae_after_mae_loss_log_{mae_epochs}mae.csv")
    loss_df.to_csv(log_file, index=False)
    print(f"MAE loss log saved to {log_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["Epoch"], loss_df["MAE Loss"], label="MAE total Loss")
    plt.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction Loss")
    plt.plot(loss_df["Epoch"], lamda * loss_df["Latent Loss"], label="lambda*Latent Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"MAEAfterCTGAN — MAE Phase\ndataset={dataset} mae_epochs={mae_epochs}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_mae_phase_{dataset}_{mae_epochs}mae.png"))
    plt.close()

    # Filter kept ratio (mae_recon mode)
    if "Filter Kept Ratio" in loss_df.columns:
        filter_df = loss_df.dropna(subset=["Filter Kept Ratio"])
        if not filter_df.empty:
            plt.figure(figsize=(10, 4))
            plt.plot(filter_df["Epoch"], filter_df["Filter Kept Ratio"],
                     label="Filter kept ratio", color="darkorange")
            plt.axhline(filter_keep_ratio, linestyle="--", color="gray",
                        label=f"filter_keep_ratio={filter_keep_ratio}")
            plt.xlabel("MAE Epoch")
            plt.ylabel("Fraction kept")
            plt.title(
                f"MAEAfterCTGAN — Filter kept ratio ({filter_mode})\ndataset={dataset}"
            )
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(results_dir, f"filter_kept_{dataset}_{mae_epochs}mae.png")
            )
            plt.close()

    # Pseudo validation AUC
    if "Pseudo Val AUC" in loss_df.columns:
        auc_df = loss_df.dropna(subset=["Pseudo Val AUC"])
        if not auc_df.empty:
            plt.figure(figsize=(10, 4))
            plt.plot(auc_df["Epoch"], auc_df["Pseudo Val AUC"],
                     label="Pseudo Val AUC", color="steelblue")
            plt.xlabel("MAE Epoch")
            plt.ylabel("AUC")
            plt.title(
                f"MAEAfterCTGAN — Pseudo Val AUC\ndataset={dataset}"
            )
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(results_dir, f"pseudo_val_auc_{dataset}_{mae_epochs}mae.png")
            )
            plt.close()

    # Validation loss
    val_loss_df = loss_df.dropna(subset=["Validation Loss"])
    if not val_loss_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(val_loss_df["Epoch"], val_loss_df["Validation Loss"], label="Val total")
        if "Validation Reconstruction Loss" in val_loss_df.columns:
            plt.plot(val_loss_df["Epoch"],
                     val_loss_df["Validation Reconstruction Loss"], label="Val RE")
        if "Validation Latent Loss" in val_loss_df.columns:
            plt.plot(val_loss_df["Epoch"],
                     lamda * val_loss_df["Validation Latent Loss"], label="lambda*Val Latent")
        plt.xlabel("MAE Epoch")
        plt.ylabel("Loss")
        plt.title(f"MAEAfterCTGAN Validation Loss\ndataset={dataset} mae_epochs={mae_epochs}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(results_dir, f"loss_mae_val_{dataset}_{mae_epochs}mae.png")
        )
        plt.close()

    print("Training completed.")
    return synthesizer


def predict_and_evaluate(scores, y_true, threshold, save_dir, filename, mode="std", k_std=3.0):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if mode == "dist":
        threshold_test = threshold
    else:
        mu, std = threshold
        threshold_test = mu + k_std * std

    y_pred = (scores > threshold_test).astype(int)
    y_true_bin = (y_true != 0).astype(int)

    unique_classes = np.unique(y_true_bin)
    auc = roc_auc_score(y_true_bin, scores) * 100 if len(unique_classes) >= 2 else 0.0
    acc = accuracy_score(y_true_bin, y_pred) * 100
    f1 = f1_score(y_true_bin, y_pred, zero_division=0) * 100
    precision = precision_score(y_true_bin, y_pred, zero_division=0) * 100
    recall = recall_score(y_true_bin, y_pred, zero_division=0) * 100
    mcc = matthews_corrcoef(y_true_bin, y_pred) * 100
    cm = confusion_matrix(y_true_bin, y_pred)

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

    print(
        f"  [{filename}] AUC={auc:.2f}% F1={f1:.2f}% MCC={mcc:.2f}% "
        f"Prec={precision:.2f}% Rec={recall:.2f}% threshold={threshold_test:.6f}"
    )
    return {
        "scores": scores,
        "y_pred": y_pred,
        "metrics": {"AUC": auc, "ACC": acc, "F1": f1,
                    "Precision": precision, "Recall": recall, "MCC": mcc},
        "confusion_matrix": cm,
    }


def _compute_total_loss(re_loss, z_norm):
    return re_loss + lamda * (z_norm ** 2)


def _evaluate_model_with_thresholds(synthesizer, X_test, y_test, thresholds, save_dir, prefix):
    thr_re, thr_z, thr_95, thr_95_total = thresholds[:4]

    re_errors, z_norm, y_true = synthesizer.get_score_mae(X_test, y_test)

    predict_and_evaluate(re_errors, y_true, thr_re,
                         save_dir=save_dir, filename=f"{prefix}_mode_re.csv", mode="re")
    predict_and_evaluate(z_norm, y_true, thr_95,
                         save_dir=save_dir, filename=f"{prefix}_mode_95.csv", mode="dist")
    total_loss = _compute_total_loss(re_errors, z_norm)
    predict_and_evaluate(total_loss, y_true, thr_95_total,
                         save_dir=save_dir, filename=f"{prefix}_mode_total95.csv", mode="dist")


def post_train_checkpoint_evaluation(results_dir, dataset, epochs):
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

    X_test_csv = pd.read_csv(x_test_path)
    y_test_csv = pd.read_csv(y_test_path)["label"]
    best_synth = MAEAfterCTGANSynthesizer.load(best_model_path)
    with open(best_threshold_path, "rb") as f:
        best_thresholds = pickle.load(f)

    _evaluate_model_with_thresholds(
        synthesizer=best_synth,
        X_test=X_test_csv,
        y_test=y_test_csv,
        thresholds=best_thresholds,
        save_dir=checkpoint_dir,
        prefix=f"test_{dataset}_best",
    )
    print("Post-train evaluation done.")


if __name__ == "__main__":
    print("Starting MAEAfterCTGAN training...")
    synthesizer = train_maeafterctgan(epochs=epochs)
    post_train_checkpoint_evaluation(
        results_dir=results_dir, dataset=dataset, epochs=epochs
    )
    print("Done.")

# =============================================================================
# GIẢI THÍCH THAM SỐ — train_MAEAfterCTGAN.py
# =============================================================================
#
# MAEAfterCTGAN: kiến trúc hai giai đoạn hoàn toàn tách biệt.
#   Vấn đề của MAECTGAN gốc: CTGAN chưa converge đã phải sinh pseudo-anomaly
#   → pseudo-anomaly kém chất lượng → MAE học từ dữ liệu nhiễu.
#   Giải pháp: Huấn luyện CTGAN đến convergence TRƯỚC, sau đó mới sinh pool
#   pseudo-anomaly cố định và huấn luyện MAE trên pool đó.
#
# ── Dữ liệu / cơ bản ─────────────────────────────────────────────────────────
#
# --d  (dataset)
#   Giá trị hợp lệ: unsw | cic_ids | nsl_kdd | cic_iot | cidds_001 | wsn_ds | bot_iot | ton_iot
#   Mặc định: unsw
#
# --e  (ctgan_epochs)   Số epoch Phase 1 (CTGAN).  Khuyến nghị: 100–300.  Mặc định: 200
# --b  (batch_size)     Mặc định: 128
# --per  Tỷ lệ normal data dùng để train (0 < per ≤ 1).  Mặc định: 1.0
# --seed  Mặc định: 42
# --time  Chỉ số lần chạy (1 → time_01 trong tên folder).  Mặc định: 1
# --n_max_samples  Giới hạn mẫu cho dataset lớn.  Mặc định: 500000
#
# ── MAE / Autoencoder ────────────────────────────────────────────────────────
#
# --lae  (lambda)   total_loss = reconstruction_loss + lae * latent_loss.  Mặc định: 0.1
# --mae  (margin)   Ngưỡng margin trong latent loss.  Mặc định: 1.0
# --mae_loss_mode  {origin | renormal}   Mặc định: renormal (khuyến nghị)
# --mae_epochs   Số epoch Phase 3 (MAE).  Khuyến nghị: 50–200.  Mặc định: 100
# --type_train  {noise | std | 3std | std_3std | noise_std | noise_3std | all}
#   Nguồn pseudo-anomaly cho MAE.  Mặc định: noise_std
#
# ── Pseudo-anomaly pool (Phase 2) ────────────────────────────────────────────
#
# --pseudo_anomaly_ratio  K%%
#   Kích thước pool pseudo-anomaly tính theo %% của normal data.
#   100.0 → pool bằng với normal data (1:1).
#   200.0 → pool gấp đôi normal data (2:1) → MAE có nhiều diverse pseudo-anomaly hơn.
#   Khuyến nghị: 100.0–200.0.  Mặc định: 100.0
#
#   Lưu ý: nếu type_train="noise_std" (2 nguồn), MAE mỗi epoch sẽ
#   lấy pseudo_ratio/2 mẫu từ std pool và pseudo_ratio/2 mẫu từ noise.
#
# ── CTGAN noise (áp dụng lúc sinh mẫu, không theo lịch epoch) ────────────────
#
# --ctgan_noise_std           Std của Gaussian noise thêm vào continuous features.
#   0.0 = không thêm noise.  0.05 = noise nhẹ giúp tăng diversity.  Mặc định: 0.05
# --ctgan_noise_feature_ratio Tỷ lệ features được thêm noise.  Mặc định: 0.5
# --ctgan_pseudo_anomaly_mode {origin | feature_outlier}.  Mặc định: origin
# --ctgan_feature_outlier_r   Hệ số r (feature_outlier mode).  Mặc định: 1.0
#
# ── Filter ───────────────────────────────────────────────────────────────────
#
# --filter_mode  {none | critic | mae_recon}
#   "none"     → không lọc (baseline).
#   "critic"   → lọc LÚC SINH MẪU (Phase 2):
#                 Sinh extra batches, giữ filter_keep_ratio với D score thấp nhất.
#                 D score thấp = discriminator nghĩ là anomaly = pseudo-anomaly tốt.
#                 Áp dụng 1 lần duy nhất khi tạo pool → không ảnh hưởng Phase 3.
#   "mae_recon"→ lọc TRONG KHI TRAIN MAE (Phase 3):
#                 Mỗi epoch, giữ filter_keep_ratio với reconstruction error cao nhất.
#                 Error cao = MAE không tái tạo được = pseudo-anomaly khó = tốt.
#   Mặc định: none
#
# --filter_keep_ratio   Tỷ lệ mẫu được GIỮ LẠI.  Mặc định: 0.7
# --filter_start_ratio  Với mae_recon: đợi filter_start_ratio * mae_epochs trước khi lọc.
#   0.5 = bắt đầu lọc từ epoch mae_epochs/2.  Mặc định: 0.5
#
# ── So sánh với CleanMAECTGAN ─────────────────────────────────────────────────
#
#   CleanMAECTGAN:   CTGAN + MAE train đồng thời (joint).
#                    Filter trong lúc joint training.
#                    Vấn đề: CTGAN chưa converge khi bắt đầu sinh pseudo-anomaly.
#   MAEAfterCTGAN:   CTGAN train xong → sinh pool → MAE train độc lập.
#                    Pseudo-anomaly chất lượng cao hơn (CTGAN đã converge).
#                    Có thể tune mae_epochs và pseudo_anomaly_ratio độc lập.
#
# ── Đầu ra (output files) ─────────────────────────────────────────────────────
#
#   results/<run_folder>/per_<per>/mae_after_<dataset>_.../
#   ├── mae_after_ctgan_loss_log_epochs_<e>.csv  — loss Phase 1 (CTGAN)
#   ├── mae_after_mae_loss_log_<mae_e>mae.csv    — loss Phase 3 (MAE)
#   ├── loss_ctgan_phase_*.png                   — CTGAN loss curve
#   ├── loss_mae_phase_*.png                     — MAE loss curve
#   ├── loss_mae_val_*.png                       — MAE validation loss
#   ├── pseudo_val_auc_*.png                     — pseudo validation AUC qua epoch
#   ├── filter_kept_*.png                        — filter kept ratio (mae_recon mode)
#   ├── MAE_After_<dataset>_epochs_<e>.pkl       — model cuối
#   ├── <dataset>_mae_after_threshold_*.pkl      — ngưỡng phát hiện
#   └── checkpoints/
#       ├── MAE_<dataset>_best.pkl               — model tốt nhất (pseudo-val AUC)
#       └── <dataset>_best_mode_*.csv
#
# ── Ví dụ lệnh chạy ──────────────────────────────────────────────────────────
#
# Baseline (không filter):
# python train_MAEAfterCTGAN.py --d unsw --e 200 --mae_epochs 100 --b 128 \
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal \
#   --type_train noise_std --pseudo_anomaly_ratio 100.0 \
#   --filter_mode none --seed 42 --time 1
#
# Với critic filter (lọc lúc sinh pool):
# python train_MAEAfterCTGAN.py --d unsw --e 200 --mae_epochs 100 --b 128 \
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal \
#   --type_train noise_std --pseudo_anomaly_ratio 100.0 \
#   --filter_mode critic --filter_keep_ratio 0.7 --seed 42 --time 2
#
# Với mae_recon filter (lọc trong khi train MAE):
# python train_MAEAfterCTGAN.py --d unsw --e 200 --mae_epochs 100 --b 128 \
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal \
#   --type_train noise_std --pseudo_anomaly_ratio 100.0 \
#   --filter_mode mae_recon --filter_keep_ratio 0.7 --filter_start_ratio 0.5 \
#   --seed 42 --time 3
#
# So sánh 3 lệnh tuần tự (Ubuntu):
# python Syn_ctgan/train_MAEAfterCTGAN.py --d unsw --e 200 --mae_epochs 100 \
#   --filter_mode none --seed 42 --time 4 && \
# python Syn_ctgan/train_MAEAfterCTGAN.py --d unsw --e 200 --mae_epochs 100 \
#   --filter_mode critic --filter_keep_ratio 0.7 --seed 42 --time 5 && \
# python Syn_ctgan/train_MAEAfterCTGAN.py --d unsw --e 200 --mae_epochs 100 \
#   --filter_mode mae_recon --filter_keep_ratio 0.7 --seed 42 --time 6
