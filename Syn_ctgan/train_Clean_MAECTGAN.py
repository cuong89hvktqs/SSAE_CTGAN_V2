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
from sdv.single_table.cleanmaectgan import CleanMAECTGANSynthesizer
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
    description="Train CleanMAECTGAN (pseudo-anomaly filtering: critic or mae_recon)"
)

# ── Dataset / training basics ────────────────────────────────────────────────
parser.add_argument('--d',   type=str,   default='unsw',   help='Dataset name')
parser.add_argument('--e',   type=int,   default=200,      help='Epochs')
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

# ── MAE schedule ─────────────────────────────────────────────────────────────
parser.add_argument(
    '--mae_start_ratio', type=float, default=0.25,
    help='Fraction of epochs before MAE pseudo-anomaly training starts (0.0-1.0).',
)

# ── CTGAN noise ───────────────────────────────────────────────────────────────
parser.add_argument(
    '--ctgan_noise_start_ratio', type=float, default=0.75,
    help='Fraction of epochs before Gaussian noise is added to CTGAN samples.',
)
parser.add_argument(
    '--ctgan_noise_std', type=float, default=0.05,
    help='Std of Gaussian noise added to continuous CTGAN features.',
)
parser.add_argument(
    '--ctgan_noise_feature_ratio', type=float, default=0.5,
    help='Fraction of continuous features perturbed per noise step.',
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

# ── Pseudo-anomaly filter (new in CleanMAECTGAN) ──────────────────────────────
parser.add_argument(
    '--filter_mode', type=str, default='none',
    choices=['none', 'critic', 'mae_recon'],
    help=(
        'Filtering strategy for pseudo-anomaly samples. '
        '"none"     = no filtering (same as MAECTGAN, good baseline). '
        '"critic"   = discard CTGAN samples with high D scores (look like normal). '
        '             Applied before buffer; uses EMA trend threshold. '
        '"mae_recon"= discard assembled pseudo-anomalies with low MAE reconstruction '
        '             error (look normal to MAE). Applied after buffer assembly. '
        'Default: none'
    ),
)
parser.add_argument(
    '--filter_keep_ratio', type=float, default=0.7,
    help=(
        'Fraction of pseudo-anomaly samples to KEEP each step. '
        '0.7 = keep 70%%, discard 30%% that look most normal. '
        '1.0 = effectively disables filtering. Default: 0.7'
    ),
)
parser.add_argument(
    '--filter_start_ratio', type=float, default=0.5,
    help=(
        'Fraction of post-MAE-start epochs to wait before filtering activates. '
        '0.5 = start filtering halfway between mae_start_epoch and final epoch. '
        'Default: 0.5'
    ),
)
parser.add_argument(
    '--critic_ema_alpha', type=float, default=0.3,
    help=(
        'EMA smoothing factor for D-score trend threshold (critic mode only). '
        '0.3 = slow adaptation (stable). 0.9 = fast (follows current batch). '
        'Default: 0.3'
    ),
)

# ── Run metadata ─────────────────────────────────────────────────────────────
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--time', type=int, default=1,  help='Run index (e.g. 1 -> time_01)')
parser.add_argument(
    '--n_max_samples', type=int, default=500_000,
    help='Max rows for large datasets (bot_iot, ton_iot). Default: 500000.'
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
if not 0.0 < args.filter_keep_ratio <= 1.0:
    parser.error('--filter_keep_ratio must be in (0, 1].')
if not 0.0 <= args.filter_start_ratio <= 1.0:
    parser.error('--filter_start_ratio must be in [0, 1].')
if not 0.0 < args.critic_ema_alpha <= 1.0:
    parser.error('--critic_ema_alpha must be in (0, 1].')
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
filter_mode = args.filter_mode
filter_keep_ratio = args.filter_keep_ratio
filter_start_ratio = args.filter_start_ratio
critic_ema_alpha = args.critic_ema_alpha
seed = args.seed
n_max_samples = args.n_max_samples
run_time = args.time
time_tag = f"{int(run_time):02d}"
run_folder = f"clean_mae_{type_train}_seed_{seed}_time_{time_tag}"


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
    " Epochs:", epochs,
    " Batch size:", batch_size,
    " Lambda:", lamda,
    " Margin:", margin,
    " MAE loss mode:", mae_loss_mode,
    " Percentage:", percentage,
    " Type train:", type_train,
    " MAE start ratio:", mae_start_ratio,
    " filter_mode:", filter_mode,
    " filter_keep_ratio:", filter_keep_ratio,
    " filter_start_ratio:", filter_start_ratio,
    " critic_ema_alpha:", critic_ema_alpha,
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
    f"clean_mae_{dataset}_{epochs}_m_{margin}_l_{lamda}"
    f"_maeloss_{mae_loss_mode}"
    f"_mstart_{mae_start_ratio}"
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


def train_clean_maectgan(epochs):
    print("Training CleanMAECTGAN, epochs =", epochs)

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
        results_dir, f"{dataset}_clean_mae_metadata_epochs_{epochs}.json"
    )
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    metadata.save_to_json(metadata_file)

    synthesizer = CleanMAECTGANSynthesizer(
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
        mae_start_ratio=mae_start_ratio,
        ctgan_noise_start_ratio=ctgan_noise_start_ratio,
        ctgan_noise_std=ctgan_noise_std,
        ctgan_noise_feature_ratio=ctgan_noise_feature_ratio,
        ctgan_pseudo_anomaly_mode=ctgan_pseudo_anomaly_mode,
        ctgan_feature_outlier_r=ctgan_feature_outlier_r,
        # ── filter ────────────────────────────────────────────────────────
        filter_mode=filter_mode,
        filter_keep_ratio=filter_keep_ratio,
        filter_start_ratio=filter_start_ratio,
        critic_ema_alpha=critic_ema_alpha,
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
        name=f"{dataset}_clean_mae_threshold_epochs_{epochs}.pkl",
    )

    file_model = os.path.join(
        results_dir, f"Clean_MAE_{dataset}_epochs_{epochs}.pkl"
    )
    synthesizer.save(file_model)

    # ── Loss curves ───────────────────────────────────────────────────────
    loss_df = synthesizer.get_loss_values()
    log_file = os.path.join(results_dir, f"clean_mae_loss_log_epochs_{epochs}.csv")
    loss_df.to_csv(log_file, index=False)
    print(f"Loss log saved to {log_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["Epoch"], loss_df["MAE Loss"], label="MAE total Loss")
    plt.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction Loss")
    plt.plot(loss_df["Epoch"], lamda * loss_df["Latent Loss"], label="lambda*Latent Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CleanMAECTGAN Training Loss\ndataset={dataset} epochs={epochs}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_mae_{dataset}_epochs_{epochs}.png"))
    plt.close()

    if "Filter Kept Ratio" in loss_df.columns:
        filter_df = loss_df.dropna(subset=["Filter Kept Ratio"])
        if not filter_df.empty:
            plt.figure(figsize=(10, 4))
            plt.plot(filter_df["Epoch"], filter_df["Filter Kept Ratio"],
                     label="Filter kept ratio", color="darkorange")
            plt.axhline(filter_keep_ratio, linestyle="--", color="gray",
                        label=f"filter_keep_ratio={filter_keep_ratio}")
            plt.xlabel("Epoch")
            plt.ylabel("Fraction kept")
            plt.title(
                f"CleanMAECTGAN — Filter kept ratio ({filter_mode})\ndataset={dataset}"
            )
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(results_dir, f"filter_kept_{dataset}_epochs_{epochs}.png")
            )
            plt.close()

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
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"CleanMAECTGAN Validation Loss\ndataset={dataset} epochs={epochs}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(results_dir, f"loss_clean_val_{dataset}_epochs_{epochs}.png")
        )
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["Epoch"], loss_df["Generator Loss"], label="Generator Loss")
    plt.plot(loss_df["Epoch"], loss_df["Discriminator Loss"], label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CleanMAECTGAN CTGAN Loss\ndataset={dataset} epochs={epochs}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_ctgan_{dataset}_epochs_{epochs}.png"))
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
    best_synth = CleanMAECTGANSynthesizer.load(best_model_path)
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
    print("Starting CleanMAECTGAN training...")
    synthesizer = train_clean_maectgan(epochs=epochs)
    post_train_checkpoint_evaluation(
        results_dir=results_dir, dataset=dataset, epochs=epochs
    )
    print("Done.")

# =============================================================================
# GIẢI THÍCH THAM SỐ — train_Clean_MAECTGAN.py
# =============================================================================
#
# CleanMAECTGAN giải quyết vấn đề "label noise thầm lặng":
#   CTGAN sinh mẫu z~N(0,I) gán nhãn y=1 (anomaly), nhưng một phần trông giống
#   normal data → nhãn sai → kéo tụt kết quả MAE.
#   Hai chiến lược lọc (chọn qua --filter_mode):
#   "critic"    → dùng D score của CTGAN để lọc (trước buffer, pac=10 vẫn dùng được)
#   "mae_recon" → dùng reconstruction error của MAE để lọc (sau buffer assembly)
#
# ── Dữ liệu / cơ bản ─────────────────────────────────────────────────────────
#
# --d  (dataset)
#   Giá trị hợp lệ: unsw | cic_ids | nsl_kdd | cic_iot | cidds_001 | wsn_ds | bot_iot | ton_iot
#   Mặc định: unsw
#
# --e  (epochs)   Số epoch.  Khuyến nghị: 100–200.  Mặc định: 200
# --b  (batch_size)  Mặc định: 128
# --per  Tỷ lệ normal data dùng để train (0 < per ≤ 1).  Mặc định: 1.0
# --seed  Mặc định: 42
# --time  Chỉ số lần chạy (1 → time_01 trong tên folder).  Mặc định: 1
# --n_max_samples  Giới hạn mẫu cho dataset lớn.  Mặc định: 500000
#
# ── MAE / Autoencoder ────────────────────────────────────────────────────────
#
# --lae  (lambda)   total_loss = reconstruction_loss + lae * latent_loss
#   Nhỏ (0.1) → ưu tiên reconstruction.  Lớn (0.5+) → ưu tiên latent separation.
#   Mặc định: 0.1
#
# --mae  (margin)   Ngưỡng margin trong latent loss.
#   Khuyến nghị: 1.0 (UNSW/CIC-IoT); 1.5–2.0 (CIC-IDS).  Mặc định: 1.0
#
# --mae_loss_mode  {origin | renormal}
#   "renormal" → MAE không học tái tạo anomaly → score cao hơn (khuyến nghị).
#   Mặc định: renormal
#
# --mae_start_ratio   Tỷ lệ epoch đầu chỉ train CTGAN (chưa train MAE).
#   Mặc định: 0.25
#
# --type_train  {noise | std | 3std | std_3std | noise_std | noise_3std | all}
#   Nguồn pseudo-anomaly cho MAE. "noise_std" là cân bằng tốt.  Mặc định: noise_std
#
# ── CTGAN noise ───────────────────────────────────────────────────────────────
#
# --ctgan_noise_start_ratio   Epoch bắt đầu thêm Gaussian noise vào CTGAN samples.
#   Mặc định: 0.75
# --ctgan_noise_std           Std của Gaussian noise.  Mặc định: 0.05
# --ctgan_noise_feature_ratio Tỷ lệ features được thêm noise.  Mặc định: 0.5
# --ctgan_pseudo_anomaly_mode {origin | feature_outlier}.  Mặc định: origin
# --ctgan_feature_outlier_r   Hệ số r (feature_outlier mode).  Mặc định: 1.0
#
# ── Filter (cơ chế chính của CleanMAECTGAN) ──────────────────────────────────
#
# --filter_mode  {none | critic | mae_recon}
#   "none"     → không lọc (baseline, giống MAECTGAN hoàn toàn). Dùng để baseline.
#   "critic"   → lọc TRƯỚC buffer dùng D score:
#                 D_score cao = mẫu trông giống normal = label noise → loại bỏ.
#                 D_score được expand từ group score (pac=10) lên per-sample level.
#                 EMA threshold tự calibrate theo trend D score qua các epoch:
#                 khi generator cải thiện, D scores có xu hướng tăng; EMA bắt kịp
#                 → threshold tự điều chỉnh theo sự trưởng thành của generator.
#   "mae_recon"→ lọc SAU buffer assembly dùng MAE reconstruction error:
#                 error thấp = MAE tái tạo được dễ = mẫu trông normal = label noise → loại.
#                 error cao  = MAE không tái tạo được = pseudo-anomaly thật → giữ.
#   Mặc định: none
#
# --filter_keep_ratio   Tỷ lệ mẫu được GIỮ LẠI mỗi step.
#   0.7 → giữ 70%, bỏ 30% trông giống normal nhất.
#   1.0 → tắt filter hoàn toàn.
#   Khuyến nghị: CIC-IDS: 0.6–0.7 | UNSW: 0.8–0.9 | CIC-IoT: 0.8–0.9
#   Mặc định: 0.7
#
# --filter_start_ratio  Tỷ lệ epoch MAE phải trôi qua trước khi bật filter.
#   Công thức: filter_start = mae_start_epoch + filter_start_ratio * (epochs - mae_start_epoch)
#   Ví dụ: e=100, mae_start_ratio=0.25, filter_start_ratio=0.5
#     → filter bật từ epoch 25 + 0.5*(100-25) = 62
#   Mục đích: D / MAE cần warmup trước khi dùng làm signal lọc.
#   Mặc định: 0.5
#
# --critic_ema_alpha  EMA smoothing cho D-score threshold (chỉ dùng với critic mode).
#   0.3 → chậm (ổn định). 0.9 → nhanh (bám sát batch hiện tại).  Mặc định: 0.3
#
# ── Đầu ra (output files) ─────────────────────────────────────────────────────
#
#   results/<run_folder>/per_<per>/clean_mae_<dataset>_<e>_.../
#   ├── clean_mae_loss_log_epochs_<e>.csv   — loss log (bao gồm "Filter Kept Ratio")
#   ├── filter_kept_<dataset>_epochs_<e>.png — biểu đồ tỷ lệ mẫu được giữ qua filter
#   ├── loss_mae_*.png  |  loss_ctgan_*.png  |  loss_clean_val_*.png
#   ├── Clean_MAE_<dataset>_epochs_<e>.pkl   — model cuối
#   ├── <dataset>_clean_mae_threshold_*.pkl  — ngưỡng phát hiện
#   └── checkpoints/
#       ├── MAE_<dataset>_best.pkl           — model tốt nhất (pseudo-val AUC)
#       └── <dataset>_best_mode_*.csv
#
#   Cột "Filter Kept Ratio" trong loss log:
#     filter_mode=none     → NaN (không lọc)
#     filter_mode=critic   → tỷ lệ mẫu CTGAN còn lại sau critic filter mỗi epoch
#     filter_mode=mae_recon→ tỷ lệ mẫu còn lại sau MAE reconstruction filter mỗi epoch
#     Lý tưởng ≈ filter_keep_ratio (0.7 → kept ratio ~0.7).
#
# ── Ví dụ lệnh chạy ──────────────────────────────────────────────────────────
#
# Baseline (không filter, đúng bằng MAECTGAN):
# python train_Clean_MAECTGAN.py --d cic_ids --e 100 --b 128 ^
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal ^
#   --type_train noise_std --mae_start_ratio 0.25 ^
#   --filter_mode none --seed 42 --time 1
#
# CIC-IDS với critic filter (lọc bằng D score, pac=10 giữ nguyên):
# python train_Clean_MAECTGAN.py --d cic_ids --e 100 --b 128 ^
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal ^
#   --type_train noise_std --mae_start_ratio 0.25 ^
#   --filter_mode critic --filter_keep_ratio 0.7 --filter_start_ratio 0.5 ^
#   --seed 42 --time 1
#
# CIC-IDS với mae_recon filter (lọc bằng MAE error):
# python train_Clean_MAECTGAN.py --d cic_ids --e 100 --b 128 ^
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal ^
#   --type_train noise_std --mae_start_ratio 0.25 ^
#   --filter_mode mae_recon --filter_keep_ratio 0.7 --filter_start_ratio 0.5 ^
#   --seed 42 --time 1
#
# UNSW với mae_recon filter (nhẹ hơn vì UNSW ít label noise):
# python train_Clean_MAECTGAN.py --d unsw --e 100 --b 128 ^
#   --lae 0.1 --mae 1.0 --mae_loss_mode renormal ^
#   --type_train noise_std --mae_start_ratio 0.25 ^
#   --filter_mode mae_recon --filter_keep_ratio 0.9 --filter_start_ratio 0.5 ^
#   --seed 42 --time 1
#
# ── Gợi ý ablation study ─────────────────────────────────────────────────────
#
# Bước 0 — Baseline (không filter):
#   --filter_mode none
#
# Bước 1 — Thử critic filter (pac=10, EMA trend):
#   --filter_mode critic --filter_keep_ratio 0.9   (filter nhẹ)
#   --filter_mode critic --filter_keep_ratio 0.7   (filter vừa)
#   --filter_mode critic --filter_keep_ratio 0.5   (filter mạnh)
#
# Bước 2 — Thử mae_recon filter (MAE reconstruction error):
#   --filter_mode mae_recon --filter_keep_ratio 0.9
#   --filter_mode mae_recon --filter_keep_ratio 0.7
#
# Bước 3 — Điều chỉnh critic_ema_alpha (chỉ cho critic mode):
#   --filter_mode critic --filter_keep_ratio 0.7 --critic_ema_alpha 0.1  (rất chậm)
#   --filter_mode critic --filter_keep_ratio 0.7 --critic_ema_alpha 0.5  (nhanh hơn)
#
# Bước 4 — Điều chỉnh filter_start_ratio:
#   --filter_start_ratio 0.0  (bắt đầu lọc ngay từ khi MAE active)
#   --filter_start_ratio 0.5  (đợi MAE warmup — khuyến nghị)
#   --filter_start_ratio 0.75 (lọc muộn, ít epoch lọc hơn)
#
# Thay đổi từng bước 1 tham số, giữ nguyên các tham số còn lại để
# cô lập được ảnh hưởng của từng tham số.
