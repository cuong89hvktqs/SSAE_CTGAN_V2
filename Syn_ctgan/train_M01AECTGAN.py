"""
 kết hợp M01AE và CTgan thành file m01aectgan.py như sau (theo hướng join):
- Mỗi mỗi epech khi huấn luyện CTGAN ta thực hiện như sau:
+ Tham số mode type_train: noise thì dùng đúng hàm add_noise như hiện tại đang có của M01AE
+ Tham số type_train: std thì add_noise bây giờ đầu vào không phải từ x_normal nữa mà từ dữ liệu do CTGAN sinh ra sau mỗi epoch huấn luyện
+ Thhm số  type_train: noise_std thì sẽ là 50% cái này 50% cái kia
+ CTGAN train epoch đầu còn yếu → sample kém → _add_noise(ctgan_sample) = double noise, vô nghĩa. Cần thêm mae_start_ratio giống MAECTGAN — đợi CTGAN warm up rồi mới bật std:
epoch 0 → 25%*epochs:   chỉ train CTGAN + M01AE dùng "noise" (fallback)
epoch 25%→ cuối:        M01AE chuyển sang dùng "std" hoặc "noise_std"
"""
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
from sdv.single_table.m01aectgan import M01AECTGANSynthesizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
)
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
    load_data_insdn,
)
from sklearn.model_selection import train_test_split
import sdv, ctgan, random
print(sdv.__file__)
print(ctgan.__file__)

import argparse

parser = argparse.ArgumentParser(
    description="Train M01AECTGAN (joint CTGAN + M01AE two-sided margin)"
)

# ── Dataset / basics ─────────────────────────────────────────────────────────
parser.add_argument("--d",   type=str,   default="unsw", help="Dataset name")
parser.add_argument("--e",   type=int,   default=200,    help="Total epochs (CTGAN + MAE joint)")
parser.add_argument("--b",   type=int,   default=128,    help="Batch size")
parser.add_argument("--per", type=float, default=1.0,    help="Fraction of normal training data (0<per<=1)")
parser.add_argument("--seed",type=int,   default=42,     help="Random seed")
parser.add_argument("--time",type=int,   default=1,      help="Run index")
parser.add_argument("--n_max_samples", type=int, default=500_000,
                    help="Max rows for large datasets")

# ── M01AE ────────────────────────────────────────────────────────────────────
parser.add_argument("--lae",  type=float, default=0.1,   help="Lambda (latent loss weight)")
parser.add_argument("--mae",  type=float, default=1.0,   help="Margin for two-sided loss")
parser.add_argument(
    "--mae_loss_mode", type=str, choices=["origin", "renormal"], default="renormal",
    help="RE loss mode: 'renormal'=normal only (recommended)"
)
parser.add_argument("--mae_lr", type=float, default=1e-3, help="M01AE learning rate")

# ── Noise ────────────────────────────────────────────────────────────────────
parser.add_argument("--noise_std",   type=float, default=0.1, help="Gaussian noise std")
parser.add_argument("--noise_ratio", type=float, default=1.0, help="Fraction of continuous features to perturb")
parser.add_argument(
    "--noise_type", type=str,
    choices=["none", "global", "feature", "feature_outlier"],
    default="global",
    help=(
        "Continuous noise type: "
        "'global'=N(0,noise_std²) [simple additive], "
        "'feature'=per-feature N(0,(noise_std·σᵢ)²) [feature-scaled], "
        "'feature_outlier'=set to μᵢ+noise_r·noise_k·σᵢ [same as M01AE default]"
    ),
)
parser.add_argument("--noise_r", type=float, default=1.0,
                    help="Outlier sign (+/-) multiplier for feature_outlier: μ + noise_r*noise_k*σ")
parser.add_argument("--noise_k", type=float, default=3.0,
                    help="Outlier scale factor for feature_outlier: μ + noise_r*noise_k*σ")
parser.add_argument(
    "--cat_noise_type", type=str, choices=["none", "shuffle", "flip", "combined"],
    default="none", help="Categorical feature noise type"
)
parser.add_argument("--cat_noise_ratio", type=float, default=1.0,
                    help="Fraction of categorical spans to perturb")

# ── Joint training ────────────────────────────────────────────────────────────
parser.add_argument(
    "--type_train", type=str, choices=["noise", "std", "noise_std"],
    default="noise_std",
    help=(
        "Pseudo-anomaly source: "
        "'noise'=add_noise(real_normal) [M01AE default, no CTGAN], "
        "'std'=add_noise(ctgan_sample) [CTGAN diversity + noise], "
        "'noise_std'=50%% noise + 50%% add_noise(ctgan_sample) [recommended]"
    ),
)
parser.add_argument(
    "--mae_start_ratio", type=float, default=0.25,
    help="Fraction of epochs CTGAN trains alone before its samples are used (warmup)"
)
parser.add_argument(
    "--patience", type=int, default=50,
    help="Early stopping patience (epochs without val improvement)"
)
parser.add_argument(
    "--val_type", type=str, choices=["loss", "auc"], default="loss",
    help=(
        "Best-checkpoint criterion for post-train evaluation: "
        "'loss'=best validation loss [default], "
        "'auc'=best validation AUC (pseudo-anomaly built the same way as type_train)"
    ),
)

args = parser.parse_args()

# ── Validation ────────────────────────────────────────────────────────────────
if not 0.0 < args.per <= 1.0:
    parser.error("--per must be in (0, 1].")
if not 0.0 <= args.mae_start_ratio < 1.0:
    parser.error("--mae_start_ratio must be in [0, 1).")
if not 0.0 < args.noise_ratio <= 1.0:
    parser.error("--noise_ratio must be in (0, 1].")
if args.noise_std < 0:
    parser.error("--noise_std must be >= 0.")
if not 0.0 < args.cat_noise_ratio <= 1.0:
    parser.error("--cat_noise_ratio must be in (0, 1].")
if args.patience < 1:
    parser.error("--patience must be >= 1.")

dataset_raw = args.d.strip().lower().replace("-", "_")
if dataset_raw == "cidds001":
    dataset_raw = "cidds_001"
dataset = dataset_raw

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
mae_start_ratio = args.mae_start_ratio
patience        = args.patience
val_type        = args.val_type
seed            = args.seed
n_max_samples   = args.n_max_samples
run_time        = args.time
time_tag        = f"{int(run_time):02d}"
run_folder      = f"m01ae_{type_train}_se_{seed}_t_{time_tag}"


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
    " Batch:", batch_size,
    " Lambda:", lamda,
    " Margin:", margin,
    " mae_loss_mode:", mae_loss_mode,
    " mae_lr:", mae_lr,
    " noise_type:", noise_type,
    " noise_std:", noise_std,
    " noise_r:", noise_r,
    " noise_k:", noise_k,
    " cat_noise_type:", cat_noise_type,
    " type_train:", type_train,
    " mae_start_ratio:", mae_start_ratio,
    " patience:", patience,
    " val_type:", val_type,
    " per:", percentage,
    " time:", time_tag,
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
}
if dataset not in DATASET_LOADERS:
    raise ValueError(f"Unsupported dataset '{args.d}'. Supported: {', '.join(DATASET_LOADERS)}")

df_full, y_full, df_test, y_test = DATASET_LOADERS[dataset]()
print("Full shape:", df_full.shape, " Test shape:", df_test.shape)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_noise_tag = noise_type
if noise_type in {"feature", "feature_outlier"}:
    _noise_tag = f"{noise_type}_r{noise_r}_k{noise_k}"
results_dir = os.path.join(
    BASE_DIR,
    f"res/{run_folder}/per_{percentage}/"
    f"{dataset}_{epochs}e_m{margin}_l{lamda}"
    f"_{mae_loss_mode}_train-{type_train}_mstart{mae_start_ratio}_noise_std{noise_std}_{_noise_tag}_conra{noise_ratio}__cat_{cat_noise_type}_catra{cat_noise_ratio}"
)
os.makedirs(results_dir, exist_ok=True)


def save_threshold(threshold, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Threshold saved to {path}")


def train_m01aectgan(epochs):
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
    meta_file = os.path.join(results_dir, f"{dataset}_m01aectgan_metadata.json")
    if os.path.exists(meta_file):
        os.remove(meta_file)
    metadata.save_to_json(meta_file)

    synthesizer = M01AECTGANSynthesizer(
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
        # M01AE
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
        # Joint
        type_train=type_train,
        mae_start_ratio=mae_start_ratio,
        patience=patience,
        # Misc
        dataset=dataset,
        path=results_dir,
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

    print(
        f"Train normal: {len(X_train_proc)}  "
        f"Val normal: {len(X_val)}  Test: {len(X_test)}"
    )

    synthesizer._fit(X_train_proc, X_train_proc, X_val)

    # ── Thresholds from validation ─────────────────────────────────────────
    thresholds = synthesizer.get_threshold(X_val)
    print("Thresholds:", thresholds)
    save_threshold(thresholds, results_dir, f"{dataset}_m01aectgan_thresholds.pkl")

    # ── Save final model ───────────────────────────────────────────────────
    model_path = os.path.join(results_dir, f"M01AECTGAN_{dataset}_{epochs}e.pkl")
    synthesizer.save(model_path)

    # ── Loss plots ────────────────────────────────────────────────────────
    loss_df = synthesizer.get_loss_values()
    loss_csv = os.path.join(results_dir, f"m01aectgan_loss_{epochs}e.csv")
    loss_df.to_csv(loss_csv, index=False)
    print(f"Loss log → {loss_csv}")

    # CTGAN loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_df["Epoch"], loss_df["Generator Loss"], label="Generator Loss")
    plt.plot(loss_df["Epoch"], loss_df["Discriminator Loss"], label="Discriminator Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"M01AECTGAN — CTGAN loss\n{dataset}  epochs={epochs}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_ctgan_{dataset}_{epochs}e.png"))
    plt.close()

    # M01AE loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_df["Epoch"], loss_df["MAE Loss"], label="MAE total")
    plt.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction")
    plt.plot(loss_df["Epoch"], lamda * loss_df["Latent Loss"], label="lambda*Latent")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"M01AECTGAN — M01AE loss\n{dataset}  epochs={epochs}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_mae_{dataset}_{epochs}e.png"))
    plt.close()

    # Validation loss + CTGAN active marker
    plt.figure(figsize=(10, 5))
    plt.plot(loss_df["Epoch"], loss_df["Validation Loss"], label="Val loss", color="steelblue")
    if "CTGAN Active" in loss_df.columns:
        switch_epoch = loss_df.loc[loss_df["CTGAN Active"] == 1, "Epoch"]
        if not switch_epoch.empty:
            plt.axvline(switch_epoch.iloc[0], color="orange", linestyle="--",
                        label=f"CTGAN active (epoch {switch_epoch.iloc[0]})")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"M01AECTGAN — Validation loss\n{dataset}  type_train={type_train}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_val_{dataset}_{epochs}e.png"))
    plt.close()

    print("Training complete.")
    return synthesizer, thresholds, X_test


def predict_and_evaluate(scores, y_true, threshold, save_dir, filename, score_mode="dist"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if score_mode == "std":
        mu, std = threshold
        thr_val = mu + 3.0 * std
    else:
        thr_val = float(threshold)

    y_true_bin = (np.asarray(y_true) != 0).astype(int)
    y_pred = (np.asarray(scores) > thr_val).astype(int)

    auc = roc_auc_score(y_true_bin, scores) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
    acc = accuracy_score(y_true_bin, y_pred) * 100
    f1  = f1_score(y_true_bin, y_pred, zero_division=0) * 100
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

    # RE score (mu + 3*std)
    predict_and_evaluate(re_errors, y_true, thresholds["threshold_re"],
                         save_dir, "test_mode_re.csv", score_mode="std")
    # Latent z-norm (mu + 3*std)
    predict_and_evaluate(z_norm, y_true, thresholds["threshold_z"],
                         save_dir, "test_mode_latent.csv", score_mode="std")
    # Latent 95th percentile
    predict_and_evaluate(z_norm, y_true, thresholds["threshold_95_latent"],
                         save_dir, "test_mode_latent95.csv", score_mode="dist")
    # Total 95th percentile
    predict_and_evaluate(total_scores, y_true, thresholds["threshold_95_total"],
                         save_dir, "test_mode_total95.csv", score_mode="dist")


def post_train_checkpoint_evaluation(results_dir, dataset, val_type="loss"):
    """Evaluate the best checkpoint selected by `val_type` ('loss' or 'auc').

    'loss' reads MAE_<dataset>_best.pkl (best validation loss — unchanged
    filenames for backward compatibility). 'auc' reads
    MAE_<dataset>_best_auc.pkl (best validation AUC, pseudo-anomaly built the
    same way as type_train). See sdv/single_table/m01aectgan.py::_save_best_model.
    """
    if val_type not in {"loss", "auc"}:
        raise ValueError("val_type must be 'loss' or 'auc'.")
    suffix = "" if val_type == "loss" else f"_{val_type}"

    print(f"Post-train evaluation on best checkpoint (val_type={val_type})...")
    x_test_path = os.path.join(results_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(results_dir, f"{dataset}_y_test.csv")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    best_model_path = os.path.join(checkpoint_dir, f"MAE_{dataset}_best{suffix}.pkl")
    best_thr_path   = os.path.join(checkpoint_dir, f"MAE_{dataset}_thresholds_best{suffix}.pkl")

    for p in [x_test_path, y_test_path, best_model_path, best_thr_path]:
        if not os.path.exists(p):
            print(f"[WARN] Missing: {p} — skip checkpoint eval.")
            return

    X_test_csv = pd.read_csv(x_test_path)
    y_test_csv = pd.read_csv(y_test_path)["label"]
    best_synth = M01AECTGANSynthesizer.load(best_model_path)
    with open(best_thr_path, "rb") as f:
        thr_re, thr_z, thr_95, thr_95_total = pickle.load(f)

    best_thresholds = {
        "threshold_re":        thr_re,
        "threshold_z":         thr_z,
        "threshold_95_latent": thr_95,
        "threshold_95_total":  thr_95_total,
    }
    run_evaluation(best_synth, X_test_csv, y_test_csv, best_thresholds, checkpoint_dir)
    print("Checkpoint evaluation done.")


if __name__ == "__main__":
    print("=== M01AECTGAN Training ===")
    synthesizer, thresholds, X_test = train_m01aectgan(epochs=epochs)
    run_evaluation(synthesizer, X_test, y_test, thresholds, results_dir)
    post_train_checkpoint_evaluation(results_dir, dataset, val_type)
    print("=== Done ===")

# =============================================================================
# GIẢI THÍCH THAM SỐ — train_M01AECTGAN.py
# =============================================================================
#
# M01AECTGAN: kết hợp CTGAN và M01AE trong joint training.
#   - CTGAN: sinh dữ liệu bao phủ toàn bộ normal distribution làm nguồn đa dạng.
#   - M01AE: two-sided margin loss (normal→‖z‖→0, anomaly→‖z‖→margin).
#   - _add_noise() áp lên mẫu CTGAN hoặc real_normal → pseudo-anomaly đa dạng hơn.
#
# =============================================================================
# NHÓM 1 — DATASET / CƠ BẢN
# =============================================================================
#
# --d  <str>   Tên dataset.
#              Các giá trị hợp lệ: unsw, cic_ids, nsl_kdd, cic_iot, cidds_001,
#              wsn_ds, bot_iot, ton_iot.
#              Mặc định: unsw
#
# --e  <int>   Số epoch huấn luyện joint (CTGAN + M01AE chạy song song mỗi epoch).
#              Mặc định: 200
#
# --b  <int>   Batch size cho cả CTGAN lẫn M01AE.
#              Mặc định: 128
#
# --per  <float ∈ (0,1]>
#              Tỉ lệ lấy mẫu từ tập normal training để huấn luyện.
#              Dùng khi muốn thử với tập nhỏ hơn để debug nhanh.
#              Mặc định: 1.0 (dùng toàn bộ)
#
# --seed  <int>
#              Random seed để tái lập kết quả (áp cho numpy, torch, random).
#              Mặc định: 42
#
# --time  <int>
#              Chỉ số lần chạy (run index), dùng để phân biệt thư mục kết quả
#              khi chạy nhiều lần với cùng seed. Ảnh hưởng đến tên folder:
#              m01ae_<type_train>_se_<seed>_t_<time>.
#              Mặc định: 1
#
# --n_max_samples  <int>
#              Số hàng tối đa lấy từ dataset lớn (bot_iot, ton_iot).
#              Giúp tránh OOM trên GPU khi dataset hàng triệu dòng.
#              Mặc định: 500000
#
# =============================================================================
# NHÓM 2 — M01AE (LOSS)
# =============================================================================
#
# --lae  <float>   (lambda)
#              Trọng số của latent loss (‖z‖) trong tổng loss M01AE:
#                  total = RE_loss + lae * latent_loss
#              Giá trị nhỏ → RE chiếm ưu thế; giá trị lớn → latent chiếm ưu thế.
#              Mặc định: 0.1
#
# --mae  <float>   (margin)
#              Ngưỡng margin trong two-sided loss:
#                  latent_loss = max(0, margin - ‖z_anomaly‖)²  (đẩy anomaly ra xa)
#                              + ‖z_normal‖²                    (kéo normal về 0)
#              Margin càng lớn → buộc anomaly phải xa gốc toạ độ nhiều hơn.
#              Mặc định: 1.0
#
# --mae_loss_mode  {origin, renormal}
#              Cách tính reconstruction loss (RE loss):
#                  renormal → chỉ tính RE trên normal samples [khuyến nghị]
#                  origin   → tính RE trên cả normal lẫn pseudo-anomaly
#              Mặc định: renormal
#
# --mae_lr  <float>
#              Learning rate cho Adam optimizer của M01AE (encoder + decoder).
#              CTGAN dùng learning rate riêng (generator_lr, discriminator_lr).
#              Mặc định: 1e-3
#
# =============================================================================
# NHÓM 3 — NOISE (continuous features)
# =============================================================================
#
# --noise_type  {none, global, feature, feature_outlier}
#              Kiểu noise thêm vào continuous features để tạo pseudo-anomaly:
#                  none            → không thêm noise (continuous giữ nguyên)
#                  global          → N(0, noise_std²) cộng vào mỗi dim được chọn
#                  feature         → N(0, (noise_std·σᵢ)²) — scale theo std thực tế
#                                    mỗi feature (σᵢ tính từ training data)
#                  feature_outlier → đặt = μᵢ + noise_r · noise_k · σᵢ (outlier cứng)
#              Mặc định: global
#
# --noise_std  <float>
#              Độ lệch chuẩn của Gaussian noise (dùng với noise_type=global/feature).
#              Với feature_outlier tham số này không có hiệu lực.
#              Mặc định: 0.1
#
# --noise_ratio  <float ∈ (0,1]>
#              Tỉ lệ continuous features được chọn để áp noise (masking ngẫu nhiên).
#              1.0 → tất cả continuous dims đều bị nhiễu; 0.5 → chỉ 50%.
#              Mặc định: 1.0
#
# --noise_r  <float>
#              Hệ số chiều (sign/direction) cho feature_outlier:
#                  x = μᵢ + noise_r · noise_k · σᵢ
#              noise_r=1 → outlier phía dương; noise_r=-1 → phía âm.
#              Mặc định: 1.0
#
# --noise_k  <float>
#              Hệ số scale cho feature_outlier:
#                  x = μᵢ + noise_r · noise_k · σᵢ
#              noise_k=3 → 3σ so với mean → nằm xa phân phối normal rõ rệt.
#              Mặc định: 3.0
#
# =============================================================================
# NHÓM 4 — NOISE (categorical features)
# =============================================================================
#
# --cat_noise_type  {none, shuffle, flip, combined}
#              Kiểu noise áp lên categorical features (one-hot encoded):
#                  none     → không áp (mặc định)
#                  shuffle  → hoán vị giá trị giữa các hàng trong batch
#                             (thay đổi category nhưng giữ one-hot hợp lệ)
#                  flip     → đảo 0↔1 trong span one-hot của mỗi feature
#                             (tốt cho binary categorical)
#                  combined → mỗi feature ngẫu nhiên chọn shuffle hoặc flip
#              Mặc định: none
#
# --cat_noise_ratio  <float ∈ (0,1]>
#              Tỉ lệ categorical feature spans được áp noise.
#              1.0 → tất cả categorical features; 0.3 → chỉ 30% features.
#              Mặc định: 1.0
#
# =============================================================================
# NHÓM 5 — JOINT TRAINING
# =============================================================================
#
# --type_train  {noise, std, noise_std}
#              Nguồn tạo pseudo-anomaly trong mỗi bước huấn luyện M01AE:
#
#                  noise     → _add_noise(real_normal)
#                              M01AE thuần; CTGAN chỉ train CTGAN loss, không đóng
#                              góp gì cho pseudo-anomaly của M01AE.
#
#                  std       → _add_noise(ctgan_sample)
#                              CTGAN sinh mẫu bao phủ vùng thưa của normal
#                              distribution; noise đẩy mẫu đó ra ngoài manifold.
#                              Pseudo-anomaly đa dạng hơn "noise" thuần.
#
#                  noise_std → 50% _add_noise(real_normal)
#                            + 50% _add_noise(ctgan_sample)
#                              Kết hợp cả hai nguồn. [khuyến nghị]
#
#              Lưu ý: CTGAN warmup (mae_start_ratio) áp dụng với std & noise_std —
#              trong giai đoạn warmup luôn fallback về "noise".
#              Mặc định: noise_std
#
# --mae_start_ratio  <float ∈ [0,1)>
#              Tỉ lệ epoch đầu để CTGAN warmup trước khi samples của nó được
#              dùng làm pseudo-anomaly:
#                  epoch < mae_start_ratio * epochs → fallback về "noise"
#                  epoch ≥ mae_start_ratio * epochs → dùng type_train đã chọn
#              Ví dụ: epochs=200, mae_start_ratio=0.25 → warmup 50 epoch.
#              Không có hiệu lực khi type_train="noise".
#              Mặc định: 0.25
#
# --patience  <int>
#              Early stopping: dừng huấn luyện khi validation loss không cải
#              thiện sau `patience` epoch liên tiếp.
#              Áp dụng cho cả CTGAN và M01AE vì joint training.
#              Mặc định: 50
#
# --val_type  {loss, auc}
#              Checkpoint nào dùng để đánh giá test set sau khi train xong.
#              Trong quá trình train, model LUÔN lưu CẢ HAI checkpoint:
#                  best-loss → checkpoints/MAE_<dataset>_best.pkl
#                  best-auc  → checkpoints/MAE_<dataset>_best_auc.pkl
#                              (AUC tính trên pseudo-anomaly val sinh ra THEO
#                               ĐÚNG type_train — nhất quán với train)
#              --val_type loss → post_train dùng best-loss checkpoint [mặc định]
#              --val_type auc  → post_train dùng best-auc checkpoint
#              Mặc định: loss
#
# =============================================================================
# NHÓM 6 — ĐẦU RA (output structure)
# =============================================================================
#
# Thư mục kết quả:
#   Syn_ctgan/res/m01ae_<type_train>_se_<seed>_t_<time>/per_<per>/
#   <dataset>_<e>e_m<mae>_l<lae>_<mae_loss_mode>_train-<type_train>
#   _mstart<mae_start_ratio>_noise_std<noise_std>_<noise_tag>_conra<noise_ratio>
#   __cat_<cat_noise_type>_catra<cat_noise_ratio>/
#   │
#   ├── m01aectgan_loss_<e>e.csv       — log loss mỗi epoch (CTGAN + MAE + Val + ValAUC)
#   ├── loss_ctgan_*.png               — Generator / Discriminator loss curve
#   ├── loss_mae_*.png                 — MAE total / RE / λ·Latent loss curve
#   ├── loss_val_*.png                 — Validation loss + CTGAN activation marker
#   ├── test_mode_re.csv               — kết quả test dùng RE score (mu+3σ)
#   ├── test_mode_latent.csv           — kết quả test dùng ‖z‖ score (mu+3σ)
#   ├── test_mode_latent95.csv         — kết quả test dùng ‖z‖ 95th percentile
#   ├── test_mode_total95.csv          — kết quả test dùng total score 95th percentile
#   ├── M01AECTGAN_<dataset>_<e>e.pkl  — final model (last epoch)
#   ├── <dataset>_m01aectgan_thresholds.pkl
#   └── checkpoints/
#       ├── MAE_<dataset>_best.pkl                — best checkpoint theo val loss
#       ├── MAE_<dataset>_best_epoch.txt
#       ├── MAE_<dataset>_thresholds_best.pkl
#       ├── MAE_<dataset>_best_auc.pkl            — best checkpoint theo val AUC
#       ├── MAE_<dataset>_best_auc_epoch.txt
#       └── MAE_<dataset>_thresholds_best_auc.pkl
#
# =============================================================================
# VÍ DỤ LỆNH CHẠY
# =============================================================================
#
# Baseline — M01AE thuần (không dùng CTGAN samples làm pseudo-anomaly):
#   python train_M01AECTGAN.py --d unsw --e 200 --b 128 \
#     --lae 0.1 --mae 1.0 --mae_loss_mode renormal \
#     --noise_type global --noise_std 0.1 --noise_ratio 0.5 \
#     --type_train noise --seed 42 --time 1 --val_type auc
#
# Recommended — noise_std (CTGAN + noise 50/50):
#   python train_M01AECTGAN.py --d unsw --e 200 --b 128 \
#     --lae 0.1 --mae 2.0 --mae_loss_mode renormal \
#     --noise_type global --noise_r 5 --noise_k 0.1 --noise_ratio 0.5 \
#     --cat_noise_type combined --cat_noise_ratio 0.1 \
#     --type_train noise_std --mae_start_ratio 0.25 \
#     --patience 50 --seed 42 --time 2 --val_type auc
#
# Dataset lớn (TON-IoT, giới hạn 500k hàng):
#   python train_M01AECTGAN.py --d ton_iot --e 150 --b 128 \
#     --n_max_samples 500000 \
#     --lae 0.1 --mae 2.0 --mae_loss_mode renormal \
#     --noise_type global --noise_r 5 --noise_k 0.1 --noise_ratio 0.5 \
#     --cat_noise_type combined --cat_noise_ratio 0.1 \
#     --type_train noise_std --mae_start_ratio 0.25 \
#     --patience 50 --seed 42 --time 2 --val_type auc
