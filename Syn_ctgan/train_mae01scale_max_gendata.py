"""Variant of train_mae01scale_max.py that ALSO dumps generated_data_snapshots.csv
(real_normal / noise_pseudo / ctgan_raw / ctgan_noise, every SNAPSHOT_EVERY
epochs once the CTGAN gate opens) -- the file MAE01ScaleMax normally disables
(see ctgan/synthesizers/mae01scale_max.py: _save_generated_snapshot is a
no-op override there, to cut per-run disk usage across many seeds).

Why this script exists: the original t_03 run's generated_data_snapshots.csv
files (seed 42, old folder layout) were deleted by reorganize_ssae_folders.py
(delete_generated_data_files(), matches any "*generated_data*" filename) when
that run was migrated to the current folder layout. They cannot be recovered
-- CTGAN sampling isn't seeded, so re-running the SAME checkpoint doesn't
reproduce the SAME generated rows either. This script re-trains fresh
(same config as t_03/t_04: m=2.0, lambda=0.1, noise_std, seed 42) with the
snapshot dump restored, into a NEW, separate folder
(res/t_03_gendata_SSAE_se_<seed>/...) so it never collides with the existing
res/t_03_SSAE_se_42/ or res/t_04_SSAE_se_*/ results.

The restore is a pure monkeypatch, done ONLY in this script -- no edits to
ctgan/synthesizers/mae01scale_max.py or sdv/single_table/mae01scale_max.py:

    class MAE01ScaleMaxGenData(MAE01ScaleMax):
        _save_generated_snapshot = MAE01Scale._save_generated_snapshot  # un-disable

    sdv_mae01scale_max_mod.MAE01ScaleMax = MAE01ScaleMaxGenData

sdv/single_table/mae01scale_max.py's _fit() does
`self._model = MAE01ScaleMax(**self._model_kwargs)`, resolving that name from
its own module globals at CALL time -- reassigning the module attribute
above is enough to make it instantiate MAE01ScaleMaxGenData instead, with
every other behaviour (the OR-gate, gate_fixed_epoch, ...) unchanged since
it's a subclass, not a reimplementation.

Everything else (CLI args, folder layout under the run dir, best_auc
checkpoint saving/eval, per-attack eval, loss plots) is IDENTICAL to
train_mae01scale_max.py -- only the run_folder prefix and the snapshot
restore differ. See that script's own docstring/comments for the rest.

Usage (same CLI as train_mae01scale_max.py, minus --time which is fixed):
    python train_mae01scale_max_gendata.py --d unsw --seed 42 \
        --e 150 --b 128 --lae 0.1 --mae 2.0 --mae_loss_mode renormal \
        --type_train noise_std --noise_std 0.3 --noise_type global --noise_ratio 0.5 \
        --cat_noise_type combined --cat_noise_ratio 0.1 --patience 50 \
        --noise_scale 1.5 --gate_stability 0.05 --gate_window 10 \
        --cov_floor 0.05 --eps_zero 0.1 --tol 0.08
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
import sdv.single_table.mae01scale_max as sdv_mae01scale_max_mod
from sdv.single_table.mae01scale_max import MAE01ScaleMaxSynthesizer
from ctgan.synthesizers.mae01scale_max import MAE01ScaleMax
from ctgan.synthesizers.mae01scale import MAE01Scale
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from val_perattack_mae01scale_max import ATTACK_LOADERS, FAR_LEVELS, evaluate_ids_metrics

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
# Monkeypatch: restore the generated_data_snapshots.csv dump for this run only
# =============================================================================

class MAE01ScaleMaxGenData(MAE01ScaleMax):
    """Same as MAE01ScaleMax (OR-gate etc. unchanged) but with the periodic
    generated_data_snapshots.csv dump restored (MAE01Scale's original
    implementation, un-overridden)."""
    _save_generated_snapshot = MAE01Scale._save_generated_snapshot


sdv_mae01scale_max_mod.MAE01ScaleMax = MAE01ScaleMaxGenData
print("[patched] sdv.single_table.mae01scale_max.MAE01ScaleMax -> MAE01ScaleMaxGenData "
      "(generated_data_snapshots.csv dump restored for this run)")


# =============================================================================
# CLI (identical to train_mae01scale_max.py, minus --time: this script always
# writes into res/t_03_gendata_SSAE_se_<seed>/, never the real t_03/t_04 runs)
# =============================================================================

parser = argparse.ArgumentParser(description="Train MAE01ScaleMax with generated_data_snapshots.csv restored")

parser.add_argument("--d",             type=str,   default="unsw")
parser.add_argument("--e",             type=int,   default=150)
parser.add_argument("--b",             type=int,   default=128)
parser.add_argument("--per",           type=float, default=1.0)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--n_max_samples", type=int,   default=500_000)

parser.add_argument("--lae",           type=float, default=0.1)
parser.add_argument("--mae",           type=float, default=2.0)
parser.add_argument("--mae_loss_mode", type=str,   default="renormal",
                    choices=["origin", "renormal"])
parser.add_argument("--mae_lr",        type=float, default=1e-3)

parser.add_argument("--noise_std",   type=float, default=0.3)
parser.add_argument("--noise_ratio", type=float, default=0.5)
parser.add_argument("--noise_type",  type=str,   default="global",
                    choices=["none", "global", "feature", "feature_outlier"])
parser.add_argument("--noise_r",     type=float, default=1.0)
parser.add_argument("--noise_k",     type=float, default=3.0)
parser.add_argument("--cat_noise_type",  type=str, default="combined",
                    choices=["none", "shuffle", "flip", "combined"])
parser.add_argument("--cat_noise_ratio", type=float, default=0.1)

parser.add_argument("--type_train", type=str, default="noise_std",
                    choices=["noise", "std", "noise_std"])
parser.add_argument("--patience",   type=int, default=50)

parser.add_argument("--noise_scale",    type=float, default=1.5)
parser.add_argument("--gate_stability", type=float, default=0.05)
parser.add_argument("--gate_window",    type=int,   default=10)
parser.add_argument("--max_clusters",   type=int,   default=None)
parser.add_argument("--snapshot_stride", type=int,  default=None,
                    help="take every Nth normal-train row when saving "
                         "generated_data_snapshots.csv (1 = all rows). "
                         "Default: 1, except nb_iot which defaults to 20.")

parser.add_argument("--cov_floor", type=float, default=0.05)
parser.add_argument("--eps_zero", type=float, default=0.1)
parser.add_argument("--gate_fixed_epoch", type=int, default=None)
parser.add_argument("--tol", type=float, default=0.08)

args = parser.parse_args()

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

max_clusters = args.max_clusters if args.max_clusters is not None else 10

if args.snapshot_stride is not None:
    snapshot_stride = args.snapshot_stride
elif dataset == "nb_iot":
    snapshot_stride = 20
else:
    snapshot_stride = 1

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
run_folder      = f"t_03_gendata_SSAE_se_{seed}"   # fixed -- never collides with real t_03/t_04

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
    "MAE01ScaleMax (gendata) | dataset:", dataset,
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
    f"_{mae_loss_mode}_{type_train}_gw{gate_window}_gs{gate_stability}"
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

    assert isinstance(synthesizer._model, MAE01ScaleMaxGenData), (
        "monkeypatch did not take effect -- synthesizer._model is "
        f"{type(synthesizer._model)}, expected MAE01ScaleMaxGenData"
    )

    loss_df  = synthesizer._model.get_loss_values()
    loss_csv = os.path.join(results_dir, f"mae01scale_max_loss_{epochs}e.csv")
    loss_df.to_csv(loss_csv, index=False)

    _gate_epoch = synthesizer._model._ctgan_gate_epoch

    def _vline(ax, epoch, label_prefix):
        if epoch is not None:
            ax.axvline(epoch, color="orange", linestyle="--",
                       label=f"{label_prefix} (epoch {epoch})")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_df["Epoch"], loss_df["Generator Loss"],     label="Generator Loss")
    ax.plot(loss_df["Epoch"], loss_df["Discriminator Loss"], label="Discriminator Loss")
    ax.plot(loss_df["Epoch"], loss_df["EMA Discriminator Loss"],
            label="EMA Disc Loss", linestyle="--", alpha=0.7)
    _vline(ax, _gate_epoch, "Gate open")
    ax.set(xlabel="Epoch", ylabel="Loss",
           title=f"MAE01ScaleMax (gendata) — CTGAN loss\n{dataset}")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"loss_ctgan_{dataset}_{epochs}e.png"))
    plt.close(fig)

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

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_df["Epoch"], loss_df["Validation Loss"], label="Val loss")
    _vline(ax, _gate_epoch, "Gate open")
    ax.set(xlabel="Epoch", ylabel="Loss",
           title=f"MAE01ScaleMax (gendata) — Validation loss\n{dataset}")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"loss_val_{dataset}_{epochs}e.png"))
    plt.close(fig)

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
                title=f"MAE01ScaleMax (gendata) — Z-norm self-scaling\n{dataset}")
        ax2.set_ylabel("mean ||z||")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"scale_noise_{dataset}_{epochs}e.png"))
        plt.close(fig)

    snapshot_path = os.path.join(results_dir, "generated_data_snapshots.csv")
    if os.path.exists(snapshot_path):
        n_rows = sum(1 for _ in open(snapshot_path, encoding="utf-8")) - 1
        print(f"generated_data_snapshots.csv: {n_rows} row(s) -> {snapshot_path}")
    else:
        print(f"[WARN] generated_data_snapshots.csv NOT found at {snapshot_path} "
              f"-- CTGAN gate may never have opened for this run (check loss_ctgan_*.png).")

    print("Training complete.")
    return synthesizer, X_test


# =============================================================================
# Evaluation (identical to train_mae01scale_max.py)
# =============================================================================

def evaluate_best_auc_checkpoint(results_dir, dataset):
    print("\n--- Post-train eval: best_auc checkpoint ---")
    x_val_path     = os.path.join(results_dir, f"{dataset}_X_val_normal.csv")
    y_val_path     = os.path.join(results_dir, f"{dataset}_y_val_normal.csv")
    x_test_path    = os.path.join(results_dir, f"{dataset}_X_test.csv")
    y_test_path    = os.path.join(results_dir, f"{dataset}_y_test.csv")
    model_path     = os.path.join(results_dir, f"MAE_{dataset}_best_auc.pkl")
    save_dir       = os.path.join(results_dir, "best_auc")

    for p in [x_val_path, y_val_path, x_test_path, y_test_path, model_path]:
        if not os.path.exists(p):
            print(f"[WARN] Missing: {p} -- skip.")
            return None

    X_val_csv  = pd.read_csv(x_val_path)
    y_val_csv  = pd.read_csv(y_val_path)["label"]
    X_test_csv = pd.read_csv(x_test_path)
    y_test_csv = pd.read_csv(y_test_path)["label"]

    with open(model_path, "rb") as f:
        best_synth = cloudpickle.load(f)

    _, _, val_total,  _      = best_synth.get_score(X_val_csv, y_val_csv)
    _, _, test_total, y_true = best_synth.get_score(X_test_csv, y_test_csv)
    y_true_bin = (np.asarray(y_true) != 0).astype(int)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "test_mode_total.csv")
    val_thresholds = {}
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for percentile, far_label in ((95, "TPR@FAR=5%"), (99, "TPR@FAR=1%")):
            threshold = float(np.percentile(val_total, percentile))
            val_thresholds[percentile] = threshold
            y_pred = (np.asarray(test_total) > threshold).astype(int)

            auc = roc_auc_score(y_true_bin, test_total) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
            mcc = matthews_corrcoef(y_true_bin, y_pred) * 100
            tpr = recall_score(y_true_bin, y_pred, zero_division=0) * 100

            w.writerow([f"Threshold_{percentile}", f"{threshold:.6f}"])
            w.writerow([far_label,                 f"{tpr:.6f}"])
            w.writerow([f"AUC_{percentile}",        f"{auc:.6f}"])
            w.writerow([f"MCC_{percentile}",        f"{mcc:.6f}"])

            print(f"  [{far_label} @ p{percentile}] AUC_{percentile}={auc:.2f}%  "
                  f"MCC_{percentile}={mcc:.2f}%  {far_label}={tpr:.2f}%  thr={threshold:.6f}")

    print(f"Checkpoint eval done -> {save_path}")
    return best_synth, X_test_csv, y_test_csv, np.asarray(test_total), y_true_bin, val_thresholds


def evaluate_per_attack(results_dir, dataset, seed, test_total, y_true_bin, val_thresholds):
    loader = ATTACK_LOADERS.get(dataset)
    if loader is None:
        print(f"[WARN] No attack-label loader for dataset '{dataset}' -- skip per_attack_seeds.csv.")
        return

    attack_labels, normal_tag = loader(seed)
    if len(attack_labels) != len(test_total):
        print(f"[WARN] attack labels ({len(attack_labels)}) != test scores "
              f"({len(test_total)}) -- skip per_attack_seeds.csv.")
        return

    normal_mask = (attack_labels == normal_tag).to_numpy()
    n_normal = int(normal_mask.sum())
    attacks = sorted(a for a in attack_labels.unique() if a != normal_tag)

    scores_normal = test_total[normal_mask]
    normal_thresholds = {
        far: np.percentile(scores_normal, 100 * (1 - far)) for far in FAR_LEVELS
    }

    rows = []
    for attack in attacks:
        atk_mask = (attack_labels == attack).to_numpy()
        sub_scores = np.concatenate([scores_normal, test_total[atk_mask]])
        sub_y = np.concatenate([
            np.zeros(n_normal, dtype=int),
            np.ones(int(atk_mask.sum()), dtype=int),
        ])
        metrics = evaluate_ids_metrics(sub_y, sub_scores, normal_thresholds, val_thresholds)
        rows.append({
            "Dataset": dataset, "Seed": seed, "Attack": attack,
            "N": int(atk_mask.sum()), "Score": "total", **metrics,
        })

    if not rows:
        print(f"[WARN] No attack types found for '{dataset}' -- skip per_attack_seeds.csv.")
        return

    save_dir = os.path.join(results_dir, "best_auc")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "per_attack_seeds.csv")
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"Per-attack eval done -> {save_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=== MAE01ScaleMax Training (with generated_data_snapshots.csv) ===")
    synthesizer, X_test = train_mae01scale_max(epochs=epochs)
    eval_result = evaluate_best_auc_checkpoint(results_dir, dataset)
    if eval_result is not None:
        _, _, _, test_total, y_true_bin, val_thresholds = eval_result
        evaluate_per_attack(results_dir, dataset, seed, test_total, y_true_bin, val_thresholds)

    # Same intermediate-CSV cleanup as train_mae01scale_max.py --
    # generated_data_snapshots.csv is NOT in this list, so it survives.
    for _fname in (
        f"{dataset}_X_val_normal.csv", f"{dataset}_y_val_normal.csv",
        f"{dataset}_X_test.csv", f"{dataset}_y_test.csv",
    ):
        _fpath = os.path.join(results_dir, _fname)
        if os.path.exists(_fpath):
            os.remove(_fpath)
    print("=== Done ===")
