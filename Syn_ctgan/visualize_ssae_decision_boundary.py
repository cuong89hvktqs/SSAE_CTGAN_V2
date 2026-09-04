# -*- coding: utf-8 -*-
"""SSAE (MAE01ScaleMax) anomaly-score map in latent PCA space, for a REAL,
already-trained checkpoint -- NO training happens here.

Originally adapted from PyOD's classic "compare_all_models" example (grid +
contourf decision-region levels). That approach was dropped: it requires
evaluating the score function at synthetic grid points, but SSAE has no
native 2D input space (unlike PyOD's toy demo) -- grid points had to be
reconstructed from a lossy 2D-PCA-of-latent-space projection, and decoding
those off-manifold points produced meaningless scores almost everywhere
except right around the real data, making the contour background pure noise.

This version keeps the same spirit (visualize where the model's decision
boundary falls relative to normal/attack data) but stays fully truthful to
what the model actually knows: only REAL test points are scored (via
get_score, the exact function that produces every reported result), each
point is colored by its own real anomaly score on a diverging colormap
centered at the real decision threshold (threshold_95_total, loaded from the
checkpoint) -- blue-ish = below threshold (would be classified normal),
red-ish = above (classified anomalous) -- and marker shape encodes the
GROUND-TRUTH label (circle = true normal, x = true attack), so mismatches
between color and shape are visually exactly the classification errors.

Usage: python visualize_ssae_decision_boundary.py [--dataset all] [--seed 42]
Output: res/figures_v2/decision_boundary/<dataset>_seed<N>_score_map.png
"""

from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

from pathlib import Path

import cloudpickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from visualize_scalemax import load_model, get_latent_z
from sklearn.decomposition import PCA

RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "decision_boundary"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
TIME_TAG = "04"   # main default-config run, same one used for Table 3 / score-distribution figures

MAX_PER_CLASS = 3000  # cap for PCA fit / scatter speed
CMAP = plt.cm.coolwarm  # blue = below threshold (normal-side), red = above (anomalous-side)


def find_run_dir(dataset, seed):
    pattern = str(RES_DIR / f"mae01scale_max_se_{seed}_t_{TIME_TAG}" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_threshold_95_total(thr_path):
    with open(thr_path, "rb") as f:
        _thr_re, _thr_z, _thr_95, thr_95_total = cloudpickle.load(f)
    return thr_95_total


def _cap(arr, n, rng):
    if len(arr) <= n:
        return arr
    idx = rng.choice(len(arr), size=n, replace=False)
    return arr[idx]


def run(dataset, seed, device="cpu"):
    run_dir = find_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] {dataset} seed={seed}: no noise_std run found for t_{TIME_TAG}")
        return False

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    x_val_path  = os.path.join(run_dir, f"{dataset}_X_val_normal.csv")
    model_path  = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")
    thr_path    = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_thresholds_best_auc.pkl")
    for p in [x_test_path, y_test_path, x_val_path, model_path, thr_path]:
        if not os.path.exists(p):
            print(f"  [skip] {dataset} seed={seed}: missing {p}")
            return False

    print("  loading model (no training) + scoring test set...")
    model = load_model(model_path, device)
    threshold = load_threshold_95_total(thr_path)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    X_val_normal = pd.read_csv(x_val_path)

    _, _, total_scores, y_true = model.get_score(X_test, y_test)
    y_true = y_true.astype(int)
    y_pred = (total_scores > threshold).astype(int)
    n_errors = int((y_pred != y_true).sum())
    print(f"  n_test={len(y_true)}  threshold={threshold:.4f}  n_errors={n_errors}")

    print("  encoding to M01AE latent space...")
    z_val_normal = get_latent_z(model, X_val_normal)
    centroid = z_val_normal.mean(axis=0, keepdims=True)
    z_test = get_latent_z(model, X_test)

    rng = np.random.default_rng(seed)
    idx_normal = _cap(np.where(y_true == 0)[0], MAX_PER_CLASS, rng)
    idx_attack = _cap(np.where(y_true == 1)[0], MAX_PER_CLASS, rng)
    fit_pool = np.concatenate([z_test[idx_normal], z_test[idx_attack]], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(fit_pool - centroid)

    def project(z):
        return (z - centroid) @ pca.components_.T

    coords_normal = project(z_test[idx_normal])
    coords_attack = project(z_test[idx_attack])
    scores_normal = total_scores[idx_normal]
    scores_attack = total_scores[idx_attack]

    all_scores = np.concatenate([scores_normal, scores_attack])
    vmin, vmax = float(all_scores.min()), float(all_scores.max())
    vcenter = float(np.clip(threshold, vmin + 1e-9, vmax - 1e-9))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    b = ax.scatter(coords_normal[:, 0], coords_normal[:, 1], c=scores_normal,
                   cmap=CMAP, norm=norm, marker='o', s=16,
                   edgecolor='k', linewidths=0.3, label='true normal')
    ax.scatter(coords_attack[:, 0], coords_attack[:, 1], c=scores_attack,
              cmap=CMAP, norm=norm, marker='x', s=22, linewidths=0.9,
              label='true attack')

    cbar = fig.colorbar(b, ax=ax)
    cbar.ax.axhline(vcenter, color='black', linewidth=1.2, linestyle='--')
    cbar.set_label('Anomaly score (dashed line = threshold)')

    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_xlabel(f"PC1  --  errors: {n_errors} / {len(y_true)}")
    ax.set_ylabel("PC2")
    ax.set_title(f"SSAE (MAE01ScaleMax) score map -- {dataset.upper()}")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_seed{seed}_score_map.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser("SSAE anomaly score map (loaded checkpoint, no training)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"\n[{name}  seed={args.seed}]")
        try:
            run(name, args.seed)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
