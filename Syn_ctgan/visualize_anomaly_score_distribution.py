"""Anomaly score distribution (histogram + KDE) for SSAE (MAE01ScaleMax):
normal vs. attack. Mirrors visualize_score_distribution_clan.py's plotting
pipeline exactly (colors, bins, KDE, threshold line, AUC-in-title, legend
style, figure size/DPI) so the two figures are directly comparable as a
CLAN-vs-SSAE pair -- only the underlying score/threshold computation differs,
because it is genuinely a different scoring mechanism per model:

    CLAN:  score(x) = -cosine_similarity(model(x), centroid)   (embedding space)
    SSAE:  score(x) = reconstruction_error + lamda * ||z||^2   (total95 score,
           see TH/latex_mae01scale_max_algorithm_section.tex's Anomaly Scoring
           section -- the primary metric adopted for all reported results),
           via the model's own get_score(X, y) (ctgan/synthesizers/m01aectgan.py).

Checkpoint used: MAE_<dataset>_best_auc.pkl -- matches the checkpoint-
selection criterion adopted in TH/latex_checkpoint_selection_comparison.tex
(best-auc, for stability across seeds), so this figure is consistent with
what the reported results tables actually use. Unlike CLAN's script (which
recomputes its percentile threshold on the fly from validation-normal
scores), SSAE's threshold_95_total is loaded from the checkpoint's saved
thresholds pickle -- same formula (95th percentile of validation-normal
scores), but this is the exact number used by all of SSAE's reported
test_mode_total95.csv results, so recomputing it independently here would
be redundant and risks drifting from the authoritative value.

Run t_04 (per_1.0, noise_std) is used -- the main default-config run, same
one used for Table 3 (checkpoint comparison).

Usage: python visualize_anomaly_score_distribution.py [--dataset all] [--seeds 42,43,44]
Output: res/figures_v2/anomaly_score_dist/<dataset>_seed<N>_score_dist.png
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cloudpickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

from visualize_scalemax import load_model

RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "anomaly_score_dist"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
SEEDS = [42, 43, 44]
TIME_TAG = "04"   # main default-config run, same one used for Table 3 (checkpoint comparison)

# Same palette/bins/figure params as visualize_score_distribution_clan.py
NORMAL_COLOR = "#3878B4"  # blue
ATTACK_COLOR = "#BF1F1F"  # red
N_BINS = 60
DPI = 150
FIG_SIZE = (7, 4.5)


def find_run_dir(dataset, seed):
    pattern = str(RES_DIR / f"mae01scale_max_se_{seed}_t_{TIME_TAG}" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_threshold_95_total(thr_path):
    with open(thr_path, "rb") as f:
        thr_re, thr_z, thr_95, thr_95_total = cloudpickle.load(f)
    return thr_95_total


def _kde_curve(scores, x_grid):
    if len(scores) < 2 or np.allclose(scores, scores[0]):
        return None
    return gaussian_kde(scores)(x_grid)


def run(dataset, seed, device="cpu"):
    run_dir = find_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] {dataset} seed={seed}: no noise_std run found for t_{TIME_TAG}")
        return False

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    model_path  = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")
    thr_path    = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_thresholds_best_auc.pkl")
    for p in [x_test_path, y_test_path, model_path, thr_path]:
        if not os.path.exists(p):
            print(f"  [skip] {dataset} seed={seed}: missing {p}")
            return False

    print("  loading model + scoring test set...")
    model = load_model(model_path, device)
    threshold = load_threshold_95_total(thr_path)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    _, _, test_scores, y_true = model.get_score(X_test, y_test)
    y_true = y_true.astype(int)

    normal_scores = test_scores[y_true == 0]
    attack_scores = test_scores[y_true == 1]
    auc = roc_auc_score(y_true, test_scores) * 100 if len(np.unique(y_true)) > 1 else float("nan")

    print(f"  normal={len(normal_scores)}  attack={len(attack_scores)}  "
          f"threshold={threshold:.4f}  AUC={auc:.2f}")

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    lo = min(normal_scores.min(), attack_scores.min()) if len(attack_scores) else normal_scores.min()
    hi = max(normal_scores.max(), attack_scores.max()) if len(attack_scores) else normal_scores.max()
    pad = 0.05 * (hi - lo + 1e-8)
    x_grid = np.linspace(lo - pad, hi + pad, 500)

    ax.hist(normal_scores, bins=N_BINS, range=(lo - pad, hi + pad), density=True,
           color=NORMAL_COLOR, alpha=0.35, label="Normal")
    kde_n = _kde_curve(normal_scores, x_grid)
    if kde_n is not None:
        ax.plot(x_grid, kde_n, color=NORMAL_COLOR, linewidth=1.8)

    if len(attack_scores):
        ax.hist(attack_scores, bins=N_BINS, range=(lo - pad, hi + pad), density=True,
               color=ATTACK_COLOR, alpha=0.35, label="Attack")
        kde_a = _kde_curve(attack_scores, x_grid)
        if kde_a is not None:
            ax.plot(x_grid, kde_a, color=ATTACK_COLOR, linewidth=1.8)

    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.3,
              label="Threshold")

    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset.upper()}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_seed{seed}_score_dist.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser("SSAE anomaly score distribution (normal vs attack)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS])
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    names = DATASETS if args.dataset == "all" else [args.dataset]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, skipped = 0, 0
    for name in names:
        for seed in seeds:
            print(f"\n[{name}  seed={seed}]")
            try:
                if run(name, seed):
                    ok += 1
                else:
                    skipped += 1
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                skipped += 1

    print(f"\ndone: {ok} dataset/seed pairs plotted, {skipped} skipped")


if __name__ == "__main__":
    main()
