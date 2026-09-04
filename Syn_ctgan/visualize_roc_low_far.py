"""ROC curve zoomed into the low-FAR region, for SSAE (MAE01ScaleMax) --
NO training happens here, only real checkpoints are loaded and scored (same
convention as visualize_anomaly_score_distribution.py /
visualize_ssae_decision_boundary.py: MAE_<dataset>_best_auc.pkl, run t_04).

Why this figure: AUC alone hides what actually matters for an IDS -- how
many true attacks are caught while the false-alarm rate (FAR = FPR) stays
low, since a real deployment can't tolerate flooding operators with false
alerts. This is exactly the TPR@FAR metric already used per-attack in this
project's val_perattack_MAE_*_tpr@far.ipynb notebooks, just shown here as a
continuous curve instead of single-point table entries.

Plot: one combined figure, one ROC curve per dataset (log-scaled x-axis,
clipped to FAR <= 5%), each annotated with:
  - full-range AUC and partial AUC restricted to FAR<=5% (pAUC, via
    sklearn's roc_auc_score(..., max_fpr=0.05) -- the standard McClish-
    normalized partial-AUC metric) in the legend;
  - a star marking the REAL operating point (FPR, TPR) at the model's own
    threshold_95_total -- i.e. where it actually operates at test time, not
    a hypothetical point on the curve;
  - two shared reference lines at FAR=1% and FAR=5%.
A companion table (printed + saved as CSV) reports interpolated TPR at
FAR in {0.1%, 1%, 5%} per dataset -- the actual numbers behind the curve.

Usage: python visualize_roc_low_far.py [--dataset all] [--seed 42]
Output: res/figures_v2/roc_low_far/roc_low_far.png
        res/figures_v2/roc_low_far/tpr_at_far.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from visualize_scalemax import load_model
from visualize_anomaly_score_distribution import find_run_dir, load_threshold_95_total

RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "roc_low_far"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
DATASET_LABEL = {
    "unsw": "UNSW-NB15", "nsl_kdd": "NSL-KDD",
    "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS",
}
DATASET_COLOR = {
    "unsw": "#1f77b4", "nsl_kdd": "#2ca02c",
    "cic_iot": "#d62728", "wsn_ds": "#9467bd",
}

FAR_MAX = 0.05          # zoom cutoff: FAR <= 5%
FAR_XMIN = 1e-4          # left edge of the log-x axis
FAR_REPORT_POINTS = [0.001, 0.01, 0.05]  # 0.1%, 1%, 5%


def score_dataset(dataset, seed, device="cpu"):
    run_dir = find_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] {dataset} seed={seed}: no noise_std run found")
        return None

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    model_path  = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")
    thr_path    = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_thresholds_best_auc.pkl")
    for p in [x_test_path, y_test_path, model_path, thr_path]:
        if not os.path.exists(p):
            print(f"  [skip] {dataset} seed={seed}: missing {p}")
            return None

    print(f"[{dataset}] loading model (no training) + scoring test set...")
    model = load_model(model_path, device)
    threshold = load_threshold_95_total(thr_path)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    _, _, total_scores, y_true = model.get_score(X_test, y_test)
    y_true = y_true.astype(int)

    return total_scores, y_true, threshold


def operating_point(total_scores, y_true, threshold):
    y_pred = (total_scores > threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return fpr, tpr


def run(datasets, seed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

    rows = []
    for dataset in datasets:
        result = score_dataset(dataset, seed)
        if result is None:
            continue
        total_scores, y_true, threshold = result
        if len(np.unique(y_true)) < 2:
            print(f"  [skip] {dataset}: only one class present in y_true")
            continue

        n_total = len(y_true)
        n_normal = int((y_true == 0).sum())
        n_attack = int((y_true == 1).sum())
        print(f"  {DATASET_LABEL[dataset]}: test set n_total={n_total}  "
              f"n_normal={n_normal}  n_attack={n_attack}")

        fpr, tpr, _ = roc_curve(y_true, total_scores)
        auc_full = roc_auc_score(y_true, total_scores)
        p_auc = roc_auc_score(y_true, total_scores, max_fpr=FAR_MAX)
        op_fpr, op_tpr = operating_point(total_scores, y_true, threshold)

        color = DATASET_COLOR[dataset]
        label = (f"{DATASET_LABEL[dataset]}  "
                f"(AUC={auc_full * 100:.2f}%, pAUC$_{{\\leq 5\\%}}$={p_auc * 100:.2f}%)")
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

        # Operating point can fall beyond the FAR<=5% zoom window (test-set
        # FAR can drift above the validation-calibrated 5% target) -- clamp
        # the marker to the right edge and annotate the real value instead
        # of letting it silently clip out of view.
        plot_op_fpr = min(op_fpr, FAR_MAX)
        ax.plot(plot_op_fpr, op_tpr, marker="*", markersize=13, color=color,
               markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
               clip_on=False, zorder=5)
        if op_fpr > FAR_MAX:
            ax.annotate(f"FAR={op_fpr * 100:.1f}%", xy=(plot_op_fpr, op_tpr),
                       xytext=(-6, 0), textcoords="offset points",
                       fontsize=7.5, color=color, ha="right", va="center",
                       fontweight="bold")

        report = {"dataset": DATASET_LABEL[dataset], "AUC_full_%": auc_full * 100,
                  f"pAUC_far<={FAR_MAX * 100:.0f}%_%": p_auc * 100,
                  "operating_FAR_%": op_fpr * 100, "operating_TPR_%": op_tpr * 100}
        for far in FAR_REPORT_POINTS:
            tpr_at = float(np.interp(far, fpr, tpr))
            report[f"TPR@FAR={far * 100:g}%_%"] = tpr_at * 100
            print(f"  {DATASET_LABEL[dataset]}: TPR@FAR={far * 100:g}% = {tpr_at * 100:.2f}%")
        print(f"  {DATASET_LABEL[dataset]}: AUC={auc_full * 100:.2f}%  "
              f"pAUC(FAR<=5%)={p_auc * 100:.2f}%  "
              f"operating point (FAR={op_fpr * 100:.2f}%, TPR={op_tpr * 100:.2f}%)")
        rows.append(report)
        

    for far in (0.01, FAR_MAX):
        ax.axvline(far, color="#888888", linestyle=":", linewidth=1, zorder=0)
        ax.text(far, 0.02, f"{far * 100:g}%", fontsize=8, color="#666666",
               ha="center", va="bottom", rotation=90,
               transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_xlim(FAR_XMIN, FAR_MAX)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Alarm Rate (FAR / FPR), log scale")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC -- low-FAR region (FAR $\\leq$ 5%)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    ax.scatter([], [], marker="*", s=120, color="white", edgecolor="black",
              label="operating point (threshold_95_total)")
    fig.tight_layout()

    out_path = OUT_DIR / f"ssae_roc_low_far_seed{seed}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nsaved -> {out_path}")

    if rows:
        csv_path = OUT_DIR / "tpr_at_far.csv"
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"saved -> {csv_path}")
    print(dataset, "n_test =", len(y_true),
      "n_normal =", int((y_true==0).sum()),
      "n_attack =", int((y_true==1).sum()))

def main():
    parser = argparse.ArgumentParser("SSAE ROC curve zoomed to low-FAR region")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    run(datasets, args.seed)


if __name__ == "__main__":
    main()
