"""Overall (whole-test-set, not per-attack) AUC / F1 / MCC summary for
NeuTraL-AD, mean+-std across seeds -- companion to eval_neutralad_perattack.py
(which only reports AUC/PR-AUC/TPR@FAR, no F1/MCC, mirroring SSAE's own
val_perattack_mae01scale_max.py convention).

Recomputes straight from the test_scores.npy / test_labels_out.npy each
run_neutralad.py run already saved -- no retraining needed, so this also
backfills MCC for runs trained before test_mcc was added to run_neutralad.py.

F1/MCC threshold: reuses NeuTraL-AD's OWN utils.compute_pre_recall_f1()
(unmodified, imported directly -- percentile(scores, 100 * normal_ratio) cut)
so F1 here matches exactly what run_neutralad.py's test_f1 already reported;
MCC is evaluated at that identical cut for a directly comparable number.

Usage: python aggregate_neutralad_overall.py [--per 1.0] [--datasets unsw,nsl_kdd,cic_iot,wsn_ds] [--seeds 42,43,44]
Output: NeuTraL-AD/results_ssae/per_<per>/overall_summary_seeds.csv
        NeuTraL-AD/results_ssae/per_<per>/overall_summary.csv   (mean+-std)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

NEUTRALAD_ROOT = Path(__file__).parent.parent / "NeuTraL-AD"
if str(NEUTRALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUTRALAD_ROOT))
from utils import compute_pre_recall_f1  # noqa: E402  (NeuTraL-AD's own, unmodified)

RESULTS_ROOT = NEUTRALAD_ROOT / "results_ssae"
DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
METRIC_COLS = ["AUC", "F1", "MCC"]


def mcc_at_paper_threshold(labels, scores):
    normal_ratio = (labels == 0).sum() / len(labels)
    threshold = np.percentile(scores, 100 * normal_ratio)
    pred = (scores > threshold).astype(int)
    return matthews_corrcoef(labels, pred)


def evaluate_run(dataset, seed, per):
    run_dir = RESULTS_ROOT / f"per_{per}" / dataset / f"seed{seed}"
    scores_path = run_dir / "test_scores.npy"
    labels_path = run_dir / "test_labels_out.npy"
    if not scores_path.exists() or not labels_path.exists():
        print(f"  [WARN] missing {run_dir} -- skip (run run_neutralad.py first).")
        return None

    scores = np.load(scores_path)
    labels = np.load(labels_path).astype(int)

    auc = roc_auc_score(labels, scores) * 100
    f1 = compute_pre_recall_f1(labels, scores) * 100
    mcc = mcc_at_paper_threshold(labels, scores) * 100
    return {"Dataset": dataset, "Seed": seed, "N": len(labels), "AUC": auc, "F1": f1, "MCC": mcc}


def summarize(df_seeds):
    rows = []
    for ds, g in df_seeds.groupby("Dataset"):
        row = {"Dataset": ds, "N": int(g["N"].iloc[0]), "Seeds": len(g)}
        for m in METRIC_COLS:
            row[f"{m} mean"] = g[m].mean()
            row[f"{m} std"] = g[m].std(ddof=0)
            row[m] = f"{g[m].mean():.2f}±{g[m].std(ddof=0):.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--seeds", type=str, default="42,43,44")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for ds in datasets:
        for seed in seeds:
            row = evaluate_run(ds, seed, args.per)
            if row is not None:
                rows.append(row)

    if not rows:
        print("No results found.")
        return

    df_seeds = pd.DataFrame(rows)
    out_dir = RESULTS_ROOT / f"per_{args.per}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_seeds.to_csv(out_dir / "overall_summary_seeds.csv", index=False)

    df_sum = summarize(df_seeds)
    df_sum.to_csv(out_dir / "overall_summary.csv", index=False)

    print(f"\n--- NeuTraL-AD overall (whole test set) | per={args.per} | mean±std across seeds ---")
    print(df_sum[["Dataset", "N"] + METRIC_COLS].to_string(index=False))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
