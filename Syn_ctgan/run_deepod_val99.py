"""DeepOD algorithms, evaluated at a validation-99th-percentile threshold
(same idea as run_deepod_val95.py, just a stricter percentile) -- separate
output tree (DeepOD/results_ssae_val99/, not results_ssae/ or results_ssae_val95/).

No retraining needed: run_deepod.py already scores BOTH the test split and
the genuine held-out normal validation split (val_data.npy) with the fitted
model. This script just re-reads results_ssae/<algo>/.../{test_scores,
test_labels_out,val_scores}.npy, recomputes the threshold as
percentile(val_scores, 99), and re-derives F1/MCC/Precision/Recall/ACC at
that threshold.

Usage: python run_deepod_val99.py --algo dsvdd [--datasets ...] [--seeds 42,43,44] [--per 1.0]
Output: DeepOD/results_ssae_val99/<algo>/per_<per>/<dataset>/seed<N>/
            test_scores.npy, test_labels_out.npy, val_scores.npy, metrics_val99.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

DEEPOD_ROOT = Path(__file__).parent.parent / "DeepOD"
RESULTS_ROOT = DEEPOD_ROOT / "results_ssae"
RESULTS_ROOT_VAL99 = DEEPOD_ROOT / "results_ssae_val99"
ALGO_NAMES = ["dsvdd", "goad", "neutral", "icl", "dif"]
PERCENTILE = 99


def metrics_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Threshold": float(threshold),
        "AUC": roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "MCC": matthews_corrcoef(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "ACC": accuracy_score(y_true, y_pred) * 100,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def run_one(algo, dataset, seed, per):
    src_dir = RESULTS_ROOT / algo / f"per_{per}" / dataset / f"seed{seed}"
    for name in ("test_scores.npy", "test_labels_out.npy", "val_scores.npy"):
        if not (src_dir / name).exists():
            raise FileNotFoundError(f"missing {src_dir / name} -- run run_deepod.py first")

    test_scores = np.load(src_dir / "test_scores.npy")
    test_labels = np.load(src_dir / "test_labels_out.npy").astype(int)
    val_scores = np.load(src_dir / "val_scores.npy")

    val_threshold = float(np.percentile(val_scores, PERCENTILE))
    m = metrics_at_threshold(test_labels, test_scores, val_threshold)

    out_dir = RESULTS_ROOT_VAL99 / algo / f"per_{per}" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_scores.npy", test_scores)
    np.save(out_dir / "test_labels_out.npy", test_labels)
    np.save(out_dir / "val_scores.npy", val_scores)
    with open(out_dir / "metrics_val99.json", "w") as f:
        json.dump(m, f, indent=2)

    print(f"[{algo} {dataset} seed={seed}] val99_threshold={val_threshold:.4f}  "
          f"AUC={m['AUC']:.2f}  F1={m['F1']:.2f}  MCC={m['MCC']:.2f}  "
          f"Precision={m['Precision']:.2f}  Recall={m['Recall']:.2f}  ACC={m['ACC']:.2f}  -> {out_dir}")
    return m


def main():
    parser = argparse.ArgumentParser("Recompute DeepOD metrics at a val-99th-percentile threshold")
    parser.add_argument("--algo", type=str, required=True, choices=ALGO_NAMES + ["all"])
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    args = parser.parse_args()

    algos = ALGO_NAMES if args.algo == "all" else [args.algo]
    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    failed = []
    for algo in algos:
        for dataset in datasets:
            for seed in seeds:
                try:
                    run_one(algo, dataset, seed, args.per)
                except Exception as e:
                    print(f"  [WARN] {algo} {dataset} seed={seed}: {e}")
                    failed.append((algo, dataset, seed, str(e)))

    if failed:
        print(f"\n[WARN] {len(failed)} run(s) missing/failed:")
        for algo, dataset, seed, err in failed:
            print(f"    {algo} {dataset} seed={seed}: {err}")


if __name__ == "__main__":
    main()
