"""Recompute SSAE's (MAE01ScaleMax) test_mode_total99.csv -- same metric
convention as test_mode_total95.csv (train_mae01scale_max.py's own
"total95" primary result), just at the validation-normal 99th percentile
instead of the 95th. No retraining: reloads the already-trained
best-val-AUC checkpoint and the already-saved X_val_normal.csv/X_test.csv/
y_test.csv from that run's results directory (all written once by
train_mae01scale_max.py at the end of its own training run), does a
forward pass to get scores, and re-derives the threshold + metrics.

Usage (point directly at one run's results directory -- the folder that
already contains checkpoints/, <dataset>_X_test.csv, <dataset>_y_test.csv,
<dataset>_X_val_normal.csv):

    python eval_ssae_val99.py --dataset cic_iot \
        --run_dir res/mae01scale_max_se_42_t_04/per_1.0/cic_iot_150e_..._scalemax-noise_std_...

Or let it auto-find the run the same way this project's other scripts do
(res/mae01scale_max_se_<seed>_t_04/per_<per>/<dataset>_*scalemax-noise_std_*):

    python eval_ssae_val99.py --dataset cic_iot --seed 42 --per 1.0

Output: <run_dir>/checkpoints/best_auc/test_mode_total99.csv
        (same 2-block CSV format as test_mode_total95.csv: Metric/Value rows
        then a Confusion Matrix block)

Run this on whichever machine actually holds the trained checkpoints (per
the project's own convention, checkpoints/MAE_<dataset>_best_auc.pkl).
"""

import argparse
import glob
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from val_perattack_mae01scale_max import load_synthesizer  # reused as-is

RES_DIR = os.path.join(BASE_DIR, "res")
TIME_TAG = "04"  # main default-config run tag, same as measure_ssae_inference_time.py
PERCENTILE = 99


def find_run_dir(dataset, seed, per):
    pattern = os.path.join(
        RES_DIR, f"mae01scale_max_se_{seed}_t_{TIME_TAG}", f"per_{per}",
        f"{dataset}_*scalemax-noise_std_*",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no run found for pattern: {pattern}")
    return matches[0]


def predict_and_evaluate(scores, y_true, threshold, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    y_true_bin = (np.asarray(y_true) != 0).astype(int)
    y_pred = (np.asarray(scores) > threshold).astype(int)

    auc = roc_auc_score(y_true_bin, scores) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
    acc = accuracy_score(y_true_bin, y_pred) * 100
    f1 = f1_score(y_true_bin, y_pred, zero_division=0) * 100
    precision = precision_score(y_true_bin, y_pred, zero_division=0) * 100
    recall = recall_score(y_true_bin, y_pred, zero_division=0) * 100
    mcc = matthews_corrcoef(y_true_bin, y_pred) * 100
    cm = confusion_matrix(y_true_bin, y_pred)

    import csv
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Threshold", f"{threshold:.6f}"])
        w.writerow(["AUC", f"{auc:.6f}"])
        w.writerow(["F1", f"{f1:.6f}"])
        w.writerow(["MCC", f"{mcc:.6f}"])
        w.writerow(["Precision", f"{precision:.6f}"])
        w.writerow(["Recall", f"{recall:.6f}"])
        w.writerow(["ACC", f"{acc:.6f}"])
        w.writerow([])
        w.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
        w.writerow(["True 0", cm[0, 0], cm[0, 1]])
        w.writerow(["True 1", cm[1, 0], cm[1, 1]])

    print(f"  [{filename}] AUC={auc:.2f}% F1={f1:.2f}% MCC={mcc:.2f}% "
          f"Prec={precision:.2f}% Rec={recall:.2f}% thr={threshold:.6f}")
    return {"AUC": auc, "F1": f1, "MCC": mcc, "Precision": precision, "Recall": recall, "ACC": acc}


def main():
    parser = argparse.ArgumentParser("Recompute SSAE test_mode_total99.csv (no retraining)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="results directory for the run (overrides --seed/--per auto-find)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per", type=str, default="1.0")
    args = parser.parse_args()

    run_dir = args.run_dir or find_run_dir(args.dataset, args.seed, args.per)
    print(f"run_dir: {run_dir}")

    x_test_path = os.path.join(run_dir, f"{args.dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{args.dataset}_y_test.csv")
    x_val_path = os.path.join(run_dir, f"{args.dataset}_X_val_normal.csv")
    model_path = os.path.join(run_dir, "checkpoints", f"MAE_{args.dataset}_best_auc.pkl")
    for p in (x_test_path, y_test_path, x_val_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    print("loading checkpoint...")
    synthesizer = load_synthesizer(model_path)

    X_val = pd.read_csv(x_val_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    print(f"scoring validation-normal ({len(X_val)} rows)...")
    _, _, total_val, _ = synthesizer.get_score(X_val, np.zeros(len(X_val)))
    threshold_99_total = float(np.percentile(total_val, PERCENTILE))
    print(f"threshold_{PERCENTILE}_total = {threshold_99_total:.6f}")

    print(f"scoring test set ({len(X_test)} rows)...")
    _, _, total_test, y_true = synthesizer.get_score(X_test, y_test)

    save_dir = os.path.join(run_dir, "checkpoints", "best_auc")
    predict_and_evaluate(total_test, y_true, threshold_99_total, save_dir, "test_mode_total99.csv")
    print(f"saved -> {os.path.join(save_dir, 'test_mode_total99.csv')}")


if __name__ == "__main__":
    main()
