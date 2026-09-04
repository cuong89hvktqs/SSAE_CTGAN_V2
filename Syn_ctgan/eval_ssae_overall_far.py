"""SSAE overall TPR@FAR=1%/5% -- ALL test-set attack samples pooled
together (not split per rare-attack-type). No retraining: reuses the
same t_04 checkpoints as the other eval_ssae_*.py scripts.

Usage: python eval_ssae_overall_far.py --seeds 42,43,44 --per 1.0
Output: res/ssae_overall_far_per<per>.csv
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from val_perattack_mae01scale_max import load_synthesizer
from eval_ssae_val99 import find_run_dir

FAR_LEVELS = (0.01, 0.05)
DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]


def evaluate_run(dataset, seed, per):
    run_dir = find_run_dir(dataset, seed, per)
    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    model_path = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")
    for p in (x_test_path, y_test_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    synthesizer = load_synthesizer(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    _, _, total_test, y_true = synthesizer.get_score(X_test, y_test)
    y_true = (np.asarray(y_true) != 0).astype(int)

    normal_mask = (y_true == 0)
    scores_normal = total_test[normal_mask]
    scores_attack = total_test[~normal_mask]

    row = dict(Dataset=dataset, Seed=seed,
               AUC=roc_auc_score(y_true, total_test) * 100 if len(np.unique(y_true)) >= 2 else 0.0)
    for far in FAR_LEVELS:
        threshold = np.percentile(scores_normal, 100 * (1 - far))
        row[f"TPR@FAR={int(far * 100)}%"] = float((scores_attack >= threshold).mean() * 100)
    return row


def main():
    parser = argparse.ArgumentParser("SSAE overall (all-attacks-pooled) TPR@FAR=1%/5%")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for dataset in DATASETS:
        for seed in seeds:
            print(f"=== {dataset} seed={seed} per={args.per} ===")
            try:
                rows.append(evaluate_run(dataset, seed, args.per))
            except Exception as e:
                print(f"  [FAILED] {e}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(BASE_DIR, "res", f"ssae_overall_far_per{args.per}.csv")
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
