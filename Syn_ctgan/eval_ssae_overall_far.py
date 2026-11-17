"""SSAE overall TPR@FAR=1%/5% -- ALL test-set attack samples pooled
together (test-normal threshold, not the val-normal threshold used by
best_auc/test_mode_total.csv). No retraining: rebuilds the exact
training-time val/test split (same as reeval_ssae_runs.py) and scores it
once with the existing t_<tag>_SSAE_se_<seed> checkpoint.

Usage: python eval_ssae_overall_far.py --time_tag 04 --seeds 42,43,44 --per 1.0
Output: res/ssae_overall_far_t<time_tag>_per<per>.csv
"""

import argparse
import glob
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reeval_ssae_runs import rebuild_val_test_split
from backfill_ssae_old_seeds import load_synthesizer

FAR_LEVELS = (0.01, 0.05)
DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]


def find_run_dir(root, time_tag, dataset, seed, per):
    pattern = os.path.join(root, f"t_{time_tag}_SSAE_se_{seed}", f"per_{per}",
                            f"{dataset}_150e_m2.0_l0.1_renormal_noise_std_*_catra0.1")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no run found for pattern: {pattern}")
    return matches[0]


def evaluate_run(root, time_tag, dataset, seed, per):
    run_dir = find_run_dir(root, time_tag, dataset, seed, per)
    model_path = os.path.join(run_dir, f"MAE_{dataset}_best_auc.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    _, _, X_test_raw, y_test = rebuild_val_test_split(dataset, seed, float(per))
    synth = load_synthesizer(model_path)
    X_test = synth._data_processor.transform(X_test_raw)
    _, _, total_test, y_true = synth.get_score(X_test, y_test)
    y_true = (np.asarray(y_true) != 0).astype(int)
    total_test = np.asarray(total_test)

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
    parser = argparse.ArgumentParser("SSAE overall (all-attacks-pooled) TPR@FAR=1%/5%, test-normal threshold")
    parser.add_argument("--time_tag", type=str, default="04")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--root", type=str, default=os.path.join(BASE_DIR, "res"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for dataset in DATASETS:
        for seed in seeds:
            print(f"=== {dataset} seed={seed} per={args.per} ===")
            try:
                rows.append(evaluate_run(args.root, args.time_tag, dataset, seed, args.per))
            except Exception as e:
                print(f"  [FAILED] {e}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(BASE_DIR, "res", f"ssae_overall_far_t{args.time_tag}_per{args.per}.csv")
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
