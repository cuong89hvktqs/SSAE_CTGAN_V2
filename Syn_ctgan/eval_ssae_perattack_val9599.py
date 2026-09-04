"""SSAE per-rare-attack AUC/F1/MCC at the val95 AND val99 thresholds (the
SAME validation-normal-percentile threshold used for the overall
AUC/F1/MCC table -- NOT the FAR-based threshold used by
eval_ssae_perattack_far.py's TPR@FAR table). For each of the 8 rare
attacks: subset = {all test-normal} + {this attack}, threshold =
percentile(val_normal_total_scores, 95 or 99), then AUC/F1/MCC on that
subset. No retraining: reuses the same t_04 checkpoints as
eval_ssae_val99.py / eval_ssae_perattack_far.py.

Usage: python eval_ssae_perattack_val9599.py --seeds 42,43,44 --per 1.0
Output: res/ssae_perattack_val95_per<per>.csv
        res/ssae_perattack_val99_per<per>.csv
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
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

from val_perattack_mae01scale_max import load_synthesizer, ATTACK_LOADERS
from eval_ssae_val99 import find_run_dir

ATTACK_PAIRS = {
    "unsw":    [("Shellcode", "Shellcode"), ("Worms", "Worms")],
    "nsl_kdd": [("R2L", "attack_3"), ("U2R", "attack_4")],
    "cic_iot": [("Recon", "Recon"), ("Spoofing", "Spoofing")],
    "wsn_ds":  [("Blackhole", "Blackhole"), ("Grayhole", "Grayhole")],
}


def metrics_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    auc = roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100
    return auc, f1, mcc


def evaluate_run(dataset, seed, per):
    run_dir = find_run_dir(dataset, seed, per)
    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    x_val_path = os.path.join(run_dir, f"{dataset}_X_val_normal.csv")
    model_path = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")
    for p in (x_test_path, y_test_path, x_val_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    synthesizer = load_synthesizer(model_path)
    X_val = pd.read_csv(x_val_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    _, _, total_val, _ = synthesizer.get_score(X_val, np.zeros(len(X_val)))
    _, _, total_test, y_true = synthesizer.get_score(X_test, y_test)

    attack_labels, normal_tag = ATTACK_LOADERS[dataset](seed)
    assert len(attack_labels) == len(y_true)

    normal_mask = (y_true == 0)
    n_normal = int(normal_mask.sum())
    scores_normal = total_test[normal_mask]

    rows95, rows99 = [], []
    for percentile, rows in ((95, rows95), (99, rows99)):
        threshold = float(np.percentile(total_val, percentile))
        for disp_name, label_value in ATTACK_PAIRS[dataset]:
            mask = (attack_labels == label_value).to_numpy()
            n_atk = int(mask.sum())
            if n_atk == 0:
                continue
            sub_scores = np.concatenate([scores_normal, total_test[mask]])
            sub_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_atk, dtype=int)])
            auc, f1, mcc = metrics_at_threshold(sub_y, sub_scores, threshold)
            rows.append(dict(Dataset=dataset, Seed=seed, Attack=disp_name, N=n_atk,
                              Threshold=threshold, AUC=auc, F1=f1, MCC=mcc))
    return rows95, rows99


def main():
    parser = argparse.ArgumentParser("SSAE per-rare-attack AUC/F1/MCC at val95/val99 thresholds")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = args.datasets.split(",")

    all95, all99 = [], []
    for dataset in datasets:
        for seed in seeds:
            print(f"=== {dataset} seed={seed} per={args.per} ===")
            try:
                r95, r99 = evaluate_run(dataset, seed, args.per)
                all95.extend(r95)
                all99.extend(r99)
            except Exception as e:
                print(f"  [FAILED] {e}")

    for tag, rows in (("val95", all95), ("val99", all99)):
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = os.path.join(BASE_DIR, "res", f"ssae_perattack_{tag}_per{args.per}.csv")
        df.to_csv(out_path, index=False)
        print(f"\nsaved: {out_path}")
        print(df.groupby(["Dataset", "Attack"])[["AUC", "F1", "MCC"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
