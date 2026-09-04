"""DeepSVDD/NeuTraL/ICL/DIF per-rare-attack AUC/F1/MCC at the val95 AND
val99 thresholds already computed by run_deepod_val95.py / run_deepod_val99.py
(no recomputation of the threshold itself -- just re-read the "Threshold"
field each already saved to metrics_val95.json / metrics_val99.json, and
apply it to the {normal + one rare attack} subset). No retraining, no even
a forward pass: pure numpy over already-saved test_scores.npy.

Attack-type labels come from val_perattack_mae01scale_max.py's own
ATTACK_LOADERS -- verified row-aligned with NeuTraL-AD's prepared
test_data.npy/test_labels.npy (same official test file, same row order;
DeepOD's run_deepod.py reuses that .npy as-is).

Usage: python eval_deepod_perattack_val9599.py --seeds 42,43,44 --per 1.0
Output: DeepOD/results_ssae/perattack_val95_per<per>.csv
        DeepOD/results_ssae/perattack_val99_per<per>.csv
"""

import argparse
import json
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

from val_perattack_mae01scale_max import ATTACK_LOADERS
from run_deepod import DEEPOD_ROOT

ALGO_NAMES = ["dsvdd", "neutral", "icl", "dif"]
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


def evaluate_one(algo, dataset, seed, per, tag):
    src_dir = DEEPOD_ROOT / "results_ssae" / algo / f"per_{per}" / dataset / f"seed{seed}"
    metrics_path = DEEPOD_ROOT / f"results_ssae_{tag}" / algo / f"per_{per}" / dataset / f"seed{seed}" / f"metrics_{tag}.json"
    test_scores_path = src_dir / "test_scores.npy"
    test_labels_path = src_dir / "test_labels_out.npy"
    for p in (metrics_path, test_scores_path, test_labels_path):
        if not p.exists():
            raise FileNotFoundError(p)

    threshold = json.loads(metrics_path.read_text())["Threshold"]
    test_scores = np.load(test_scores_path)
    y_true = np.load(test_labels_path).astype(int)

    attack_labels, normal_tag = ATTACK_LOADERS[dataset](seed)
    assert len(attack_labels) == len(y_true)

    normal_mask = (y_true == 0)
    n_normal = int(normal_mask.sum())
    scores_normal = test_scores[normal_mask]

    rows = []
    for disp_name, label_value in ATTACK_PAIRS[dataset]:
        mask = (attack_labels == label_value).to_numpy()
        n_atk = int(mask.sum())
        if n_atk == 0:
            continue
        sub_scores = np.concatenate([scores_normal, test_scores[mask]])
        sub_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_atk, dtype=int)])
        auc, f1, mcc = metrics_at_threshold(sub_y, sub_scores, threshold)
        rows.append(dict(Algo=algo, Dataset=dataset, Seed=seed, Attack=disp_name, N=n_atk,
                          Threshold=threshold, AUC=auc, F1=f1, MCC=mcc))
    return rows


def main():
    parser = argparse.ArgumentParser("DeepOD per-rare-attack AUC/F1/MCC at val95/val99 thresholds")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--algos", type=str, default=",".join(ALGO_NAMES))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    algos = args.algos.split(",")
    datasets = list(ATTACK_PAIRS.keys())

    for tag in ("val95", "val99"):
        rows = []
        for algo in algos:
            for dataset in datasets:
                for seed in seeds:
                    try:
                        rows.extend(evaluate_one(algo, dataset, seed, args.per, tag))
                    except Exception as e:
                        print(f"  [FAILED] {algo} {dataset} seed={seed} {tag}: {e}")
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = DEEPOD_ROOT / "results_ssae" / f"perattack_{tag}_per{args.per}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
