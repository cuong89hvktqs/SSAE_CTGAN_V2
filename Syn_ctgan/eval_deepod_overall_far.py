"""DeepSVDD/NeuTraL/ICL/DIF overall TPR@FAR=1%/5% -- ALL test-set attack
samples pooled together (not split per rare-attack-type). Threshold =
percentile(test-normal scores, 99 / 95) -- same FAR convention as the
per-attack TPR@FAR tables, just applied to the whole attack population
instead of one rare attack subset at a time. Pure numpy over already-saved
test_scores.npy/test_labels_out.npy -- no retraining, no forward pass.

Usage: python eval_deepod_overall_far.py --seeds 42,43,44 --per 1.0
Output: DeepOD/results_ssae/overall_far_per<per>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEEPOD_ROOT = Path(__file__).parent.parent / "DeepOD"
ALGO_NAMES = ["dsvdd", "neutral", "icl", "dif"]
DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
FAR_LEVELS = (0.01, 0.05)


def evaluate_one(algo, dataset, seed, per):
    src_dir = DEEPOD_ROOT / "results_ssae" / algo / f"per_{per}" / dataset / f"seed{seed}"
    test_scores = np.load(src_dir / "test_scores.npy")
    y_true = np.load(src_dir / "test_labels_out.npy").astype(int)

    normal_mask = (y_true == 0)
    scores_normal = test_scores[normal_mask]
    scores_attack = test_scores[~normal_mask]

    row = dict(Algo=algo, Dataset=dataset, Seed=seed,
               AUC=roc_auc_score(y_true, test_scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0)
    for far in FAR_LEVELS:
        threshold = np.percentile(scores_normal, 100 * (1 - far))
        row[f"TPR@FAR={int(far * 100)}%"] = float((scores_attack >= threshold).mean() * 100)
    return row


def main():
    parser = argparse.ArgumentParser("DeepOD overall (all-attacks-pooled) TPR@FAR=1%/5%")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--algos", type=str, default=",".join(ALGO_NAMES))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    algos = args.algos.split(",")

    rows = []
    for algo in algos:
        for dataset in DATASETS:
            for seed in seeds:
                try:
                    rows.append(evaluate_one(algo, dataset, seed, args.per))
                except Exception as e:
                    print(f"  [FAILED] {algo} {dataset} seed={seed}: {e}")

    df = pd.DataFrame(rows)
    out_path = DEEPOD_ROOT / "results_ssae" / f"overall_far_per{args.per}.csv"
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
