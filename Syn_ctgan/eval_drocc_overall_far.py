"""DROCC overall TPR@FAR=1%/5% -- ALL test-set attack samples pooled
together (not split per rare-attack-type). No retraining: reuses
eval_drocc.py's own get_data()/load_drocc_model()/anomaly_score() against
the current official checkpoints (log_ssae/).

Usage: python eval_drocc_overall_far.py --seeds 42,43,44 --per 1.0
Output: DROCC_icml2020/tabular/results_ssae/overall_far_per<per>.csv
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eval_drocc import DATASETS, DROCC_TABULAR, SAMPLE_THRES, get_data, load_drocc_model, anomaly_score

FAR_LEVELS = (0.01, 0.05)


def evaluate_run(dataset, seed, device, data_root, log_root):
    data_dir = data_root / dataset / f"seed{seed}"
    model_path = log_root / dataset / f"seed{seed}" / "model_epilepsy.pt"
    train_path, val_path = data_dir / "train_data.npy", data_dir / "val_data.npy"
    for p in [train_path, val_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    train_data = np.load(train_path)
    mean, std = train_data.mean(0), train_data.std(0)

    cfg = DATASETS[dataset]
    x_train, y_train, x_val, y_val, x_te, y_te, x_zd, y_zd, label_mapping, feature_names = get_data(
        data_path=str(cfg["data_path"]), target=cfg["target"], drop=cfg["drop_cols"],
        class_zero=cfg["class_zero"], sample_thres=SAMPLE_THRES, split_seed=seed,
        test_ratio=0.2, val_ratio=0.2, anomaly_detection=True, n_samples=cfg["total_rows"],
        sample_seed=seed,
    )
    if x_zd is not None:
        x_te = np.concatenate([x_te, x_zd])
        y_te = np.concatenate([y_te, y_zd])

    model = load_drocc_model(model_path, x_te.shape[1], device)
    scores_test = anomaly_score(model, x_te, mean, std, device)
    y_true = (y_te != 0).astype(int)

    normal_mask = (y_true == 0)
    scores_normal = scores_test[normal_mask]
    scores_attack = scores_test[~normal_mask]

    row = dict(Dataset=dataset, Seed=seed,
               AUC=roc_auc_score(y_true, scores_test) * 100 if len(np.unique(y_true)) >= 2 else 0.0)
    for far in FAR_LEVELS:
        threshold = np.percentile(scores_normal, 100 * (1 - far))
        row[f"TPR@FAR={int(far * 100)}%"] = float((scores_attack >= threshold).mean() * 100)
    return row


def main():
    parser = argparse.ArgumentParser("DROCC overall (all-attacks-pooled) TPR@FAR=1%/5%")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    args = parser.parse_args()

    import torch as T
    device = "cuda" if T.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    data_root = DROCC_TABULAR / "data_ssae" / f"per_{args.per}"
    log_root = DROCC_TABULAR / "log_ssae" / f"per_{args.per}"

    rows = []
    for dataset in DATASETS:
        for seed in seeds:
            print(f"=== {dataset} seed={seed} per={args.per} ===")
            try:
                rows.append(evaluate_run(dataset, seed, device, data_root, log_root))
            except Exception as e:
                print(f"  [FAILED] {e}")

    df = pd.DataFrame(rows)
    out_path = DROCC_TABULAR / "results_ssae" / f"overall_far_per{args.per}.csv"
    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
