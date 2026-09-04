"""DROCC per-rare-attack AUC/F1/MCC at the val95 AND val99 thresholds (the
SAME validation-normal-percentile threshold used for the overall table --
NOT the FAR-based threshold eval_drocc.py's own per-attack loop uses).
No retraining: reuses eval_drocc.py's own load_drocc_model()/anomaly_score()
/get_data() against the current official checkpoints (log_ssae/), same ones
val99_summary uses.

Usage: python eval_drocc_perattack_val9599.py --seeds 42,43,44 --per 1.0
Output: DROCC_icml2020/tabular/results_ssae/perattack_val95_per<per>.csv
        DROCC_icml2020/tabular/results_ssae/perattack_val99_per<per>.csv
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

from eval_drocc import DATASETS, DROCC_TABULAR, SAMPLE_THRES, get_data, load_drocc_model, anomaly_score

ATTACK_PAIRS = {
    "unsw":    [("Shellcode", "shellcode"), ("Worms", "worms")],
    "nsl_kdd": [("R2L", "3"), ("U2R", "4")],
    "cic_iot": [("Recon", "recon"), ("Spoofing", "spoofing")],
    "wsn_ds":  [("Blackhole", "blackhole"), ("Grayhole", "grayhole")],
}


def metrics_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) > threshold).astype(int)
    auc = roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100
    return auc, f1, mcc


def evaluate_run(dataset, seed, device, data_root, log_root):
    data_dir = data_root / dataset / f"seed{seed}"
    model_path = log_root / dataset / f"seed{seed}" / "model_epilepsy.pt"
    train_path, val_path = data_dir / "train_data.npy", data_dir / "val_data.npy"
    for p in [train_path, val_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    train_data = np.load(train_path)
    val_data = np.load(val_path)
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
    label_names = {v: k for k, v in label_mapping.items()}
    attack_labels = pd.Series([label_names[y] for y in y_te])

    model = load_drocc_model(model_path, x_te.shape[1], device)
    scores_val = anomaly_score(model, val_data, mean, std, device)
    scores_test = anomaly_score(model, x_te, mean, std, device)

    y_true_bin = (y_te != 0).astype(int)
    normal_mask = (y_true_bin == 0)
    n_normal = int(normal_mask.sum())
    scores_normal = scores_test[normal_mask]

    out = {}
    for percentile in (95, 99):
        threshold = float(np.percentile(scores_val, percentile))
        rows = []
        for disp_name, clan_label in ATTACK_PAIRS[dataset]:
            mask = (attack_labels == clan_label).to_numpy()
            n_atk = int(mask.sum())
            if n_atk == 0:
                continue
            sub_scores = np.concatenate([scores_normal, scores_test[mask]])
            sub_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_atk, dtype=int)])
            auc, f1, mcc = metrics_at_threshold(sub_y, sub_scores, threshold)
            rows.append(dict(Dataset=dataset, Seed=seed, Attack=disp_name, N=n_atk,
                              Threshold=threshold, AUC=auc, F1=f1, MCC=mcc))
        out[f"val{percentile}"] = rows
    return out


def main():
    parser = argparse.ArgumentParser("DROCC per-rare-attack AUC/F1/MCC at val95/val99 thresholds")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    args = parser.parse_args()

    import torch as T
    device = "cuda" if T.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    data_root = DROCC_TABULAR / "data_ssae" / f"per_{args.per}"
    log_root = DROCC_TABULAR / "log_ssae" / f"per_{args.per}"

    all95, all99 = [], []
    for dataset in ATTACK_PAIRS:
        for seed in seeds:
            print(f"=== {dataset} seed={seed} per={args.per} ===")
            try:
                out = evaluate_run(dataset, seed, device, data_root, log_root)
                all95.extend(out["val95"])
                all99.extend(out["val99"])
            except Exception as e:
                print(f"  [FAILED] {e}")

    out_dir = DROCC_TABULAR / "results_ssae"
    for tag, rows in (("val95", all95), ("val99", all99)):
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = out_dir / f"perattack_{tag}_per{args.per}.csv"
        df.to_csv(out_path, index=False)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
