"""SSAE per-rare-attack TPR@FAR=1%/5%, restricted to the t_04 default-config
run (unlike val_perattack_mae01scale_max.py's own find_run_dirs(), which
globs "mae01scale_max_se_*_t_*" -- ANY t_XX tag -- and can silently mix in
leftover ablation-sweep configs from t_02/t_03 under the same seed label).
No retraining: reloads each run's own checkpoints/MAE_<dataset>_best_auc.pkl
and already-saved <dataset>_X_test.csv/_y_test.csv (same checkpoints used by
eval_ssae_val99.py's overall AUC/F1/MCC table, so both tables are guaranteed
to describe the SAME model).

Threshold convention (independent of the val95/val99 choice used for the
overall table): each TPR@FAR threshold is a percentile of the FULL
TEST-normal "total" score distribution (matches val_perattack_mae01scale_max
.py's own evaluate_ids_metrics -- FAR=1% -> 99th percentile of test-normal
scores, FAR=5% -> 95th percentile).

The 2 rare/hard attacks per dataset match this project's established
RARE_ATTACKS convention (UNSW-NB15: Shellcode/Worms, NSL-KDD: R2L/U2R,
CIC-IoT 2023: Recon/Spoofing, WSN-DS: Blackhole/Grayhole).

Usage: python eval_ssae_perattack_far.py --seeds 42,43,44 --per 1.0
Output: res/ssae_perattack_far_per<per>.csv
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

from val_perattack_mae01scale_max import load_synthesizer, ATTACK_LOADERS, FAR_LEVELS
from eval_ssae_val99 import find_run_dir  # t_04-specific run discovery

ATTACK_PAIRS = {
    "unsw":    [("Shellcode", "Shellcode"), ("Worms", "Worms")],
    "nsl_kdd": [("R2L", "attack_3"), ("U2R", "attack_4")],
    "cic_iot": [("Recon", "Recon"), ("Spoofing", "Spoofing")],
    "wsn_ds":  [("Blackhole", "Blackhole"), ("Grayhole", "Grayhole")],
}


def evaluate_ids_metrics(y_true, scores, normal_thresholds):
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    results = {
        "AUC": roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0,
        "PR-AUC": average_precision_score(y_true, scores) * 100,
    }
    for far, threshold in normal_thresholds.items():
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results[f"TPR@FAR={int(far * 100)}%"] = tp / (tp + fn + 1e-12) * 100
    return results


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

    attack_labels, normal_tag = ATTACK_LOADERS[dataset](seed)
    assert len(attack_labels) == len(y_true), (
        f"{dataset} seed={seed}: attack-label length {len(attack_labels)} != test size {len(y_true)}"
    )

    normal_mask = (y_true == 0)
    n_normal = int(normal_mask.sum())
    scores_normal = total_test[normal_mask]
    normal_thresholds = {far: float(np.percentile(scores_normal, 100 * (1 - far))) for far in FAR_LEVELS}

    rows = []
    for disp_name, label_value in ATTACK_PAIRS[dataset]:
        mask = (attack_labels == label_value).to_numpy()
        n_atk = int(mask.sum())
        if n_atk == 0:
            print(f"  [warn] {dataset} seed={seed}: no rows for attack '{label_value}'")
            continue
        sub_scores = np.concatenate([scores_normal, total_test[mask]])
        sub_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_atk, dtype=int)])
        metrics = evaluate_ids_metrics(sub_y, sub_scores, normal_thresholds)
        rows.append(dict(Dataset=dataset, Seed=seed, Attack=disp_name, N=n_atk, **metrics))
    return rows


def main():
    parser = argparse.ArgumentParser("SSAE per-rare-attack TPR@FAR (t_04 default-config runs only)")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = args.datasets.split(",")

    all_rows = []
    for dataset in datasets:
        for seed in seeds:
            print(f"=== {dataset} seed={seed} per={args.per} ===")
            try:
                all_rows.extend(evaluate_run(dataset, seed, args.per))
            except Exception as e:
                print(f"  [FAILED] {e}")

    if not all_rows:
        print("nothing evaluated")
        return

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(BASE_DIR, "res", f"ssae_perattack_far_per{args.per}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")

    summary = df.groupby(["Dataset", "Attack"])[["AUC", "PR-AUC", "TPR@FAR=1%", "TPR@FAR=5%"]].agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
