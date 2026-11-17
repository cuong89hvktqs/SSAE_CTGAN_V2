"""Re-evaluate already-trained SSAE (MAE01ScaleMax) runs into the current
standard result layout -- no retraining.

For every run folder res/t_<tag>_SSAE_se_<seed>/per_1.0/<dataset>_.../ that
holds a MAE_<dataset>_best_auc.pkl checkpoint:
  1. Rebuild the EXACT training-time val/test split (same loader, same seed,
     same 80/20 normal-only split as train_mae01scale_max.py), score it ONCE
     with the checkpoint, and write
       best_auc/test_mode_total.csv   -- same schema/logic as
                                         evaluate_best_auc_checkpoint()
       best_auc/per_attack_seeds.csv  -- same schema/logic as
                                         evaluate_per_attack() (incl. MCC_95/99)
     Scoring once guarantees AUC_95 == AUC_99 (see CLAUDE.md, data_transformer
     noise).
  2. Delete every old result file of that run: the top-level test_mode_*.csv,
     MAE_<dataset>_best_epoch.txt (old best_loss artifact), and everything
     inside best_auc/ other than the 2 new files (incl. best_auc/old/).
     Checkpoint .pkl, loss plots/csv and metadata json are kept.

Dry-run by default (prints the plan only); pass --apply to evaluate + delete.

Usage:
    python reeval_ssae_runs.py --time_tag 05 --seeds 42,43,44            # dry run
    python reeval_ssae_runs.py --time_tag 05 --seeds 42,43,44 --apply
"""

import argparse
import csv
import glob
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from backfill_ssae_old_seeds import load_synthesizer
from process_data import (
    load_data_cic_ids, load_data_cic_iot, load_data_nb_iot, load_data_nsl_kdd,
    load_data_unsw, load_data_wsn_ds,
)
from val_perattack_mae01scale_max import ATTACK_LOADERS, FAR_LEVELS, evaluate_ids_metrics

# Same loader calls as train_mae01scale_max.py's DATASET_LOADERS
DATASET_LOADERS = {
    "unsw":    lambda seed: load_data_unsw(),
    "cic_ids": lambda seed: load_data_cic_ids(),
    "nsl_kdd": lambda seed: load_data_nsl_kdd(),
    "cic_iot": lambda seed: load_data_cic_iot(),
    "wsn_ds":  lambda seed: load_data_wsn_ds(seed=seed, test_size=0.2),
    "nb_iot":  lambda seed: load_data_nb_iot(seed=seed, test_size=0.2),
}
NEW_OUTPUT_FILES = {"test_mode_total.csv", "per_attack_seeds.csv"}


def infer_dataset(run_name):
    for ds in sorted(DATASET_LOADERS, key=len, reverse=True):
        if run_name.startswith(ds + "_"):
            return ds
    return None


def find_runs(root, time_tag, seeds, per):
    runs = []
    for seed in seeds:
        pattern = os.path.join(root, f"t_{time_tag}_SSAE_se_{seed}", f"per_{per}", "*")
        for run_dir in sorted(glob.glob(pattern)):
            if not os.path.isdir(run_dir):
                continue
            dataset = infer_dataset(os.path.basename(run_dir))
            if dataset is None:
                print(f"[WARN] unknown dataset prefix, skip: {run_dir}")
                continue
            runs.append((seed, dataset, run_dir))
    return runs


def rebuild_val_test_split(dataset, seed, percentage):
    """Line-for-line copy of train_mae01scale_max.py's split."""
    df_full, y_full, df_test, y_test = DATASET_LOADERS[dataset](seed)

    n_test = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)

    normal_mask = y_train == 0
    X_train_normal_full = X_train_raw.loc[normal_mask].copy()
    y_train_normal_full = y_train.loc[normal_mask].copy()

    sampled_idx = X_train_normal_full.sample(frac=percentage, random_state=seed).index
    X_normal_sub = X_train_normal_full.loc[sampled_idx].reset_index(drop=True)
    y_normal_sub = y_train_normal_full.loc[sampled_idx].reset_index(drop=True)

    _, X_val_raw, _, y_val = train_test_split(
        X_normal_sub, y_normal_sub, test_size=0.2, random_state=seed
    )
    X_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    X_test_raw.columns = X_test_raw.columns.map(str)

    return X_val_raw, y_val.reset_index(drop=True), X_test_raw, y_test.reset_index(drop=True)


def score_run(run_dir, dataset, seed, percentage):
    model_path = os.path.join(run_dir, f"MAE_{dataset}_best_auc.pkl")
    X_val_raw, y_val, X_test_raw, y_test = rebuild_val_test_split(dataset, seed, percentage)

    synth = load_synthesizer(model_path)
    X_val = synth._data_processor.transform(X_val_raw)
    X_test = synth._data_processor.transform(X_test_raw)

    _, _, val_total, _ = synth.get_score(X_val, y_val)
    _, _, test_total, y_true = synth.get_score(X_test, y_test)
    y_true_bin = (np.asarray(y_true) != 0).astype(int)
    return np.asarray(val_total), np.asarray(test_total), y_true_bin


def write_test_mode_total(save_path, val_total, test_total, y_true_bin):
    """Same as evaluate_best_auc_checkpoint() in train_mae01scale_max.py."""
    val_thresholds = {}
    auc = roc_auc_score(y_true_bin, test_total) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for percentile, far_label in ((95, "TPR@FAR=5%"), (99, "TPR@FAR=1%")):
            threshold = float(np.percentile(val_total, percentile))
            val_thresholds[percentile] = threshold
            y_pred = (test_total > threshold).astype(int)
            mcc = matthews_corrcoef(y_true_bin, y_pred) * 100
            tpr = recall_score(y_true_bin, y_pred, zero_division=0) * 100

            w.writerow([f"Threshold_{percentile}", f"{threshold:.6f}"])
            w.writerow([far_label,                 f"{tpr:.6f}"])
            w.writerow([f"AUC_{percentile}",        f"{auc:.6f}"])
            w.writerow([f"MCC_{percentile}",        f"{mcc:.6f}"])
            print(f"    [{far_label} @ p{percentile}] AUC={auc:.2f}  MCC={mcc:.2f}  TPR={tpr:.2f}")
    return val_thresholds


def write_per_attack(save_path, dataset, seed, test_total, val_thresholds):
    """Same as evaluate_per_attack() in train_mae01scale_max.py."""
    attack_labels, normal_tag = ATTACK_LOADERS[dataset](seed)
    if len(attack_labels) != len(test_total):
        print(f"    [ERROR] attack labels ({len(attack_labels)}) != test scores "
              f"({len(test_total)}) -- skip per_attack_seeds.csv.")
        return False

    normal_mask = (attack_labels == normal_tag).to_numpy()
    n_normal = int(normal_mask.sum())
    attacks = sorted(a for a in attack_labels.unique() if a != normal_tag)

    scores_normal = test_total[normal_mask]
    normal_thresholds = {
        far: np.percentile(scores_normal, 100 * (1 - far)) for far in FAR_LEVELS
    }

    rows = []
    for attack in attacks:
        atk_mask = (attack_labels == attack).to_numpy()
        sub_scores = np.concatenate([scores_normal, test_total[atk_mask]])
        sub_y = np.concatenate([
            np.zeros(n_normal, dtype=int),
            np.ones(int(atk_mask.sum()), dtype=int),
        ])
        metrics = evaluate_ids_metrics(sub_y, sub_scores, normal_thresholds, val_thresholds)
        rows.append({
            "Dataset": dataset, "Seed": seed, "Attack": attack,
            "N": int(atk_mask.sum()), "Score": "total", **metrics,
        })

    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"    per_attack_seeds.csv: {len(rows)} attack rows")
    return True


def old_result_paths(run_dir, dataset):
    """Every old-format result file/folder of a run (never the checkpoint)."""
    paths = sorted(glob.glob(os.path.join(run_dir, "test_mode_*.csv")))
    best_epoch = os.path.join(run_dir, f"MAE_{dataset}_best_epoch.txt")
    if os.path.exists(best_epoch):
        paths.append(best_epoch)
    best_auc_dir = os.path.join(run_dir, "best_auc")
    if os.path.isdir(best_auc_dir):
        for name in sorted(os.listdir(best_auc_dir)):
            if name not in NEW_OUTPUT_FILES:
                paths.append(os.path.join(best_auc_dir, name))
    return paths


def remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def main():
    parser = argparse.ArgumentParser("Re-evaluate SSAE runs into the standard best_auc/ layout")
    parser.add_argument("--time_tag", type=str, default="05")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--root", type=str, default=os.path.join(SCRIPT_DIR, "res"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    runs = find_runs(args.root, args.time_tag, seeds, args.per)
    print(f"{len(runs)} run(s) under t_{args.time_tag}, seeds={seeds}, per_{args.per}"
          f"{'' if args.apply else '  [DRY RUN]'}")

    failed = []
    for seed, dataset, run_dir in runs:
        print(f"\n[seed={seed}] {os.path.basename(run_dir)}")
        model_path = os.path.join(run_dir, f"MAE_{dataset}_best_auc.pkl")
        if not os.path.exists(model_path):
            print(f"    [WARN] missing {model_path} -- skip (nothing deleted).")
            failed.append(run_dir)
            continue

        old_paths = old_result_paths(run_dir, dataset)
        for p in old_paths:
            print(f"    delete {os.path.relpath(p, run_dir)}")
        if not args.apply:
            continue

        best_auc_dir = os.path.join(run_dir, "best_auc")
        os.makedirs(best_auc_dir, exist_ok=True)
        try:
            val_total, test_total, y_true_bin = score_run(run_dir, dataset, seed, float(args.per))
            val_thresholds = write_test_mode_total(
                os.path.join(best_auc_dir, "test_mode_total.csv"), val_total, test_total, y_true_bin)
            ok = write_per_attack(
                os.path.join(best_auc_dir, "per_attack_seeds.csv"), dataset, seed, test_total, val_thresholds)
        except Exception as e:
            print(f"    [FAILED] {e!r} -- old results kept.")
            failed.append(run_dir)
            continue
        if not ok:
            failed.append(run_dir)
            continue

        # Only delete old results once both new files exist.
        for p in old_paths:
            remove_path(p)

    print(f"\nDone. {len(runs) - len(failed)}/{len(runs)} run(s) OK.")
    for p in failed:
        print(f"  FAILED/SKIPPED: {p}")


if __name__ == "__main__":
    main()
