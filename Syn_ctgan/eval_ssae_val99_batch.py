"""Batch version of eval_ssae_val99.py: auto-discovers EVERY run/scenario
folder under res/mae01scale_max_se_<seed>_t_<tag>/per_*/ (e.g. the t_02
ablation-sweep tree, which holds multiple hyperparameter scenarios per
seed/per -- unlike t_04's single default-config run) and recomputes
test_mode_total99.csv for each one, exactly like eval_ssae_val99.py does
for a single run. No retraining: reloads each run's own already-trained
checkpoints/MAE_<dataset>_best_auc.pkl and already-saved
<dataset>_X_val_normal.csv / <dataset>_X_test.csv / <dataset>_y_test.csv.

Dataset name is inferred from the scenario folder's name (it must start
with one of DATASETS below followed by "_", e.g.
"unsw_150e_m2.0_l0.1_..." -> "unsw").

Usage (run on the machine that actually holds the checkpoints):
    python eval_ssae_val99_batch.py --seeds 42,43,44 --time_tag 02
    python eval_ssae_val99_batch.py --seeds 42,43,44 --time_tag 02 --root res

Output per scenario: <run_dir>/checkpoints/best_auc/test_mode_total99.csv
Aggregated summary:  res/ssae_val99_t<tag>_seeds.csv
                     (Seed, Per, Dataset, ScenarioDir, AUC, F1, MCC, ...)
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

PERCENTILE = 99
DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds", "cic_ids", "nb_iot"]


def infer_dataset(scenario_dir_name):
    for ds in sorted(DATASETS, key=len, reverse=True):  # longest first (nsl_kdd before nb_iot etc.)
        if scenario_dir_name == ds or scenario_dir_name.startswith(ds + "_"):
            return ds
    return None


def find_scenarios(root, seed, time_tag):
    pattern = os.path.join(root, f"mae01scale_max_se_{seed}_t_{time_tag}", "per_*", "*")
    for run_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(run_dir):
            continue
        per = os.path.basename(os.path.dirname(run_dir)).replace("per_", "")
        scenario_name = os.path.basename(run_dir)
        dataset = infer_dataset(scenario_name)
        if dataset is None:
            print(f"  [skip] cannot infer dataset from folder name: {scenario_name}")
            continue
        yield dict(run_dir=run_dir, per=per, dataset=dataset, scenario=scenario_name)


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

    return {"AUC": auc, "F1": f1, "MCC": mcc, "Precision": precision, "Recall": recall, "ACC": acc}


def evaluate_one(run_dir, dataset):
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
    threshold_99_total = float(np.percentile(total_val, PERCENTILE))

    _, _, total_test, y_true = synthesizer.get_score(X_test, y_test)

    save_dir = os.path.join(run_dir, "checkpoints", "best_auc")
    m = predict_and_evaluate(total_test, y_true, threshold_99_total, save_dir, "test_mode_total99.csv")
    m["Threshold"] = threshold_99_total
    return m


def main():
    parser = argparse.ArgumentParser("Batch-recompute test_mode_total99.csv for every scenario under a t_<tag> tree")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--time_tag", type=str, default="02")
    parser.add_argument("--root", type=str, default=os.path.join(BASE_DIR, "res"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        scenarios = list(find_scenarios(args.root, seed, args.time_tag))
        print(f"\n=== seed={seed}: {len(scenarios)} scenario(s) found under t_{args.time_tag} ===")
        for sc in scenarios:
            print(f"  [{sc['dataset']} per={sc['per']}] {sc['scenario']}")
            try:
                m = evaluate_one(sc["run_dir"], sc["dataset"])
                rows.append(dict(Seed=seed, Per=sc["per"], Dataset=sc["dataset"],
                                  ScenarioDir=sc["scenario"], **m))
                print(f"    -> AUC={m['AUC']:.2f}  F1={m['F1']:.2f}  MCC={m['MCC']:.2f}  "
                      f"Precision={m['Precision']:.2f}  Recall={m['Recall']:.2f}  ACC={m['ACC']:.2f}")
            except Exception as e:
                print(f"    [FAILED] {e}")

    if not rows:
        print("\nnothing evaluated")
        return

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.root, f"ssae_val99_t{args.time_tag}_seeds.csv")
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(df.to_string(index=False))

    print("\nmean +/- std per (Dataset, Per, ScenarioDir):")
    summary = df.groupby(["Dataset", "Per", "ScenarioDir"])[["AUC", "F1", "MCC"]].agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
