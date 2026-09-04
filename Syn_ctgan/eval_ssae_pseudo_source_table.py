"""Pseudo-intrusion-source ablation table (paper's "Table 11: Impact of
pseudo-intrusion sources"): P (perturbation-based anomalies alone,
type_train="noise"), Q (CTGAN-generated anomalies alone, type_train="std"),
P+Q (the full model combining both, type_train="noise_std" -- this IS
SSAE's own standard default config, same checkpoints used everywhere else
in this project). Adds MCC(val99) and TPR@FAR=1% alongside the paper's own
AUC row, at the SAME default hyperparameters (margin=2.0, lambda=0.1) the
other two vary independently in the lambda/margin ablation tables.

No retraining: reloads each of the 3 scenarios' own already-trained
checkpoints/MAE_<dataset>_best_auc.pkl under
res/mae01scale_max_se_{42,43,44}_t_04/per_1.0/ (SSAE's usual t_04
default-config tree -- P and Q are simply the OTHER type_train variants
already sitting alongside the noise_std/P+Q run in that same tree).

Threshold convention (same as eval_ssae_ablation_lambda.py):
  - AUC       : threshold-independent.
  - TPR@FAR=1%: threshold = 99th percentile of TEST-normal scores.
  - MCC       : threshold = 99th percentile of VALIDATION-normal scores (val99).

Usage: python eval_ssae_pseudo_source_table.py --seeds 42,43,44 --per 1.0 \
           --margin 2.0 --lamda 0.1
Output: res/pseudo_source_table_per<per>.csv
"""

import argparse
import glob
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef

from val_perattack_mae01scale_max import load_synthesizer

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
DATASET_LABELS = {"unsw": "UNSW", "nsl_kdd": "NSL-KDD", "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS"}
SOURCE_LABEL = {"noise": "P", "std": "Q", "noise_std": "P+Q"}
SOURCE_ORDER = ["P", "Q", "P+Q"]
# order matters: try "noise_std" before "noise" (substring of it)
TYPE_TRAIN_RE = re.compile(r"scalemax-(noise_std|noise|std)_gw")
MARGIN_LAMBDA_RE = re.compile(r"_m([0-9.]+)_l([0-9.]+)_")


def find_scenarios(root, seed, time_tag, per, margin, lamda):
    pattern = os.path.join(root, f"mae01scale_max_se_{seed}_t_{time_tag}", f"per_{per}", "*")
    for run_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        dataset = next((ds for ds in sorted(DATASETS, key=len, reverse=True)
                         if name == ds or name.startswith(ds + "_")), None)
        if dataset is None:
            continue
        ml = MARGIN_LAMBDA_RE.search(name)
        if not ml or float(ml.group(1)) != margin or float(ml.group(2)) != lamda:
            continue
        tt = TYPE_TRAIN_RE.search(name)
        if not tt:
            continue
        source = SOURCE_LABEL.get(tt.group(1))
        if source is None:
            continue
        yield dict(run_dir=run_dir, dataset=dataset, source=source, scenario=name)


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
    _, _, total_test, y_true = synthesizer.get_score(X_test, y_test)
    y_true = (np.asarray(y_true) != 0).astype(int)

    auc = roc_auc_score(y_true, total_test) * 100 if len(np.unique(y_true)) >= 2 else 0.0

    normal_mask = (y_true == 0)
    scores_normal_test = total_test[normal_mask]
    scores_attack = total_test[~normal_mask]
    threshold_far1 = np.percentile(scores_normal_test, 99)
    tpr_far1 = float((scores_attack >= threshold_far1).mean() * 100)

    threshold_val99 = np.percentile(total_val, 99)
    y_pred = (total_test >= threshold_val99).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred) * 100

    return dict(AUC=auc, **{"TPR@FAR=1%": tpr_far1}, MCC=mcc)


def main():
    parser = argparse.ArgumentParser("SSAE pseudo-intrusion-source table (P / Q / P+Q)")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--time_tag", type=str, default="04")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--root", type=str, default=os.path.join(BASE_DIR, "res"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        for sc in find_scenarios(args.root, seed, args.time_tag, args.per, args.margin, args.lamda):
            print(f"[seed={seed}] {sc['dataset']} source={sc['source']} -> {sc['scenario']}")
            try:
                m = evaluate_one(sc["run_dir"], sc["dataset"])
                rows.append(dict(Seed=seed, Dataset=sc["dataset"], Source=sc["source"], **m))
            except Exception as e:
                print(f"  [FAILED] {e}")

    if not rows:
        print("nothing evaluated")
        return

    df = pd.DataFrame(rows)
    raw_path = os.path.join(args.root, f"pseudo_source_raw_per{args.per}.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nsaved raw: {raw_path}")

    metrics = ["AUC", "MCC", "TPR@FAR=1%"]
    table_rows = []
    for ds in DATASETS:
        for metric in metrics:
            row = {"Dataset": DATASET_LABELS[ds], "Metric": metric}
            for src in SOURCE_ORDER:
                sub = df[(df.Dataset == ds) & (df.Source == src)]
                row[src] = f"{sub[metric].mean():.2f}±{sub[metric].std():.2f}" if len(sub) else ""
            table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table_path = os.path.join(args.root, f"pseudo_source_table_per{args.per}.csv")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"saved table: {table_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
