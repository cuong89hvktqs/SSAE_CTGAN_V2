"""Margin-ablation table: AUC / TPR@FAR=1% / MCC(val99) for every margin
value found under the t_02 ablation-sweep tree, at a FIXED lambda (default
0.1 -- the project's own default/main-config lambda). Mirror of
eval_ssae_ablation_lambda.py -- see that script's docstring for the full
threshold-convention explanation (AUC threshold-independent; TPR@FAR=1%
from test-normal 99th percentile; MCC from val99 = validation-normal 99th
percentile).

Usage:
    python eval_ssae_ablation_margin.py --seeds 42,43,44 --time_tag 02 \
        --lamda 0.1 --per 1.0
Output: res/ablation_margin_t<tag>_per<per>.csv          (raw per-seed rows)
        res/ablation_margin_t<tag>_per<per>_table.csv    (final pivoted table)
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
DATASET_LABELS = {"unsw": "UNSW-NB15", "nsl_kdd": "NSL-KDD", "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS"}
SCENARIO_RE = re.compile(r"_m([0-9.]+)_l([0-9.]+)_")


def infer_dataset(scenario_name):
    for ds in sorted(DATASETS, key=len, reverse=True):
        if scenario_name == ds or scenario_name.startswith(ds + "_"):
            return ds
    return None


def find_scenarios(root, seed, time_tag, per):
    pattern = os.path.join(root, f"mae01scale_max_se_{seed}_t_{time_tag}", f"per_{per}", "*")
    for run_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(run_dir):
            continue
        scenario_name = os.path.basename(run_dir)
        dataset = infer_dataset(scenario_name)
        if dataset is None:
            continue
        m = SCENARIO_RE.search(scenario_name)
        if not m:
            continue
        margin, lamda = float(m.group(1)), float(m.group(2))
        yield dict(run_dir=run_dir, dataset=dataset, margin=margin, lamda=lamda, scenario=scenario_name)


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
    parser = argparse.ArgumentParser("SSAE margin-ablation table (AUC / TPR@FAR=1% / MCC@val99)")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--time_tag", type=str, default="02")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--lamda", type=float, default=0.1, help="fixed lambda to hold while sweeping margin")
    parser.add_argument("--root", type=str, default=os.path.join(BASE_DIR, "res"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        for sc in find_scenarios(args.root, seed, args.time_tag, args.per):
            if sc["lamda"] != args.lamda:
                continue
            print(f"[seed={seed}] {sc['dataset']} m={sc['margin']} l={sc['lamda']} -> {sc['scenario']}")
            try:
                m = evaluate_one(sc["run_dir"], sc["dataset"])
                rows.append(dict(Seed=seed, Dataset=sc["dataset"], Margin=sc["margin"], **m))
            except Exception as e:
                print(f"  [FAILED] {e}")

    if not rows:
        print("nothing evaluated -- check --lamda / --time_tag / --per")
        return

    df = pd.DataFrame(rows)
    raw_path = os.path.join(args.root, f"ablation_margin_t{args.time_tag}_per{args.per}.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nsaved raw: {raw_path}")

    margins = sorted(df.Margin.unique())
    metrics = ["TPR@FAR=1%", "AUC", "MCC"]
    table_rows = []
    for ds in DATASETS:
        for metric in metrics:
            row = {"Dataset": DATASET_LABELS[ds], "Metric": metric}
            for m_val in margins:
                sub = df[(df.Dataset == ds) & (df.Margin == m_val)]
                if len(sub):
                    row[f"m={m_val}"] = f"{sub[metric].mean():.2f}±{sub[metric].std():.2f}"
                else:
                    row[f"m={m_val}"] = ""
            table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table_path = os.path.join(args.root, f"ablation_margin_t{args.time_tag}_per{args.per}_table.csv")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"saved table: {table_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
