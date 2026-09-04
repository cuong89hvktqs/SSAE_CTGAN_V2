"""Lambda-ablation table: AUC / TPR@FAR=1% / MCC(val99) for every lambda
value found under the t_02 ablation-sweep tree, at a FIXED margin (default
2.0 -- the project's own default/main-config margin).

Run this on the machine that actually holds the checkpoints (per the
project's convention, checkpoints/MAE_<dataset>_best_auc.pkl) -- e.g. the
Ubuntu machine holding res/mae01scale_max_se_{42,43,44}_t_02/.

No retraining: reloads each scenario's own already-trained checkpoint and
already-saved <dataset>_X_val_normal.csv/_X_test.csv/_y_test.csv (same
files train_mae01scale_max.py's own post-training eval writes). For each
scenario:
  - AUC       : roc_auc_score on the full test set (threshold-independent,
                does not change with lambda/margin choice of THRESHOLD --
                still varies across ablation runs because the underlying
                SCORES themselves differ per lambda/margin).
  - TPR@FAR=1%: threshold = 99th percentile of TEST-normal scores (the
                project's standard FAR convention, independent of val95/99).
  - MCC       : threshold = 99th percentile of VALIDATION-normal scores
                (the "val99" convention -- same one used everywhere else
                in this project's overall AUC/F1/MCC tables).

Scenario folders are matched by name pattern "..._m<margin>_l<lambda>_..."
(train_mae01scale_max.py's own results_dir naming, e.g.
"unsw_150e_m2.0_l0.1_renormal_scalemax-noise_std_..."); dataset name is
inferred from the folder's own prefix.

Usage:
    python eval_ssae_ablation_lambda.py --seeds 42,43,44 --time_tag 02 \
        --margin 2.0 --per 1.0
Output: res/ablation_lambda_t<tag>_per<per>.csv   (raw per-seed rows)
        res/ablation_lambda_t<tag>_per<per>_table.csv  (final pivoted table,
        same Dataset x Metric row layout / lambda-value column layout as
        the reference image)
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
    parser = argparse.ArgumentParser("SSAE lambda-ablation table (AUC / TPR@FAR=1% / MCC@val99)")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--time_tag", type=str, default="02")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--margin", type=float, default=2.0, help="fixed margin to hold while sweeping lambda")
    parser.add_argument("--root", type=str, default=os.path.join(BASE_DIR, "res"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        for sc in find_scenarios(args.root, seed, args.time_tag, args.per):
            if sc["margin"] != args.margin:
                continue
            print(f"[seed={seed}] {sc['dataset']} m={sc['margin']} l={sc['lamda']} -> {sc['scenario']}")
            try:
                m = evaluate_one(sc["run_dir"], sc["dataset"])
                rows.append(dict(Seed=seed, Dataset=sc["dataset"], Lambda=sc["lamda"], **m))
            except Exception as e:
                print(f"  [FAILED] {e}")

    if not rows:
        print("nothing evaluated -- check --margin / --time_tag / --per")
        return

    df = pd.DataFrame(rows)
    raw_path = os.path.join(args.root, f"ablation_lambda_t{args.time_tag}_per{args.per}.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nsaved raw: {raw_path}")

    lambdas = sorted(df.Lambda.unique())
    metrics = ["TPR@FAR=1%", "AUC", "MCC"]
    table_rows = []
    for ds in DATASETS:
        for metric in metrics:
            row = {"Dataset": DATASET_LABELS[ds], "Metric": metric}
            for lam in lambdas:
                sub = df[(df.Dataset == ds) & (df.Lambda == lam)]
                if len(sub):
                    row[f"lambda={lam}"] = f"{sub[metric].mean():.2f}±{sub[metric].std():.2f}"
                else:
                    row[f"lambda={lam}"] = ""
            table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table_path = os.path.join(args.root, f"ablation_lambda_t{args.time_tag}_per{args.per}_table.csv")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"saved table: {table_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
