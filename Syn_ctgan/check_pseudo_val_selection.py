"""Reuses already-trained t_01 (lambda ablation) / t_02 (margin ablation) /
t_04 (baseline) checkpoints -- NO retraining. For every finished run, reads
the per-epoch training log `mae01scale_max_loss_150e.csv` and takes
max(Validation AUC): this is the EXACT SAME quantity the training loop uses
every epoch to pick the best_auc checkpoint (see
`M01AECTGANSynthesizer._evaluate_pseudo_validation_auc` in
ctgan/synthesizers/m01aectgan.py) -- ROC-AUC separating real val-normal rows
from pseudo-anomalies built the same way as type_train. It never touches a
real attack label.

Purpose: answer Reviewer 1 Comment 1 (was m, lambda selected using the test
set?) without retraining anything. If the (m=2.0, lambda=0.1) baseline is
also the arg-best under this pseudo-validation-only criterion, that is
direct evidence the selection is label-free and not test-set-driven.

Usage:
    python check_pseudo_val_selection.py [--seeds 42,43,...,51]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(SCRIPT_DIR, "res")

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
DATASET_DISPLAY = {"unsw": "UNSW-NB15", "nsl_kdd": "NSL-KDD", "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS"}

LAMBDAS = ["0.0", "0.01", "0.05", "0.1", "0.5", "1.0"]
MARGINS = ["1.0", "2.0", "3.0", "4.0", "5.0", "9.0"]


def first_match(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def run_dir(time_tag, dataset, m, lam):
    return first_match(os.path.join(
        RES_DIR, f"t_{time_tag}_SSAE_se_*", "per_1.0",
        f"{dataset}_150e_m{m}_l{lam}_renormal_noise_std_*_catra0.1"))


def best_pseudo_val_auc(time_tag, dataset, seed, m, lam):
    d = first_match(os.path.join(
        RES_DIR, f"t_{time_tag}_SSAE_se_{seed}", "per_1.0",
        f"{dataset}_150e_m{m}_l{lam}_renormal_noise_std_*_catra0.1"))
    if d is None:
        return None
    csv_path = os.path.join(d, "mae01scale_max_loss_150e.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "Validation AUC" not in df.columns or df["Validation AUC"].isna().all():
        return None
    return float(df["Validation AUC"].max())


def sweep(sweep_name, values, m_for, lam_for, time_tag, seeds, baseline_value):
    print(f"\n=== {sweep_name} sweep (pseudo-validation AUC, max over epochs, {len(seeds)} seeds) ===")
    header = ["Dataset"] + [f"{sweep_name}={v}" for v in values]
    rows = []
    best_matches_baseline = []
    for ds in DATASETS:
        cell_means = []
        for v in values:
            m, lam = (m_for(v), lam_for(v))
            tag = time_tag(v)
            vals = [best_pseudo_val_auc(tag, ds, s, m, lam) for s in seeds]
            vals = [x for x in vals if x is not None]
            cell_means.append(np.mean(vals) if vals else float("nan"))
        rows.append(cell_means)
        if all(not np.isnan(x) for x in cell_means):
            best_idx = int(np.argmax(cell_means))
            best_val = values[best_idx]
            best_matches_baseline.append((ds, best_val, best_val == baseline_value))
    df_out = pd.DataFrame(rows, columns=[f"{sweep_name}={v}" for v in values])
    df_out.insert(0, "Dataset", [DATASET_DISPLAY[d] for d in DATASETS])
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nArg-best {sweep_name} per dataset (by pseudo-val AUC) vs paper baseline ({sweep_name}={baseline_value}):")
    for ds, best_val, matches in best_matches_baseline:
        flag = "MATCHES baseline" if matches else "DIFFERS from baseline"
        print(f"  {DATASET_DISPLAY[ds]}: best={best_val}  -> {flag}")
    return df_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # Lambda sweep: m fixed at 2.0, lambda in LAMBDAS; 0.1 comes from t_04 (baseline), rest from t_01.
    sweep(
        "lambda", LAMBDAS,
        m_for=lambda v: "2.0",
        lam_for=lambda v: v,
        time_tag=lambda v: "04" if v == "0.1" else "01",
        seeds=seeds,
        baseline_value="0.1",
    )

    # Margin sweep: lambda fixed at 0.1, m in MARGINS; 2.0 comes from t_04 (baseline), rest from t_02.
    sweep(
        "m", MARGINS,
        m_for=lambda v: v,
        lam_for=lambda v: "0.1",
        time_tag=lambda v: "04" if v == "2.0" else "02",
        seeds=seeds,
        baseline_value="2.0",
    )


if __name__ == "__main__":
    main()
