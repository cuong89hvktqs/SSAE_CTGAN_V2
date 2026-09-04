"""Overall (whole-test-set) AUC/F1/MCC/Precision/Recall/ACC summary for a
DeepOD algorithm at the validation-95th-percentile threshold, mean+-std
across seeds. Reads metrics_val95.json written by run_deepod_val95.py.

Usage: python aggregate_deepod_val95_overall.py --algo dsvdd [--per 1.0]
Output: DeepOD/results_ssae_val95/<algo>/per_<per>/overall_summary_seeds.csv
        DeepOD/results_ssae_val95/<algo>/per_<per>/overall_summary.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DEEPOD_ROOT = Path(__file__).parent.parent / "DeepOD"
RESULTS_ROOT_VAL95 = DEEPOD_ROOT / "results_ssae_val95"
DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
METRIC_COLS = ["AUC", "F1", "MCC", "Precision", "Recall", "ACC"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algo", type=str, required=True,
                        choices=["dsvdd", "goad", "neutral", "icl", "dif"])
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--seeds", type=str, default="42,43,44")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for ds in datasets:
        for seed in seeds:
            p = RESULTS_ROOT_VAL95 / args.algo / f"per_{args.per}" / ds / f"seed{seed}" / "metrics_val95.json"
            if not p.exists():
                print(f"  [WARN] missing {p} -- skip (run run_deepod_val95.py first).")
                continue
            with open(p) as f:
                m = json.load(f)
            n_test = m["TN"] + m["FP"] + m["FN"] + m["TP"]
            rows.append({"Dataset": ds, "Seed": seed, "N": n_test, "Threshold": m["Threshold"],
                        **{k: m[k] for k in METRIC_COLS}})

    if not rows:
        print("No results found.")
        return

    df_seeds = pd.DataFrame(rows)
    out_dir = RESULTS_ROOT_VAL95 / args.algo / f"per_{args.per}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_seeds.to_csv(out_dir / "overall_summary_seeds.csv", index=False)

    rows_sum = []
    for ds, g in df_seeds.groupby("Dataset"):
        row = {"Dataset": ds, "N": int(g["N"].iloc[0]), "Seeds": len(g)}
        for m in METRIC_COLS:
            row[f"{m} mean"] = g[m].mean()
            row[f"{m} std"] = g[m].std(ddof=0)
            row[m] = f"{g[m].mean():.2f}±{g[m].std(ddof=0):.2f}"
        rows_sum.append(row)
    df_sum = pd.DataFrame(rows_sum)
    df_sum.to_csv(out_dir / "overall_summary.csv", index=False)

    print(f"\n--- {args.algo} overall @ val95 threshold | per={args.per} | mean±std across seeds ---")
    print(df_sum[["Dataset", "N"] + METRIC_COLS].to_string(index=False))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
