"""Per-attack evaluation for a DeepOD algorithm at the validation-95th-
percentile threshold (see run_deepod_val95.py). Same structure as
eval_deepod_perattack.py, but reports Precision/Recall/F1/MCC at the ONE
fixed val95 threshold instead of a %FAR sweep, mirroring
eval_neutralad_perattack_val95.py.

Usage: python eval_deepod_perattack_val95.py --algo dsvdd [--per 1.0]
Outputs (under DeepOD/results_ssae_val95/<algo>/per_attack_results/per<per>/):
    <ds>_per_attack_seeds.csv, <ds>_per_attack_summary.csv, all_datasets_per_attack_summary.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

BASE_DIR = Path(__file__).parent
NEUTRALAD_ROOT = BASE_DIR.parent / "NeuTraL-AD"
DEEPOD_ROOT = BASE_DIR.parent / "DeepOD"
DATA_ROOT = NEUTRALAD_ROOT / "DATA" / "ssae"
RESULTS_ROOT_VAL95 = DEEPOD_ROOT / "results_ssae_val95"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
METRIC_COLS = ["AUC", "PR-AUC", "Precision", "Recall", "F1", "MCC"]


def metrics_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0,
        "PR-AUC": average_precision_score(y_true, scores) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "F1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "MCC": matthews_corrcoef(y_true, y_pred) * 100,
    }


def evaluate_run(algo, dataset, seed, per):
    res_dir = RESULTS_ROOT_VAL95 / algo / f"per_{per}" / dataset / f"seed{seed}"
    data_dir = DATA_ROOT / f"per_{per}" / dataset / f"seed{seed}"

    metrics_path = res_dir / "metrics_val95.json"
    scores_path = res_dir / "test_scores.npy"
    labels_path = res_dir / "test_labels_out.npy"
    attacks_path = data_dir / "test_attack_types.npy"
    tag_path = data_dir / "normal_tag.txt"
    for p in [metrics_path, scores_path, labels_path, attacks_path, tag_path]:
        if not p.exists():
            print(f"  [WARN] missing {p} -- skip run (run run_deepod_val95.py first).")
            return []

    with open(metrics_path) as f:
        val_threshold = json.load(f)["Threshold"]

    scores = np.load(scores_path)
    labels = np.load(labels_path).astype(int)
    attack_labels = np.load(attacks_path, allow_pickle=True)
    normal_tag = tag_path.read_text().strip()

    if len(scores) != len(attack_labels):
        print(f"  [ERROR] {algo} {dataset} seed={seed}: scores/attack_types length mismatch -- skip run.")
        return []
    y_from_attack = (attack_labels != normal_tag).astype(int)
    mismatch = int((y_from_attack != labels).sum())
    if mismatch > 0:
        print(f"  [ERROR] {algo} {dataset} seed={seed}: binary label mismatch on {mismatch} rows -- skip run.")
        return []

    normal_mask = attack_labels == normal_tag
    n_normal = int(normal_mask.sum())
    attacks = sorted(a for a in np.unique(attack_labels) if a != normal_tag)
    print(f"  seed={seed}: {n_normal} normal, {len(attacks)} attack types, val95_threshold={val_threshold:.4f}")

    scores_normal = scores[normal_mask]
    rows = []
    for attack in attacks:
        atk_mask = attack_labels == attack
        sub_scores = np.concatenate([scores_normal, scores[atk_mask]])
        sub_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(int(atk_mask.sum()), dtype=int)])
        metrics = metrics_at_threshold(sub_y, sub_scores, val_threshold)
        rows.append({"Dataset": dataset, "Seed": seed, "Attack": attack,
                     "N": int(atk_mask.sum()), "Threshold": val_threshold, **metrics})
    return rows


def summarize(df_seeds):
    rows = []
    for (ds, attack), g in df_seeds.groupby(["Dataset", "Attack"]):
        row = {"Dataset": ds, "Attack": attack, "N": int(g["N"].iloc[0]), "Seeds": len(g)}
        for m in METRIC_COLS:
            row[f"{m} mean"] = g[m].mean()
            row[f"{m} std"] = g[m].std(ddof=0)
            row[m] = f"{g[m].mean():.2f}±{g[m].std(ddof=0):.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algo", type=str, required=True,
                        choices=["dsvdd", "goad", "neutral", "icl", "dif"])
    parser.add_argument("--d", type=str, default="all", help=f"dataset ({', '.join(DATASETS)}) or 'all'")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    args = parser.parse_args()

    targets = DATASETS if args.d == "all" else [args.d.strip().lower().replace("-", "_")]
    seeds = [int(s) for s in args.seeds.split(",")]

    save_dir = RESULTS_ROOT_VAL95 / args.algo / "per_attack_results" / f"per{args.per}"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for ds in targets:
        print(f"\n================ {args.algo} | {ds} (val95) ================")
        rows = []
        for seed in seeds:
            rows.extend(evaluate_run(args.algo, ds, seed, args.per))
        if not rows:
            print(f"  [WARN] no results for {ds}.")
            continue

        df_seeds = pd.DataFrame(rows)
        df_seeds.to_csv(save_dir / f"{ds}_per_attack_seeds.csv", index=False)
        df_sum = summarize(df_seeds)
        df_sum.to_csv(save_dir / f"{ds}_per_attack_summary.csv", index=False)
        all_summaries.append(df_sum)

        print(f"\n--- {ds} | {args.algo} @ val95 | mean±std over {int(df_sum['Seeds'].iloc[0])} seeds ---")
        print(df_sum[["Attack", "N"] + METRIC_COLS].to_string(index=False))

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            save_dir / "all_datasets_per_attack_summary.csv", index=False
        )
        print(f"\nSaved results to: {save_dir}")


if __name__ == "__main__":
    main()
