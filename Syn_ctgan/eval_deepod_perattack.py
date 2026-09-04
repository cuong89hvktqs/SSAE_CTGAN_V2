"""Per-attack evaluation for a DeepOD algorithm (dsvdd/goad/neutral/icl/dif)
-- same metric definitions and CSV output convention as
val_perattack_mae01scale_max.py / eval_neutralad_perattack.py, so all three
are directly comparable:
    AUC (%), PR-AUC (%), TPR@FAR=1%, TPR@FAR=5%
computed per (dataset, seed, attack type) on {all normal test rows} +
{rows of that attack type}, FAR thresholds calibrated once from the
percentile of ALL normal test scores.

Reads, per (algo, dataset, seed):
    DeepOD/results_ssae/<algo>/per_<per>/<dataset>/seed<seed>/
        test_scores.npy, test_labels_out.npy      -- from run_deepod.py
    NeuTraL-AD/DATA/ssae/per_<per>/<dataset>/seed<seed>/
        test_attack_types.npy, normal_tag.txt      -- from prepare_neutralad_data.py
        (reused as-is -- same row order, same encoding, algorithm-agnostic)

Usage:
    python eval_deepod_perattack.py --algo dsvdd            # all 4 datasets
    python eval_deepod_perattack.py --algo goad --d nsl_kdd
Outputs (under DeepOD/results_ssae/<algo>/per_attack_results/per<per>/):
    <ds>_per_attack_seeds.csv
    <ds>_per_attack_summary.csv
    all_datasets_per_attack_summary.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

BASE_DIR = Path(__file__).parent
NEUTRALAD_ROOT = BASE_DIR.parent / "NeuTraL-AD"
DEEPOD_ROOT = BASE_DIR.parent / "DeepOD"
DATA_ROOT = NEUTRALAD_ROOT / "DATA" / "ssae"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
FAR_LEVELS = (0.01, 0.05)
METRIC_COLS = ["AUC", "PR-AUC"] + [f"TPR@FAR={int(f * 100)}%" for f in FAR_LEVELS]


def evaluate_ids_metrics(y_true, scores, normal_thresholds):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    results = {}
    results["AUC"] = roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0
    results["PR-AUC"] = average_precision_score(y_true, scores) * 100
    for far, threshold in normal_thresholds.items():
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results[f"TPR@FAR={int(far * 100)}%"] = tp / (tp + fn + 1e-12) * 100
    return results


def evaluate_run(algo, dataset, seed, per, results_root):
    res_dir = results_root / algo / f"per_{per}" / dataset / f"seed{seed}"
    data_dir = DATA_ROOT / f"per_{per}" / dataset / f"seed{seed}"

    scores_path = res_dir / "test_scores.npy"
    labels_path = res_dir / "test_labels_out.npy"
    attacks_path = data_dir / "test_attack_types.npy"
    tag_path = data_dir / "normal_tag.txt"
    for p in [scores_path, labels_path, attacks_path, tag_path]:
        if not p.exists():
            print(f"  [WARN] missing {p} -- skip run.")
            return []

    scores = np.load(scores_path)
    labels = np.load(labels_path).astype(int)
    attack_labels = np.load(attacks_path, allow_pickle=True)
    normal_tag = tag_path.read_text().strip()

    if len(scores) != len(attack_labels):
        print(f"  [ERROR] {algo} {dataset} seed={seed}: scores ({len(scores)}) != "
              f"attack_types ({len(attack_labels)}) -- skip run.")
        return []
    y_from_attack = (attack_labels != normal_tag).astype(int)
    mismatch = int((y_from_attack != labels).sum())
    if mismatch > 0:
        print(f"  [ERROR] {algo} {dataset} seed={seed}: binary label mismatch on "
              f"{mismatch} rows -- alignment broken, skip run.")
        return []

    normal_mask = attack_labels == normal_tag
    n_normal = int(normal_mask.sum())
    attacks = sorted(a for a in np.unique(attack_labels) if a != normal_tag)
    print(f"  seed={seed}: {n_normal} normal, {len(attacks)} attack types")

    scores_normal = scores[normal_mask]
    normal_thresholds = {
        far: np.percentile(scores_normal, 100 * (1 - far)) for far in FAR_LEVELS
    }

    rows = []
    for attack in attacks:
        atk_mask = attack_labels == attack
        sub_scores = np.concatenate([scores_normal, scores[atk_mask]])
        sub_y = np.concatenate([
            np.zeros(n_normal, dtype=int),
            np.ones(int(atk_mask.sum()), dtype=int),
        ])
        metrics = evaluate_ids_metrics(sub_y, sub_scores, normal_thresholds)
        rows.append({
            "Dataset": dataset, "Seed": seed, "Attack": attack,
            "N": int(atk_mask.sum()), "Score": algo, **metrics,
        })
    return rows


def summarize(df_seeds):
    rows = []
    for (ds, attack, score), g in df_seeds.groupby(["Dataset", "Attack", "Score"]):
        row = {"Dataset": ds, "Attack": attack, "Score": score,
               "N": int(g["N"].iloc[0]), "Seeds": len(g)}
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
    parser.add_argument("--results-root", type=str, default=str(DEEPOD_ROOT / "results_ssae"))
    args = parser.parse_args()

    results_root = Path(args.results_root)
    targets = DATASETS if args.d == "all" else [args.d.strip().lower().replace("-", "_")]
    seeds = [int(s) for s in args.seeds.split(",")]

    save_dir = results_root / args.algo / "per_attack_results" / f"per{args.per}"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for ds in targets:
        print(f"\n================ {args.algo} | {ds} ================")
        rows = []
        for seed in seeds:
            rows.extend(evaluate_run(args.algo, ds, seed, args.per, results_root))
        if not rows:
            print(f"  [WARN] no results for {ds}.")
            continue

        df_seeds = pd.DataFrame(rows)
        df_seeds.to_csv(save_dir / f"{ds}_per_attack_seeds.csv", index=False)

        df_sum = summarize(df_seeds)
        df_sum.to_csv(save_dir / f"{ds}_per_attack_summary.csv", index=False)
        all_summaries.append(df_sum)

        print(f"\n--- {ds} | {args.algo} | mean±std over {int(df_sum['Seeds'].iloc[0])} seeds ---")
        print(df_sum[["Attack", "N"] + METRIC_COLS].to_string(index=False))

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            save_dir / "all_datasets_per_attack_summary.csv", index=False
        )
        print(f"\nSaved results to: {save_dir}")


if __name__ == "__main__":
    main()
