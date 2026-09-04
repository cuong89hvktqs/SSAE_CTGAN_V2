"""Evaluation for DROCC -- brings its results to the same report format used
throughout this project for SSAE/CLAN, so the three are directly comparable:
  - Overall: AUC, F1, MCC, Precision, Recall, ACC at a 95th-percentile
    threshold (mirrors train_mae01scale_max.py's "total95" checkpoint eval).
  - Per rare/hard attack (the 2 paper-highlighted attacks per dataset, same
    ATTACK_PAIRS as visualize_centroid_distance_vs_clan.py): AUC, PR-AUC,
    TPR@FAR=1%, TPR@FAR=5% (mirrors val_perattack_mae01scale_max.py).

This script does NOT touch DROCC_icml2020/ -- it only (a) loads the trained
checkpoint DROCC_icml2020/tabular/log_ssae/<dataset>/seed<seed>/model_epilepsy.pt
via DROCC's own unmodified TwoLayer class, and (b) re-implements scoring
(DROCC's own train-set standardization, from one_class_main_tabular.py's
load_data()) purely for evaluation on top of it. Test-set reload uses
clan_style_get_data.get_data() -- same standalone, CLAN-independent
reimplementation prepare_drocc_data.py uses -- so nothing here depends on
the CLAN-main repo either.

Threshold sourcing intentionally differs by metric, matching each reference
script's own convention exactly rather than inventing a new one:
  - The single "total95" threshold (for F1/MCC/Precision/Recall/ACC) comes
    from the 95th percentile of anomaly score on val_data.npy -- a normal
    slice DROCC never trained on (see prepare_drocc_data.py's docstring).
  - Each TPR@FAR threshold comes from a percentile of the FULL TEST-normal
    score distribution, exactly as val_perattack_mae01scale_max.py does it
    (a different, but standard, convention for that specific metric).

Score convention: DROCC's classifier is trained so real normal points ->
label 1 (see loss_tabular.py / process_dataset.py). Its raw output is
sigmoid(logit) = P(normal), so anomaly_score = 1 - sigmoid(logit) here,
making "higher = more anomalous" -- consistent with this project's own
score conventions (total_score, 1-cos_sim, ...) even though DROCC's own
internal label convention (1=normal) is inverted from the rest of the
project (see CLAUDE.md).

Designed to be called once per (dataset, seed) right after that run's
training finishes (see run_drocc_datasets.sh), not just once at the end for
everything: every call MERGES its freshly-computed row(s) into whatever
<dataset>_seeds.csv / <dataset>_per_attack_seeds.csv already exist on disk
(replacing any prior row for the same seed/attack, keeping everything else),
recomputes that dataset's summary from the merged set, and recomputes the
all_datasets_summary.csv / all_datasets_per_attack_summary.csv rollups by
rescanning every *_summary.csv already on disk -- so results are visible and
up to date after each individual run, no separate final aggregation pass
needed. Calling it with multiple --seeds at once (the old batch style) still
works identically, just as a multi-row merge in one call.

--per (default 1.0) selects which data_ssae/per_<per>/ tree to read and
which results_ssae/per_<per>/ tree to write, matching prepare_drocc_data.py
/ run_drocc_datasets.sh's own --per (see clan_style_get_data.py's docstring
for the percentage semantics). It does NOT change how the test set itself
is built (percentage only ever shrinks train/val), just which prepared
data/checkpoint tree this call reads.

Usage: python eval_drocc.py --dataset <name> [--seeds 42,43,44] [--per 1.0]
Output: DROCC_icml2020/tabular/results_ssae/per_<per>/<dataset>_seeds.csv
        DROCC_icml2020/tabular/results_ssae/per_<per>/<dataset>_summary.csv
        DROCC_icml2020/tabular/results_ssae/per_<per>/<dataset>_per_attack_seeds.csv
        DROCC_icml2020/tabular/results_ssae/per_<per>/<dataset>_per_attack_summary.csv
        DROCC_icml2020/tabular/results_ssae/per_<per>/all_datasets_summary.csv
        DROCC_icml2020/tabular/results_ssae/per_<per>/all_datasets_per_attack_summary.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as T
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

from clan_style_get_data import DATASETS, SAMPLE_THRES, get_data

DROCC_ROOT = Path(__file__).parent.parent / "DROCC_icml2020"
DROCC_TABULAR = DROCC_ROOT / "tabular"

# Needed to `from models import TwoLayer` -- models.py itself does
# `from base.base_net import BaseNet`, a top-level import that only resolves
# when tabular/ is on sys.path (exactly how one_class_main_tabular.py runs
# it, since DROCC's own scripts are always invoked with CWD=tabular/).
if str(DROCC_TABULAR) not in sys.path:
    sys.path.insert(0, str(DROCC_TABULAR))
from models import TwoLayer  # noqa: E402  (DROCC's own, unmodified)

HD = 128  # must match --hd used in run_drocc_datasets.sh

# Same 2 rare/hard attacks per dataset used throughout this project
# (visualize_distance_to_centroid_vs_attacks.py, visualize_tsne_gendata.py,
# visualize_centroid_distance_vs_clan.py) -- (display name, CLAN-space label).
ATTACK_PAIRS = {
    "unsw":    [("Shellcode", "shellcode"), ("Worms", "worms")],
    "nsl_kdd": [("R2L", "3"), ("U2R", "4")],
    "cic_iot": [("Recon", "recon"), ("Spoofing", "spoofing")],
    "wsn_ds":  [("Blackhole", "blackhole"), ("Grayhole", "grayhole")],
}

FAR_LEVELS = (0.01, 0.05)  # TPR@FAR=1%, TPR@FAR=5%


def load_drocc_model(model_path, input_dim, device):
    model = TwoLayer(input_dim=input_dim, num_classes=1, num_hidden_nodes=HD).to(device)
    state = T.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@T.no_grad()
def anomaly_score(model, X, mean, std, device, chunk=8192):
    """1 - sigmoid(logit) -- see module docstring for the score-direction
    rationale. X is raw (CLAN-MinMax-space) features; standardized here with
    DROCC's own train mean/std, exactly as one_class_main_tabular.py's
    load_data() does it."""
    X_std = (X - mean) / (std + 1e-4)
    X_t = T.tensor(X_std, dtype=T.float32)
    scores = []
    for i in range(0, len(X_t), chunk):
        xb = X_t[i:i + chunk].to(device)
        logits = model(xb).squeeze(1)
        scores.append((1.0 - T.sigmoid(logits)).cpu().numpy())
    return np.concatenate(scores)


def predict_and_evaluate(scores, y_true_bin, threshold):
    """Same metric set/format as train_mae01scale_max.py's predict_and_evaluate."""
    y_pred = (scores > threshold).astype(int)
    auc = roc_auc_score(y_true_bin, scores) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
    acc = accuracy_score(y_true_bin, y_pred) * 100
    f1 = f1_score(y_true_bin, y_pred, zero_division=0) * 100
    precision = precision_score(y_true_bin, y_pred, zero_division=0) * 100
    recall = recall_score(y_true_bin, y_pred, zero_division=0) * 100
    mcc = matthews_corrcoef(y_true_bin, y_pred) * 100
    return {"Threshold": float(threshold), "AUC": auc, "F1": f1, "MCC": mcc,
            "Precision": precision, "Recall": recall, "ACC": acc}


def evaluate_ids_metrics(y_true, scores, normal_thresholds):
    """Same as val_perattack_mae01scale_max.py's evaluate_ids_metrics."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    results = {"AUC": roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0,
               "PR-AUC": average_precision_score(y_true, scores) * 100}
    for far, threshold in normal_thresholds.items():
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results[f"TPR@FAR={int(far * 100)}%"] = tp / (tp + fn + 1e-12) * 100
    return results


def evaluate_run(dataset, seed, device, data_root, log_root, percentile=95):
    data_dir = data_root / dataset / f"seed{seed}"
    model_path = log_root / dataset / f"seed{seed}" / "model_epilepsy.pt"
    train_path, val_path = data_dir / "train_data.npy", data_dir / "val_data.npy"
    for p in [train_path, val_path, model_path]:
        if not p.exists():
            print(f"  [skip] {dataset} seed={seed}: missing {p}")
            return None, None

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    mean, std = train_data.mean(0), train_data.std(0)

    cfg = DATASETS[dataset]
    x_train, y_train, x_val, y_val, x_te, y_te, x_zd, y_zd, label_mapping, feature_names = get_data(
        data_path=str(cfg["data_path"]), target=cfg["target"], drop=cfg["drop_cols"],
        class_zero=cfg["class_zero"], sample_thres=SAMPLE_THRES, split_seed=seed,
        test_ratio=0.2, val_ratio=0.2, anomaly_detection=True, n_samples=cfg["total_rows"],
        sample_seed=seed,
    )
    if x_zd is not None:
        x_te = np.concatenate([x_te, x_zd])
        y_te = np.concatenate([y_te, y_zd])
    label_names = {v: k for k, v in label_mapping.items()}
    attack_labels = pd.Series([label_names[y] for y in y_te])

    model = load_drocc_model(model_path, x_te.shape[1], device)
    scores_val = anomaly_score(model, val_data, mean, std, device)
    scores_test = anomaly_score(model, x_te, mean, std, device)

    y_true_bin = (y_te != 0).astype(int)  # project convention: 0=normal, 1=attack
    threshold_95 = float(np.percentile(scores_val, percentile))
    overall = predict_and_evaluate(scores_test, y_true_bin, threshold_95)
    overall.update({"Dataset": dataset, "Seed": seed, "N_test": len(y_te)})

    normal_mask = (y_te == 0)
    n_normal = int(normal_mask.sum())
    scores_normal = scores_test[normal_mask]
    normal_thresholds = {far: np.percentile(scores_normal, 100 * (1 - far)) for far in FAR_LEVELS}

    attack_rows = []
    for disp_name, clan_label in ATTACK_PAIRS[dataset]:
        mask = (attack_labels == clan_label).to_numpy()
        n_atk = int(mask.sum())
        if n_atk == 0:
            print(f"  [warn] {dataset} seed={seed}: no rows for attack '{clan_label}'")
            continue
        sub_scores = np.concatenate([scores_normal, scores_test[mask]])
        sub_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_atk, dtype=int)])
        metrics = evaluate_ids_metrics(sub_y, sub_scores, normal_thresholds)
        attack_rows.append({"Dataset": dataset, "Seed": seed, "Attack": disp_name,
                            "N": n_atk, **metrics})

    print(f"  [{dataset} seed={seed}] AUC={overall['AUC']:.2f}% F1={overall['F1']:.2f}% "
          f"MCC={overall['MCC']:.2f}% thr={threshold_95:.4f}")
    return overall, attack_rows


OVERALL_COLS = ["AUC", "F1", "MCC", "Precision", "Recall", "ACC"]
ATTACK_COLS = ["AUC", "PR-AUC"] + [f"TPR@FAR={int(f * 100)}%" for f in FAR_LEVELS]


def summarize(df, group_cols, metric_cols):
    rows = []
    for key, g in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["Seeds"] = len(g)
        if "N" in g.columns:
            row["N"] = int(g["N"].iloc[0])
        for m in metric_cols:
            row[f"{m} mean"] = g[m].mean()
            row[f"{m} std"] = g[m].std(ddof=0)
            row[m] = f"{g[m].mean():.2f}\u00b1{g[m].std(ddof=0):.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def merge_rows(new_df, csv_path, key_cols):
    """Read any existing csv_path, replace rows sharing new_df's key_cols
    with the fresh ones (so re-evaluating a seed overwrites its old row
    instead of duplicating it), keep everything else, and save."""
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        keep_mask = ~old_df.set_index(key_cols).index.isin(new_df.set_index(key_cols).index)
        merged = pd.concat([old_df[keep_mask], new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)
    return merged


def rebuild_all_datasets_rollup(out_dir):
    """Rescan every <dataset>_summary.csv / <dataset>_per_attack_summary.csv
    already on disk (not just the dataset(s) this call touched) so the
    cross-dataset rollup stays current even when called per single dataset."""
    overall_parts, attack_parts = [], []
    for name in DATASETS:
        p_overall = out_dir / f"{name}_summary.csv"
        if p_overall.exists():
            overall_parts.append(pd.read_csv(p_overall))
        p_attack = out_dir / f"{name}_per_attack_summary.csv"
        if p_attack.exists():
            attack_parts.append(pd.read_csv(p_attack))
    if overall_parts:
        pd.concat(overall_parts, ignore_index=True).to_csv(out_dir / "all_datasets_summary.csv", index=False)
    if attack_parts:
        pd.concat(attack_parts, ignore_index=True).to_csv(out_dir / "all_datasets_per_attack_summary.csv", index=False)


def evaluate_and_report(dataset, seeds, device, data_root, log_root, out_dir, percentile=95):
    """Evaluate `seeds` for `dataset`, merge into that dataset's on-disk
    seeds/summary CSVs, and refresh the all-datasets rollup. Safe to call
    repeatedly (e.g. once per finished training run) -- see module docstring."""
    print(f"\n================ {dataset} ================")
    overall_rows, attack_rows = [], []
    for seed in seeds:
        overall, attacks = evaluate_run(dataset, seed, device, data_root, log_root, percentile=percentile)
        if overall is not None:
            overall_rows.append(overall)
            attack_rows.extend(attacks)

    out_dir.mkdir(parents=True, exist_ok=True)

    if overall_rows:
        merged_overall = merge_rows(pd.DataFrame(overall_rows), out_dir / f"{dataset}_seeds.csv", ["Seed"])
        sum_overall = summarize(merged_overall, ["Dataset"], OVERALL_COLS)
        sum_overall.to_csv(out_dir / f"{dataset}_summary.csv", index=False)
        print(f"\n--- {dataset} | overall | mean\u00b1std over {len(merged_overall)} seed(s) ---")
        print(sum_overall[OVERALL_COLS].to_string(index=False))

    if attack_rows:
        merged_attack = merge_rows(pd.DataFrame(attack_rows), out_dir / f"{dataset}_per_attack_seeds.csv",
                                   ["Seed", "Attack"])
        sum_attack = summarize(merged_attack, ["Dataset", "Attack"], ATTACK_COLS)
        sum_attack.to_csv(out_dir / f"{dataset}_per_attack_summary.csv", index=False)
        print(f"\n--- {dataset} | per rare attack ---")
        print(sum_attack[["Attack", "N"] + ATTACK_COLS].to_string(index=False))

    rebuild_all_datasets_rollup(out_dir)


def main():
    parser = argparse.ArgumentParser("DROCC evaluation (AUC/F1/MCC + per-rare-attack TPR@FAR)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS.keys()])
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0",
                        help="which data_ssae/per_<per>/ tree to evaluate (matches prepare_drocc_data.py --per)")
    parser.add_argument("--percentile", type=float, default=95,
                        help="percentile of val-normal scores used as the overall threshold "
                             "(F1/MCC/Precision/Recall/ACC); does not affect AUC or the "
                             "per-attack TPR@FAR metrics. Non-95 values are written to a "
                             "separate results_ssae_val<P>/ tree so the val95 tree is untouched.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    data_root = DROCC_TABULAR / "data_ssae" / f"per_{args.per}"
    log_root = DROCC_TABULAR / "log_ssae" / f"per_{args.per}"
    results_dirname = "results_ssae" if args.percentile == 95 else f"results_ssae_val{int(args.percentile)}"
    out_dir = DROCC_TABULAR / results_dirname / f"per_{args.per}"

    device = "cuda" if T.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for dataset in names:
        evaluate_and_report(dataset, seeds, device, data_root, log_root, out_dir, percentile=args.percentile)

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
