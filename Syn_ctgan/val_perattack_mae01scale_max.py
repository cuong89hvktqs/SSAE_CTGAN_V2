"""Per-attack evaluation for MAE01ScaleMax (best val-AUC checkpoint).

Based on val_perattack_MAE_nsl_kdd_tpr@far.ipynb, generalized to 6 datasets:
    unsw, nsl_kdd, cic_iot, wsn_ds, cic_ids, nb_iot

For each dataset and each seed run under res/:
  1. Load the saved (already transformed) <ds>_X_test.csv — row-aligned with
     the loader's test set.
  2. Rebuild the per-attack (multiclass) labels of that same test set from
     the raw data, using the exact cleaning steps of process_data.py.
  3. Load checkpoints/MAE_<ds>_best_auc.pkl (best model by validation AUC).
  4. Score the full test set ONCE using the "total" score (re + lambda*||z||^2
     -- same formula as test_mode_total95.csv, the primary metric convention
     used throughout this project), then for each attack type evaluate on
     the subset {all normal test samples} + {samples of that attack}:
         AUC (%), PR-AUC (%), TPR@FAR=1%, TPR@FAR=5%
  5. Aggregate mean±std across seeds.

Outputs (under res/per_attack_results/):
    <ds>_per_attack_seeds.csv     -- one row per (seed, attack)
    <ds>_per_attack_summary.csv   -- mean±std across seeds
    all_datasets_per_attack_summary.csv

Usage:
    python val_perattack_mae01scale_max.py                 # all 5 datasets
    python val_perattack_mae01scale_max.py --d nsl_kdd     # one dataset
    python val_perattack_mae01scale_max.py --root res
"""

import argparse
import glob
import os
import re
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

import cloudpickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_synthesizer(model_path):
    """Load a pickled MAE01ScaleMax synthesizer, remapping tensors to
    whatever device is actually available on THIS machine.

    Checkpoints are saved with tensors pinned to the training machine's
    device (e.g. 'cuda'). Loading on a machine without a working CUDA
    device (no GPU, or a broken/mismatched driver) crashes deep inside
    torch's storage deserialization unless map_location is forced -- and
    cloudpickle.load() doesn't expose that parameter, so torch.load is
    monkey-patched for the duration of this call.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": device})
    try:
        with open(model_path, "rb") as f:
            synthesizer = cloudpickle.load(f)
    finally:
        torch.load = _orig_load
    synthesizer._model.set_device(device)
    return synthesizer

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds", "cic_ids","nb_iot"]
FAR_LEVELS = (0.01, 0.05)  # TPR@FAR=1%, TPR@FAR=5%


# =============================================================================
# Per-attack label loaders
# Each takes the run's training seed and returns
# (attack_labels: pd.Series[str], normal_tag: str), row-aligned with the test
# set produced by the corresponding loader in process_data.py.
# NOTE: wsn_ds is the only loader whose test split depends on the seed
# (train_mae01scale_max.py calls load_data_wsn_ds(seed=seed)); the fixed
# train/test datasets ignore it.
# =============================================================================

def attack_labels_unsw(seed=None):
    test_data = pd.read_csv(os.path.join(BASE_DIR, "data/unsw/UNSW_NB15_testing-set.csv"))
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.dropna(inplace=True)
    labels = test_data["attack_cat"].astype(str).str.strip().reset_index(drop=True)
    return labels, "Normal"


def attack_labels_nsl_kdd(seed=None):
    test_data = pd.read_csv(os.path.join(BASE_DIR, "data/NSLKDD/NSLKDD_Test.csv"), header=None)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.dropna(inplace=True)
    raw = test_data.iloc[:, -1].astype(int).reset_index(drop=True)
    labels = raw.map(lambda v: "normal" if v == 0 else f"attack_{v}")
    return labels, "normal"


def attack_labels_cic_iot(seed=None):
    test_data = pd.read_csv(os.path.join(BASE_DIR, "data/cic_iot/test.csv"), header=None).dropna()
    labels = test_data.iloc[:, -1].astype(str).str.strip().reset_index(drop=True)
    return labels, "Benign"


def attack_labels_cic_ids(seed=None):
    test_data = pd.read_csv(
        os.path.join(BASE_DIR, "data/cic-ids/CIC-IDS-2017-Test-10percent.csv"),
        sep=",", header=None,
    )
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.dropna(inplace=True)
    labels = test_data[85].astype(str).str.strip().reset_index(drop=True)
    return labels, "BENIGN"


def attack_labels_wsn_ds(seed, test_size=0.2):
    # Mirror load_data_wsn_ds(seed=seed) exactly — the training script passes
    # the run's training seed, so the test split is seed-dependent.
    path = os.path.join(BASE_DIR, "data", "wsn-ds", "WSN-DS.csv")
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    attack = df["Attack type"].astype(str).str.strip()
    y = (attack != "Normal").astype(int)
    X = df.drop(columns=[" id", "Attack type"], errors="ignore")

    # Same split call => same row selection; attack rides along.
    _, _, _, _, _, attack_test = train_test_split(
        X, y, attack, test_size=test_size, random_state=seed, stratify=y
    )
    return attack_test.reset_index(drop=True), "Normal"


def attack_labels_nb_iot(seed, test_size=0.2):
    # Mirror load_data_nb_iot(seed=seed, test_size=0.2) exactly (same file
    # concat order, same cleaning, same stratified split), carrying a
    # per-attack-type label alongside the binary label so it stays
    # row-aligned with X_test.
    # Raw nb-iot files have no label column at all, so both the binary
    # label and the attack-type name are synthesized here from the
    # source file/folder, same as the real loader.
    data_dir = os.path.join(BASE_DIR, "data", "nb-iot")

    device_dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "benign_traffic.csv"))
    )
    if not device_dirs:
        raise FileNotFoundError(f"No device folders with benign_traffic.csv found under {data_dir}")

    frames = []
    attack_names = []  # one entry per row, built in the same order as frames
    for device_dir in device_dirs:
        df_benign = pd.read_csv(os.path.join(device_dir, "benign_traffic.csv"))
        df_benign["label"] = 0
        frames.append(df_benign)
        attack_names.extend(["benign"] * len(df_benign))

        for attack_type, prefix in (("gafgyt_attacks", "gafgyt"), ("mirai_attacks", "mirai")):
            attack_dir = os.path.join(device_dir, attack_type)
            if not os.path.isdir(attack_dir):
                continue
            for attack_csv in sorted(glob.glob(os.path.join(attack_dir, "*.csv"))):
                df_attack = pd.read_csv(attack_csv)
                df_attack["label"] = 1
                frames.append(df_attack)
                name = f"{prefix}_{os.path.splitext(os.path.basename(attack_csv))[0]}"
                attack_names.extend([name] * len(df_attack))

    df = pd.concat(frames, ignore_index=True)
    attack_names = pd.Series(attack_names, name="attack")

    # Same row-survival mask as df.replace([inf,-inf], nan).dropna() in
    # process_data.py, applied identically to the parallel attack_names series.
    keep_mask = df.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    df = df[keep_mask].reset_index(drop=True)
    attack_names = attack_names[keep_mask].reset_index(drop=True)

    y_all = df["label"].astype(int)

    _, attack_test, _, _y_test = train_test_split(
        attack_names, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )
    return attack_test.reset_index(drop=True), "benign"


ATTACK_LOADERS = {
    "unsw":    attack_labels_unsw,
    "nsl_kdd": attack_labels_nsl_kdd,
    "cic_iot": attack_labels_cic_iot,
    "wsn_ds":  attack_labels_wsn_ds,
    "cic_ids": attack_labels_cic_ids,
    "nb_iot":  attack_labels_nb_iot,
}


# =============================================================================
# Metrics (same definitions as the notebook)
# =============================================================================

def evaluate_ids_metrics(y_true, scores, normal_thresholds, val_thresholds=None):
    """AUC (%), PR-AUC (%), TPR@FAR levels, and (optionally) MCC at the
    val95/val99 thresholds.

    normal_thresholds: {far: threshold} precomputed from the FULL normal
    test scores (identical to the notebook: subset normals == all normals).
    val_thresholds: optional {95: threshold_95, 99: threshold_99} from the
    val-normal percentile (same thresholds as best_auc/test_mode_total.csv)
    -- when given, adds MCC_95/MCC_99 to the result using the SAME
    convention as evaluate_best_auc_checkpoint() (strict ">" cutoff).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    results = {}

    results["AUC"] = roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0
    results["PR-AUC"] = average_precision_score(y_true, scores) * 100

    for far, threshold in normal_thresholds.items():
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results[f"TPR@FAR={int(far * 100)}%"] = tp / (tp + fn + 1e-12) * 100

    if val_thresholds:
        for percentile, threshold in val_thresholds.items():
            y_pred = (scores > threshold).astype(int)
            results[f"MCC_{percentile}"] = matthews_corrcoef(y_true, y_pred) * 100

    return results


# =============================================================================
# Per-run evaluation
# =============================================================================

def find_run_dirs(root, dataset, per="1.0"):
    # res holds THREE type_train variants per (dataset, seed) for
    # most datasets (noise / noise_std / std, from an earlier ablation study)
    # -- filter explicitly to "noise_std" (the standard/recommended config
    # used throughout this project) or every seed gets evaluated 3x with
    # results silently mixed together under the same seed label.
    per_folder = f"per_{per}"
    pattern = os.path.join(root, "mae01scale_max_se_*_t_*", per_folder,
                           f"{dataset}_*scalemax-noise_std_*")
    matches = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if matches:
        return matches

    # Nothing matched -- widen the search (drop the noise_std filter) so we
    # can tell the user exactly what IS on disk instead of a bare "not found".
    wide_pattern = os.path.join(root, "mae01scale_max_se_*_t_*", per_folder, f"{dataset}_*")
    wide_matches = sorted(d for d in glob.glob(wide_pattern) if os.path.isdir(d))
    if wide_matches:
        print(f"  [WARN] no 'scalemax-noise_std' run found for {dataset} under "
              f"'{per_folder}', but {len(wide_matches)} other run dir(s) exist "
              f"here (different type_train variant, or naming mismatch):")
        for d in wide_matches:
            print(f"           {d}")
    else:
        seed_dirs = sorted(glob.glob(os.path.join(root, "mae01scale_max_se_*_t_*")))
        print(f"  [WARN] no run dir at all matches '{dataset}_*' under "
              f"'{per_folder}' in any of {len(seed_dirs)} seed folder(s) in {root}:")
        for d in seed_dirs:
            per1 = os.path.join(d, per_folder)
            contents = sorted(os.listdir(per1)) if os.path.isdir(per1) else [f"<no {per_folder} folder>"]
            print(f"           {d}/{per_folder}/ -> {contents}")
    return []


def seed_from_path(path):
    m = re.search(r"_se_(\d+)_t_", path.replace("\\", "/"))
    return int(m.group(1)) if m else -1


def evaluate_run(run_dir, dataset, label_cache):
    """Evaluate one seed run with its best val-AUC model. Returns list of row dicts."""
    seed = seed_from_path(run_dir)
    if seed not in label_cache:
        label_cache[seed] = ATTACK_LOADERS[dataset](seed)
    attack_labels, normal_tag = label_cache[seed]

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    model_path  = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")  # best val AUC

    for p in [x_test_path, y_test_path, model_path]:
        if not os.path.exists(p):
            print(f"  [WARN] missing {p} -- skip run.")
            return []

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    # ── Alignment sanity checks ──────────────────────────────────────────────
    if len(attack_labels) != len(X_test):
        print(f"  [ERROR] attack labels ({len(attack_labels)}) != X_test rows "
              f"({len(X_test)}) -- skip run.")
        return []
    y_from_attack = (attack_labels != normal_tag).astype(int).to_numpy()
    mismatch = int((y_from_attack != y_test.to_numpy()).sum())
    if mismatch > 0:
        print(f"  [ERROR] binary label mismatch on {mismatch} rows -- alignment "
              f"broken, skip run.")
        return []

    synthesizer = load_synthesizer(model_path)

    # Score the FULL test set once; per-sample scores are independent,
    # so subsetting afterwards is identical to the notebook's per-attack calls.
    # Only the "total" score (re + lambda*||z||^2) is used -- same formula as
    # test_mode_total95.csv, the primary metric convention used throughout
    # this project. The "latent"-only score is intentionally not computed.
    _, _, total_scores, _ = synthesizer.get_score(X_test, y_test)
    total_scores = np.asarray(total_scores)

    normal_mask = (attack_labels == normal_tag).to_numpy()
    n_normal = int(normal_mask.sum())
    attacks = sorted(a for a in attack_labels.unique() if a != normal_tag)
    print(f"  seed={seed}: {n_normal} normal, {len(attacks)} attack types")

    scores_normal = total_scores[normal_mask]
    # threshold per FAR from percentile of ALL normal test scores
    normal_thresholds = {
        far: np.percentile(scores_normal, 100 * (1 - far)) for far in FAR_LEVELS
    }

    rows = []
    for attack in attacks:
        atk_mask = (attack_labels == attack).to_numpy()
        sub_scores = np.concatenate([scores_normal, total_scores[atk_mask]])
        sub_y = np.concatenate([
            np.zeros(n_normal, dtype=int),
            np.ones(int(atk_mask.sum()), dtype=int),
        ])
        metrics = evaluate_ids_metrics(sub_y, sub_scores, normal_thresholds)
        rows.append({
            "Dataset": dataset,
            "Seed": seed,
            "Attack": attack,
            "N": int(atk_mask.sum()),
            "Score": "total",
            **metrics,
        })
    return rows


# =============================================================================
# Aggregation
# =============================================================================

METRIC_COLS = ["AUC", "PR-AUC"] + [f"TPR@FAR={int(f * 100)}%" for f in FAR_LEVELS]


def summarize(df_seeds):
    """mean±std across seeds per (Dataset, Attack, Score)."""
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--d", type=str, default="all",
                        help=f"dataset ({', '.join(DATASETS)}) or 'all'")
    parser.add_argument("--root", type=str, default="res",
                        help="folder containing mae01scale_max_se_* runs")
    parser.add_argument("--per", type=str, default="1.0",
                        help="training-data fraction tag used at train time "
                             "(--per flag in train_mae01scale_max.py) -- "
                             "selects the 'per_<value>' subfolder, e.g. "
                             "'1.0' -> per_1.0, '0.5' -> per_0.5")
    args = parser.parse_args()

    root = args.root if os.path.isabs(args.root) else os.path.join(BASE_DIR, args.root)
    if not os.path.isdir(root):
        sys.exit(f"Root folder not found: {root}")

    targets = DATASETS if args.d == "all" else [args.d.strip().lower().replace("-", "_")]
    for ds in targets:
        if ds not in ATTACK_LOADERS:
            sys.exit(f"Unsupported dataset '{ds}'. Supported: {', '.join(DATASETS)}")

    # One subfolder per --per value (e.g. per_attack_results/per1.0/,
    # per_attack_results/per0.5/) so results from different training-data
    # fractions never overwrite each other.
    save_dir = os.path.join(root, "per_attack_results", f"per{args.per}")
    os.makedirs(save_dir, exist_ok=True)

    all_summaries = []
    for ds in targets:
        print(f"\n================ {ds} ================")
        run_dirs = find_run_dirs(root, ds, per=args.per)
        if not run_dirs:
            print(f"  [WARN] no runs found for {ds} -- skip.")
            continue

        label_cache = {}  # seed -> (attack_labels, normal_tag); wsn_ds is seed-dependent
        rows = []
        for run_dir in run_dirs:
            rows.extend(evaluate_run(run_dir, ds, label_cache))
        if not rows:
            print(f"  [WARN] no results for {ds}.")
            continue

        df_seeds = pd.DataFrame(rows)
        df_seeds.to_csv(os.path.join(save_dir, f"{ds}_per_attack_seeds.csv"), index=False)

        df_sum = summarize(df_seeds)
        df_sum.to_csv(os.path.join(save_dir, f"{ds}_per_attack_summary.csv"), index=False)
        all_summaries.append(df_sum)

        print(f"\n--- {ds} | score=total | best val-AUC | mean±std over "
              f"{int(df_sum['Seeds'].iloc[0])} seeds ---")
        print(df_sum[["Attack", "N"] + METRIC_COLS].to_string(index=False))

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            os.path.join(save_dir, "all_datasets_per_attack_summary.csv"), index=False
        )
        print(f"\nSaved results to: {save_dir}")


if __name__ == "__main__":
    main()
