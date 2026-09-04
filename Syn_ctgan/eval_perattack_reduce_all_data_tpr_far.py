import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


# =========================
# User configuration
# =========================
# DATASETS = ["unsw", "cic_ids", "nsl_kdd", "cic_iot", "cidds_001"]
# SEEDS = [42, 43]
DATASETS = ["cidds_001"]
SEEDS = [42]

# These values mirror train_reduce_ctgan.py arguments and build result paths.
EPOCHS = 100
BATCH_SIZE = 128
LAMDA = 0.1
MARGIN = 1.0
MAE_LOSS_MODE = "renormal"
PERCENTAGE = 1.0
TYPE_TRAIN = "noise_std"
MAE_START_RATIO = 0.25
CTGAN_NOISE_START_RATIO = 0.75
CTGAN_NOISE_STD = 0.05
CTGAN_NOISE_FEATURE_RATIO = 0.5
CTGAN_PSEUDO_ANOMALY_MODE = "origin"
CTGAN_FEATURE_OUTLIER_R = 1.0
TIME = 1

USE_BEST_CHECKPOINT = True
EPOCH_CHECKPOINT = None
ALLOW_FALLBACK_RESULT_DIR = True

FAR_LEVELS = (0.01, 0.10)
MODEL_NAME = "SSAE-Reduce"


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from process_data import load_cidds_001_test_attacks  # noqa: E402
from sdv.single_table import REDUCEMAECTGANSynthesizer  # noqa: E402


def normalize_dataset_name(name):
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "cidds001":
        return "cidds_001"
    return normalized


def time_tag():
    return f"{int(TIME):02d}"


def result_dir_name(dataset):
    prefix = f"reduce_mae_{dataset}_{EPOCHS}_m_{MARGIN}_l_{LAMDA}"
    common = (
        f"{prefix}_maeloss_{MAE_LOSS_MODE}"
        f"_mstart_{MAE_START_RATIO}_nstart_{CTGAN_NOISE_START_RATIO}"
        f"_nfeat_{CTGAN_NOISE_FEATURE_RATIO}"
        f"_pmode_{CTGAN_PSEUDO_ANOMALY_MODE}"
    )
    if CTGAN_PSEUDO_ANOMALY_MODE == "feature_outlier":
        return f"{common}_r_{CTGAN_FEATURE_OUTLIER_R}"
    return f"{common}_nstd_{CTGAN_NOISE_STD}"


def csv_result_name(dataset, seed):
    parts = [
        "reduce",
        dataset,
        str(EPOCHS),
        f"b_{BATCH_SIZE}",
        f"m_{MARGIN}",
        f"l_{LAMDA}",
        f"maeloss_{MAE_LOSS_MODE}",
        f"per_{PERCENTAGE}",
        f"type_{TYPE_TRAIN}",
        f"mstart_{MAE_START_RATIO}",
        f"nstart_{CTGAN_NOISE_START_RATIO}",
        f"nfeat_{CTGAN_NOISE_FEATURE_RATIO}",
        f"pmode_{CTGAN_PSEUDO_ANOMALY_MODE}",
    ]
    if CTGAN_PSEUDO_ANOMALY_MODE == "feature_outlier":
        parts.append(f"r_{CTGAN_FEATURE_OUTLIER_R}")
    else:
        parts.append(f"nstd_{CTGAN_NOISE_STD}")
    parts.extend([f"seed_{seed}", f"time_{time_tag()}", "per_attack"])
    return "_".join(parts) + ".csv"


def summary_csv_result_name():
    normalized_datasets = [normalize_dataset_name(dataset) for dataset in DATASETS]
    dataset_tag = "all_datasets" if len(normalized_datasets) != 1 else normalized_datasets[0]
    seed_tag = "all_seeds" if len(SEEDS) != 1 else f"seed_{SEEDS[0]}"
    parts = [
        "reduce",
        dataset_tag,
        str(EPOCHS),
        f"b_{BATCH_SIZE}",
        f"m_{MARGIN}",
        f"l_{LAMDA}",
        f"maeloss_{MAE_LOSS_MODE}",
        f"per_{PERCENTAGE}",
        f"type_{TYPE_TRAIN}",
        f"mstart_{MAE_START_RATIO}",
        f"nstart_{CTGAN_NOISE_START_RATIO}",
        f"nfeat_{CTGAN_NOISE_FEATURE_RATIO}",
        f"pmode_{CTGAN_PSEUDO_ANOMALY_MODE}",
    ]
    if CTGAN_PSEUDO_ANOMALY_MODE == "feature_outlier":
        parts.append(f"r_{CTGAN_FEATURE_OUTLIER_R}")
    else:
        parts.append(f"nstd_{CTGAN_NOISE_STD}")
    parts.extend([seed_tag, f"time_{time_tag()}", "per_attack"])
    return "_".join(parts) + ".csv"


def checkpoint_path(result_dir, dataset):
    checkpoint_dir = result_dir / "checkpoints"
    if USE_BEST_CHECKPOINT:
        return checkpoint_dir / f"MAE_{dataset}_best.pkl"
    if EPOCH_CHECKPOINT is None:
        return checkpoint_dir / f"MAE_{dataset}_epoch_NONE.pkl"
    return checkpoint_dir / f"MAE_{dataset}_epoch_{EPOCH_CHECKPOINT}.pkl"


def _candidate_score(path, dataset):
    name = path.name
    score = 0
    required_prefix = f"reduce_mae_{dataset}_{EPOCHS}_m_{MARGIN}_l_{LAMDA}"
    if name.startswith(required_prefix):
        score += 100
    if f"_maeloss_{MAE_LOSS_MODE}" in name:
        score += 15
    if f"_mstart_{MAE_START_RATIO}" in name:
        score += 8
    if f"_nstart_{CTGAN_NOISE_START_RATIO}" in name:
        score += 8
    if f"_nfeat_{CTGAN_NOISE_FEATURE_RATIO}" in name:
        score += 6
    if f"_pmode_{CTGAN_PSEUDO_ANOMALY_MODE}" in name:
        score += 12
    if CTGAN_PSEUDO_ANOMALY_MODE == "feature_outlier":
        if f"_r_{CTGAN_FEATURE_OUTLIER_R}" in name:
            score += 5
    elif f"_nstd_{CTGAN_NOISE_STD}" in name:
        score += 5
    return score


def find_result_dir(dataset, seed):
    run_folder = f"{TYPE_TRAIN}_seed_{seed}_time_{time_tag()}"
    per_dir = BASE_DIR / "results" / run_folder / f"per_{PERCENTAGE}"
    exact = per_dir / result_dir_name(dataset)
    if exact.exists() and checkpoint_path(exact, dataset).exists():
        return exact
    if exact.exists():
        print(f"[WARN] Exact directory exists but checkpoint is missing: {checkpoint_path(exact, dataset)}")

    if not ALLOW_FALLBACK_RESULT_DIR or not per_dir.exists():
        raise FileNotFoundError(f"Missing result directory: {exact}")

    prefix = f"reduce_mae_{dataset}_{EPOCHS}_m_{MARGIN}_l_{LAMDA}"
    candidates = [
        path for path in per_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    candidates = [
        path for path in candidates
        if checkpoint_path(path, dataset).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No reduce result directory with checkpoint matching '{prefix}' under: {per_dir}"
        )

    candidates.sort(
        key=lambda path: (_candidate_score(path, dataset), path.stat().st_mtime),
        reverse=True,
    )
    chosen = candidates[0]
    print(f"[WARN] Exact directory not found for {dataset}, seed={seed}.")
    print(f"       Expected: {exact}")
    print(f"       Using:    {chosen}")
    return chosen


def load_synthesizer(result_dir, dataset):
    if not USE_BEST_CHECKPOINT and EPOCH_CHECKPOINT is None:
        raise ValueError("Set EPOCH_CHECKPOINT when USE_BEST_CHECKPOINT is False.")

    model_path = checkpoint_path(result_dir, dataset)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    return REDUCEMAECTGANSynthesizer.load(str(model_path)), model_path


def _clean_frame(df):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


def _string_columns(df):
    df = df.copy()
    df.columns = df.columns.map(str)
    return df


def load_unsw_attacks():
    test_path = BASE_DIR / "data" / "unsw" / "UNSW_NB15_testing-set.csv"
    test_data = _clean_frame(pd.read_csv(test_path))
    attack_labels = test_data["attack_cat"].copy()
    x_test = test_data.drop(["id", "attack_cat", "label"], axis=1)
    return _string_columns(x_test), attack_labels, "Normal"


def load_cic_ids_attacks():
    test_path = BASE_DIR / "data" / "cic-ids" / "CIC-IDS-2017-Test-10percent.csv"
    test_data = _clean_frame(pd.read_csv(test_path, sep=",", header=None))
    attack_labels = test_data[85].copy()
    x_test = test_data.drop([0, 1, 2, 4, 7, 85], axis=1)
    return _string_columns(x_test), attack_labels, "BENIGN"


def load_nsl_kdd_attacks():
    test_path = BASE_DIR / "data" / "NSLKDD" / "NSLKDD_Test.csv"
    test_data = _clean_frame(pd.read_csv(test_path, header=None))
    attack_labels = test_data.iloc[:, -1].astype(int)
    x_test = test_data.iloc[:, :-1]
    return _string_columns(x_test), attack_labels, 0


def load_cic_iot_attacks():
    test_path = BASE_DIR / "data" / "cic_iot" / "test.csv"
    test_data = _clean_frame(pd.read_csv(test_path, header=None))
    attack_labels = test_data.iloc[:, -1].copy()
    x_test = test_data.iloc[:, :-1]
    return _string_columns(x_test), attack_labels, "Benign"


def load_cidds_001_attacks(seed):
    x_test, attack_labels, normal_name = load_cidds_001_test_attacks(
        seed=seed,
        test_size=0.4,
    )
    return _string_columns(x_test), attack_labels, normal_name


def load_attack_groups(dataset, seed):
    loaders = {
        "unsw": load_unsw_attacks,
        "cic_ids": load_cic_ids_attacks,
        "nsl_kdd": load_nsl_kdd_attacks,
        "cic_iot": load_cic_iot_attacks,
    }
    if dataset == "cidds_001":
        x_test, attack_labels, normal_name = load_cidds_001_attacks(seed)
    elif dataset in loaders:
        x_test, attack_labels, normal_name = loaders[dataset]()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    normal_mask = attack_labels == normal_name
    x_normal = x_test.loc[normal_mask].reset_index(drop=True)
    if x_normal.empty:
        raise ValueError(f"No normal samples found for dataset={dataset}.")

    attack_names = sorted(
        [label for label in pd.unique(attack_labels) if label != normal_name],
        key=lambda value: str(value),
    )
    return x_test, attack_labels, normal_name, x_normal, attack_names


def evaluate_ids_metrics(y_true, scores, far_levels=FAR_LEVELS):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        auc = 0.0
        pr_auc = 0.0
    else:
        auc = roc_auc_score(y_true, scores) * 100.0
        pr_auc = average_precision_score(y_true, scores) * 100.0

    results = {"AUC": auc, "PR-AUC": pr_auc}
    scores_normal = scores[y_true == 0]
    if scores_normal.size == 0:
        for far in far_levels:
            results[f"TPR@FAR={int(far * 100)}%"] = 0.0
        return results

    for far in far_levels:
        threshold = np.percentile(scores_normal, 100.0 * (1.0 - far))
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        results[f"TPR@FAR={int(far * 100)}%"] = tpr * 100.0

    return results


def score_combined(synthesizer, x_normal, x_attack):
    y_normal = np.zeros(len(x_normal), dtype=int)
    y_attack = np.ones(len(x_attack), dtype=int)
    x_combined = pd.concat([x_normal, x_attack], ignore_index=True)
    y_combined = np.concatenate([y_normal, y_attack])

    _, z_norm_scores, _ = synthesizer.get_score_mae(x_combined, y_combined)
    return y_combined, z_norm_scores


def evaluate_dataset_seed(dataset, seed, attack_cache):
    dataset = normalize_dataset_name(dataset)
    result_dir = find_result_dir(dataset, seed)
    synthesizer, model_path = load_synthesizer(result_dir, dataset)

    cache_key = (dataset, seed)
    if cache_key not in attack_cache:
        attack_cache[cache_key] = load_attack_groups(dataset, seed)

    _, attack_labels, normal_name, x_normal, attack_names = attack_cache[cache_key]
    rows = []
    print(
        f"[INFO] dataset={dataset} seed={seed} normal={normal_name} "
        f"normal_samples={len(x_normal)} attacks={len(attack_names)}"
    )

    for attack_name in attack_names:
        attack_mask = attack_labels == attack_name
        x_test, _, _, _, _ = attack_cache[cache_key]
        x_attack = x_test.loc[attack_mask].reset_index(drop=True)
        if x_attack.empty:
            print(f"[WARN] Skip empty attack={attack_name} dataset={dataset}")
            continue

        print(f"[RUN] dataset={dataset} seed={seed} attack={attack_name} n={len(x_attack)}")
        y_true, scores = score_combined(synthesizer, x_normal, x_attack)
        metrics = evaluate_ids_metrics(y_true, scores)
        rows.append({
            "Dataset": dataset,
            "Seed": seed,
            "Time": time_tag(),
            "Model": MODEL_NAME,
            "Attack": attack_name,
            "NormalSamples": len(x_normal),
            "AttackSamples": len(x_attack),
            "ResultDir": str(result_dir),
            "Checkpoint": str(model_path),
            **metrics,
        })

    df = pd.DataFrame(rows)
    save_dir = result_dir / "per_attack_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / csv_result_name(dataset, seed)
    df.to_csv(out_file, index=False)
    print(f"[SAVE] {out_file}")
    return df


def main():
    attack_cache = {}
    all_frames = []
    errors = []

    for seed in SEEDS:
        for dataset_input in DATASETS:
            dataset = normalize_dataset_name(dataset_input)
            try:
                df = evaluate_dataset_seed(dataset, seed, attack_cache)
                if not df.empty:
                    all_frames.append(df)
            except Exception as exc:
                message = f"dataset={dataset} seed={seed}: {exc}"
                errors.append(message)
                print(f"[ERROR] {message}")

    summary = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    summary_dir = BASE_DIR / "results" / "per_attack_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / summary_csv_result_name()
    summary.to_csv(summary_file, index=False)
    print(f"[SAVE] {summary_file}")

    if errors:
        error_file = summary_dir / "reduce_all_datasets_all_seeds_per_attack_errors.txt"
        error_file.write_text("\n".join(errors), encoding="utf-8")
        print(f"[WARN] Some runs failed. See: {error_file}")


if __name__ == "__main__":
    main()
