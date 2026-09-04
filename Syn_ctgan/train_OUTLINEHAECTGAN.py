import argparse
import csv
import json
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from process_data import load_data_cic_ids, load_data_cic_iot, load_data_nsl_kdd, load_data_unsw
from sdv.metadata import SingleTableMetadata
from sdv.single_table import OUTLINEHAECTGANSynthesizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset: str):
    if dataset == "unsw":
        return load_data_unsw()
    if dataset == "cic_ids":
        return load_data_cic_ids()
    if dataset == "nsl_kdd":
        return load_data_nsl_kdd()
    if dataset == "cic_iot":
        return load_data_cic_iot()
    raise ValueError(f"Unsupported dataset: {dataset}")


def save_pickle(obj, save_dir: str, name: str) -> str:
    path = os.path.abspath(os.path.join(save_dir, name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(_path_for_io(path), "wb") as f:
        pickle.dump(obj, f)
    return path


def _path_for_io(path: str) -> str:
    """Return a Windows long-path-safe path for file IO."""
    abs_path = os.path.abspath(path)
    if os.name == "nt" and len(abs_path) >= 240 and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path


def save_tsne_test_normal_vs_ae_generated(
    x_test,
    y_test_bin,
    generated_csv_path: str,
    save_dir: str,
    dataset: str,
    ctgan_transformer=None,
    seed: int = 42,
    max_per_class: int = 2000,
    perplexity: float = 30.0,
):
    """Run t-SNE on real normal test rows and generated AE-train samples."""
    generated_csv_io_path = _path_for_io(generated_csv_path)
    if not os.path.exists(generated_csv_io_path):
        print(f"[TSNE] Skip: missing generated sample file: {generated_csv_path}")
        return None

    y_test_bin = np.asarray(y_test_bin).astype(int)
    normal_mask = y_test_bin == 0
    generated_df = pd.read_csv(generated_csv_io_path)
    if isinstance(x_test, pd.DataFrame):
        if set(x_test.columns).issubset(set(generated_df.columns)):
            generated_df = generated_df.loc[:, x_test.columns]
        x_normal_raw = x_test.loc[normal_mask].reset_index(drop=True)
    else:
        x_normal_raw = pd.DataFrame(np.asarray(x_test)[normal_mask])

    if x_normal_raw.shape[0] == 0:
        print("[TSNE] Skip: no normal rows found in test data.")
        return None
    if generated_df.shape[0] == 0:
        print("[TSNE] Skip: generated sample file is empty.")
        return None

    if ctgan_transformer is not None:
        try:
            x_normal = ctgan_transformer.transform(x_normal_raw)
            x_generated = ctgan_transformer.transform(generated_df)
        except Exception as exc:
            print(f"[TSNE] Skip: CTGAN transform failed: {exc}")
            return None

        if isinstance(x_normal, pd.DataFrame):
            x_normal = x_normal.to_numpy(dtype=np.float32)
        else:
            x_normal = np.asarray(x_normal, dtype=np.float32)

        if isinstance(x_generated, pd.DataFrame):
            x_generated = x_generated.to_numpy(dtype=np.float32)
        else:
            x_generated = np.asarray(x_generated, dtype=np.float32)
    else:
        x_normal = x_normal_raw.to_numpy(dtype=np.float32)
        x_generated = generated_df.to_numpy(dtype=np.float32)

    if x_normal.shape[1] != x_generated.shape[1]:
        print(
            "[TSNE] Skip: feature dimension mismatch.",
            "test_normal_dim=", x_normal.shape[1],
            "generated_dim=", x_generated.shape[1],
        )
        return None
    print(
        "[TSNE] Input shapes after transform:",
        "test_normal=", x_normal.shape,
        "generated=", x_generated.shape,
    )

    max_per_class = max(1, int(max_per_class))
    rng = np.random.default_rng(seed)
    if x_normal.shape[0] > max_per_class:
        idx = rng.choice(x_normal.shape[0], size=max_per_class, replace=False)
        x_normal = x_normal[idx]
    if x_generated.shape[0] > max_per_class:
        idx = rng.choice(x_generated.shape[0], size=max_per_class, replace=False)
        x_generated = x_generated[idx]

    x_all = np.concatenate([x_normal, x_generated], axis=0)
    labels = np.array(
        ["test_normal_real"] * x_normal.shape[0]
        + ["ae_generated"] * x_generated.shape[0]
    )
    x_all = np.nan_to_num(x_all, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    if x_all.shape[0] < 4:
        print("[TSNE] Skip: need at least 4 rows for t-SNE.")
        return None

    effective_perplexity = min(float(perplexity), max(1.0, (x_all.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        random_state=seed,
    )
    emb = tsne.fit_transform(x_all)

    os.makedirs(save_dir, exist_ok=True)
    emb_df = pd.DataFrame(
        {
            "tsne_1": emb[:, 0],
            "tsne_2": emb[:, 1],
            "source": labels,
        }
    )
    csv_path = os.path.abspath(os.path.join(save_dir, f"{dataset}_tsne_test_normal_vs_ae_generated.csv"))
    png_path = os.path.abspath(os.path.join(save_dir, f"{dataset}_tsne_test_normal_vs_ae_generated.png"))
    emb_df.to_csv(_path_for_io(csv_path), index=False)

    plt.figure(figsize=(8, 6))
    for source, color in [
        ("test_normal_real", "#1f77b4"),
        ("ae_generated", "#ff7f0e"),
    ]:
        mask = emb_df["source"].to_numpy() == source
        plt.scatter(
            emb_df.loc[mask, "tsne_1"],
            emb_df.loc[mask, "tsne_2"],
            s=8,
            alpha=0.55,
            label=source,
            c=color,
            edgecolors="none",
        )
    plt.title(f"t-SNE test normal vs AE generated - {dataset}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(_path_for_io(png_path), dpi=180)
    plt.close()

    print(f"[TSNE] Saved csv: {csv_path}")
    print(f"[TSNE] Saved figure: {png_path}")
    return csv_path, png_path


def _metadata_sdtypes(metadata):
    if metadata is None:
        return {}

    try:
        columns = metadata.to_dict().get("columns", {})
    except Exception:
        return {}

    return {
        str(column): column_info.get("sdtype")
        for column, column_info in columns.items()
        if isinstance(column_info, dict)
    }


def _category_value_counts(series, max_categories=12):
    values = series.astype("object").where(series.notna(), "<NA>").map(str)
    return values.value_counts(normalize=True).head(max_categories)


def _sample_for_kde(values, max_points=5000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= max_points:
        return values

    idx = np.linspace(0, values.size - 1, max_points).astype(int)
    return values[idx]


def _gaussian_kde_density(values, grid):
    values = _sample_for_kde(values)
    if values.size == 0:
        return None
    if np.unique(values).size < 2:
        density = np.zeros_like(grid, dtype=np.float64)
        nearest = int(np.argmin(np.abs(grid - values[0])))
        density[nearest] = 1.0
        return density

    std = np.std(values, ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    robust_std = min(std, iqr / 1.349) if iqr > 0 else std
    bandwidth = 0.9 * robust_std * (values.size ** (-1.0 / 5.0))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = max(np.std(values, ddof=0), 1e-3)

    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * (z ** 2)).mean(axis=1)
    density /= bandwidth * np.sqrt(2.0 * np.pi)
    return density


def save_feature_distribution_test_normal_vs_ae_generated(
    x_test,
    y_test_bin,
    generated_csv_path: str,
    save_dir: str,
    dataset: str,
    metadata=None,
    max_categories: int = 12,
):
    """Plot selected feature distributions for test-normal and AE-generated rows."""
    generated_csv_io_path = _path_for_io(generated_csv_path)
    if not os.path.exists(generated_csv_io_path):
        print(f"[FEATURE-DIST] Skip: missing generated sample file: {generated_csv_path}")
        return None

    if not isinstance(x_test, pd.DataFrame):
        x_test = pd.DataFrame(np.asarray(x_test))

    y_test_bin = np.asarray(y_test_bin).astype(int)
    normal_mask = y_test_bin == 0
    x_normal = x_test.loc[normal_mask].reset_index(drop=True)
    generated_df = pd.read_csv(generated_csv_io_path)

    common_cols = [col for col in x_normal.columns if col in generated_df.columns]
    if not common_cols:
        print("[FEATURE-DIST] Skip: no common columns between normal and generated data.")
        return None

    x_normal = x_normal.loc[:, common_cols]
    generated_df = generated_df.loc[:, common_cols]
    if x_normal.shape[0] == 0 or generated_df.shape[0] == 0:
        print("[FEATURE-DIST] Skip: normal or generated data is empty.")
        return None

    sdtypes = _metadata_sdtypes(metadata)
    numeric_cols = [
        col for col in common_cols
        if pd.api.types.is_numeric_dtype(x_normal[col])
        and pd.api.types.is_numeric_dtype(generated_df[col])
    ]
    metadata_continuous_cols = [
        col for col in numeric_cols
        if sdtypes.get(str(col)) in {"numerical", "datetime"}
    ]
    fallback_continuous_cols = [
        col for col in numeric_cols
        if col not in metadata_continuous_cols
        and pd.concat([x_normal[col], generated_df[col]], ignore_index=True).nunique(dropna=True) > 20
    ]
    continuous_cols = (metadata_continuous_cols + fallback_continuous_cols)[:2]

    metadata_category_cols = [
        col for col in common_cols
        if sdtypes.get(str(col)) in {"categorical", "boolean"}
    ]
    dtype_category_cols = [
        col for col in common_cols
        if col not in metadata_category_cols
        and (
            pd.api.types.is_object_dtype(x_normal[col])
            or pd.api.types.is_categorical_dtype(x_normal[col])
            or pd.api.types.is_bool_dtype(x_normal[col])
        )
    ]
    low_cardinality_numeric_cols = [
        col for col in numeric_cols
        if col not in continuous_cols
        and col not in metadata_category_cols
        and pd.concat([x_normal[col], generated_df[col]], ignore_index=True).nunique(dropna=True) <= max_categories
    ]
    category_cols = (
        metadata_category_cols
        + dtype_category_cols
        + low_cardinality_numeric_cols
    )[:2]

    selected = [
        ("continuous", col) for col in continuous_cols
    ] + [
        ("categorical", col) for col in category_cols
    ]
    if not selected:
        print("[FEATURE-DIST] Skip: no suitable continuous or categorical features found.")
        return None

    while len(selected) < 4:
        selected.append(("empty", None))

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (feature_type, col) in zip(axes, selected):
        if col is None:
            ax.axis("off")
            continue

        if feature_type == "continuous":
            real_vals = pd.to_numeric(x_normal[col], errors="coerce").dropna().to_numpy()
            gen_vals = pd.to_numeric(generated_df[col], errors="coerce").dropna().to_numpy()
            if real_vals.size == 0 or gen_vals.size == 0:
                ax.axis("off")
                continue

            combined = np.concatenate([real_vals, gen_vals])
            combined = combined[np.isfinite(combined)]
            if combined.size == 0:
                ax.axis("off")
                continue
            if np.unique(combined).size < 2:
                x_min = combined[0] - 1.0
                x_max = combined[0] + 1.0
            else:
                x_min = float(np.min(combined))
                x_max = float(np.max(combined))
                pad = max((x_max - x_min) * 0.05, 1e-6)
                x_min -= pad
                x_max += pad

            grid = np.linspace(x_min, x_max, 256)
            real_density = _gaussian_kde_density(real_vals, grid)
            gen_density = _gaussian_kde_density(gen_vals, grid)

            if real_density is not None:
                ax.plot(grid, real_density, color="#1f77b4", linewidth=2.0, label="test_normal_real")
                ax.fill_between(grid, real_density, color="#1f77b4", alpha=0.12)
            if gen_density is not None:
                ax.plot(grid, gen_density, color="#ff7f0e", linewidth=2.0, label="ae_generated")
                ax.fill_between(grid, gen_density, color="#ff7f0e", alpha=0.12)
            ax.set_ylabel("Density")
        else:
            real_counts = _category_value_counts(x_normal[col], max_categories=max_categories)
            gen_counts = _category_value_counts(generated_df[col], max_categories=max_categories)
            categories = list((real_counts.add(gen_counts, fill_value=0)).sort_values(ascending=False).index)
            categories = categories[:max_categories]
            x_pos = np.arange(len(categories))
            width = 0.4
            ax.bar(
                x_pos - width / 2,
                real_counts.reindex(categories, fill_value=0).to_numpy(),
                width=width,
                label="test_normal_real",
            )
            ax.bar(
                x_pos + width / 2,
                gen_counts.reindex(categories, fill_value=0).to_numpy(),
                width=width,
                label="ae_generated",
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Ratio")

        ax.set_title(f"{feature_type}: {col}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"Feature distributions: test normal vs AE generated - {dataset}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = os.path.abspath(
        os.path.join(save_dir, f"{dataset}_feature_dist_test_normal_vs_ae_generated.png")
    )
    csv_path = os.path.abspath(
        os.path.join(save_dir, f"{dataset}_feature_dist_selected_features.csv")
    )
    fig.savefig(_path_for_io(png_path), dpi=180)
    plt.close(fig)

    pd.DataFrame(
        [
            {"feature_type": feature_type, "feature": col}
            for feature_type, col in selected
            if col is not None
        ]
    ).to_csv(_path_for_io(csv_path), index=False)

    print(f"[FEATURE-DIST] Saved figure: {png_path}")
    print(f"[FEATURE-DIST] Saved selected features: {csv_path}")
    return csv_path, png_path


def predict_and_evaluate(
    scores: np.ndarray,
    y_true,
    threshold,
    save_dir: str,
    filename: str,
    mode: str = "dist",
    k_std: float = 3.0,
):
    y_true = np.asarray(y_true)
    y_true_bin = (y_true != 0).astype(int)

    if mode in {"std", "re"}:
        mu, std = threshold
        threshold_test = float(mu + k_std * std)
    else:
        threshold_test = float(threshold)

    y_pred = (scores > threshold_test).astype(int)

    unique_classes = np.unique(y_true_bin)
    if len(unique_classes) < 2:
        auc = 0.0
    else:
        auc = float(roc_auc_score(y_true_bin, scores) * 100.0)
    acc = float(accuracy_score(y_true_bin, y_pred) * 100.0)
    f1 = float(f1_score(y_true_bin, y_pred, zero_division=0) * 100.0)
    precision = float(precision_score(y_true_bin, y_pred, zero_division=0) * 100.0)
    recall = float(recall_score(y_true_bin, y_pred, zero_division=0) * 100.0)
    mcc = float(matthews_corrcoef(y_true_bin, y_pred) * 100.0)
    cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])

    save_path = os.path.abspath(os.path.join(save_dir, filename))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(_path_for_io(save_path), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Threshold", f"{threshold_test:.6f}"])
        writer.writerow(["AUC", f"{auc:.6f}"])
        writer.writerow(["F1", f"{f1:.6f}"])
        writer.writerow(["MCC", f"{mcc:.6f}"])
        writer.writerow(["True 0", cm[0, 0], cm[0, 1]])
        writer.writerow(["True 1", cm[1, 0], cm[1, 1]])
        writer.writerow(["Precision", f"{precision:.6f}"])
        writer.writerow(["Recall", f"{recall:.6f}"])
        writer.writerow(["ACC", f"{acc:.6f}"])
        writer.writerow([])
        writer.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
       
        writer.writerow([])
        writer.writerow(["CM_TN", int(cm[0, 0])])
        writer.writerow(["CM_FP", int(cm[0, 1])])
        writer.writerow(["CM_FN", int(cm[1, 0])])
        writer.writerow(["CM_TP", int(cm[1, 1])])

    print(f"[EVAL] Saved: {save_path}")
    return {
        "Threshold": threshold_test,
        "AUC": auc,
        "F1": f1,
        "MCC": mcc,
        "Precision": precision,
        "Recall": recall,
        "ACC": acc,
        "ConfusionMatrix": [
            [int(cm[0, 0]), int(cm[0, 1])],
            [int(cm[1, 0]), int(cm[1, 1])],
        ],
    }


def evaluate_best_by_val(
    results_dir: str,
    dataset: str,
    lamda: float,
    loss_method: str = "original",
    type_val: str = "val_one_class",
):
    x_test = pd.read_csv(os.path.join(results_dir, f"{dataset}_X_test.csv"))
    y_test = pd.read_csv(os.path.join(results_dir, f"{dataset}_y_test.csv"))["label"]
    y_test_bin = (np.asarray(y_test) != 0).astype(int)
    n_test_normal = int((y_test_bin == 0).sum())
    n_test_abnormal = int((y_test_bin == 1).sum())
    print(f"[DATA] Test labels: normal={n_test_normal}, abnormal={n_test_abnormal}, total={len(y_test_bin)}")

    best_model_path = os.path.join(results_dir, "checkpoints", f"OUTLINEHA_{dataset}_best.pkl")
    best_threshold_path = os.path.join(
        results_dir,
        "checkpoints",
        f"OUTLINEHA_{dataset}_thresholds_best.pkl",
    )
    best_epoch_path = os.path.join(results_dir, "checkpoints", f"OUTLINEHA_{dataset}_best_epoch.txt")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Missing best-by-val model: {best_model_path}")
    if not os.path.exists(best_threshold_path):
        raise FileNotFoundError(f"Missing best-by-val thresholds: {best_threshold_path}")

    with open(best_threshold_path, "rb") as f:
        threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss = pickle.load(f)

    best_epoch = None
    if os.path.exists(best_epoch_path):
        with open(best_epoch_path, "r", encoding="utf-8") as f:
            best_epoch = f.read().strip()

    synthesizer = OUTLINEHAECTGANSynthesizer.load(best_model_path)
    test_re_loss, test_z_norm, test_y_true = synthesizer.get_score_ha(x_test, y_test)
    test_total_loss = test_re_loss + lamda * (test_z_norm ** 2)

    eval_dir = os.path.join(results_dir, "checkpoints")
    prefix = "eval"
    if best_epoch:
        prefix = f"ep{best_epoch}"

    metrics_95 = predict_and_evaluate(
        test_z_norm,
        test_y_true,
        threshold_95,
        save_dir=eval_dir,
        filename=f"{prefix}_mode_95.csv",
        mode="dist",
    )
    total_mode_file_suffix = "mode_total95"
    if type_val == "val_multi_class":
        total_mode_file_suffix = "mode_auc"

    metrics_total95 = predict_and_evaluate(
        test_total_loss,
        test_y_true,
        threshold_95_total_loss,
        save_dir=eval_dir,
        filename=f"{prefix}_{total_mode_file_suffix}.csv",
        mode="dist",
    )
    metrics_re = predict_and_evaluate(
        test_re_loss,
        test_y_true,
        threshold_re,
        save_dir=eval_dir,
        filename=f"{prefix}_mode_re.csv",
        mode="re",
        k_std=3.0,
    )
    metrics_std = predict_and_evaluate(
        test_z_norm,
        test_y_true,
        threshold_z_0,
        save_dir=eval_dir,
        filename=f"{prefix}_mode_std_3.0.csv",
        mode="std",
        k_std=3.0,
    )

    summary = {
        "loss_method": loss_method,
        "best_epoch": best_epoch,
        "best_selection_metric": (
            "val_auc_total" if type_val == "val_multi_class" else "oneclass_avg_total_loss"
        ),
        "mode95_threshold": metrics_95["Threshold"],
        "mode95_confusion_matrix": metrics_95["ConfusionMatrix"],
        "test_mode95_auc": metrics_95["AUC"],
        "test_mode95_f1": metrics_95["F1"],
        "test_mode95_mcc": metrics_95["MCC"],
        "mode_total_threshold": metrics_total95["Threshold"],
        "mode_total_confusion_matrix": metrics_total95["ConfusionMatrix"],
        "test_total_auc": metrics_total95["AUC"],
        "test_total_f1": metrics_total95["F1"],
        "test_total_mcc": metrics_total95["MCC"],
        "mode_total95_threshold": metrics_total95["Threshold"],
        "mode_total95_confusion_matrix": metrics_total95["ConfusionMatrix"],
        "test_total95_auc": metrics_total95["AUC"],
        "test_total95_f1": metrics_total95["F1"],
        "test_total95_mcc": metrics_total95["MCC"],
        "mode_re_threshold": metrics_re["Threshold"],
        "mode_re_confusion_matrix": metrics_re["ConfusionMatrix"],
        "mode_std_threshold": metrics_std["Threshold"],
        "mode_std_confusion_matrix": metrics_std["ConfusionMatrix"],
    }
    summary_path = os.path.abspath(os.path.join(results_dir, f"best_by_val_test_summary_{loss_method}.json"))
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(_path_for_io(summary_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] Saved summary: {summary_path}")


def build_results_dir(args) -> str:
    time_tag = f"{int(args.time):02d}"
    run_folder = f"{args.type_train}_seed_{args.seed}_time_{time_tag}"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    loss_tag = {
        "re_normal_only": "renormal",
        "original": "orig",
        "asymmetric": "asym",
        "contrastive": "contrast",
        "dual": "dual",
        "focal": "focal",
    }.get(args.loss_method, args.loss_method)
    val_tag = {
        "val_one_class": "oneclass",
        "val_multi_class": "multiclass",
    }.get(args.type_val, args.type_val)
    boundary_tag = ""
    if getattr(args, "use_boundary_anomaly", False):
        boundary_tag = f"_bbeta_{args.boundary_beta}__bbprob_{args.boundary_dim_prob}"
    outlier_mode = getattr(args, "outlier_mode", "tail_abs")
    outlier_tag_name = {
        "tail_abs": "tail_abs",
        "truncnorm": "trunc",
        "boundary_band": "boundary_band",
    }.get(outlier_mode, outlier_mode)
    outlier_tag = f"_outlier_{outlier_tag_name}"
    if outlier_mode == "boundary_band":
        outlier_tag += f"_bbprob_{args.boundary_dim_prob}_w{args.boundary_band_width}"
    buffer_tag = f"buff_{args.buffer_multiplier}" if args.use_buffer else "no_buff"
    return os.path.join(
        base_dir,
        f"results/{run_folder}/per_{args.per}/"
        f"outlineha_{args.d}_{args.e}_mae_{args.mae}_lae_{args.lae}_{buffer_tag}_k1_{args.k1}_k2_{args.k2}_{loss_tag}_{val_tag}{boundary_tag}{outlier_tag}",
    )


def train_outlinehaectgan_best_by_val(args):
    set_seed(args.seed)
    if args.use_buffer and args.buffer_multiplier < 1:
        raise ValueError("--buffer_multiplier must be >= 1.")
    if args.k1 <= 0:
        raise ValueError("--k1 must be > 0.")
    if args.k2 <= 0:
        raise ValueError("--k2 must be > 0.")
    if not 0.0 <= args.boundary_dim_prob <= 1.0:
        raise ValueError("--boundary_dim_prob must be in [0, 1].")
    if args.boundary_band_width < 0:
        raise ValueError("--boundary_band_width must be >= 0.")
    if args.ae_train_sample_max_rows < 0:
        raise ValueError("--ae_train_sample_max_rows must be >= 0.")
    if args.tsne_max_per_class < 1:
        raise ValueError("--tsne_max_per_class must be >= 1.")
    if args.tsne_perplexity <= 0:
        raise ValueError("--tsne_perplexity must be > 0.")

    results_dir = build_results_dir(args)
    save_ae_train_samples = not args.no_save_ae_train_samples
    ae_train_sample_save_path = os.path.join(
        results_dir,
        f"{args.d}_outlineha_ae_train_generated_samples.csv",
    )

    if args.skip_test_eval == "test_only":
        print("[MODE] test_only: skip training, evaluate from saved best-by-val artifacts.")
        evaluate_best_by_val(
            results_dir=results_dir,
            dataset=args.d,
            lamda=args.lae,
            loss_method=args.loss_method,
            type_val=args.type_val,
        )
        print("[DONE] Test-only evaluation completed.")
        return

    print(
        "Dataset:", args.d,
        "Epochs:", args.e,
        "Batch size:", args.b,
        "Use buffer:", args.use_buffer,
        "Buffer multiplier:", args.buffer_multiplier,
        "k1:", args.k1,
        "k2:", args.k2,
        "Type val:", args.type_val,
        "Lamda:", args.lae,
        "Margin:", args.mae,
        "Percentage:", args.per,
        "Type train:", args.type_train,
        "Loss method:", args.loss_method,
        "Focal gamma:", args.focal_gamma if args.loss_method == "focal" else "N/A",
        "Outlier mode:", args.outlier_mode,
        "Boundary dim/perturb prob:", args.boundary_dim_prob,
        "Boundary band width:", args.boundary_band_width,
        "Use boundary anomaly:", args.use_boundary_anomaly,
        "Boundary beta:", args.boundary_beta,
        "Save AE train samples:", save_ae_train_samples,
        "AE train sample max rows:", args.ae_train_sample_max_rows,
        "Run TSNE:", not args.no_tsne_ae_train_samples,
        "Run feature distribution plot:", save_ae_train_samples,
        "TSNE max per class:", args.tsne_max_per_class,
        "TSNE perplexity:", args.tsne_perplexity,
        "Best metric:", "val_auc_total (val_multi_class) / avg_val_total_loss (val_one_class)",
    )

    df_full, y_full, df_test, _ = load_dataset(args.d)
    df_full = df_full.copy()
    df_test = df_test.copy()
    df_full.columns = df_full.columns.map(str)
    df_test.columns = df_test.columns.map(str)
    print("Full shape:", df_full.shape, "y_full shape:", y_full.shape)
    print("Test shape:", df_test.shape)

    os.makedirs(results_dir, exist_ok=True)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_full)
    metadata_file = os.path.join(results_dir, f"{args.d}_OUTLINEHA_bestbyval_metadata_epochs_{args.e}.json")
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    metadata.save_to_json(metadata_file)

    synthesizer = OUTLINEHAECTGANSynthesizer(
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        pac=8,
        cuda=True,
        epochs=args.e,
        batch_size=args.b,
        metadata=metadata,
        lamda=args.lae,
        margin=args.mae,
        dataset=args.d,
        path=results_dir,
        type_train=args.type_train,
        loss_method=args.loss_method,
        focal_gamma=args.focal_gamma,
        buffer_multiplier=args.buffer_multiplier,
        use_buffer=args.use_buffer,
        type_val=args.type_val,
        k1=args.k1,
        k2=args.k2,
        use_boundary_anomaly=args.use_boundary_anomaly,
        boundary_beta=args.boundary_beta,
        outlier_mode=args.outlier_mode,
        boundary_dim_prob=args.boundary_dim_prob,
        boundary_band_width=args.boundary_band_width,
        save_ae_train_samples=save_ae_train_samples,
        ae_train_sample_max_rows=args.ae_train_sample_max_rows,
        ae_train_sample_save_path=ae_train_sample_save_path,
    )

    n_test = len(df_test)
    x_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    x_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)
    y_test = y_full.iloc[-n_test:].reset_index(drop=True)
    y_test_bin = (np.asarray(y_test) != 0).astype(int)
    n_test_normal = int((y_test_bin == 0).sum())
    n_test_abnormal = int((y_test_bin == 1).sum())
    x_train_raw.columns = x_train_raw.columns.map(str)
    x_test_raw.columns = x_test_raw.columns.map(str)

    x_train = synthesizer.preprocess(x_train_raw)
    x_test = synthesizer._data_processor.transform(x_test_raw)
    if x_test.shape[1] == 0:
        raise ValueError(
            "Transformed x_test has 0 columns. Check train/test column-name alignment before transform."
        )

    x_test_file = os.path.join(results_dir, f"{args.d}_X_test.csv")
    y_test_file = os.path.join(results_dir, f"{args.d}_y_test.csv")
    x_test.to_csv(x_test_file, index=False)
    y_test.to_frame(name="label").to_csv(y_test_file, index=False)

    x_train_normal_full = x_train[y_train == 0].copy()
    y_train_normal_full = y_train[y_train == 0].copy()
    if args.per < 1.0:
        sampled_index = x_train_normal_full.sample(frac=args.per, random_state=args.seed).index
        x_normal_subset = x_train_normal_full.loc[sampled_index].reset_index(drop=True)
        y_normal_subset = y_train_normal_full.loc[sampled_index].reset_index(drop=True)
    else:
        x_normal_subset = x_train_normal_full.reset_index(drop=True)
        y_normal_subset = y_train_normal_full.reset_index(drop=True)

    x_train_normal, x_val, y_train_normal, y_val = train_test_split(
        x_normal_subset,
        y_normal_subset,
        test_size=0.2,
        random_state=args.seed,
    )

    x_val_file = os.path.join(results_dir, f"{args.d}_X_val_normal.csv")
    y_val_file = os.path.join(results_dir, f"{args.d}_y_val_normal.csv")
    x_val.to_csv(x_val_file, index=False)
    y_val.to_frame(name="label").to_csv(y_val_file, index=False)

    print("Train normal shape:", x_train_normal.shape)
    print("Val normal shape:", x_val.shape)
    print("Test shape:", x_test.shape)
    print(f"[DATA] Test labels: normal={n_test_normal}, abnormal={n_test_abnormal}, total={len(y_test_bin)}")

    synthesizer._fit(x_normal_subset, x_train_normal, x_val, val_labels=y_val)

    last_model_path = os.path.join(results_dir, f"OUTLINEHA_{args.d}_epochs_{args.e}.pkl")
    synthesizer.save(last_model_path)
    print(f"[SAVE] Last model: {last_model_path}")

    loss_df = synthesizer.get_loss_values()
    loss_log_path = os.path.join(results_dir, f"outlineha_bestbyval_loss_log_epochs_{args.e}.csv")
    loss_df.to_csv(loss_log_path, index=False)
    print(f"[SAVE] Loss log: {loss_log_path}")

    plt.figure(figsize=(10, 6))
    total_loss_col = "HAE Loss" if "HAE Loss" in loss_df.columns else "MAE Loss"
    plt.plot(loss_df["Epoch"], loss_df[total_loss_col], label="OUTLINEHA total Loss")
    plt.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction Loss")
    plt.plot(loss_df["Epoch"], loss_df["Latent Loss"], label="Latent Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"OUTLINEHA Loss Curves ({args.loss_method}) - {args.d}")
    plt.legend()
    plt.grid(True)
    loss_fig = os.path.join(results_dir, f"loss_outlineha_bestbyval_{args.d}_epochs_{args.e}.png")
    plt.savefig(loss_fig)
    plt.close()
    print(f"[SAVE] Loss figure: {loss_fig}")

    tsne_outputs = None
    feature_distribution_outputs = None
    run_config = {
        "dataset": args.d,
        "epochs": args.e,
        "loss_method": args.loss_method,
        "focal_gamma": args.focal_gamma,
        "use_buffer": args.use_buffer,
        "buffer_multiplier": args.buffer_multiplier,
        "k1": args.k1,
        "k2": args.k2,
        "type_val": args.type_val,
        "outlier_mode": args.outlier_mode,
        "boundary_dim_prob": args.boundary_dim_prob,
        "boundary_band_width": args.boundary_band_width,
        "use_boundary_anomaly": args.use_boundary_anomaly,
        "boundary_beta": args.boundary_beta,
        "save_ae_train_samples": save_ae_train_samples,
        "ae_train_sample_max_rows": args.ae_train_sample_max_rows,
        "ae_train_sample_save_path": ae_train_sample_save_path,
        "tsne_outputs": tsne_outputs,
        "feature_distribution_outputs": feature_distribution_outputs,
        "tsne_max_per_class": args.tsne_max_per_class,
        "tsne_perplexity": args.tsne_perplexity,
        "best_selection_metric": (
            "val_auc_total" if args.type_val == "val_multi_class" else "oneclass_avg_total_loss"
        ),
        "results_dir": results_dir,
    }
    save_pickle(run_config, save_dir=results_dir, name="run_config_bestbyval.pkl")

    if args.skip_test_eval == "train_test":
        evaluate_best_by_val(
            results_dir=results_dir,
            dataset=args.d,
            lamda=args.lae,
            loss_method=args.loss_method,
            type_val=args.type_val,
        )
    if save_ae_train_samples and not args.no_tsne_ae_train_samples:
        tsne_outputs = save_tsne_test_normal_vs_ae_generated(
            x_test=x_test,
            y_test_bin=y_test_bin,
            generated_csv_path=ae_train_sample_save_path,
            save_dir=results_dir,
            dataset=args.d,
            ctgan_transformer=synthesizer._model._transformer,
            seed=args.seed,
            max_per_class=args.tsne_max_per_class,
            perplexity=args.tsne_perplexity,
        )
    if save_ae_train_samples:
        feature_distribution_outputs = save_feature_distribution_test_normal_vs_ae_generated(
            x_test=x_test,
            y_test_bin=y_test_bin,
            generated_csv_path=ae_train_sample_save_path,
            save_dir=results_dir,
            dataset=args.d,
            metadata=metadata,
        )
    run_config["tsne_outputs"] = tsne_outputs
    run_config["feature_distribution_outputs"] = feature_distribution_outputs
    save_pickle(run_config, save_dir=results_dir, name="run_config_bestbyval.pkl")
    print("[DONE] Training with best-by-val checkpoint completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train OUTLINEHA-CTGAN with multiple loss methods."
    )
    parser.add_argument("--d", type=str, default="unsw", help="Dataset")
    parser.add_argument("--e", type=int, default=200, help="Epochs")
    parser.add_argument("--b", type=int, default=128, help="Batch size")
    parser.add_argument("--lae", type=float, default=0.1, help="Lambda")
    parser.add_argument("--mae", type=float, default=1.0, help="Margin")
    parser.add_argument("--per", type=float, default=1.0, help="Fraction of normal train for OUTLINEHA")
    parser.add_argument(
        "--type_train",
        type=str,
        choices=["all", "noise", "std", "3std", "std_3std", "noise_std", "noise_3std"],
        default="all",
        help="Type train OUTLINEHA",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--time", type=int, default=1, help="Run index")
    parser.add_argument(
        "--loss_method",
        type=str,
        choices=["original", "asymmetric", "dual", "contrastive", "focal", "re_normal_only"],
        default="original",
        help="Loss method: original, asymmetric, dual, contrastive, focal, re_normal_only",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss mode (ignored by other modes).",
    )
    parser.add_argument(
        "--buffer_multiplier",
        type=int,
        default=2,
        help="Buffer size multiplier n: max buffer size = n * batch_size.",
    )
    buffer_group = parser.add_mutually_exclusive_group()
    buffer_group.add_argument(
        "--buff",
        dest="use_buffer",
        action="store_true",
        help="Enable replay buffer for OUTLINEHA generated samples. This is the default.",
    )
    #Có sử dụng buffer hay không, nếu không sử dụng buffer thì chỉ train HAE trên batch hiện tại được tạo ra, không giữ lại các batch trước đó để train tiếp
    buffer_group.add_argument(
        "--no_buff",
        dest="use_buffer",
        action="store_false",
        help="Disable replay buffer and train HAE from the current generated batch only.",
    )
    parser.set_defaults(use_buffer=True)
    #k1 và k2 là hệ số nhân để xác định ngưỡng outlier trong không gian latent, ảnh hưởng đến cách tạo mẫu outlier cho huấn luyện HAE. K1 áp dụng cho buffer fakez bình thường, trong khi k2 áp dụng cho buffer fakez bất thường. Các giá trị k1 và k2 lớn hơn sẽ tạo ra các mẫu outlier xa hơn từ phân phối latent của dữ liệu thật, trong khi các giá trị nhỏ hơn sẽ tạo ra các mẫu gần hơn. Điều này ảnh hưởng đến độ khó của nhiệm vụ phát hiện bất thường mà HAE phải học để giải quyết.
    parser.add_argument(
        "--k1",
        type=float,
        default=1.0,
        help="Outlier multiplier for fakez_buffer_nor.",
    )
    parser.add_argument(
        "--k2",
        type=float,
        default=3.0,
        help="Outlier multiplier for fakez_buffer_ano.",
    )
    parser.add_argument(
        "--outlier_mode",
        type=str,
        choices=["tail_abs", "truncnorm", "boundary_band"],
        default="tail_abs",
        help="Latent outlier sampling mode: tail_abs, truncnorm, or boundary_band.",
    )
    parser.add_argument(
        "--boundary_dim_prob",
        type=float,
        default=0.2,
        help=(
            "Probability for boundary behavior: latent dimensions moved near boundary "
            "in boundary_band mode, and continuous features perturbed in boundary anomaly mode."
        ),
    )
    parser.add_argument(
        "--boundary_band_width",
        type=float,
        default=0.1,
        help="For boundary_band mode, width of the near-boundary latent band.",
    )
    parser.add_argument(
        "--type_val",
        type=str,
        choices=["val_one_class", "val_multi_class"],
        default="val_one_class",
        help="Validation mode: val_one_class or val_multi_class.",
    )
    parser.add_argument(
        "--use_boundary_anomaly",
        action="store_true",
        help="Create HAE pseudo-anomalies by moving real normal rows toward CTGAN fake rows.",
    )
    parser.add_argument(
        "--boundary_beta",
        type=float,
        default=0.1,
        help="Step size for CTGAN boundary pseudo-anomaly generation.",
    )
    parser.add_argument(
        "--no_save_ae_train_samples",
        action="store_true",
        help="Disable saving generated samples used to train AE.",
    )
    parser.add_argument(
        "--ae_train_sample_max_rows",
        type=int,
        default=20000,
        help="Maximum generated AE-train rows to save for t-SNE analysis.",
    )
    parser.add_argument(
        "--no_tsne_ae_train_samples",
        action="store_true",
        help="Disable t-SNE plot for test-normal rows and generated AE-train rows.",
    )
    parser.add_argument(
        "--tsne_max_per_class",
        type=int,
        default=2000,
        help="Maximum rows per source for t-SNE: test normal and AE generated.",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. It is automatically reduced for small samples.",
    )
    parser.add_argument(
        "--skip_test_eval",
        type=str,
        choices=["train_only", "test_only", "train_test"],
        default="train_test",
        help="train_only, test_only, or train_test",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_outlinehaectgan_best_by_val(parse_args())

#python train_OUTLINEHAECTGAN.py --d unsw --e 200 --b 128 --lae 0.1 --mae 1.0 --per 0.1 --type_train std --loss_method original --type_val val_one_class --buff --buffer_multiplier 8 --k1 1.0 --k2 3.0 --outlier_mode boundary_band --boundary_dim_prob 0.0 --seed 42 --time 1
"""
#Tham số chạy đnag tốt nhất:python train_OUTLINEHAECTGAN.py --d unsw --e 200 --b 128 --lae 0.1 --mae 1.0 --per 0.1 --type_train all --loss_method re_normal_only --type_val val_one_class --buff --buffer_multiplier 2 --k1 1.0 --k2 1.0 --outlier_mode tail_abs --use_boundary_anomaly --boundary_beta 0.1 --boundary_dim_prob 0.1 --seed 42 --time 1

#python train_OUTLINEHAECTGAN.py --d unsw --e 200 --b 128 --lae 0.1 --mae 1.0 --per 0.1 --type_train all --loss_method re_normal_only --type_val val_one_class --buff --buffer_multiplier 2 --k1 1.0 --k2 1.0 --outlier_mode tail_abs --use_boundary_anomaly --boundary_beta 0.1 --boundary_dim_prob 0.5 --seed 42 --time 1

Ý nghĩa cấu hình:

--d unsw: dataset UNSW.
--loss_method re_normal_only: reconstruction loss chỉ tính trên mẫu normal.
--type_val val_one_class: validation chỉ gồm normal.
--buff --buffer_multiplier 2: bật buffer kích thước nhỏ 2 * batch_size.
--k1 1.0 --k2 1.0: cả hai luồng fake trong type_train=all đều dùng mức outlier k=1.
--outlier_mode tail_abs: sinh latent outlier theo đuôi ngoài biên.
--type_train all: huấn luyện bằng cả std, 3std/luồng fake thứ hai và noise mix.
--use_boundary_anomaly --boundary_beta 0.1: đẩy mẫu real về hướng fake với bước beta=0.1.
--boundary_dim_prob 0.5: áp dụng perturbation boundary lên khoảng 50% feature liên tục.
--per 0.1: dùng 10% normal train.
"""
