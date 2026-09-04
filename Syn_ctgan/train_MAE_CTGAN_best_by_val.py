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
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from process_data import load_data_cic_ids, load_data_cic_iot, load_data_nsl_kdd, load_data_unsw
from sdv.metadata import SingleTableMetadata
from sdv.single_table import MAECTGANSynthesizer


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
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


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

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Threshold", f"{threshold_test:.6f}"])
        writer.writerow(["AUC", f"{auc:.6f}"])
        writer.writerow(["F1", f"{f1:.6f}"])
        writer.writerow(["MCC", f"{mcc:.6f}"])
        writer.writerow(["Precision", f"{precision:.6f}"])
        writer.writerow(["Recall", f"{recall:.6f}"])
        writer.writerow(["ACC", f"{acc:.6f}"])
        writer.writerow([])
        writer.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
        writer.writerow(["True 0", cm[0, 0], cm[0, 1]])
        writer.writerow(["True 1", cm[1, 0], cm[1, 1]])

    print(f"[EVAL] Saved: {save_path}")
    return {
        "AUC": auc,
        "F1": f1,
        "MCC": mcc,
        "Precision": precision,
        "Recall": recall,
        "ACC": acc,
    }


def evaluate_best_by_val(results_dir: str, dataset: str, lamda: float):
    x_test = pd.read_csv(os.path.join(results_dir, f"{dataset}_X_test.csv"))
    y_test = pd.read_csv(os.path.join(results_dir, f"{dataset}_y_test.csv"))["label"]

    best_model_path = os.path.join(results_dir, "checkpoints", f"MAE_{dataset}_best.pkl")
    best_threshold_path = os.path.join(
        results_dir,
        "checkpoints",
        f"MAE_{dataset}_thresholds_best.pkl",
    )
    best_epoch_path = os.path.join(results_dir, "checkpoints", f"MAE_{dataset}_best_epoch.txt")

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

    synthesizer = MAECTGANSynthesizer.load(best_model_path)
    test_re_loss, test_z_norm, test_y_true = synthesizer.get_score_mae(x_test, y_test)
    test_total_loss = test_re_loss + lamda * (test_z_norm ** 2)

    eval_dir = os.path.join(results_dir, "checkpoints")
    prefix = f"test_{dataset}_best_by_val"
    if best_epoch:
        prefix = f"{prefix}_epoch_{best_epoch}"

    metrics_95 = predict_and_evaluate(
        test_z_norm,
        test_y_true,
        threshold_95,
        save_dir=eval_dir,
        filename=f"{prefix}_mode_95.csv",
        mode="dist",
    )
    metrics_total95 = predict_and_evaluate(
        test_total_loss,
        test_y_true,
        threshold_95_total_loss,
        save_dir=eval_dir,
        filename=f"{prefix}_mode_total95.csv",
        mode="dist",
    )
    predict_and_evaluate(
        test_re_loss,
        test_y_true,
        threshold_re,
        save_dir=eval_dir,
        filename=f"{prefix}_mode_re.csv",
        mode="re",
        k_std=3.0,
    )
    predict_and_evaluate(
        test_z_norm,
        test_y_true,
        threshold_z_0,
        save_dir=eval_dir,
        filename=f"{prefix}_mode_std_3.0.csv",
        mode="std",
        k_std=3.0,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_selection_metric": "avg_val_total_loss",
        "test_mode95_auc": metrics_95["AUC"],
        "test_mode95_f1": metrics_95["F1"],
        "test_mode95_mcc": metrics_95["MCC"],
        "test_total95_auc": metrics_total95["AUC"],
        "test_total95_f1": metrics_total95["F1"],
        "test_total95_mcc": metrics_total95["MCC"],
    }
    summary_path = os.path.join(results_dir, "best_by_val_test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[EVAL] Saved summary: {summary_path}")


def build_results_dir(args) -> str:
    time_tag = f"{int(args.time):02d}"
    run_folder = f"{args.type_train}_seed_{args.seed}_time_{time_tag}"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        base_dir,
        f"results/{run_folder}/percentage_{args.per}/"
        f"mae_bestbyval_{args.d}_{args.e}_margin_{args.mae}_lambda_{args.lae}",
    )


def train_mae_ctgan_best_by_val(args):
    set_seed(args.seed)

    results_dir = build_results_dir(args)

    # --skip_test_eval test_only: skip training and run test evaluation only.
    if args.skip_test_eval == "test_only":
        print("[MODE] test_only: skip training, evaluate from saved best-by-val artifacts.")
        evaluate_best_by_val(results_dir=results_dir, dataset=args.d, lamda=args.lae)
        print("[DONE] Test-only evaluation completed.")
        return

    print(
        "Dataset:", args.d,
        "Epochs:", args.e,
        "Batch size:", args.b,
        "Lamda:", args.lae,
        "Margin:", args.mae,
        "Percentage:", args.per,
        "Type train:", args.type_train,
        "Best metric:", "avg_val_total_loss",
    )

    df_full, y_full, df_test, _ = load_dataset(args.d)
    # Keep feature schemas consistent across datasets (e.g., NSL-KDD integer column names).
    # The SDV data processor may canonicalize names to strings during preprocess.
    df_full = df_full.copy()
    df_test = df_test.copy()
    df_full.columns = df_full.columns.map(str)
    df_test.columns = df_test.columns.map(str)
    print("Full shape:", df_full.shape, "y_full shape:", y_full.shape)
    print("Test shape:", df_test.shape)

    os.makedirs(results_dir, exist_ok=True)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_full)
    metadata_file = os.path.join(results_dir, f"{args.d}_MAE_bestbyval_metadata_epochs_{args.e}.json")
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    metadata.save_to_json(metadata_file)

    synthesizer = MAECTGANSynthesizer(
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
    )

    n_test = len(df_test)
    x_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    x_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)
    y_test = y_full.iloc[-n_test:].reset_index(drop=True)
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

    # MAECTGAN core saves the best model by mean total loss on normal validation data.
    synthesizer._fit(x_train, x_train_normal, x_val)

    # Save last model for reference.
    last_model_path = os.path.join(results_dir, f"MAE_{args.d}_epochs_{args.e}.pkl")
    synthesizer.save(last_model_path)
    print(f"[SAVE] Last model: {last_model_path}")

    loss_df = synthesizer.get_loss_values()
    loss_log_path = os.path.join(results_dir, f"mae_bestbyval_loss_log_epochs_{args.e}.csv")
    loss_df.to_csv(loss_log_path, index=False)
    print(f"[SAVE] Loss log: {loss_log_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["Epoch"], loss_df["MAE Loss"], label="MAE total Loss")
    plt.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction Loss")
    plt.plot(loss_df["Epoch"], loss_df["Latent Loss"], label="Latent Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"MAE Loss Curves - {args.d}")
    plt.legend()
    plt.grid(True)
    loss_fig = os.path.join(results_dir, f"loss_mae_bestbyval_{args.d}_epochs_{args.e}.png")
    plt.savefig(loss_fig)
    plt.close()
    print(f"[SAVE] Loss figure: {loss_fig}")

    save_pickle(
        {
            "dataset": args.d,
            "epochs": args.e,
            "best_selection_metric": "avg_val_total_loss",
            "results_dir": results_dir,
        },
        save_dir=results_dir,
        name="run_config_bestbyval.pkl",
    )

    if args.skip_test_eval == "train_test":
        evaluate_best_by_val(results_dir=results_dir, dataset=args.d, lamda=args.lae)

    print("[DONE] Training with best-by-val checkpoint completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MAE-CTGAN (save only best model by avg_val_total_loss on val)."
    )
    parser.add_argument("--d", type=str, default="unsw", help="Dataset")
    parser.add_argument("--e", type=int, default=200, help="Epochs")
    parser.add_argument("--b", type=int, default=128, help="Batch size")
    parser.add_argument("--lae", type=float, default=0.1, help="Lambda")
    parser.add_argument("--mae", type=float, default=5.0, help="Margin")
    parser.add_argument("--per", type=float, default=1.0, help="Fraction of normal train for MAE")
    parser.add_argument(
        "--type_train",
        type=str,
        choices=["all", "noise", "std", "3std", "std_3std", "noise_std", "noise_3std"],
        default="all",
        help="Type train MAE",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--time", type=int, default=1, help="Run index (e.g., 1 -> time_01)")
    parser.add_argument(
        "--skip_test_eval",
        type=str,
        choices=["train_only", "test_only", "train_test"],
        default="train_test",
        help=(
            "train_only: train only (skip test evaluation). "
            "test_only: do not train, only evaluate test from existing best-by-val artifacts. "
            "train_test: train then evaluate test."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_mae_ctgan_best_by_val(parse_args())
