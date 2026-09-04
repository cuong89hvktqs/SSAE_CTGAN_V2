import argparse
import csv
import os
import pickle
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import REDUCEMAECTGANSynthesizer
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

from process_data import (  # noqa: E402
    load_data_cic_ids,
    load_data_cic_iot,
    load_data_cidds_001,
    load_data_nsl_kdd,
    load_data_unsw,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train reduced-space MAE-CTGAN")
    parser.add_argument("--d", type=str, default="unsw", help="Dataset name")
    parser.add_argument("--e", type=int, default=200, help="Training epochs")
    parser.add_argument("--b", type=int, default=128, help="Batch size")
    parser.add_argument("--lae", type=float, default=0.1, help="MAE lambda")
    parser.add_argument("--mae", type=float, default=1.0, help="MAE margin")
    parser.add_argument(
        "--mae_loss_mode",
        type=str,
        choices=["origin", "renormal"],
        default="origin",
        help="origin uses reconstruction loss for all samples; renormal uses normal only.",
    )
    parser.add_argument("--per", type=float, default=1.0, help="Normal train fraction")
    parser.add_argument(
        "--type_train",
        type=str,
        choices=["all", "noise", "std", "3std", "std_3std", "noise_std", "noise_3std"],
        default="all",
        help="Pseudo anomaly source for MAE training",
    )
    parser.add_argument(
        "--mae_start_ratio",
        type=float,
        default=0.0,
        help="Fraction of epochs before pseudo anomalies train MAE.",
    )
    parser.add_argument(
        "--ctgan_noise_start_ratio",
        type=float,
        default=1.0,
        help="Fraction of epochs before Gaussian noise is added to CTGAN samples.",
    )
    parser.add_argument(
        "--ctgan_noise_std",
        type=float,
        default=0.05,
        help="Gaussian noise std for generated continuous CTGAN features.",
    )
    parser.add_argument(
        "--ctgan_noise_feature_ratio",
        type=float,
        default=1.0,
        help="Fraction of generated continuous CTGAN features to perturb.",
    )
    parser.add_argument(
        "--ctgan_pseudo_anomaly_mode",
        type=str,
        choices=["origin", "feature_outlier"],
        default="origin",
        help="How generated CTGAN samples become MAE pseudo anomalies.",
    )
    parser.add_argument(
        "--ctgan_feature_outlier_r",
        type=float,
        default=1.0,
        help="Outlier coefficient for feature_outlier mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--time", type=int, default=1, help="Run index")
    args = parser.parse_args()

    if not 0.0 <= args.mae_start_ratio <= 1.0:
        parser.error("--mae_start_ratio must be in [0, 1].")
    if args.mae_start_ratio >= 1.0:
        parser.error("--mae_start_ratio must be less than 1.0.")
    if not 0.0 <= args.ctgan_noise_start_ratio <= 1.0:
        parser.error("--ctgan_noise_start_ratio must be in [0, 1].")
    if args.ctgan_noise_std < 0.0:
        parser.error("--ctgan_noise_std must be greater than or equal to 0.")
    if not 0.0 <= args.ctgan_noise_feature_ratio <= 1.0:
        parser.error("--ctgan_noise_feature_ratio must be in [0, 1].")
    if args.ctgan_feature_outlier_r < 0.0:
        parser.error("--ctgan_feature_outlier_r must be greater than or equal to 0.")
    if not 0.0 < args.per <= 1.0:
        parser.error("--per must be in (0, 1].")
    return args


def normalize_dataset_name(name):
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "cidds001":
        return "cidds_001"
    return normalized


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def result_dir_name(args, dataset):
    prefix = f"reduce_mae_{dataset}_{args.e}_m_{args.mae}_l_{args.lae}"
    common = (
        f"{prefix}_maeloss_{args.mae_loss_mode}"
        f"_mstart_{args.mae_start_ratio}_nstart_{args.ctgan_noise_start_ratio}"
        f"_nfeat_{args.ctgan_noise_feature_ratio}"
        f"_pmode_{args.ctgan_pseudo_anomaly_mode}"
    )
    if args.ctgan_pseudo_anomaly_mode == "feature_outlier":
        return f"{common}_r_{args.ctgan_feature_outlier_r}"
    return f"{common}_nstd_{args.ctgan_noise_std}"


def save_threshold(threshold, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Threshold saved to {path}")


def _compute_total_loss_from_outputs(test_re_loss, test_z_norm, lamda):
    return test_re_loss + lamda * (test_z_norm ** 2)


def predict_and_evaluate(
    scores,
    y_true,
    threshold,
    save_dir,
    filename,
    k_std=3.0,
    mode="std",
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    y_true = np.asarray(y_true)

    if mode in {"re", "std"}:
        mu, std = threshold
        threshold_test = mu + k_std * std
    elif mode == "dist":
        threshold_test = threshold
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")

    y_pred = (scores > threshold_test).astype(int)
    y_true = (y_true != 0).astype(int)

    if len(np.unique(y_true)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(y_true, scores) * 100.0
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    precision = precision_score(y_true, y_pred, zero_division=0) * 100.0
    recall = recall_score(y_true, y_pred, zero_division=0) * 100.0
    mcc = matthews_corrcoef(y_true, y_pred) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
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
        writer.writerow(["True 0", cm[0, 0], cm[0, 1]])
        writer.writerow(["True 1", cm[1, 0], cm[1, 1]])

    print(f"Evaluation results saved to {save_path}")
    return {
        "re_errors": scores,
        "y_pred": y_pred,
        "metrics": {
            "AUC": auc,
            "ACC": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "MCC": mcc,
        },
        "confusion_matrix": cm,
    }


def _evaluate_model_with_thresholds(
    synthesizer,
    X_test_scaled,
    y_test,
    thresholds,
    save_dir,
    filename_prefix,
    lamda,
):
    threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss = thresholds
    test_re_loss, test_z_norm, test_y_true = synthesizer.get_score_mae(
        X_test_scaled,
        y_test,
        already_scaled=True,
    )

    predict_and_evaluate(
        test_re_loss,
        test_y_true,
        threshold_re,
        save_dir=save_dir,
        filename=f"{filename_prefix}_mode_re.csv",
        mode="re",
    )
    result_95 = predict_and_evaluate(
        test_z_norm,
        test_y_true,
        threshold_95,
        save_dir=save_dir,
        filename=f"{filename_prefix}_mode_95.csv",
        mode="dist",
    )
    test_total_loss = _compute_total_loss_from_outputs(test_re_loss, test_z_norm, lamda)
    result_total95 = predict_and_evaluate(
        test_total_loss,
        test_y_true,
        threshold_95_total_loss,
        save_dir=save_dir,
        filename=f"{filename_prefix}_mode_total95.csv",
        mode="dist",
    )
    return {"result_95": result_95, "result_total95": result_total95}


def post_train_checkpoint_evaluation(results_dir, dataset, lamda):
    print("Evaluating validation-selected best checkpoint on held-out test data...")
    x_test_path = os.path.join(results_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(results_dir, f"{dataset}_y_test.csv")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    best_model_path = os.path.join(checkpoint_dir, f"MAE_{dataset}_best.pkl")
    best_threshold_path = os.path.join(checkpoint_dir, f"MAE_{dataset}_thresholds_best.pkl")

    for path in [x_test_path, y_test_path, best_model_path, best_threshold_path]:
        if not os.path.exists(path):
            print(f"[WARN] Missing file, skip post-train eval: {path}")
            return

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    best_synth = REDUCEMAECTGANSynthesizer.load(best_model_path)
    with open(best_threshold_path, "rb") as f:
        best_thresholds = pickle.load(f)

    _evaluate_model_with_thresholds(
        synthesizer=best_synth,
        X_test_scaled=X_test,
        y_test=y_test,
        thresholds=best_thresholds,
        save_dir=checkpoint_dir,
        filename_prefix=f"test_{dataset}_best",
        lamda=lamda,
    )
    print("Held-out test evaluation done.")


def train_reduce_ctgan(args):
    dataset = normalize_dataset_name(args.d)
    set_seed(args.seed)

    loaders = {
        "unsw": load_data_unsw,
        "cic_ids": load_data_cic_ids,
        "nsl_kdd": load_data_nsl_kdd,
        "cic_iot": load_data_cic_iot,
        "cidds_001": lambda: load_data_cidds_001(seed=args.seed, test_size=0.4),
    }
    if dataset not in loaders:
        supported = ", ".join(loaders)
        raise ValueError(f"Unsupported dataset '{args.d}'. Supported datasets: {supported}")

    df_full, y_full, df_test, y_test = loaders[dataset]()
    print("Test shape:", df_test.shape)
    print("y_test shape:", y_test.shape)
    print("Full shape:", df_full.shape)
    print("y_full shape:", y_full.shape)

    time_tag = f"{int(args.time):02d}"
    run_folder = f"{args.type_train}_seed_{args.seed}_time_{time_tag}"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(
        base_dir,
        "results",
        run_folder,
        f"per_{args.per}",
        result_dir_name(args, dataset),
    )
    os.makedirs(results_dir, exist_ok=True)

    n_test = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    X_test_raw.columns = X_test_raw.columns.map(str)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)
    y_test = y_full.iloc[-n_test:].reset_index(drop=True)

    normal_mask = y_train == 0
    X_train_normal_raw_full = X_train_raw.loc[normal_mask].copy()
    y_train_normal_full = y_train.loc[normal_mask].copy()
    sampled_index = X_train_normal_raw_full.sample(
        frac=args.per,
        random_state=args.seed,
    ).index
    X_normal_subset_raw = X_train_normal_raw_full.loc[sampled_index].reset_index(drop=True)
    y_normal_subset = y_train_normal_full.loc[sampled_index].reset_index(drop=True)
    X_train_normal_raw, X_val_raw, _, y_val = train_test_split(
        X_normal_subset_raw,
        y_normal_subset,
        test_size=0.2,
        random_state=args.seed,
    )

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_train_normal_raw)
    metadata_file = os.path.join(
        results_dir,
        f"{dataset}_REDUCE_MAE_metadata_epochs_{args.e}.json",
    )
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    metadata.save_to_json(metadata_file)

    synthesizer = REDUCEMAECTGANSynthesizer(
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
        mae_loss_mode=args.mae_loss_mode,
        dataset=dataset,
        path=results_dir,
        type_train=args.type_train,
        mae_start_ratio=args.mae_start_ratio,
        ctgan_noise_start_ratio=args.ctgan_noise_start_ratio,
        ctgan_noise_std=args.ctgan_noise_std,
        ctgan_noise_feature_ratio=args.ctgan_noise_feature_ratio,
        ctgan_pseudo_anomaly_mode=args.ctgan_pseudo_anomaly_mode,
        ctgan_feature_outlier_r=args.ctgan_feature_outlier_r,
    )

    X_train_source = synthesizer.preprocess(X_train_normal_raw)
    X_val_source = synthesizer._data_processor.transform(X_val_raw)
    X_test_source = synthesizer._data_processor.transform(X_test_raw)

    print("Number of normal samples in validation data:", len(X_val_source), X_val_source.shape)
    print("Number of normal samples in training data:", len(X_train_source), X_train_source.shape)
    print("Number of samples in test data:", len(X_test_source), X_test_source.shape)

    synthesizer._fit(X_train_source, X_train_source, X_val_source)

    X_val_mae = synthesizer.transform_to_mae_space(X_val_source, already_source=True)
    X_test_mae = synthesizer.transform_to_mae_space(X_test_source, already_source=True)
    X_test_mae.to_csv(os.path.join(results_dir, f"{dataset}_X_test.csv"), index=False)
    y_test.to_frame(name="label").to_csv(
        os.path.join(results_dir, f"{dataset}_y_test.csv"),
        index=False,
    )
    X_val_mae.to_csv(os.path.join(results_dir, f"{dataset}_X_val_normal.csv"), index=False)
    y_val.to_frame(name="label").to_csv(
        os.path.join(results_dir, f"{dataset}_y_val_normal.csv"),
        index=False,
    )

    (
        avg_loss,
        threshold_re,
        threshold_z_0,
        threshold_95,
        threshold_95_total_loss,
    ) = synthesizer.get_threshold_mae(X_val_mae, already_scaled=True)
    print("Threshold_re:", threshold_re)
    print("Threshold_z:", threshold_z_0)
    print("Threshold_95:", threshold_95)
    print("Threshold_95_total_loss:", threshold_95_total_loss)
    thresholds = (threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss)
    save_threshold(
        thresholds,
        save_dir=results_dir,
        name=f"{dataset}_mae_threshold_re_epochs_{args.e}.pkl",
    )
    save_threshold(
        thresholds,
        save_dir=results_dir,
        name=f"{dataset}_mae_threshold_z_epochs_{args.e}.pkl",
    )

    file_model = os.path.join(results_dir, f"REDUCE_MAE_{dataset}_epochs_{args.e}.pkl")
    synthesizer.save(file_model)

    loss_df = synthesizer.get_loss_values()
    loss_df.to_csv(os.path.join(results_dir, f"reduce_mae_loss_log_epochs_{args.e}.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["Epoch"], loss_df["MAE Loss"], label="MAE total Loss")
    plt.plot(loss_df["Epoch"], loss_df["Reconstruction Loss"], label="Reconstruction Loss")
    plt.plot(loss_df["Epoch"], args.lae * loss_df["Latent Loss"], label="lambda*Latent Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Reduced MAE Loss Curves\ndataset={dataset}_epochs={args.e}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_reduce_mae_dataset_{dataset}_epochs_{args.e}.png"))
    plt.close()

    val_loss_df = loss_df.dropna(subset=["Validation Loss"])
    if not val_loss_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(val_loss_df["Epoch"], val_loss_df["Validation Loss"], label="Validation total Loss")
        if "Validation Reconstruction Loss" in val_loss_df.columns:
            plt.plot(
                val_loss_df["Epoch"],
                val_loss_df["Validation Reconstruction Loss"],
                label="Validation Reconstruction Loss",
            )
        if "Validation Latent Loss" in val_loss_df.columns:
            plt.plot(
                val_loss_df["Epoch"],
                args.lae * val_loss_df["Validation Latent Loss"],
                label="lambda*Validation Latent Loss",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Reduced Validation Loss Curves\ndataset={dataset}_epochs={args.e}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"loss_reduce_mae_val_dataset_{dataset}_epochs_{args.e}.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["Epoch"], loss_df["Generator Loss"], label="Generator Loss")
    plt.plot(loss_df["Epoch"], loss_df["Discriminator Loss"], label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CTGAN Loss Curves\ndataset={dataset}_epochs={args.e}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_reduce_ctgan_dataset_{dataset}_epochs_{args.e}.png"))
    plt.close()

    return synthesizer, results_dir, dataset


def main():
    args = parse_args()
    print("Starting reduced-space MAE-CTGAN training...")
    _, results_dir, dataset = train_reduce_ctgan(args)
    post_train_checkpoint_evaluation(results_dir=results_dir, dataset=dataset, lamda=args.lae)
    print("Finished reduced-space MAE-CTGAN training.")


if __name__ == "__main__":
    main()

# Example:
# python Syn_ctgan/train_reduce_ctgan.py --d unsw --e 200 --b 128 --lae 0.1 --mae 1.0 --mae_loss_mode renormal --per 1.0 --type_train noise_std --mae_start_ratio 0.25 --ctgan_noise_start_ratio 0.75 --ctgan_noise_std 0.05 --ctgan_noise_feature_ratio 0.5 --ctgan_pseudo_anomaly_mode origin --seed 42 --time 1
