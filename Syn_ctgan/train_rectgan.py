import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import RECTGANSynthesizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,matthews_corrcoef
from sdv.errors import  InvalidDataTypeError
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import csv
from process_data import load_data_unsw,load_data_cic_ids,load_data_nsl_kdd,load_data_cic_iot
from sklearn.model_selection import train_test_split
import sdv
print(sdv.__file__)
import random
import ctgan
print(ctgan.__file__)

# ÄÆ°á»ng dáº«n dá»¯ liá»‡u vÃ  cÃ¡c tham sá»‘ sáº½ nháº­n tá»« dÃ²ng lá»‡nh
# python train_rectgan.py --d unsw --e 200 --b 128 --r 1.0 --per 0.1 --type_train noise_3std --seed 42 --time 1
import argparse
parser = argparse.ArgumentParser(description="Train RECTGAN")
parser.add_argument('--d', type=str, default='unsw', help='TÃªn dataset sá»­ dá»¥ng')
parser.add_argument('--e', type=int, default=200, help='Sá»‘ epoch huáº¥n luyá»‡n')
parser.add_argument('--b', type=int, default=128, help='Batch size')
parser.add_argument(
    '--r',
    '--re_abnormal_target',
    dest='re_abnormal_target',
    type=float,
    default=1.0,
    help='Target R in abnormal reconstruction loss: mean((R - sqrt((x_hat - x)^2))^2).',
)
parser.add_argument('--per', type=float, default=1.0, help='Percentage train set')  # ThÃªm tham sá»‘ GPU ID
parser.add_argument('--type_train', type=str, choices=["all","noise","std","3std","std_3std","noise_std","noise_3std"], default="all", help='Type train REC')  # ThÃªm tham sá»‘ GPU ID
parser.add_argument(
    '--rec_start_ratio',
    type=float,
    default=0.0,
    help='Fraction of epochs completed before generated CTGAN samples train REC (0.0-1.0).',
)
parser.add_argument(
    '--ctgan_noise_start_ratio',
    type=float,
    default=1.0,
    help='Fraction of epochs completed before Gaussian noise is added to continuous CTGAN sample features (0.0-1.0).',
)
parser.add_argument(
    '--ctgan_noise_std',
    type=float,
    default=0.05,
    help='Standard deviation of Gaussian noise added to generated continuous CTGAN features.',
)
parser.add_argument(
    '--ctgan_noise_feature_ratio',
    type=float,
    default=1.0,
    help='Fraction of generated continuous CTGAN features perturbed by Gaussian noise (0.0-1.0).',
)
parser.add_argument(
    '--ctgan_pseudo_anomaly_mode',
    type=str,
    choices=['origin', 'feature_outlier'],
    default='origin',
    help='How CTGAN generated samples are converted into REC pseudo-anomalies.',
)
parser.add_argument(
    '--ctgan_feature_outlier_r',
    type=float,
    default=1.0,
    help='Outlier coefficient r for feature_outlier mode: continuous = mean + r * std.',
)
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--time', type=int, default=1, help='Run index (e.g., 1 -> time_01)')
args = parser.parse_args()

if not 0.0 <= args.rec_start_ratio <= 1.0:
    parser.error('--rec_start_ratio must be in [0, 1].')
if not 0.0 <= args.ctgan_noise_start_ratio <= 1.0:
    parser.error('--ctgan_noise_start_ratio must be in [0, 1].')
if args.ctgan_noise_std < 0.0:
    parser.error('--ctgan_noise_std must be greater than or equal to 0.')
if not 0.0 <= args.ctgan_noise_feature_ratio <= 1.0:
    parser.error('--ctgan_noise_feature_ratio must be in [0, 1].')
if args.ctgan_feature_outlier_r < 0.0:
    parser.error('--ctgan_feature_outlier_r must be greater than or equal to 0.')
if args.re_abnormal_target < 0.0:
    parser.error('--r/--re_abnormal_target must be greater than or equal to 0.')
if args.rec_start_ratio >= 1.0:
    parser.error('--rec_start_ratio must be less than 1.0 so REC is trained for at least one epoch.')
if not 0.0 < args.per <= 1.0:
    parser.error('--per must be in (0, 1].')
    
dataset = args.d
epochs = args.e
batch_size = args.b
re_abnormal_target = args.re_abnormal_target
percentage = args.per
type_train = args.type_train
rec_start_ratio = args.rec_start_ratio
ctgan_noise_start_ratio = args.ctgan_noise_start_ratio
ctgan_noise_std = args.ctgan_noise_std
ctgan_noise_feature_ratio = args.ctgan_noise_feature_ratio
ctgan_pseudo_anomaly_mode = (
    'feature_outlier'
    if args.ctgan_pseudo_anomaly_mode == 'feature_outliner'
    else args.ctgan_pseudo_anomaly_mode
)
ctgan_feature_outlier_r = args.ctgan_feature_outlier_r
seed = args.seed
run_time = args.time
time_tag = f"{int(run_time):02d}"
run_folder = f"{type_train}_seed_{seed}_time_{time_tag}"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed)

print(
    "Dataset:", dataset,
    " Epochs:", epochs,
    " Batch size:", batch_size,
    " R:", re_abnormal_target,
    " Percentage:", percentage,
    " Type train:", type_train,
    " REC start ratio:", rec_start_ratio,
    " CTGAN noise start ratio:", ctgan_noise_start_ratio,
    " CTGAN noise std:", ctgan_noise_std,
    " CTGAN noise feature ratio:", ctgan_noise_feature_ratio,
    " CTGAN pseudo-anomaly mode:", ctgan_pseudo_anomaly_mode,
    " CTGAN feature outlier r:", ctgan_feature_outlier_r,
    " Run time:", time_tag,
)
if dataset=="unsw":
    df_full, y_full, df_test, y_test = load_data_unsw()
elif dataset=="cic_ids":
    df_full, y_full, df_test, y_test = load_data_cic_ids()  
elif dataset=="nsl_kdd":
    df_full, y_full, df_test, y_test = load_data_nsl_kdd()
elif dataset=="cic_iot":
    df_full, y_full, df_test, y_test = load_data_cic_iot()


print("Test shape:", df_test.shape)
print("y_test shape:", y_test.shape)
print("Full shape:", df_full.shape)
print("y_full shape:", y_full.shape)

# In sá»‘ lÆ°á»£ng máº«u cá»§a má»—i nhÃ£n trong táº­p train
unique_labels, counts = np.unique(y_full, return_counts=True)
print("Sá»‘ lÆ°á»£ng máº«u cá»§a má»—i nhÃ£n trong táº­p train:")
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")
unique_labels, counts = np.unique(y_test, return_counts=True)
print("Sá»‘ lÆ°á»£ng máº«u cá»§a má»—i nhÃ£n trong táº­p test:")
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")

# Táº¡o thÆ° má»¥c lÆ°u káº¿t quáº£ náº¿u chÆ°a cÃ³
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if(ctgan_pseudo_anomaly_mode=="feature_outlier"):
    results_dir = os.path.join(
        BASE_DIR,
        f"results/{run_folder}/per_{percentage}/"+
        f"rec_{dataset}_{epochs}_r_{re_abnormal_target}"
        f"_rstart_{rec_start_ratio}_nstart_{ctgan_noise_start_ratio}"
        f"_nfeat_{ctgan_noise_feature_ratio}"
        f"_pmode_{ctgan_pseudo_anomaly_mode}_r_{ctgan_feature_outlier_r}"
    )
else:
    results_dir = os.path.join(
        BASE_DIR,
        f"results/{run_folder}/per_{percentage}/"+
        f"rec_{dataset}_{epochs}_r_{re_abnormal_target}"
        f"_rstart_{rec_start_ratio}_nstart_{ctgan_noise_start_ratio}"
        f"_nfeat_{ctgan_noise_feature_ratio}"
        f"_pmode_{ctgan_pseudo_anomaly_mode}_nstd_{ctgan_noise_std}"
    )
if not os.path.exists(results_dir):
	os.makedirs(results_dir)
 

def save_threshold(threshold, save_dir=f"results/{run_folder}/per_{percentage}", name="threshold.pkl"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Threshold saved to {path}")
    
def load_threshold(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def train_rec_ctgan(epochs):
    print("Training RECTGAN model with epochs =", epochs)
    # Táº¡o metadata
    # Split off held-out test rows before fitting metadata or any transformer.
    n_test = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    X_test_raw.columns = X_test_raw.columns.map(str)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)
    y_test = y_full.iloc[-n_test:].reset_index(drop=True)

    # One-class training: only normal training records may shape the model or its transforms.
    normal_mask = y_train == 0
    X_train_normal_raw_full = X_train_raw.loc[normal_mask].copy()
    y_train_normal_full = y_train.loc[normal_mask].copy()
    sampled_index = X_train_normal_raw_full.sample(
        frac=percentage,
        random_state=seed,
    ).index
    X_normal_subset_raw = X_train_normal_raw_full.loc[sampled_index].reset_index(drop=True)
    y_normal_subset = y_train_normal_full.loc[sampled_index].reset_index(drop=True)
    X_train_normal_raw, X_val_raw, y_train_normal, y_val = train_test_split(
        X_normal_subset_raw,
        y_normal_subset,
        test_size=0.2,
        random_state=seed,
    )

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_train_normal_raw)
    # Save metadata to JSON
    metadata_file = os.path.join(
        results_dir,
        f"{dataset}_REC_metadata_epochs_{epochs}.json"
    )
    # XÃ³a file cÅ© náº¿u Ä‘Ã£ tá»“n táº¡i
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    metadata.save_to_json(metadata_file)
    # Khá»Ÿi táº¡o vÃ  huáº¥n luyá»‡n mÃ´ hÃ¬nh RECTGAN  
    synthesizer = RECTGANSynthesizer(
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
        epochs=epochs,
        batch_size=batch_size,
        metadata=metadata,
        re_abnormal_target=re_abnormal_target,
        dataset=dataset,
        path=results_dir,
        type_train=type_train,
        rec_start_ratio=rec_start_ratio,
        ctgan_noise_start_ratio=ctgan_noise_start_ratio,
        ctgan_noise_std=ctgan_noise_std,
        ctgan_noise_feature_ratio=ctgan_noise_feature_ratio,
        ctgan_pseudo_anomaly_mode=ctgan_pseudo_anomaly_mode,
        ctgan_feature_outlier_r=ctgan_feature_outlier_r,
    )
    # Split raw train/test first to avoid preprocessing leakage from test into train
    # Fit preprocessing on normal model-training rows only, then transform validation and test.
    X_train_normal = synthesizer.preprocess(X_train_normal_raw)
    X_val = synthesizer._data_processor.transform(X_val_raw)
    X_test = synthesizer._data_processor.transform(X_test_raw)


    # LÆ°u X_train, X_test, y_train, y_test ra file CSV
    #x_train_file = os.path.join(results_dir, f'{dataset}_X_train.csv')
    #y_train_file = os.path.join(results_dir, f'{dataset}_y_train.csv')
    x_test_file = os.path.join(results_dir, f'{dataset}_X_test.csv')
    y_test_file = os.path.join(results_dir, f'{dataset}_y_test.csv')
    
    #X_train.to_csv(x_train_file, index=False)
    #y_train.to_frame(name='label').to_csv(y_train_file, index=False)
    X_test.to_csv(x_test_file, index=False)
    y_test.to_frame(name='label').to_csv(y_test_file, index=False)
    
    # Tá»« táº­p train chá»‰ láº¥y cÃ¡c báº£n ghi nhÃ£n lÃ  bÃ¬nh thÆ°á»ng
    #Láº¥y 10% dá»¯ liá»‡u bÃ¬nh thÆ°á»ng Ä‘á»ƒ tÃ­nh threshold
    
    
    
    #X_train_normal,X_val,y_train_normal,y_val = train_test_split(X_train_normal_full, y_train[y_train == 0], test_size=0.2, random_state=42)
    
    X_val_file = os.path.join(results_dir, f'{dataset}_X_val_normal.csv')
    y_val_file = os.path.join(results_dir, f'{dataset}_y_val_normal.csv')
    X_val.to_csv(X_val_file, index=False)
    y_val.to_frame(name='label').to_csv(y_val_file, index=False)
    
    print("Number of normal samples in validation data:", len(X_val), " shape:", X_val.shape)
    
    print("Number of normal samples in training data:", len(X_train_normal), " shape:", X_train_normal.shape)
    print("Number of samples in test data:", len(X_test), " shape:", X_test.shape)
    unique_labels, counts = np.unique(y_test, return_counts=True)
    print("Sá»‘ lÆ°á»£ng máº«u cá»§a má»—i nhÃ£n trong táº­p test sau khi xá»­ lÃ½ vÃ  lÆ°u trá»¯:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} samples")
    
    synthesizer._fit(X_train_normal, X_train_normal, X_val)
    #synthesizer.fit(X_train_normal)
    
    # TÃ­nh threshold tá»« táº­p validation
    avg_loss, threshold_re, threshold_re95 = synthesizer.get_threshold_rec(X_val)
    print("Threshold_re:", threshold_re)
    print("Threshold_re95:", threshold_re95)
    save_threshold(
        {
            "threshold_re": threshold_re,
            "threshold_re95": threshold_re95,
            "threshold_auc": None,
            "pseudo_val_auc": None,
        },
        save_dir=results_dir,
        name=f"{dataset}_rec_thresholds_epochs_{epochs}.pkl"
    )
    
     # LÆ°u mÃ´ hÃ¬nh
    
    
    file_model = os.path.join(results_dir, f'REC_{dataset}_epochs_{epochs}.pkl')
    synthesizer.save(file_model)
    #synthetic_data = synthesizer.sample(num_rows=1000)
    #file_gen_data = os.path.join(results_dir, f'REC_{dataset}_epochs_{epochs}.csv')
    #synthetic_data.to_csv(file_gen_data, index=False)
    
    
    # Váº½ vÃ  lÆ°u biá»ƒu Ä‘á»“ loss
    loss_df = synthesizer.get_loss_values()
    plt.figure(figsize=(10,6))
    #plt.plot(loss_df['Epoch'], loss_df['Generator Loss'], label='Generator Loss')
    #plt.plot(loss_df['Epoch'], loss_df['Discriminator Loss'], label='Discriminator Loss')
    plt.plot(loss_df['Epoch'], loss_df['REC Loss'], label='REC total Loss')
    plt.plot(loss_df['Epoch'], loss_df['Normal Reconstruction Loss'], label='Normal Reconstruction Loss')
    plt.plot(loss_df['Epoch'], loss_df['Abnormal Reconstruction Loss'], label='Abnormal Reconstruction Target Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curves\ndataset={dataset}_epochs={epochs}')
    plt.legend()
    plt.grid(True)
    loss_fig_name = os.path.join(results_dir, f'loss_rec_dataset_{dataset}_epochs_{epochs}.png')
    plt.savefig(loss_fig_name)
    plt.close()

    val_loss_df = loss_df.dropna(subset=['Validation Reconstruction Loss'])
    if not val_loss_df.empty:
        plt.figure(figsize=(10,6))
        plt.plot(
            val_loss_df['Epoch'],
            val_loss_df['Validation Reconstruction Loss'],
            label='Validation Reconstruction Loss'
        )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Validation Loss Curves\ndataset={dataset}_epochs={epochs}')
        plt.legend()
        plt.grid(True)
        val_loss_fig_name = os.path.join(results_dir, f'loss_rec_val_dataset_{dataset}_epochs_{epochs}.png')
        plt.savefig(val_loss_fig_name)
        plt.close()
        print(f"Validation loss figure saved to {val_loss_fig_name}")

    print("Training completed.")
    # LÆ°u toÃ n bá»™ loss_df ra file CSV
    log_file = f"{results_dir}/rec_loss_log_epochs_{epochs}.csv"
    loss_df.to_csv(log_file, index=False)
    print(f"Loss log saved to {log_file}")
    
    
    # Váº½ vÃ  lÆ°u biá»ƒu Ä‘á»“ loss
    plt.figure(figsize=(10,6))
    plt.plot(loss_df['Epoch'], loss_df['Generator Loss'], label='Generator Loss')
    plt.plot(loss_df['Epoch'], loss_df['Discriminator Loss'], label='Discriminator Loss')  
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curves\ndataset={dataset}_epochs={epochs}')
    plt.legend()
    plt.grid(True)
    loss_fig_name = os.path.join(results_dir, f'loss_ctgan_dataset_{dataset}_epochs_{epochs}.png')
    plt.savefig(loss_fig_name)
    plt.close()

    print(f"Loss log saved to {log_file}")
    
    
    print("Test evaluation is deferred to the validation-selected best checkpoint.")
    
    
    
    return synthesizer



def predict_and_evaluate(
    Z_norm_test,
    y_true,
    threshold,
    save_dir="results",
    filename="test_results.csv",
    k_std=3.0,
    mode="std"
):
    """
    AE-based anomaly detection evaluation
    - Predict anomaly using reconstruction error
    - Compute AUC, ACC, F1, Precision, Recall
    - Compute confusion matrix
    - Save everything to TXT
    """
    # -------------------------
    # Prediction
    # -------------------------
    print(
        "Mean score normal:",
        Z_norm_test[y_true == 0].mean()
    )
    print(
        "Mean score anomaly:",
        Z_norm_test[y_true == 1].mean()
    )
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print(f"Anomaly score shape: {Z_norm_test.shape}")
    #y_true = np.concatenate(y_true)
    if mode=="re":
        mu, std = threshold
        threshold_test = mu + k_std * std
    elif mode=="std":
        mu, std = threshold
        threshold_test = mu + k_std * std
    elif mode=="dist":
        threshold_test = threshold
    
    y_pred = (Z_norm_test > threshold_test).astype(int)
    print("y_pred_shape:", y_pred.shape )
    
    print("Predicted label distribution in test set:")
    labels, counts = np.unique(y_pred, return_counts=True)
    for label, count in zip(labels, counts):
        print(f"Label {label}: {count} samples")
    
    
    
    y_true = (y_true != 0).astype(int)  # 0 = normal
    print("True label distribution in test set:")
    
    labels, counts = np.unique(y_true, return_counts=True)
    for label, count in zip(labels, counts):
        print(f"Label {label}: {count} samples")
    # =========================
    # 3. METRICS
    # =========================
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc = 0000  # Default AUC for binary classification with only one class
    else:
        auc = roc_auc_score(y_true, Z_norm_test)*100
    acc = accuracy_score(y_true, y_pred)*100
    f1 = f1_score(y_true, y_pred, zero_division=0)*100
    precision = precision_score(y_true, y_pred, zero_division=0)*100
    recall = recall_score(y_true, y_pred, zero_division=0)*100

    cm = confusion_matrix(y_true, y_pred)
    mcc=matthews_corrcoef(y_true, y_pred)*100

    # =========================
    # 4. SAVE TO TXT
    # =========================
    import csv
    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Threshold", f"{threshold_test:.6f}"])
        writer.writerow(["AUC", f"{auc:.6f}"])
        writer.writerow(["F1", f"{f1:.6f}"])
        writer.writerow(["MCC", f"{mcc:.6f}"])
        writer.writerow(["True 0", cm[0,0], cm[0,1]])
        writer.writerow(["True 1", cm[1,0], cm[1,1]])
        writer.writerow(["Precision", f"{precision:.6f}"])
        writer.writerow(["Recall", f"{recall:.6f}"])
        writer.writerow(["ACC", f"{acc:.6f}"])
        writer.writerow([])
        writer.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
        writer.writerow(["True 0", cm[0,0], cm[0,1]])
        writer.writerow(["True 1", cm[1,0], cm[1,1]])
    

    print(f"Evaluation results saved to {save_path}")

    return {
        "re_errors": Z_norm_test    ,
        "y_pred": y_pred,
        "metrics": {
            "AUC": auc,
            "ACC": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "MCC": mcc
        },
        "confusion_matrix": cm
    }


def _evaluate_model_with_thresholds(synthesizer, X_test, y_test, thresholds, save_dir, filename_prefix):
    """Evaluate one loaded model and save all standard CSV outputs."""
    if isinstance(thresholds, dict):
        threshold_re = thresholds["threshold_re"]
        threshold_re95 = thresholds["threshold_re95"]
        threshold_auc = thresholds.get("threshold_auc")
        checkpoint_kind = thresholds.get("checkpoint_kind", "unknown")
    else:
        threshold_re, threshold_re95 = thresholds[:2]
        threshold_auc = None
        checkpoint_kind = "unknown"

    test_re_loss, test_y_true = synthesizer.get_score_rec(X_test, y_test)

    result_re_std = predict_and_evaluate(
        test_re_loss,
        test_y_true,
        threshold_re,
        save_dir=save_dir,
        filename=f"{filename_prefix}_mode_re_std.csv",
        mode="re"
    )
    result_re95 = predict_and_evaluate(
        test_re_loss,
        test_y_true,
        threshold_re95,
        save_dir=save_dir,
        filename=f"{filename_prefix}_mode_re95.csv",
        mode="dist"
    )

    results = {
        "checkpoint_kind": checkpoint_kind,
        "thresholds": thresholds,
        "result_re_std": result_re_std,
        "result_re95": result_re95,
    }
    if threshold_auc is not None and np.isfinite(threshold_auc):
        results["result_auc"] = predict_and_evaluate(
            test_re_loss,
            test_y_true,
            threshold_auc,
            save_dir=save_dir,
            filename=f"{filename_prefix}_mode_auc.csv",
            mode="dist"
        )
    result_path = os.path.join(save_dir, f"{filename_prefix}_evaluation_results.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Evaluation result pickle saved to {result_path}")
    return results


def _save_best_result_csv(best_result, out_csv):
    metrics = best_result["metrics"]
    cm = best_result["confusion_matrix"]
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["AUC", f"{metrics['AUC']:.6f}"])
        writer.writerow(["F1", f"{metrics['F1']:.6f}"])
        writer.writerow(["MCC", f"{metrics['MCC']:.6f}"])
        writer.writerow(["True 0", cm[0,0], cm[0,1]])
        writer.writerow(["True 1", cm[1,0], cm[1,1]])
        writer.writerow(["Precision", f"{metrics['Precision']:.6f}"])
        writer.writerow(["Recall", f"{metrics['Recall']:.6f}"])
        writer.writerow(["ACC", f"{metrics['ACC']:.6f}"])
        writer.writerow([])
        writer.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
        writer.writerow(["True 0", cm[0,0], cm[0,1]])
        writer.writerow(["True 1", cm[1,0], cm[1,1]])
    print(f"Evaluation results saved to {out_csv}")


def post_train_checkpoint_evaluation(results_dir, dataset, epochs):
    """Evaluate both validation-selected checkpoints on held-out test data."""
    del epochs
    print("Evaluating validation-selected best_auc and best_re95 checkpoints on held-out test data...")
    x_test_path = os.path.join(results_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(results_dir, f"{dataset}_y_test.csv")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")

    for path in [x_test_path, y_test_path]:
        if not os.path.exists(path):
            print(f"[WARN] Missing file, skip post-train eval: {path}")
            return

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    for checkpoint_kind in ["auc", "re95"]:
        best_model_path = os.path.join(
            checkpoint_dir,
            f"REC_{dataset}_best_{checkpoint_kind}.pkl",
        )
        best_threshold_path = os.path.join(
            checkpoint_dir,
            f"REC_{dataset}_thresholds_best_{checkpoint_kind}.pkl",
        )
        if not os.path.exists(best_model_path) or not os.path.exists(best_threshold_path):
            print(
                f"[WARN] Missing {checkpoint_kind} checkpoint or thresholds, skip: "
                f"{best_model_path} / {best_threshold_path}"
            )
            continue

        best_synth = RECTGANSynthesizer.load(best_model_path)
        with open(best_threshold_path, "rb") as f:
            best_thresholds = pickle.load(f)

        _evaluate_model_with_thresholds(
            synthesizer=best_synth,
            X_test=X_test,
            y_test=y_test,
            thresholds=best_thresholds,
            save_dir=checkpoint_dir,
            filename_prefix=f"test_{dataset}_best_{checkpoint_kind}",
        )
    print("Held-out test evaluation done.")


if __name__ == "__main__":
    print("Starting training process with x% normal...")
    synthesizer = train_rec_ctgan(epochs=epochs)
    post_train_checkpoint_evaluation(results_dir=results_dir, dataset=dataset, epochs=epochs)
    print("Finishing training process with x% normal...")
# Full PowerShell command example:
# python Syn_ctgan/train_rectgan.py --d unsw --e 200 --b 128 --r 1.0 --per 0.1 --type_train noise_3std --rec_start_ratio 0.25 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_std 0.0 --seed 42 --time 1

"""
RECTGAN parameters:

--r / --re_abnormal_target:
Target R for pseudo anomaly reconstruction loss. Normal samples use ordinary reconstruction loss. Pseudo anomalies use mean((R - sqrt((x_hat - x)^2))^2). Default is 1.0.

Checkpoint selection:
Each run saves both best checkpoints:
- REC_<dataset>_best_auc.pkl: best by pseudo validation AUC, with REC_<dataset>_thresholds_best_auc.pkl.
- REC_<dataset>_best_re95.pkl: best by lowest normal validation reconstruction loss, with REC_<dataset>_thresholds_best_re95.pkl.
Post-train evaluation writes CSV and pickle results for both checkpoints in the checkpoints folder.

--type_train:
Pseudo anomaly source for REC: all, noise, std, 3std, std_3std, noise_std, noise_3std.

--rec_start_ratio:
Fraction of epochs before starting REC pseudo-anomaly training. Example: --e 200 --rec_start_ratio 0.25 starts REC around epoch 50.

--ctgan_noise_start_ratio, --ctgan_noise_std, --ctgan_noise_feature_ratio:
Gaussian perturbation schedule and strength for generated CTGAN samples when ctgan_pseudo_anomaly_mode=origin.

Recommended first run:
python Syn_ctgan/train_rectgan.py --d unsw --e 200 --b 128 --r 1.0 --per 0.1 --type_train noise_3std --rec_start_ratio 0.25 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_std 0.0 --seed 42 --time 1
"""
