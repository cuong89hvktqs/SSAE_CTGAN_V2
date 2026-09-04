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
from sdv.single_table import MAECTGANSynthesizer
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

# Đường dẫn dữ liệu và các tham số sẽ nhận từ dòng lệnh
#python train_MABMAE_CTGAN.py --d unsw --e 200 --b 128 --lae 0.1 --mae 5.0 --per 0.1 --type_train std --seed 42 --time 1
import argparse
parser = argparse.ArgumentParser(description="Train MAE-CTGAN")
parser.add_argument('--d', type=str, default='unsw', help='Tên dataset sử dụng')
parser.add_argument('--e', type=int, default=200, help='Số epoch huấn luyện')
parser.add_argument('--b', type=int, default=128, help='Batch size')
parser.add_argument('--lae', type=float, default=0.1, help='Lamda')
parser.add_argument('--mae', type=float, default=5.0, help='Margin')
parser.add_argument(
    '--mae_loss_mode',
    type=str,
    choices=['origin', 'renormal'],
    default='origin',
    help='MAE training loss mode. origin uses reconstruction loss for all samples; renormal uses reconstruction loss only for normal samples.',
)
parser.add_argument('--per', type=float, default=1.0, help='Percentage train set')  # Thêm tham số GPU ID
parser.add_argument('--type_train', type=str, choices=["all","noise","std","3std","std_3std","noise_std","noise_3std"], default="all", help='Type train MAE')  # Thêm tham số GPU ID
parser.add_argument(
    '--mae_start_ratio',
    type=float,
    default=0.0,
    help='Fraction of epochs completed before generated CTGAN samples train MAE (0.0-1.0).',
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
    help='How CTGAN generated samples are converted into MAE pseudo-anomalies.',
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

if not 0.0 <= args.mae_start_ratio <= 1.0:
    parser.error('--mae_start_ratio must be in [0, 1].')
if not 0.0 <= args.ctgan_noise_start_ratio <= 1.0:
    parser.error('--ctgan_noise_start_ratio must be in [0, 1].')
if args.ctgan_noise_std < 0.0:
    parser.error('--ctgan_noise_std must be greater than or equal to 0.')
if not 0.0 <= args.ctgan_noise_feature_ratio <= 1.0:
    parser.error('--ctgan_noise_feature_ratio must be in [0, 1].')
if args.ctgan_feature_outlier_r < 0.0:
    parser.error('--ctgan_feature_outlier_r must be greater than or equal to 0.')
if args.mae_start_ratio >= 1.0:
    parser.error('--mae_start_ratio must be less than 1.0 so MAE is trained for at least one epoch.')
if not 0.0 < args.per <= 1.0:
    parser.error('--per must be in (0, 1].')
    
dataset = args.d
epochs = args.e
batch_size = args.b
lamda = args.lae
margin = args.mae
mae_loss_mode = args.mae_loss_mode
percentage = args.per
type_train = args.type_train
mae_start_ratio = args.mae_start_ratio
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
    " Lamda:", lamda,
    " Margin:", margin,
    " MAE loss mode:", mae_loss_mode,
    " Percentage:", percentage,
    " Type train:", type_train,
    " MAE start ratio:", mae_start_ratio,
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

# In số lượng mẫu của mỗi nhãn trong tập train
unique_labels, counts = np.unique(y_full, return_counts=True)
print("Số lượng mẫu của mỗi nhãn trong tập train:")
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")
unique_labels, counts = np.unique(y_test, return_counts=True)
print("Số lượng mẫu của mỗi nhãn trong tập test:")
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")

# Tạo thư mục lưu kết quả nếu chưa có
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if(ctgan_pseudo_anomaly_mode=="feature_outlier"):
    results_dir = os.path.join(
        BASE_DIR,
        f"results/{run_folder}/per_{percentage}/"+
        f"mae_{dataset}_{epochs}_m_{margin}_l_{lamda}"
        f"_maeloss_{mae_loss_mode}"
        f"_mstart_{mae_start_ratio}_nstart_{ctgan_noise_start_ratio}"
        f"_nfeat_{ctgan_noise_feature_ratio}"
        f"_pmode_{ctgan_pseudo_anomaly_mode}_r_{ctgan_feature_outlier_r}"
    )
else:
    results_dir = os.path.join(
        BASE_DIR,
        f"results/{run_folder}/per_{percentage}/"+
        f"mae_{dataset}_{epochs}_m_{margin}_l_{lamda}"
        f"_maeloss_{mae_loss_mode}"
        f"_mstart_{mae_start_ratio}_nstart_{ctgan_noise_start_ratio}"
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
    
def train_mae_ctgan(epochs):
    print("Training MAE-CTGAN model with epochs =", epochs)
    # Tạo metadata
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
        f"{dataset}_MAE_metadata_epochs_{epochs}.json"
    )
    # Xóa file cũ nếu đã tồn tại
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    metadata.save_to_json(metadata_file)
    # Khởi tạo và huấn luyện mô hình MAE-CTGAN  
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
        epochs=epochs,
        batch_size=batch_size,
        metadata=metadata,
        lamda=lamda,
        margin=margin,
        mae_loss_mode=mae_loss_mode,
        dataset=dataset,
        path=results_dir,
        type_train=type_train,
        mae_start_ratio=mae_start_ratio,
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


    # Lưu X_train, X_test, y_train, y_test ra file CSV
    #x_train_file = os.path.join(results_dir, f'{dataset}_X_train.csv')
    #y_train_file = os.path.join(results_dir, f'{dataset}_y_train.csv')
    x_test_file = os.path.join(results_dir, f'{dataset}_X_test.csv')
    y_test_file = os.path.join(results_dir, f'{dataset}_y_test.csv')
    
    #X_train.to_csv(x_train_file, index=False)
    #y_train.to_frame(name='label').to_csv(y_train_file, index=False)
    X_test.to_csv(x_test_file, index=False)
    y_test.to_frame(name='label').to_csv(y_test_file, index=False)
    
    # Từ tập train chỉ lấy các bản ghi nhãn là bình thường
    #Lấy 10% dữ liệu bình thường để tính threshold
    
    
    
    #X_train_normal,X_val,y_train_normal,y_val = train_test_split(X_train_normal_full, y_train[y_train == 0], test_size=0.2, random_state=42)
    
    X_val_file = os.path.join(results_dir, f'{dataset}_X_val_normal.csv')
    y_val_file = os.path.join(results_dir, f'{dataset}_y_val_normal.csv')
    X_val.to_csv(X_val_file, index=False)
    y_val.to_frame(name='label').to_csv(y_val_file, index=False)
    
    print("Number of normal samples in validation data:", len(X_val), " shape:", X_val.shape)
    
    print("Number of normal samples in training data:", len(X_train_normal), " shape:", X_train_normal.shape)
    print("Number of samples in test data:", len(X_test), " shape:", X_test.shape)
    unique_labels, counts = np.unique(y_test, return_counts=True)
    print("Số lượng mẫu của mỗi nhãn trong tập test sau khi xử lý và lưu trữ:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} samples")
    
    synthesizer._fit(X_train_normal, X_train_normal, X_val)
    #synthesizer.fit(X_train_normal)
    
    # Tính threshold từ tập validation
    avg_loss,threshold_re, threshold_z_0,threshold_95, threshold_95_total_loss = synthesizer.get_threshold_mae(X_val)
    print("Threshold_re:", threshold_re)
    print("Threshold_z:", threshold_z_0)
    print("Threshold_95:", threshold_95)
    print("Threshold_95_total_loss:", threshold_95_total_loss)
    save_threshold(
        (threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss),
        save_dir=results_dir,
        name=f"{dataset}_mae_threshold_re_epochs_{epochs}.pkl"
    )
    save_threshold(
        (threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss),
        save_dir=results_dir,
        name=f"{dataset}_mae_threshold_z_epochs_{epochs}.pkl"
    )
    
     # Lưu mô hình
    
    
    file_model = os.path.join(results_dir, f'MAE_{dataset}_epochs_{epochs}.pkl')
    synthesizer.save(file_model)
    #synthetic_data = synthesizer.sample(num_rows=1000)
    #file_gen_data = os.path.join(results_dir, f'MAE_{dataset}_epochs_{epochs}.csv')
    #synthetic_data.to_csv(file_gen_data, index=False)
    
    
    # Vẽ và lưu biểu đồ loss
    loss_df = synthesizer.get_loss_values()
    plt.figure(figsize=(10,6))
    #plt.plot(loss_df['Epoch'], loss_df['Generator Loss'], label='Generator Loss')
    #plt.plot(loss_df['Epoch'], loss_df['Discriminator Loss'], label='Discriminator Loss')
    plt.plot(loss_df['Epoch'], loss_df['MAE Loss'], label='MAE total Loss')
    plt.plot(loss_df['Epoch'], loss_df['Reconstruction Loss'], label='Reconstruction Loss')
    plt.plot(loss_df['Epoch'], loss_df['Latent Loss'], label='Latent Loss')   
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curves\ndataset={dataset}_epochs={epochs}')
    plt.legend()
    plt.grid(True)
    loss_fig_name = os.path.join(results_dir, f'loss_mae_dataset_{dataset}_epochs_{epochs}.png')
    plt.savefig(loss_fig_name)
    plt.close()
    print("Training completed.")
    # Lưu toàn bộ loss_df ra file CSV
    log_file = f"{results_dir}/mae_loss_log_epochs_{epochs}.csv"
    loss_df.to_csv(log_file, index=False)
    print(f"Loss log saved to {log_file}")
    
    
    # Vẽ và lưu biểu đồ loss
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
        "Mean z_norm normal:",
        Z_norm_test[y_true == 0].mean()
    )
    print(
        "Mean z_norm anomaly:",
        Z_norm_test[y_true == 1].mean()
    )
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print(f"Latent errors shape: {Z_norm_test.shape}")
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


def _compute_total_loss_from_outputs(test_re_loss, test_z_norm):
    """total95 inference score (no label usage): total_loss = re + lamda * (z_norm^2)."""
    return test_re_loss + lamda * (test_z_norm ** 2)


def _evaluate_model_with_thresholds(synthesizer, X_test, y_test, thresholds, save_dir, filename_prefix):
    """Evaluate one loaded model and save all standard CSV outputs."""
    threshold_re, threshold_z_0, threshold_95, threshold_95_total_loss = thresholds

    test_re_loss, test_z_norm, test_y_true = synthesizer.get_score_mae(X_test, y_test)

    # k_std_list = [3.0, 3.5, 4.0]
    # for k_std in k_std_list:
    #     predict_and_evaluate(
    #         test_z_norm,
    #         test_y_true,
    #         threshold_z_0,
    #         save_dir=save_dir,
    #         filename=f"{filename_prefix}_k_std_{k_std}.csv",
    #         k_std=k_std,
    #         mode="std"
    #     )

    predict_and_evaluate(
        test_re_loss,
        test_y_true,
        threshold_re,
        save_dir=save_dir,
        filename=f"{filename_prefix}_mode_re.csv",
        mode="re"
    )
    result_95 = predict_and_evaluate(
        test_z_norm,
        test_y_true,
        threshold_95,
        save_dir=save_dir,
        filename=f"{filename_prefix}_mode_95.csv",
        mode="dist"
    )

    test_total_loss = _compute_total_loss_from_outputs(test_re_loss, test_z_norm)
    result_total95 = predict_and_evaluate(
        test_total_loss,
        test_y_true,
        threshold_95_total_loss,
        save_dir=save_dir,
        filename=f"{filename_prefix}_mode_total95.csv",
        mode="dist"
    )
    return {"result_95": result_95, "result_total95": result_total95}


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
    """Evaluate the checkpoint selected by validation exactly once on held-out test data."""
    del epochs
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
    best_synth = MAECTGANSynthesizer.load(best_model_path)
    with open(best_threshold_path, "rb") as f:
        best_thresholds = pickle.load(f)

    _evaluate_model_with_thresholds(
        synthesizer=best_synth,
        X_test=X_test,
        y_test=y_test,
        thresholds=best_thresholds,
        save_dir=checkpoint_dir,
        filename_prefix=f"test_{dataset}_best"
    )
    print("Held-out test evaluation done.")


if __name__ == "__main__":
    print("Starting training process with x% normal...")
    synthesizer = train_mae_ctgan(epochs=epochs)
    post_train_checkpoint_evaluation(results_dir=results_dir, dataset=dataset, epochs=epochs)
    print("Finishing training process with x% normal...")

# Full PowerShell command example:
# Origin mode (existing Gaussian-noise behavior):
# python train_MAE_CTGAN.py --d unsw --e 200 --b 128 --lae 0.1 --mae 5.0 --mae_loss_mode origin --per 0.1 --type_train std --mae_start_ratio 0.25 --ctgan_noise_start_ratio 0.75 --ctgan_noise_std 0.1 --ctgan_noise_feature_ratio 1.0 --ctgan_pseudo_anomaly_mode origin --seed 42 --time 1
# Feature outlier mode (continuous = CTGAN mean + r * CTGAN std, non-continuous spans from real data):
# python train_MAE_CTGAN.py --d unsw --e 200 --b 128 --lae 0.1 --mae 5.0 --mae_loss_mode renormal --per 0.1 --type_train std --mae_start_ratio 0.25 --ctgan_pseudo_anomaly_mode feature_outlier --ctgan_feature_outlier_r 1.0 --seed 42 --time 1



#python train_MAE_CTGAN.py --d unsw --e 200 --b 128 --lae 0.1 --mae 5.0 --mae_loss_mode renormal --per 0.1 --type_train std --mae_start_ratio 0.25 --ctgan_pseudo_anomaly_mode feature_outlier --ctgan_feature_outlier_r 1.0 --seed 42 --time 1


"""

Các tham số trong train_MAE_CTGAN.py:

--lae:
Hệ số lambda của MAE loss. Default là 0.1. Loss của MAE có dạng:
loss = reconstruction_loss + lambda * latent_loss.
Lambda càng lớn thì mô hình càng nhấn mạnh việc kéo normal về gần tâm latent và đẩy pseudo anomaly ra xa margin.

--mae:
Margin của MAE. Default là 5.0. Normal được kéo về latent norm gần 0, còn pseudo anomaly bị phạt nếu latent norm nhỏ hơn margin. Margin quá lớn có thể làm mô hình đẩy anomaly giả quá mạnh; margin quá nhỏ có thể tách normal/anomaly chưa rõ.

--mae_loss_mode:
Cách tính reconstruction loss cho MAE. Default là "origin". Có 2 lựa chọn:
origin: reconstruction loss tính trên cả normal và pseudo anomaly. Cách này có thể làm AE học reconstruct cả anomaly giả, đôi khi làm giảm khả năng phát hiện anomaly.
renormal: reconstruction loss chỉ tính trên normal. Pseudo anomaly chỉ tham gia latent margin loss. Cách này thường hợp lý hơn cho anomaly detection.

--per:
Tỉ lệ dữ liệu normal train được sử dụng. Default là 1.0, tức dùng toàn bộ normal train. Ví dụ --per 0.1 nghĩa là chỉ dùng 10% normal train.

--type_train:
Chọn loại pseudo anomaly dùng để train MAE. Default là "all". Các lựa chọn:
all: dùng cả CTGAN sample từ 3std, perturbed real normal, và CTGAN sample từ std.
noise: chỉ dùng perturbed real normal.
std: dùng CTGAN sample từ z ~ N(0,I).
3std: dùng CTGAN sample từ z ~ N(0,3I).
std_3std: dùng cả CTGAN sample từ z ~ N(0,I) và z ~ N(0,3I).
noise_std: dùng perturbed real normal và CTGAN sample từ z ~ N(0,I).
noise_3std: dùng perturbed real normal và CTGAN sample từ z ~ N(0,3I).
Lưu ý: std là sample từ prior bình thường của CTGAN nên nếu CTGAN học tốt normal, label std là anomaly có thể gây nhiễu. Vì vậy noise_3std hoặc 3std thường đáng thử hơn all.

--mae_start_ratio:
Tỉ lệ epoch trước khi bắt đầu train MAE bằng pseudo anomaly. Default là 0.0. Ví dụ --e 200 --mae_start_ratio 0.25 nghĩa là MAE bắt đầu train từ khoảng epoch 50. Nếu để 0.0, MAE học ngay từ đầu khi CTGAN còn chưa ổn định, có thể làm pseudo anomaly kém chất lượng.

--ctgan_noise_start_ratio:
Tỉ lệ epoch trước khi bắt đầu thêm Gaussian noise vào continuous feature của CTGAN sample. Default là 1.0. Ví dụ --e 200 --ctgan_noise_start_ratio 0.75 nghĩa là noise bắt đầu được thêm từ khoảng epoch 150. 
Tham số này chỉ có tác dụng khi MAE đã active và pseudo anomaly mode là origin.

--ctgan_noise_std:
Độ lệch chuẩn của Gaussian noise thêm vào continuous feature của CTGAN sample. Default là 0.1. 
Giá trị càng lớn thì pseudo anomaly càng bị nhiễu mạnh. Nếu muốn chạy gần với ý tưởng chỉ dùng z ~ N(0,I) và z ~ N(0,3I), có thể đặt --ctgan_noise_std 0.0.

--ctgan_noise_feature_ratio:
Tỉ lệ continuous feature bị thêm Gaussian noise. Default là 1.0, tức thêm noise vào toàn bộ continuous feature. Ví dụ 0.3 nghĩa là chỉ chọn khoảng 30% continuous feature để thêm noise.

--ctgan_pseudo_anomaly_mode:
Cách tạo pseudo anomaly từ CTGAN sample. Default là "origin". Có 2 lựa chọn:
origin: dùng sample CTGAN bình thường, sample từ z ~ N(0,3I), và có thể thêm Gaussian noise.
feature_outlier: tạo anomaly bằng cách lấy real sample làm nền, sau đó thay continuous feature bằng giá trị dạng mean + r * std dựa trên generated sample.

--ctgan_feature_outlier_r:
Hệ số outlier khi dùng --ctgan_pseudo_anomaly_mode feature_outlier. Default là 1.0. Công thức trong code là continuous = mean + r * std. r càng lớn thì feature càng bị đẩy xa phân phối generated sample. Tham số này chủ yếu có ý nghĩa khi mode là feature_outlier.

--seed:
Random seed. Default là 42. Dùng để cố định random của Python, NumPy và PyTorch, đồng thời ảnh hưởng đến split train/validation và quá trình train.

--time:
Chỉ số lần chạy. Default là 1. Tham số này chủ yếu dùng để đặt tên thư mục kết quả dạng time_01, time_02,... giúp phân biệt các lần chạy cùng cấu hình.

Các tham số cố định trong file, hiện chưa chỉnh được từ command line:
embedding_dim=128: kích thước vector noise đầu vào generator của CTGAN.
generator_dim=(256, 256): kiến trúc generator gồm 2 block ẩn kích thước 256.
discriminator_dim=(256, 256): kiến trúc discriminator gồm 2 lớp ẩn kích thước 256.
generator_lr=2e-4: learning rate của generator.
generator_decay=1e-6: weight decay của generator.
discriminator_lr=2e-4: learning rate của discriminator.
discriminator_decay=1e-6: weight decay của discriminator.
discriminator_steps=1: số bước train discriminator cho mỗi bước train generator.
log_frequency=True: dùng log-frequency sampling cho categorical condition của CTGAN.
verbose=True: in log trong quá trình train.
pac=8: số sample được pack lại khi đưa vào discriminator. Batch size nên chia hết cho pac.
cuda=True: ưu tiên chạy GPU nếu có.

Một cấu hình nên thử:
python Syn_ctgan/train_MAE_CTGAN.py --d unsw --e 200 --b 128 --lae 0.1 --mae 5.0 --mae_loss_mode renormal --per 1.0 --type_train noise_3std --mae_start_ratio 0.25 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_std 0.0 --seed 42 --time 1

"""
