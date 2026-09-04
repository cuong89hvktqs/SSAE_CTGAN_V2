import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from load_data import load_data_unsw, load_data_cic_ids,load_data_nsl_kdd,load_data_cic_iot
from utils.tsne import tsne_2d_2label_and_save, run_tsne_3d__2label_and_save,visualize_1d_with_labels,plot_boxplot
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef
)
from models.ae import SAE, AE
from models.centroid import CentroidBasedOneClassClassifier
import csv
import copy

def split_by_percent_random(X, y, percent, random_state=42):
    np.random.seed(random_state)
    idx = np.random.permutation(len(X))
    n = int(len(X) * percent)
    selected = idx[:n]
    return np.array(X)[selected], np.array(y)[selected]
def train_ae_with_early_stopping(
    X, Y,
    sae_model,
    batch_size=128,
    lr=1e-3,
    max_epochs=500,
    patience=100, #dừng sau 60 eporch loss không giảm tiếp
    device="cuda" if torch.cuda.is_available() else "cpu",
    lamda=1.0,
    path=None
):
    """
    Train AE using NORMAL samples only
    Use validation for early stopping
    Compute anomaly threshold = mean + k * std (on val)
    """
    X_normal = X[Y == 0]
    Y_normal = Y[Y == 0]
    
    print(f"normal samples TRAIN: {X_normal.shape}")
    #rng = np.random.default_rng(seed=42)
    #n = len(X_normal)
    #idx = rng.choice(n, size=int(0.1 * n), replace=False)
    #X_normal_10_percent = X_normal[idx]
    #Y_normal_10_percent = Y_normal[idx]
    #X_normal = X_normal_10_percent
    #Y_normal = Y_normal_10_percent
    print(f"Training SAE on normal samples only {percentage*100}% data: {X_normal.shape}")
    
    
    # =========================
    # 2. SPLIT TRAIN / VAL
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(X_normal, Y_normal, test_size=0.2, random_state=42, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    print(f"Data shape: {X_normal.shape} Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    # =========================
    # 3. MODEL + OPT
    # =========================
    sae_model = sae_model.to(device)
    optimizer = torch.optim.Adadelta(sae_model.parameters())
    #optimizer = torch.optim.Adam(sae_model.parameters(), lr=lr)
    
    criterion = nn.MSELoss(reduction="none")

    # =========================
    # 4. EARLY STOPPING
    # =========================
    best_val_loss = float("inf")
    patience_counter = 0

    # =========================
    # 5. TRAIN LOOP
    # =========================
    # DataFrame lưu loss qua các epoch
    loss_history_log = []

    for epoch in range(max_epochs):
        sae_model.train()
        train_loss_list = []
        train_re_loss_list = []   
        train_latent_loss_list = []

        for (xb,yb) in train_loader:
            xb = xb.to(device)

            latent, xb_hat = sae_model(xb)
            re_loss = torch.mean((xb_hat - xb) ** 2)
            latent_loss = torch.mean(latent** 2)
            loss = re_loss + lamda*latent_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            train_re_loss_list.append(re_loss.item())
            train_latent_loss_list.append(latent_loss.item())

        avg_train_loss = np.mean(train_loss_list)
        avg_train_re_loss = np.mean(train_re_loss_list)  
        avg_train_latent_loss = np.mean(train_latent_loss_list)
        # Lưu vào history
        loss_history_log.append({
            'epoch': epoch,
            'total_loss': avg_train_loss,
            're_loss': avg_train_re_loss,
            'latent_loss': avg_train_latent_loss
        })
        
        
        # ===== VALIDATION =====
        sae_model.eval()
        val_losses_list = []
        val_re_losses_list = []
        val_latent_losses_list = []

        with torch.no_grad():
            for (xb, yb) in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                latent, xb_hat = sae_model(xb)
                """
                val_re_loss,val_latent_loss = sae_model.loss_function(xb, yb)
                val_loss= val_re_loss + lamda* val_latent_loss
                val_losses_list.append(val_loss.item())
                val_re_losses_list.append(val_re_loss.item())
                val_latent_losses_list.append(val_latent_loss.item())
                
                """
                
                val_re_loss= criterion(xb_hat, xb).mean(dim=1)  # reconstruction error per sample
                
                val_latent_loss= torch.mean(latent ** 2, dim=1)  # latent error per sample
                
                
                val_loss= val_re_loss + lamda * val_latent_loss
                val_losses_list.append(val_loss.cpu().numpy())
                val_re_losses_list.append(val_re_loss.cpu().numpy())
                val_latent_losses_list.append(lamda * val_latent_loss.cpu().numpy())
                
        val_losses_list=np.concatenate(val_losses_list)
        val_re_losses_list=np.concatenate(val_re_losses_list)
        val_latent_losses_list=np.concatenate(val_latent_losses_list)

        val_loss_avg = np.mean(val_losses_list)
        val_latent_loss_avg = np.mean(val_latent_losses_list)

        print(f"[Epoch {epoch:03d}] Train total: {avg_train_loss:.6f} | Train re: {avg_train_re_loss:.6f} | Train latent: {avg_train_latent_loss:.6f} | Val total: {val_loss_avg:.6f}")

        # ===== EARLY STOP =====
        if val_loss_avg  < best_val_loss - 1e-6: # tim đươcn model tốt hơn
            best_val_loss = val_loss_avg 
            patience_counter = 0

            best_state = copy.deepcopy(sae_model.state_dict())
            
            mu_val= np.mean(val_losses_list)
            std_val= np.std(val_losses_list)
            threshold_val=(mu_val, std_val) if val_losses_list is not None else None
            
            mu_val_re= np.mean(val_re_losses_list)
            std_val_re= np.std(val_re_losses_list)
            threshold_val_re=(mu_val_re, std_val_re) if val_re_losses_list is not None else None
            
            mu_val_latent= np.mean(val_latent_losses_list)
            std_val_latent= np.std(val_latent_losses_list)
            threshold_val_latent=(mu_val_latent, std_val_latent) if val_latent_losses_list is not None else None  
            threshold_val_latent_95=(np.percentile(val_latent_losses_list,95), std_val_latent) if val_latent_losses_list is not None else None
              
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        if epoch%10==0:
            checkpoint_dir = f"{path}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"[Epoch {epoch:03d}] Train total: {avg_train_loss:.6f} | Train re: {avg_train_re_loss:.6f} | Train latent: {avg_train_latent_loss:.6f} | Val total: {val_loss_avg:.6f}")
            save_ae_model(sae_model, save_dir=checkpoint_dir, model_name="sae_model_epoch_{}.pt".format(epoch))
            mu_val= np.mean(val_losses_list)
            std_val= np.std(val_losses_list)
            threshold_val=(mu_val, std_val) if val_losses_list is not None else None
            mu_val_re= np.mean(val_re_losses_list)
            std_val_re= np.std(val_re_losses_list)
            threshold_val_re=(mu_val_re, std_val_re) if val_re_losses_list is not None else None
            mu_val_latent= np.mean(val_latent_losses_list)
            std_val_latent= np.std(val_latent_losses_list)
            threshold_val_latent=(mu_val_latent, std_val_latent) if val_latent_losses_list is not None else None  
            threshold_val_latent_95=(np.percentile(val_latent_losses_list,95), std_val_latent) if val_latent_losses_list is not None else None
            save_threshold(threshold_val, save_dir=checkpoint_dir, name="sae_threshold_epoch_{}.pkl".format(epoch))
            save_threshold(threshold_val_re, save_dir=checkpoint_dir, name="sae_threshold_re_epoch_{}.pkl".format(epoch))
            save_threshold(threshold_val_latent, save_dir=checkpoint_dir, name="sae_threshold_latent_epoch_{}.pkl".format(epoch))
            save_threshold(threshold_val_latent_95, save_dir=checkpoint_dir, name="sae_threshold_latent_95_epoch_{}.pkl".format(epoch))


    
    sae_model.load_state_dict(best_state)

    

    return sae_model, threshold_val, threshold_val_re, threshold_val_latent,threshold_val_latent_95, loss_history_log

def save_ae_model(model, save_dir="results", model_name="ae_model.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    torch.save({
        "model_state_dict": model.state_dict(),
    }, save_path)
    print(f"AE model saved to {save_path}")
    
def load_sae_model(model, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Fix lỗi UnpicklingError do weights_only=True mặc định ở torch 2.6
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"SAE model loaded from {model_path}")
    return model
import pickle

def save_threshold(threshold, save_dir="results", name="threshold.pkl"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Threshold saved to {path}")
def load_threshold(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_and_evaluate(
    sae_model,
    X_test,
    y_test,
    threshold,
    save_dir="results",
    filename="test_results.csv",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=125,
    mode="latent" #re #all
):
    """
    AE-based anomaly detection evaluation
    - Predict anomaly using reconstruction error
    - Compute AUC, ACC, F1, Precision, Recall
    - Compute confusion matrix
    - Save everything to TXT
    """

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    sae_model.eval()
    sae_model.to(device)

    # -------------------------
    # Convert to tensor
    # -------------------------
    if isinstance(X_test, np.ndarray):
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    else:
        X_tensor = X_test.to(device)
    
    if isinstance(y_test, np.ndarray):
        y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    else:
        y_tensor = y_test.to(device)    
    

    # -------------------------
    # Forward AE → Reconstruction Error
    # -------------------------
    criterion = nn.MSELoss(reduction="none")
    
    test_loader = DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False
    )

    re_errors = []
    lattent_errors = []
    loss_all_errors = []
    y_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            latent, xb_hat = sae_model(xb)
            re = criterion(xb_hat, xb).mean(dim=1)  # reconstruction error per sample
            latent_loss=torch.mean(latent ** 2, dim=1)
            lattent_errors.append(latent_loss.cpu().numpy())
            loss_all= re + sae_model.lamda * latent_loss
            loss_all_errors.append(loss_all.cpu().numpy())
            re_errors.append(re.cpu().numpy())
            y_true.append(yb.cpu().numpy())
            
    re_errors = np.concatenate(re_errors)
    latent_errors = np.concatenate(lattent_errors)
    loss_all_errors = np.concatenate(loss_all_errors)
    print(f"Reconstruction errors shape: {re_errors.shape}")
    print(f"Latent errors shape: {latent_errors.shape}")
    y_true = np.concatenate(y_true)


    
    # -------------------------
    # Prediction
    # -------------------------
    if mode=="latent":
        mu, std = threshold
        threshold_test = mu + 3.0 * std
        y_pred = (latent_errors > threshold_test).astype(int)
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
            auc = roc_auc_score(y_true, latent_errors)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        mcc= matthews_corrcoef(y_true, y_pred)
        
    elif mode=="re":#re
        mu, std = threshold
        threshold_test = mu + 3.0 * std
        y_pred = (re_errors > threshold_test).astype(int)
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
            auc = roc_auc_score(y_true, re_errors)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        mcc= matthews_corrcoef(y_true, y_pred)  
    elif mode=="all":
        mu, std = threshold
        threshold_test = mu + 3.0 * std
        y_pred = (loss_all_errors > threshold_test).astype(int)
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
            auc = roc_auc_score(y_true, loss_all_errors)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        mcc= matthews_corrcoef(y_true, y_pred)  
    elif mode=="95":
        threshold_test,std = threshold
        y_pred = (latent_errors > threshold_test).astype(int)
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
            auc = roc_auc_score(y_true, latent_errors)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        mcc= matthews_corrcoef(y_true, y_pred)  

    # =========================
    # 4. SAVE TO TXT
    # =========================
    # =========================
    # 4. SAVE TO CSV
    # =========================
    import csv
    with open(save_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"SAE Anomaly Detection Results"])
        writer.writerow([])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["AUC", f"{auc:.6f}"])
        writer.writerow(["ACC", f"{acc:.6f}"])
        writer.writerow(["F1", f"{f1:.6f}"])
        writer.writerow(["Precision", f"{precision:.6f}"])
        writer.writerow(["Recall", f"{recall:.6f}"])
        writer.writerow(["MCC", f"{mcc:.6f}"])
        writer.writerow([])
        writer.writerow(["Confusion Matrix (y_true x y_pred)"])
        writer.writerow(["", "Pred 0", "Pred 1"])
        writer.writerow(["True 0", cm[0,0], cm[0,1]])
        writer.writerow(["True 1", cm[1,0], cm[1,1]])

    print(f"Evaluation results saved to {save_path}")

    return {
        "re_errors": re_errors,
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
    
def train_model(path, data_train, label_train, epochs=500, lamda=1.0, batch_size=128):
    n_features = data_train.shape[1]
    sae = SAE(n_features=n_features, lamda=lamda)
    print("Start SAE training. Lambda =", lamda)
    sae_best_model, threshold_total, threshold_re, threshold_latent, threshold_latent_95,loss_history_log= train_ae_with_early_stopping(
        data_train, label_train,
        sae_model=sae,
        batch_size=batch_size,
        lr=1e-4,
        max_epochs=epochs,
        patience=100,
        lamda=lamda,
        path=path
    )
    
    # Vẽ loss sau khi huấn luyện
    import matplotlib.pyplot as plt
    df_loss = pd.DataFrame(loss_history_log)
    plt.figure(figsize=(10,6))
    plt.plot(df_loss['epoch'], df_loss['total_loss'], label='Total Loss')
    plt.plot(df_loss['epoch'], df_loss['re_loss'], label='Reconstruction Loss')
    plt.plot(df_loss['epoch'], df_loss['latent_loss'], label='Latent Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path}/sae_train_loss.png')
    plt.close()
    print(f'Saved training loss plot to {path}/sae_train_loss.png')
    print("Finished SAE training....")
    print("Saving SAE model and thresholds re:{}, latent: {}, latent 95: {}, total: {} to {}".format(threshold_re, threshold_latent, threshold_latent_95, threshold_total, path))
    save_ae_model(sae_best_model, save_dir=path, model_name="sae_model.pt")
    save_threshold(threshold_total, save_dir=path, name="sae_threshold.pkl")
    save_threshold(threshold_re, save_dir=path, name="sae_threshold_re.pkl")
    save_threshold(threshold_latent, save_dir=path, name="sae_threshold_latent.pkl")
    save_threshold(threshold_latent_95, save_dir=path, name="sae_threshold_latent_95.pkl")
    
 
def test_model(path, sae_model, threshold, data_test, label_test, device,filename="sae_test_results.csv",mode="latent"):
    print("Testing...")
    results = predict_and_evaluate(
        sae_model=sae_model,
        X_test=data_test,
        y_test=label_test,
        threshold=threshold,
        save_dir=path,
        filename=filename,
        device=device,
        mode=mode
    )
    print(results)
    print(f"Test results saved to {path}/{filename}")
    return results
    
   

if __name__ == "__main__":
    
    
    import argparse
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="AE Classification")
    parser.add_argument('--d', type=str, default="nsl_kdd", choices=["nsl_kdd", "cic_ids", "unsw", "cic_iot"], help="Dataset to use")
    parser.add_argument('--e', type=int, default=300, help="Number of epochs")
    parser.add_argument('--b', type=int, default=32, help="Batch size")
    parser.add_argument('--m', type=str, default="train", choices=["train", "test"], help="Mode: train or test")
    parser.add_argument('--l', type=float, default=10, help="Lambda latent value")
    parser.add_argument('--per', type=float, default=1.0, help="Percentage of training data to use")
    args = parser.parse_args()
    
    dataset= args.d
    epochs=args.e
    lamda=args.l
    batch_size=args.b
    mode=args.m
    percentage=args.per
    
    path=f"results/percentage_{percentage}/sae_{dataset}_epochs{epochs}_lamda{lamda}"
    if not os.path.exists(path):
        os.makedirs(path)
    
    if dataset=="unsw":
        print(f"Using device: {device}")
        (
            _, _,
            normal_train_data_scaled, normal_train_labels,
            test_data_scaled, test_labels,
            gen_df_lamda1, gen_df_lamda4
        ) = load_data_unsw()
    elif dataset=="cic_ids":
        _,_,normal_train_data_scaled, normal_train_labels, test_data_scaled, test_labels = load_data_cic_ids()
    
    elif dataset=="nsl_kdd":
        _,_,normal_train_data_scaled, normal_train_labels, test_data_scaled, test_labels = load_data_nsl_kdd()
    elif dataset=="cic_iot":
        _,_,normal_train_data_scaled, normal_train_labels, test_data_scaled, test_labels = load_data_cic_iot()
        
    n_features = normal_train_data_scaled.shape[1]
    normal_train_data_scaled, normal_train_labels = split_by_percent_random(normal_train_data_scaled, normal_train_labels, percent=percentage, random_state=42)
    print("SAE Model Summary: dataset={}, n_features={} epochs={}  batch_size={} mode={}".format(dataset, n_features, epochs, batch_size, mode))
    
    if mode=="train":
    
        train_model(path,normal_train_data_scaled, normal_train_labels, epochs=epochs,lamda=lamda, batch_size=batch_size)

        
        sae = SAE(n_features,lamda=lamda).to(device)
        sae = load_sae_model(
            model=sae,
            model_path=f"{path}/sae_model.pt",
            device=device
        )
        threshold_latent = load_threshold(path=f"{path}/sae_threshold_latent.pkl")
        test_model(path,sae, threshold_latent, test_data_scaled, test_labels, device, filename="sae_test_latent_results.csv", mode="latent")
        threshold_re = load_threshold(path=f"{path}/sae_threshold_re.pkl")
        test_model(path,sae, threshold_re, test_data_scaled, test_labels, device, filename="sae_test_re_results.csv", mode="re")
        threshold_total = load_threshold(path=f"{path}/sae_threshold.pkl")
        test_model(path,sae, threshold_total, test_data_scaled, test_labels, device, filename="sae_test_total_results.csv", mode="all")
        threshold_latent_95 = load_threshold(path=f"{path}/sae_threshold_latent_95.pkl")
        test_model(path,sae, threshold_latent_95, test_data_scaled, test_labels, device, filename="sae_test_latent_95_results.csv", mode="95")
        
        #============================CENTROID BASED ONE CLASS CLASSIFIER==============================
        print("Testing Centroid Based One Class Classifier...")
        CEN = CentroidBasedOneClassClassifier(threshold=0.95, metric="euclidean")
        
        latent_normal_train, _ = sae(torch.tensor(normal_train_data_scaled, dtype=torch.float32).to(device))
        latent_normal_train = latent_normal_train.detach().cpu().numpy()
        latent_test, _ = sae(torch.tensor(test_data_scaled, dtype=torch.float32).to(device))     
        latent_test = latent_test.detach().cpu().numpy()
        
        # ===== Centroid on latent space =====
        CEN = CentroidBasedOneClassClassifier(threshold=0.95, metric="euclidean")
        CEN.fit(latent_normal_train)

        # continuous anomaly score
        score_cen = CEN.get_density(latent_test)

        # binary prediction (threshold-based)
        y_pred = CEN.predict(latent_test).astype(int)

        # ground truth
        y_true = (test_labels != 0).astype(int)

        # metrics
        auc = roc_auc_score(y_true, score_cen)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        print("AUC: ", auc)
        print("ACC: ", acc)
        print("F1: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Confusion Matrix: ", cm)
        print("MCC: ", mcc)
        
        with open(os.path.join(path, "cen_sae_test_results.csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"AE Anomaly Detection Results"])
            writer.writerow([])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["AUC", f"{auc:.6f}"])
            writer.writerow(["ACC", f"{acc:.6f}"])
            writer.writerow(["F1", f"{f1:.6f}"])
            writer.writerow(["Precision", f"{precision:.6f}"])
            writer.writerow(["Recall", f"{recall:.6f}"])
            writer.writerow(["MCC", f"{mcc:.6f}"])
            writer.writerow([])
            writer.writerow(["Confusion Matrix (y_true x y_pred)"])
            writer.writerow(["", "Pred 0", "Pred 1"])
            writer.writerow(["True 0", cm[0,0], cm[0,1]])
            writer.writerow(["True 1", cm[1,0], cm[1,1]])

        print(f"Centroid-Based One-Class Classifier SAE Test results saved to {path}/cen_sae_test_results.csv")
        
        #============================CENTROID BASED ONE CLASS CLASSIFIER==============================
        print("Testing SAE models at different epochs...")
        path_checkpoints=f"{path}/checkpoints"
        best_result=None
        best_epoch=None
        for i in range(epochs):
            if i % 10 == 0:
                
                model_path=f"{path_checkpoints}/sae_model_epoch_{i}.pt"
                if not os.path.exists(model_path):
                    continue
                sae = load_sae_model(
                    model=SAE(n_features,lamda=lamda).to(device),
                    model_path=f"{path_checkpoints}/sae_model_epoch_{i}.pt",
                    device=device
                )
                threshold_latent = load_threshold(path=f"{path_checkpoints}/sae_threshold_latent_epoch_{i}.pkl")
                threshold_latent_95 = load_threshold(path=f"{path_checkpoints}/sae_threshold_latent_95_epoch_{i}.pkl")
                results_latent = test_model(path_checkpoints,sae, threshold_latent, test_data_scaled, test_labels, device, filename="sae_test_latent_results_epoch_{}.csv".format(i), mode="latent")
                results_latent_95 = test_model(path_checkpoints,sae, threshold_latent_95, test_data_scaled, test_labels, device, filename="sae_test_latent_95_results_epoch_{}.csv".format(i), mode="95")
                if( best_result is None or results_latent_95["metrics"]['AUC'] > best_result["metrics"]['AUC']):
                    best_result=results_latent_95
                    best_epoch=i

        # Visualize latent space with t-SNE
        #In ra file csv best epoch
        re_errors = best_result["re_errors"]
        y_pred = best_result["y_pred"]

        metrics = best_result["metrics"]
        auc = metrics["AUC"]
        acc = metrics["ACC"]
        f1 = metrics["F1"]
        precision = metrics["Precision"]
        recall = metrics["Recall"]
        mcc = metrics["MCC"]

        cm = best_result["confusion_matrix"]

        #In ra file csv best epoch 
        re_errors = best_result["re_errors"]
        y_pred = best_result["y_pred"]

        metrics = best_result["metrics"]
        auc = metrics["AUC"]
        acc = metrics["ACC"]
        f1 = metrics["F1"]
        precision = metrics["Precision"]
        recall = metrics["Recall"]
        mcc = metrics["MCC"]

        cm = best_result["confusion_matrix"]
        best_epoch_file=f"{path_checkpoints}/best_auc_epoch_{best_epoch}.csv"
        with open(best_epoch_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["Metric", "Value"])
            writer.writerow(["AUC", f"{auc:.6f}"])
            writer.writerow(["ACC", f"{acc:.6f}"])
            writer.writerow(["F1", f"{f1:.6f}"])
            writer.writerow(["Precision", f"{precision:.6f}"])
            writer.writerow(["Recall", f"{recall:.6f}"])
            writer.writerow(["MCC", f"{mcc:.6f}"])
            writer.writerow([])
            writer.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
            writer.writerow(["True 0", cm[0,0], cm[0,1]])
            writer.writerow(["True 1", cm[1,0], cm[1,1]])
        print(f"Evaluation results saved to {best_epoch_file}")
        
    elif mode=="test":
        sae = SAE(n_features,lamda=lamda).to(device)
        sae = load_sae_model(
            model=sae,
            model_path=f"{path}/sae_model.pt",
            device=device
        )
        threshold_latent = load_threshold(path=f"{path}/sae_threshold_latent.pkl")
        test_model(path,sae, threshold_latent, test_data_scaled, test_labels, device, filename="sae_test_latent_results.csv", mode="latent")
        threshold_re = load_threshold(path=f"{path}/sae_threshold_re.pkl")
        test_model(path,sae, threshold_re, test_data_scaled, test_labels, device, filename="sae_test_re_results.csv", mode="re")
        threshold_total = load_threshold(path=f"{path}/sae_threshold.pkl")
        test_model(path,sae, threshold_total, test_data_scaled, test_labels, device, filename="sae_test_total_results.csv", mode="all") 
        
        threshold_latent_95 = load_threshold(path=f"{path}/sae_threshold_latent_95.pkl")
        test_model(path,sae, threshold_latent_95, test_data_scaled, test_labels, device, filename="sae_test_latent_95_results.csv", mode="95")
        
        #============================CENTROID BASED ONE CLASS CLASSIFIER==============================
        print("Testing Centroid Based One Class Classifier...")
        CEN = CentroidBasedOneClassClassifier(threshold=0.95, metric="euclidean")
        
        latent_normal_train, _ = sae(torch.tensor(normal_train_data_scaled, dtype=torch.float32).to(device))
        latent_normal_train = latent_normal_train.detach().cpu().numpy()
        latent_test, _ = sae(torch.tensor(test_data_scaled, dtype=torch.float32).to(device))     
        latent_test = latent_test.detach().cpu().numpy()
        
        # ===== Centroid on latent space =====
        CEN = CentroidBasedOneClassClassifier(threshold=0.95, metric="euclidean")
        CEN.fit(latent_normal_train)

        # continuous anomaly score
        score_cen = CEN.get_density(latent_test)

        # binary prediction (threshold-based)
        y_pred = CEN.predict(latent_test).astype(int)

        # ground truth
        y_true = (test_labels != 0).astype(int)

        # metrics
        auc = roc_auc_score(y_true, score_cen)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        print("AUC: ", auc)
        print("ACC: ", acc)
        print("F1: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Confusion Matrix: ", cm)
        print("MCC: ", mcc)
        
        with open(os.path.join(path, "cen_sae_test_results.csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"SAE Anomaly Detection Results"])
            writer.writerow([])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["AUC", f"{auc:.6f}"])
            writer.writerow(["ACC", f"{acc:.6f}"])
            writer.writerow(["F1", f"{f1:.6f}"])
            writer.writerow(["Precision", f"{precision:.6f}"])
            writer.writerow(["Recall", f"{recall:.6f}"])
            writer.writerow(["MCC", f"{mcc:.6f}"])
            writer.writerow([])
            writer.writerow(["Confusion Matrix (y_true x y_pred)"])
            writer.writerow(["", "Pred 0", "Pred 1"])
            writer.writerow(["True 0", cm[0,0], cm[0,1]])
            writer.writerow(["True 1", cm[1,0], cm[1,1]])

        print(f"Centroid-Based One-Class Classifier SAE Test results saved to {path}/cen_sae_test_results.csv")

         #============================CENTROID BASED ONE CLASS CLASSIFIER==============================
        print("Testing SAE models at different epochs...")
        path_checkpoints=f"{path}/checkpoints"
        best_result=None
        best_epoch=None
        for i in range(epochs):
            if i % 10 == 0:
                model_path=f"{path_checkpoints}/sae_model_epoch_{i}.pt"
                if not os.path.exists(model_path):
                    continue
                sae = load_sae_model(
                    model=SAE(n_features,lamda=lamda).to(device),
                    model_path=f"{path_checkpoints}/sae_model_epoch_{i}.pt",
                    device=device
                )
                threshold_latent = load_threshold(path=f"{path_checkpoints}/sae_threshold_latent_epoch_{i}.pkl")
                threshold_latent_95 = load_threshold(path=f"{path_checkpoints}/sae_threshold_latent_95_epoch_{i}.pkl")
                results_latent = test_model(path_checkpoints,sae, threshold_latent, test_data_scaled, test_labels, device, filename="sae_test_latent_results_epoch_{}.csv".format(i), mode="latent")
                results_latent_95 = test_model(path_checkpoints,sae, threshold_latent_95, test_data_scaled, test_labels, device, filename="sae_test_latent_95_results_epoch_{}.csv".format(i), mode="95")
                if( best_result is None or results_latent_95["metrics"]['AUC'] > best_result["metrics"]['AUC']):
                    best_result=results_latent_95
                    best_epoch=i

        # Visualize latent space with t-SNE
        #In ra file csv best epoch
        re_errors = best_result["re_errors"]
        y_pred = best_result["y_pred"]

        metrics = best_result["metrics"]
        auc = metrics["AUC"]
        acc = metrics["ACC"]
        f1 = metrics["F1"]
        precision = metrics["Precision"]
        recall = metrics["Recall"]
        mcc = metrics["MCC"]

        cm = best_result["confusion_matrix"]

        #In ra file csv best epoch 
        re_errors = best_result["re_errors"]
        y_pred = best_result["y_pred"]

        metrics = best_result["metrics"]
        auc = metrics["AUC"]
        acc = metrics["ACC"]
        f1 = metrics["F1"]
        precision = metrics["Precision"]
        recall = metrics["Recall"]
        mcc = metrics["MCC"]

        cm = best_result["confusion_matrix"]
        best_epoch_file=f"{path_checkpoints}/best_auc_epoch_{best_epoch}.csv"
        with open(best_epoch_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["Metric", "Value"])
            writer.writerow(["AUC", f"{auc:.6f}"])
            writer.writerow(["ACC", f"{acc:.6f}"])
            writer.writerow(["F1", f"{f1:.6f}"])
            writer.writerow(["Precision", f"{precision:.6f}"])
            writer.writerow(["Recall", f"{recall:.6f}"])
            writer.writerow(["MCC", f"{mcc:.6f}"])
            writer.writerow([])
            writer.writerow(["Confusion Matrix", "Pred 0", "Pred 1"])
            writer.writerow(["True 0", cm[0,0], cm[0,1]])
            writer.writerow(["True 1", cm[1,0], cm[1,1]])
        print(f"Evaluation results saved to {best_epoch_file}")
        
        # Visualize latent space with t-SNE
        """
        lattent, _ = sae(torch.tensor(test_data_scaled, dtype=torch.float32).to(device))
        lattent = lattent.detach().cpu().numpy()
        tsne_2d_2label_and_save(lattent, test_labels, f"{path}/test_sae_latent_tsne_2d.png")
        run_tsne_3d__2label_and_save(lattent, test_labels, f"{path}/test_sae_latent_tsne_3d.png")
        lattents, lables=sae.get_score_latent_distance(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(test_data_scaled, dtype=torch.float32), torch.tensor(test_labels)), batch_size=batch_size), device=device)
        visualize_1d_with_labels(lattents, lables, f"{path}/test_sae_latent_distance_tsne_2d.png")
        plot_boxplot(lattents, lables, f"{path}/test_sae_latent_distance_boxplot.png", title="SAE Latent Distance Boxplot")
        """
        
        
    