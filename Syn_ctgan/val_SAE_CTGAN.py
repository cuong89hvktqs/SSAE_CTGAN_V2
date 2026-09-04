import sys
import os
sys.path.insert(0, r"D:\CTGAN_SUAMODEL\SDV_D1_D2_detect_lamda")  # Æ°u tiÃªn folder custom
import pandas as pd
import pickle
from sdv.single_table import SAECTGANSynthesizer
import numpy as np

def load_threshold(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def classify_test_data(model_path, x_test_path, y_test_path,k_std,threshold_path):
    # Load test data
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)['label']

    # Load model
    synthesizer = SAECTGANSynthesizer.load(model_path)

    # Load threshold
    threshold_re, threshold_z = load_threshold(threshold_path) # threshold_z

    # TÃ­nh score
    test_re_loss, test_latent_loss, test_y_true = synthesizer.get_score_sae(X_test, y_test)

    # PhÃ¢n loáº¡i
    print("Reconstruction threshold:", threshold_re)  #
    print("Latent threshold:", threshold_z)
    #print(threshold_z)  #
    mu, std = threshold_z
    print(mu, std)
    threshold_test = mu + k_std * std
    y_pred = (test_latent_loss > threshold_test).astype(int)
    y_true = (test_y_true != 0).astype(int)

    # In káº¿t quáº£
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
    
     # =========================
    # 3. METRICS
    # =========================
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc = 0000  # Default AUC for binary classification with only one class
    else:
        auc = roc_auc_score(y_true, y_pred)*100
    acc = accuracy_score(y_true, y_pred)*100
    f1 = f1_score(y_true, y_pred, zero_division=0)*100
    precision = precision_score(y_true, y_pred, zero_division=0)*100
    recall = recall_score(y_true, y_pred, zero_division=0)*100
    cm = confusion_matrix(y_true, y_pred)
    mcc=matthews_corrcoef(y_true, y_pred)*100
    print("AUC:", auc)
    print("Accuracy:", acc)
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    
    
    print("MCC:", mcc)
    print("Confusion matrix:\n", cm)
    
    
    
    # =========================
    # 4. SAVE TO CSV
    # =========================
    import csv
    save_path = f"{path}/test_std_{k_std}.csv"
    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
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
        writer.writerow(["True 0", cm[0,0], cm[0,1]])
        writer.writerow(["True 1", cm[1,0], cm[1,1]])
    
    

# VÃ­ dá»¥ sá»­ dá»¥ng:
# classify_test_data(
#     model_path="results_5/SAE_unsw_epochs_5.pkl",
#     x_test_path="results_5/unsw_X_test.csv",
#     y_test_path="results_5/unsw_y_test.csv",
#     threshold_path="results_5/unsw_sae_threshold_z_epochs_5.pkl"
# )
if __name__ == "__main__":
    dataset = "cic_ids"  # "unsw"  "cic_ids"
    path=f"results/{dataset}"
    epochs=200
    k_std_list=[2.0,3.0,4.0,5.0]
    for k_std in k_std_list:
        classify_test_data(
            model_path=f"{path}/SAE_{dataset}_epochs_{epochs}.pkl",
            x_test_path=f"{path}/{dataset}_X_test.csv",
            y_test_path=f"{path}/{dataset}_y_test.csv",
            threshold_path=f"{path}/{dataset}_sae_threshold_z_epochs_{epochs}.pkl",
            k_std=k_std
        )
    
    
