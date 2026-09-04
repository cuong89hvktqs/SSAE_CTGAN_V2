import sys
import os
sys.path.insert(0, r"D:\CTGAN_SUAMODEL\SDV_D1_D2_detect_lamda")
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from ctgan.synthesizers.lcctgan import LCCTGAN
import cloudpickle

# Đường dẫn dữ liệu và model
path_test = "./data/unsw/UNSW_NB15_testing-set.csv"
results_dir = "results"
model_path = os.path.join(results_dir, "LCCTGAN_detect_lamda_1.0_lamda_detect_0.1_beta_g_0.3_noise_0.1_epochs_400.pkl")

test_data = pd.read_csv(path_test)
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.dropna(inplace=True)


drop_cols = ["id", "attack_cat", "label"]
df_test_x = test_data.copy()
for idx in drop_cols:
    if idx in df_test_x.columns:
        df_test_x = df_test_x.drop([idx], axis=1)

y_test = test_data['label'].astype(int)

# Load model
with open(model_path, "rb") as f:
    synthesizer = cloudpickle.load(f)

# Tính anomaly score
score = synthesizer._anomaly_score(df_test_x)
score = np.array(score)
y_test_bin = (y_test > 0).astype(int)

thresholds = [0.5, 0.65, 0.7, 0.8]
for threshold in thresholds:
    y_pred = (score <= threshold).astype(int)
    acc = accuracy_score(y_test_bin, y_pred)
    precision = precision_score(y_test_bin, y_pred)
    recall = recall_score(y_test_bin, y_pred)
    f1 = f1_score(y_test_bin, y_pred)
    auc = roc_auc_score(y_test_bin, score)
    cm = confusion_matrix(y_test_bin, y_pred)
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"AUC-ROC: {auc}")
    print(f"Confusion Matrix:\n{cm}")
    print("---------------------------")

    # Vẽ và lưu histogram anomaly score
    plt.figure(figsize=(8, 5))
    plt.hist(score[y_test_bin == 0], bins=50, alpha=0.6, label='Normal (y=0)')
    plt.hist(score[y_test_bin == 1], bins=50, alpha=0.6, label='Anomaly (y=1)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title(f'Histogram of Anomaly Scores\nThreshold={threshold}')
    plt.legend()
    plt.grid(True)
    hist_fig_name = os.path.join(results_dir, f'histogram_eval_thresh_{threshold}.png')
    plt.savefig(hist_fig_name)
    plt.close()

    # Vẽ boxplot anomaly score
    plt.figure(figsize=(8, 5))
    plt.boxplot([
        score[y_test_bin == 0].ravel(),
        score[y_test_bin == 1].ravel()
    ], labels=['Normal (y=0)', 'Anomaly (y=1)'])
    plt.title(f'Boxplot of Anomaly Scores\nThreshold={threshold}')
    plt.ylabel('Anomaly Score')
    plt.grid(True)
    box_fig_name = os.path.join(results_dir, f'boxplot_eval_thresh_{threshold}.png')
    plt.savefig(box_fig_name)
    plt.close()

    # Vẽ mean anomaly score
    mean_normal = np.mean(score[y_test_bin == 0])
    mean_anomaly = np.mean(score[y_test_bin == 1])
    plt.figure(figsize=(6, 4))
    plt.bar(['Normal', 'Anomaly'], [mean_normal, mean_anomaly], color=['g', 'r'])
    plt.ylabel('Mean Anomaly Score')
    plt.title(f'Mean Anomaly Score\nThreshold={threshold}')
    mean_fig_name = os.path.join(results_dir, f'mean_eval_thresh_{threshold}.png')
    plt.savefig(mean_fig_name)
    plt.close()

