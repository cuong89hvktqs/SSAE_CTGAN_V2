import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local custom packages

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sdv.errors import  InvalidDataTypeError
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch


def load_data_unsw():
    # Đường dẫn dữ liệu
    
    path_train = "./data/unsw/UNSW_NB15_training-set.csv"
    path_test = "./data/unsw/UNSW_NB15_testing-set.csv"
    train_data = pd.read_csv(path_train)
    test_data = pd.read_csv(path_test)

    # Nối train và test lại với nhau
    full_data = pd.concat([train_data, test_data], ignore_index=True)
    full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_data.dropna(inplace=True)
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data.dropna(inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.dropna(inplace=True)

    # Chuẩn bị dữ liệu
    df_full = full_data.copy()
    df_test = test_data.copy()

    # Tạo nhãn nhị phân an toàn
    y_full = (df_full['label'] != 0).astype(int)  # 0 = normal, 1 = anomaly
    y_test = (df_test['label'] != 0).astype(int)

    
    drop_cols = ["id", "attack_cat", "label"]
    for idx in drop_cols:
        df_full = df_full.drop([idx], axis=1)
        df_test = df_test.drop([idx], axis=1)
    
    return df_full, y_full, df_test, y_test

def load_data_cic_ids():
    fileDir = os.path.dirname(os.path.abspath("__file__"))
    pathTrain = os.path.join(fileDir, "./data/cic-ids/CIC-IDS-2017-Train-10percent.csv")
    pathTest = os.path.join(fileDir, "./data/cic-ids/CIC-IDS-2017-Test-10percent.csv")
    TrainData = pd.read_csv(pathTrain, sep=",", header=None)
    TestData = pd.read_csv(pathTest, sep=",", header=None)
    # Loại bỏ giá trị NaN / inf
    TrainData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TrainData.dropna(inplace=True)
    TestData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TestData.dropna(inplace=True)

    y_train=TrainData[85]
    y_test=TestData[85]
    
    y_train.loc[y_train != 'BENIGN'] = 1
    y_train.loc[y_train == 'BENIGN'] = 0
    y_train = y_train.astype(int)
    y_test.loc[y_test != 'BENIGN'] = 1
    y_test.loc[y_test == 'BENIGN'] = 0
    y_test = y_test.astype(int)
    
    X_full = pd.concat([TrainData, TestData], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    X_test = TestData
    droppedColumn = [0, 1, 2, 4, 7, 85]
    for col in droppedColumn:
        X_full = X_full.drop([col], axis=1)
        X_test = X_test.drop([col], axis=1)
    return X_full, y_full, X_test, y_test






def load_data_nsl_kdd():
    
    fileDir = os.path.dirname(os.path.abspath("__file__"))
    pathTrain = os.path.join(fileDir, "./data/NSLKDD/NSLKDD_Train.csv")
    pathTest = os.path.join(fileDir, "./data/NSLKDD/NSLKDD_Test.csv")

    # Đọc dữ liệu với pandas để lấy DataFrame
    TrainData = pd.read_csv(pathTrain, header=None)
    TestData = pd.read_csv(pathTest, header=None)
    # Loại bỏ giá trị NaN / inf
    TrainData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TrainData.dropna(inplace=True)
    TestData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TestData.dropna(inplace=True)

    print("train size", TrainData.shape)
    print("test size", TestData.shape)
   

    y_train = TrainData.iloc[:, -1].copy()
    y_test = TestData.iloc[:, -1].copy()

    y_train[y_train != 0] = 1
    y_test[y_test != 0] = 1
    
    

    X_train = TrainData.iloc[:, 0:-1]
    X_test = TestData.iloc[:, 0:-1]
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    normal_train_data_scaled = X_full[y_full == 0]
    normal_train_labels = y_full[y_full == 0]
    normaL_test_data_scaled = X_test[y_test == 0]
    normal_test_labels = y_test[y_test == 0]

    print(
        "Final Size: Train {},Test {}, Normal Train{} , Normal Test{}".format(
            (X_full.shape, y_full.shape),
            (X_test.shape, y_test.shape),
            (normal_train_data_scaled.shape, normal_train_labels.shape),
            (normaL_test_data_scaled.shape, normal_test_labels.shape))
    )

    return X_full, y_full, X_test, y_test

def load_data_cic_iot():

    fileDir = os.path.dirname(os.path.abspath("__file__"))
    pathTrain = os.path.join(fileDir, "./data/cic_iot/train.csv")
    pathTest  = os.path.join(fileDir, "./data/cic_iot/test.csv")

    TrainData = pd.read_csv(pathTrain, header=None).dropna()
    TestData  = pd.read_csv(pathTest,  header=None).dropna()

    
    # Labels: 0 = Benign, 1 = Attack
    y_train = (TrainData.iloc[:, -1] != "Benign").astype(int)
    y_test  = (TestData.iloc[:, -1]  != "Benign").astype(int)

    # Features
    X_train = TrainData.iloc[:, :-1]
    X_test  = TestData.iloc[:, :-1]

    # Full data (for unsupervised / visualization only)
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    normal_train_data_scaled = X_full[y_full == 0]
    normal_train_labels = y_full[y_full == 0]
    normal_test_data_scaled = X_test[y_test == 0]
    normal_test_labels = y_test[y_test == 0]
    print(
        "Final Size: Train {},Test {}, Normal Train{} , Normal Test{}".format(
            (X_full.shape, y_full.shape),
            (X_test.shape, y_test.shape),
            (normal_train_data_scaled.shape, normal_train_labels.shape),
            (normal_test_data_scaled.shape, normal_test_labels.shape))
    )
    
    return X_full, y_full, X_test, y_test


def _load_cidds_001_raw():
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "CIDDS-001",
    )
    paths = [
        os.path.join(data_dir, "cidds-001-openstack.parquet"),
        os.path.join(data_dir, "cidds-001-externalserver.parquet"),
    ]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing CIDDS-001 parquet files: {missing}")

    frames = [pd.read_parquet(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    return data.reset_index(drop=True)


def split_cidds_001(seed=42, test_size=0.4):
    """
    Preprocess CIDDS-001 and split train/test with a deterministic 6:4 ratio.

    Features exclude all target/leakage columns:
    - label
    - attack_type
    - attack_id

    Binary label convention matches the other loaders:
    - 0 = benign
    - 1 = attack
    """
    data = _load_cidds_001_raw()
    attack_labels = data["attack_type"].astype(str)
    y_binary = (attack_labels != "benign").astype(int)
    features = data.drop(columns=["label", "attack_type", "attack_id"])

    stratify_labels = attack_labels
    if stratify_labels.value_counts().min() < 2:
        stratify_labels = y_binary

    X_train, X_test, y_train, y_test, attack_train, attack_test = train_test_split(
        features,
        y_binary,
        attack_labels,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        attack_train.reset_index(drop=True),
        attack_test.reset_index(drop=True),
    )


def load_data_cidds_001(seed=42, test_size=0.4):
    X_train, X_test, y_train, y_test, _, _ = split_cidds_001(
        seed=seed,
        test_size=test_size,
    )
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    normal_train_data = X_train[y_train == 0]
    normal_test_data = X_test[y_test == 0]
    print(
        "Final Size: Train {}, Test {}, Normal Train {}, Normal Test {}".format(
            (X_train.shape, y_train.shape),
            (X_test.shape, y_test.shape),
            (normal_train_data.shape, y_train[y_train == 0].shape),
            (normal_test_data.shape, y_test[y_test == 0].shape),
        )
    )

    return X_full, y_full, X_test, y_test


def load_cidds_001_test_attacks(seed=42, test_size=0.4):
    _, X_test, _, _, _, attack_test = split_cidds_001(
        seed=seed,
        test_size=test_size,
    )
    return X_test, attack_test, "benign"


def save_per_attack_cic_ids(save_dir="./data/cic-ids/per_attack_test"):
    """
    Lưu từng loại attack trong CIC-IDS-2017 test set
    - KHÔNG tạo nhãn giả
    - Feature và label lưu thành 2 file riêng
    - Feature đã drop cột không cần thiết
    """

    fileDir = os.path.dirname(os.path.abspath("__file__"))
    pathTest = os.path.join(
        fileDir, "./data/cic-ids/CIC-IDS-2017-Test-10percent.csv"
    )

    # -------------------------
    # 1) Load test data
    # -------------------------
    TestData = pd.read_csv(pathTest, sep=",", header=None)

    TestData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TestData.dropna(inplace=True)

    # -------------------------
    # 2) Tách label gốc (attack name)
    # -------------------------
    y_test = TestData[85]        # label string: BENIGN / attack_name

    # -------------------------
    # 3) Drop cột không cần thiết (GIỐNG load_data_cic_ids)
    # -------------------------
    droppedColumn = [0, 1, 2, 4, 7, 85]
    X_test = TestData.drop(columns=droppedColumn)

    # -------------------------
    # 4) Lấy danh sách attack types (loại BENIGN)
    # -------------------------
    attack_types = sorted(y_test.unique())
    attack_types = [a for a in attack_types if a != "BENIGN"]

    os.makedirs(save_dir, exist_ok=True)

    print(f"Found {len(attack_types)} attack types")

    # -------------------------
    # 5) Lưu từng attack
    # -------------------------
    for atk in attack_types:
        idx = (y_test == atk)

        X_atk = X_test.loc[idx].reset_index(drop=True)
        y_atk = y_test.loc[idx].reset_index(drop=True)

        atk_name = atk.replace(" ", "_").replace("/", "_")

        X_file = os.path.join(save_dir, f"{atk_name}.csv")
        #y_file = os.path.join(save_dir, f"{atk_name}_y.csv")

        # lưu feature (KHÔNG label)
        X_atk.to_csv(X_file, index=False)

        # lưu label (dùng đúng lệnh bạn yêu cầu)
        #y_atk.to_frame(name="label").to_csv(y_file, index=False)

        print(f"[Saved] {atk_name}: X={X_atk.shape}, y={y_atk.shape}")

    print("Done.")
import os
import glob
import pandas as pd
import numpy as np

def load_all_attacks_from_dir(attack_dir):
    """
    Đọc toàn bộ các file attack (*.csv) trong thư mục
    - Mỗi file = 1 loại tấn công
    - Tên tấn công = tên file (không .csv)
    - Trả về cả X và y (y toàn 1)

    Trả về:
        attacks : list[dict]
        [
          {
            "attack": attack_name,
            "X": DataFrame,
            "y": Series (toàn 1)
          },
          ...
        ]
    """

    attacks = []

    csv_files = sorted(glob.glob(os.path.join(attack_dir, "*.csv")))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {attack_dir}")

    for file_path in csv_files:
        attack_name = os.path.splitext(os.path.basename(file_path))[0]

        X_attack = pd.read_csv(file_path)

        # y toàn 1 (attack)
        y_attack = pd.Series(
            np.ones(len(X_attack), dtype=int),
            name="label"
        )

        attacks.append({
            "attack": attack_name,
            "X": X_attack,
            "y": y_attack
        })

        print(f"[Loaded] {attack_name} | X={X_attack.shape}, y={y_attack.shape}")

    return attacks
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# WSN-DS
# ─────────────────────────────────────────────────────────────────────────────

def load_data_wsn_ds(seed=42, test_size=0.2):
    """Load WSN-DS dataset (Wireless Sensor Network intrusion detection).

    File  : data/wsn-ds/WSN-DS.csv
    Label : 'Attack type' — 'Normal' → 0, everything else → 1
    Drop  : ' id' (row identifier), 'Attack type' (label)
    Split : stratified 80 / 20 train-test by seed
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wsn-ds")
    path = os.path.join(data_dir, "WSN-DS.csv")

    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    y = (df["Attack type"].str.strip() != "Normal").astype(int)
    X = df.drop(columns=[" id", "Attack type"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # X_full: train first, test last (required by training script convention)
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    print(
        "WSN-DS — Train {}, Test {}, Normal Train {}, Normal Test {}".format(
            (X_train.shape, y_train.shape),
            (X_test.shape,  y_test.shape),
            (X_train[y_train == 0].shape, y_train[y_train == 0].shape),
            (X_test[y_test   == 0].shape, y_test[y_test   == 0].shape),
        )
    )
    return X_full, y_full, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# InSDN (SDN intrusion detection — Normal_data + OVS + metasploitable-2)
# ─────────────────────────────────────────────────────────────────────────────

# Columns to drop from InSDN:
#   - Metadata (non-feature identifiers)
#   - Constant columns across the full dataset (unique == 1 → zero variance)
_INSDN_DROP_COLS = [
    # metadata
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    # constant (zero-variance) across all 3 files
    "Fwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "CWE Flag Count", "ECE Flag Cnt",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Init Fwd Win Byts", "Fwd Seg Size Min",
    # label
    "Label",
]


def load_data_insdn(seed=42, test_size=0.2):
    """Load InSDN dataset (SDN-based intrusion detection).

    Files  : data/InSDN/Normal_data.csv   (68 424 normal flows)
             data/InSDN/OVS.csv           (138 722 attack flows: DoS, DDoS, Probe, BFA, Web-Attack, BOTNET)
             data/InSDN/metasploitable-2.csv (136 743 attack flows: DDoS, Probe, DoS, BFA, U2R)
    Label  : 'Label' — 'Normal' → 0, everything else → 1
    Drop   : metadata columns (Flow ID, Src/Dst IP, Timestamp) +
             12 constant columns (zero variance across full dataset)
    Split  : stratified train/test by seed  (default test_size=0.2)
    Return : X_full, y_full, X_test, y_test
             (X_full = concat[train, test] — train rows first, as required by training script)
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "InSDN")

    normal = pd.read_csv(os.path.join(data_dir, "Normal_data.csv"),      low_memory=False)
    ovs    = pd.read_csv(os.path.join(data_dir, "OVS.csv"),              low_memory=False)
    meta   = pd.read_csv(os.path.join(data_dir, "metasploitable-2.csv"), low_memory=False)

    df = pd.concat([normal, ovs, meta], ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # Binary label: Normal=0, all attacks=1
    y = (df["Label"].str.strip() != "Normal").astype(int)
    X = df.drop(columns=_INSDN_DROP_COLS, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    print(
        "InSDN — Train {}, Test {}, Normal Train {}, Normal Test {}".format(
            (X_train.shape, y_train.shape),
            (X_test.shape,  y_test.shape),
            (X_train[y_train == 0].shape, y_train[y_train == 0].shape),
            (X_test[y_test   == 0].shape, y_test[y_test   == 0].shape),
        )
    )
    return X_full, y_full, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper for NF-BoT-IoT-v3 / NF-ToN-IoT-v3 (identical schema)
# ─────────────────────────────────────────────────────────────────────────────

_NETFLOW_V3_DROP_COLS = [
    "FLOW_START_MILLISECONDS",  # absolute timestamp → not a feature
    "FLOW_END_MILLISECONDS",    # absolute timestamp → not a feature
    "IPV4_SRC_ADDR",            # raw IP string → non-numeric
    "IPV4_DST_ADDR",            # raw IP string → non-numeric
    "Label",                    # numeric label column
    "Attack",                   # string attack name
]


def _load_netflow_v3(csv_path, dataset_name, n_max_samples=500_000,
                     seed=42, test_size=0.2):
    """Shared loader for NF-BoT-IoT-v3 and NF-ToN-IoT-v3.

    Both files share the same 55-column NF v3 schema.
    Large files are down-sampled to n_max_samples rows (stratified) before splitting.

    Returns: (X_full, y_full, X_test, y_test)
    """
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    y_all = df["Label"].astype(int)
    X_all = df.drop(columns=_NETFLOW_V3_DROP_COLS, errors="ignore")

    # Stratified sub-sampling to keep label ratio
    if len(df) > n_max_samples:
        print(f"[{dataset_name}] Sampling {n_max_samples:,} from {len(df):,} rows (seed={seed})")
        _, X_all, _, y_all = train_test_split(
            X_all, y_all,
            test_size=n_max_samples,
            random_state=seed,
            stratify=y_all,
        )
        X_all = X_all.reset_index(drop=True)
        y_all = y_all.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    print(
        "{} — Train {}, Test {}, Normal Train {}, Normal Test {}".format(
            dataset_name,
            (X_train.shape, y_train.shape),
            (X_test.shape,  y_test.shape),
            (X_train[y_train == 0].shape, y_train[y_train == 0].shape),
            (X_test[y_test   == 0].shape, y_test[y_test   == 0].shape),
        )
    )
    return X_full, y_full, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# BoT-IoT
# ─────────────────────────────────────────────────────────────────────────────

def load_data_bot_iot(seed=42, n_max_samples=500_000, test_size=0.2):
    """Load NF-BoT-IoT-v3 (default 500 K rows sampled by seed).

    File  : data/bot_iot/NF-BoT-IoT-v3.csv
    Label : 'Label' — 0 = Benign, 1 = Attack
    Drop  : timestamps, raw IPs, 'Label', 'Attack'
    Split : stratified 80 / 20
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bot_iot")
    path = os.path.join(data_dir, "NF-BoT-IoT-v3.csv")
    return _load_netflow_v3(path, "BoT-IoT", n_max_samples=n_max_samples,
                            seed=seed, test_size=test_size)


# ─────────────────────────────────────────────────────────────────────────────
# ToN-IoT
# ─────────────────────────────────────────────────────────────────────────────

def load_data_ton_iot(seed=42, n_max_samples=500_000, test_size=0.4):
    """Load NF-ToN-IoT-v3 (default 500 K rows sampled by seed).

    File  : data/ton_iot/NF-ToN-IoT-v3.csv
    Label : 'Label' — 0 = Benign, 1 = Attack
    Drop  : timestamps, raw IPs, 'Label', 'Attack'
    Split : stratified 60 / 40
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ton_iot")
    path = os.path.join(data_dir, "NF-ToN-IoT-v3.csv")
    return _load_netflow_v3(path, "ToN-IoT", n_max_samples=n_max_samples,
                            seed=seed, test_size=test_size)


# ─────────────────────────────────────────────────────────────────────────────
# N-BaIoT (nb_iot) — per-device IoT botnet traffic (Gafgyt + Mirai)
# ─────────────────────────────────────────────────────────────────────────────

def load_data_nb_iot(seed=42, test_size=0.2):
    """Load N-BaIoT (nb_iot): per-device benign traffic + Gafgyt/Mirai attacks.

    Dir     : data/nb-iot/<device>/benign_traffic.csv         (label 0)
              data/nb-iot/<device>/gafgyt_attacks/*.csv       (label 1)
              data/nb-iot/<device>/mirai_attacks/*.csv        (label 1; only
                  devices that support Mirai have this folder, e.g.
                  Ennio_Doorbell has none)
    Devices : any subdirectory of data/nb-iot containing benign_traffic.csv
              is auto-discovered (currently Danmini_Doorbell, Ecobee_Thermostat,
              Ennio_Doorbell, Provision_PT_737E_Security_Camera,
              Provision_PT_838_Security_Camera -- the last currently has no
              attack files yet and contributes normal-only rows).
    Features: 115 pre-extracted statistical flow features, identical schema
              across every file -- no metadata/IP/timestamp columns to drop.
    Label   : synthesized from the source file/folder (raw CSVs carry no
              label column themselves).
    Sample  : no down-sampling -- uses the full raw data (~3.1M rows across
              ~49 files), same convention as load_data_unsw/load_data_wsn_ds.
    Split   : stratified train/test by seed (default test_size=0.2)
    Return  : X_full, y_full, X_test, y_test
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nb-iot")

    device_dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "benign_traffic.csv"))
    )
    if not device_dirs:
        raise FileNotFoundError(f"No device folders with benign_traffic.csv found under {data_dir}")

    frames = []
    for device_dir in device_dirs:
        df_benign = pd.read_csv(os.path.join(device_dir, "benign_traffic.csv"))
        df_benign["label"] = 0
        frames.append(df_benign)

        for attack_type in ("gafgyt_attacks", "mirai_attacks"):
            attack_dir = os.path.join(device_dir, attack_type)
            if not os.path.isdir(attack_dir):
                continue
            for attack_csv in sorted(glob.glob(os.path.join(attack_dir, "*.csv"))):
                df_attack = pd.read_csv(attack_csv)
                df_attack["label"] = 1
                frames.append(df_attack)

    df = pd.concat(frames, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    y_all = df["label"].astype(int)
    X_all = df.drop(columns=["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    print(
        "N-BaIoT — Train {}, Test {}, Normal Train {}, Normal Test {}".format(
            (X_train.shape, y_train.shape),
            (X_test.shape,  y_test.shape),
            (X_train[y_train == 0].shape, y_train[y_train == 0].shape),
            (X_test[y_test   == 0].shape, y_test[y_test   == 0].shape),
        )
    )
    return X_full, y_full, X_test, y_test


def load_data_ctu13_08(test_size=0.2, random_state=42):

    fileDir = os.path.dirname(os.path.abspath("__file__"))
    pathData = os.path.join(fileDir, "./data/ctu13-08/CTU13_08.csv")

    # Đọc dữ liệu
    Data = pd.read_csv(pathData, header=None)

    # Loại bỏ NaN / inf
    Data.replace([np.inf, -np.inf], np.nan, inplace=True)
    Data.dropna(inplace=True)

    print("data size", Data.shape)

    # Feature + label
    X = Data.iloc[:, 0:-1]
    y = Data.iloc[:, -1].copy()

    # Mapping label:
    # 1 -> normal -> 0
    # 2 -> attack -> 1
    y[y == 1] = 0
    y[y == 2] = 1

    y = y.astype(np.int64)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Full train giữ nguyên raw data
    X_full = X_train.reset_index(drop=True)
    y_full = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Normal samples
    normal_train_data = X_full[y_full == 0]
    normal_train_labels = y_full[y_full == 0]

    normal_test_data = X_test[y_test == 0]
    normal_test_labels = y_test[y_test == 0]

    print(
        "Final Size: Train {}, Test {}, Normal Train {}, Normal Test {}".format(
            (X_full.shape, y_full.shape),
            (X_test.shape, y_test.shape),
            (normal_train_data.shape, normal_train_labels.shape),
            (normal_test_data.shape, normal_test_labels.shape)
        )
    )

    return X_full, y_full, X_test, y_test
