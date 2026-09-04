"""Centroid-to-Normal distance for ALL 4 WSN-DS attack types (Blackhole,
Grayhole, Flooding, TDMA), SSAE space only -- a quick offshoot of
visualize_centroid_distance_vs_clan_normal.py's methodology (same
z-normalized distance-to-fixed-normal-centroid metric, same per-epoch
resampling-for-comparable-noise convention), but:
  - only WSN-DS (the other 3 datasets' "hard" attacks are already covered
    by the 2-rare-attack convention elsewhere)
  - all 4 real attack types, not just the 2 rare ones (Blackhole/Grayhole)
  - real-attack lines only -- no CTGAN/noise generated sources, no CLAN

Usage: python visualize_centroid_distance_wsn_ds_4attacks.py
Output: res/figures_v2/centroid_distance_vs_clan/v4_normal/wsn_ds_seed42_4attacks.png
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import val_perattack_mae01scale_max as vp

RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "centroid_distance_vs_clan" / "v4_normal"

DATASET = "wsn_ds"
SEED = 42
TIME_TAG = "t_03"

ATTACKS = ["Blackhole", "Grayhole", "Flooding", "TDMA"]
ATTACK_COLORS = {
    "Blackhole": "#d62728",  # red
    "Grayhole":  "#e377c2",  # pink
    "Flooding":  "#1f77b4",  # blue
    "TDMA":      "#ff7f0e",  # orange
}


def find_snapshot_csv(dataset):
    pattern = str(RES_DIR / f"mae01scale_max_se_{SEED}_{TIME_TAG}" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*" / "generated_data_snapshots.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def numeric_columns(df):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c != "epoch"]


def select_informative_columns(normal_df, feat_cols, rel_threshold=1e-3, abs_std_floor=1e-2):
    X = normal_df[feat_cols].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    scale = np.maximum(np.abs(mean), 1.0)
    keep = (std >= rel_threshold * scale) & (std >= abs_std_floor)
    dropped = [c for c, k in zip(feat_cols, keep) if not k]
    if dropped:
        print(f"    [INFO] excluding {len(dropped)} near-constant column(s): {dropped}")
    kept_cols = [c for c, k in zip(feat_cols, keep) if k]
    return kept_cols, std[keep]


def centroid_distance_to_fixed(X_a, centroid_fixed, std_ref):
    c_a = X_a.mean(axis=0)
    return float(np.linalg.norm((c_a - centroid_fixed) / std_ref))


def resample_batch(pool, batch_size, rng):
    replace = len(pool) < batch_size
    idx = rng.choice(len(pool), size=batch_size, replace=replace)
    return pool[idx]


def main():
    csv_path = find_snapshot_csv(DATASET)
    if csv_path is None:
        print(f"[skip] {DATASET}: no generated_data_snapshots.csv found")
        return
    run_dir = os.path.dirname(csv_path)

    x_test_path = os.path.join(run_dir, f"{DATASET}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{DATASET}_y_test.csv")
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print(f"[skip] {DATASET}: missing X_test/y_test.csv in {run_dir}")
        return

    print(f"[{DATASET}]")
    df = pd.read_csv(csv_path)
    all_feat_cols = numeric_columns(df)
    epochs = sorted(df["epoch"].unique())

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    normal_test_df = X_test[y_test.to_numpy() == 0].reset_index(drop=True)
    all_feat_cols = [c for c in all_feat_cols if c in normal_test_df.columns]
    feat_cols, std_ref = select_informative_columns(normal_test_df, all_feat_cols)
    std_ref[std_ref < 1e-6] = 1e-6
    centroid = normal_test_df[feat_cols].mean(axis=0).to_numpy(dtype=np.float64)
    print(f"  centroid_normal from full test-normal set (n={len(normal_test_df)})")

    attack_labels, normal_tag = vp.ATTACK_LOADERS[DATASET](SEED)
    if len(attack_labels) != len(X_test):
        print(f"  [ERROR] attack labels ({len(attack_labels)}) != X_test rows "
              f"({len(X_test)}) -- skip.")
        return
    y_from_attack = (attack_labels != normal_tag).astype(int).to_numpy()
    if int((y_from_attack != y_test.to_numpy()).sum()) > 0:
        print(f"  [ERROR] label mismatch -- alignment broken, skip.")
        return

    attack_lines = []  # (name, color, dist_list)
    for name in ATTACKS:
        mask = (attack_labels == name).to_numpy()
        pool = X_test[mask][feat_cols].to_numpy(dtype=np.float64)
        if len(pool) == 0:
            print(f"  [warn] no rows for attack '{name}' -- skip")
            continue
        rng_atk = np.random.default_rng(SEED)
        dist = []
        for epoch in epochs:
            epoch_df = df[df["epoch"] == epoch]
            batch_size = len(epoch_df[epoch_df["source"] == "real_normal"])
            dist.append(centroid_distance_to_fixed(
                resample_batch(pool, batch_size, rng_atk), centroid, std_ref))
        attack_lines.append((name, ATTACK_COLORS[name], dist))
        print(f"  {name}: n={len(pool)}  mean_dist={np.mean(dist):.3f}")

    if not attack_lines:
        print("  [warn] no attack lines available -- skip image")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for name, color, dist in attack_lines:
        ax.plot(epochs, dist, marker="^", markersize=4, color=color, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Centroid distance to Normal")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = OUT_DIR / f"{DATASET}_seed{SEED}_4attacks.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
