"""Distance-to-real-normal-centroid: pseudo-anomaly sources vs. real rare
attacks, over training epochs, per dataset (seed 42).

Extends visualize_distance_to_centroid.py: same z-normalized Euclidean
distance metric (raw feature space, fixed real-normal centroid/std from the
first snapshot epoch -- see that script's docstring for the full mechanism),
but:
  - only 2 pseudo-anomaly sources are plotted: ctgan_noise (G) and
    noise_pseudo (P) -- ctgan_raw is dropped, it isn't a training signal;
  - 2 REAL rare attack types per dataset (same set/labels as
    visualize_tsne_gendata.py's RARE_ATTACKS, matching the reference table:
    UNSW-NB15: Shellcode/Worms, NSL-KDD: R2L/U2R, CIC-IoT 2023:
    Recon/Spoofing, WSN-DS: Blackhole/Grayhole) are added as horizontal
    reference lines -- these are real X_test rows, not model output, so
    their raw feature values (and hence distance-to-centroid) do not depend
    on epoch; each is drawn as a flat mean line + shaded +/-1 std band
    across the whole epoch axis, using the same red/pink convention as the
    t-SNE figures (visualize_tsne_gendata.py) for cross-figure consistency.

Purpose: lets a reader directly compare how far the *synthetic* pseudo-
anomalies (P/G) sit from the normal centroid against how far *genuine*
attacks sit -- i.e. whether the training signal's calibration is realistic.

Usage: python visualize_distance_to_centroid_vs_attacks.py
Output: res/figures_v2/distance_to_centroid_vs_attacks/<dataset>_seed42_distance_to_centroid_vs_attacks.png
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
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

import val_perattack_mae01scale_max as vp  # reuses ATTACK_LOADERS

RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "distance_to_centroid_vs_attacks"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
SEED = 42
TIME_TAG = "t_03"

# Same 2 rare/hard attacks per dataset as visualize_tsne_gendata.py (matches
# the reference table: UNSW-NB15 Shellcode/Worms, NSL-KDD R2L/U2R,
# CIC-IoT 2023 Recon/Spoofing, WSN-DS Blackhole/Grayhole).
RARE_ATTACKS = {
    "unsw":    ["Shellcode", "Worms"],
    "nsl_kdd": ["attack_3", "attack_4"],
    "cic_iot": ["Recon", "Spoofing"],
    "wsn_ds":  ["Blackhole", "Grayhole"],
}
DISPLAY_NAME = {"attack_3": "R2L", "attack_4": "U2R"}  # cosmetic only

# Pseudo-anomaly sources (ctgan_raw dropped -- not itself a training signal)
SOURCES = ("ctgan_noise", "noise_pseudo")
COLORS = {
    "ctgan_noise":  "#8C3FBF",  # purple
    "noise_pseudo": "#E58C19",  # orange
}
LABELS = {
    "ctgan_noise":  "CTGAN + noise",
    "noise_pseudo": "Real-normal + noise",
}
# Same red/pink convention as visualize_tsne_gendata.py's attack palette.
ATTACK_COLORS = ["#d62728", "#e377c2"]  # red, pink


def find_snapshot_csv(dataset):
    pattern = str(RES_DIR / f"mae01scale_max_se_{SEED}_{TIME_TAG}" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*" / "generated_data_snapshots.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def numeric_columns(df):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c != "epoch"]


def compute_distances(df, feat_cols, centroid, std):
    X = df[feat_cols].to_numpy(dtype=np.float64)
    z = (X - centroid) / std
    return np.linalg.norm(z, axis=1)


def run(dataset):
    csv_path = find_snapshot_csv(dataset)
    if csv_path is None:
        print(f"[skip] {dataset}: no generated_data_snapshots.csv found "
              f"(run may be on a different machine)")
        return
    run_dir = os.path.dirname(csv_path)

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print(f"[skip] {dataset}: missing X_test/y_test.csv in {run_dir}")
        return

    print(f"[{dataset}] loading {csv_path}")
    df = pd.read_csv(csv_path)
    feat_cols = numeric_columns(df)
    epochs = sorted(df["epoch"].unique())

    first_epoch_normal = df[(df["epoch"] == epochs[0]) & (df["source"] == "real_normal")]
    centroid = first_epoch_normal[feat_cols].mean(axis=0).to_numpy(dtype=np.float64)
    std = first_epoch_normal[feat_cols].std(axis=0).to_numpy(dtype=np.float64)
    std[std < 1e-8] = 1e-8  # avoid division by ~0 for near-constant columns

    baseline_dist = compute_distances(first_epoch_normal, feat_cols, centroid, std)
    baseline_mean = float(baseline_dist.mean())

    # ── Pseudo-anomaly sources: per-epoch mean/std (same as base script) ──
    stats = {src: {"mean": [], "std": []} for src in SOURCES}
    for epoch in epochs:
        epoch_df = df[df["epoch"] == epoch]
        for src in SOURCES:
            sub = epoch_df[epoch_df["source"] == src]
            d = compute_distances(sub, feat_cols, centroid, std)
            stats[src]["mean"].append(d.mean())
            stats[src]["std"].append(d.std())

    # ── Real rare attacks: constant across epochs (real X_test rows) ──────
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    attack_labels, normal_tag = vp.ATTACK_LOADERS[dataset](SEED)
    if len(attack_labels) != len(X_test):
        print(f"  [ERROR] attack labels ({len(attack_labels)}) != X_test rows "
              f"({len(X_test)}) -- skip.")
        return
    y_from_attack = (attack_labels != normal_tag).astype(int).to_numpy()
    if int((y_from_attack != y_test.to_numpy()).sum()) > 0:
        print(f"  [ERROR] label mismatch -- alignment broken, skip.")
        return

    attack_feat_cols = [c for c in feat_cols if c in X_test.columns]
    attack_stats = {}
    for name in RARE_ATTACKS[dataset]:
        mask = (attack_labels == name).to_numpy()
        n = int(mask.sum())
        if n == 0:
            print(f"  [WARN] no rows found for attack '{name}' in {dataset} test set -- skip")
            continue
        d = compute_distances(X_test[mask].reset_index(drop=True), attack_feat_cols, centroid, std)
        attack_stats[name] = (float(d.mean()), float(d.std()), n)
        print(f"  attack '{DISPLAY_NAME.get(name, name)}': n={n} "
              f"mean_dist={d.mean():.3f} std={d.std():.3f}")

    # ── Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for src in SOURCES:
        mean = np.array(stats[src]["mean"])
        std_arr = np.array(stats[src]["std"])
        color = COLORS[src]
        ax.plot(epochs, mean, marker="o", markersize=4, color=color, label=LABELS[src])
        ax.fill_between(epochs, mean - std_arr, mean + std_arr, color=color, alpha=0.15)

    x_lo, x_hi = min(epochs), max(epochs)
    for i, name in enumerate(RARE_ATTACKS[dataset]):
        if name not in attack_stats:
            continue
        mean, std_a, n = attack_stats[name]
        color = ATTACK_COLORS[i % len(ATTACK_COLORS)]
        disp = DISPLAY_NAME.get(name, name)
        ax.axhline(mean, color=color, linestyle="-.", linewidth=1.5,
                  label=f"{disp} (real attack, n={n})")
        ax.axhspan(mean - std_a, mean + std_a, color=color, alpha=0.12)

    ax.axhline(baseline_mean, color="#555555", linestyle="--", linewidth=1.2,
              label="Real-normal intrinsic spread")

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean z-normalized distance to\nreal-normal centroid")
    ax.set_title(f"{dataset.upper()} (seed {SEED})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_seed{SEED}_distance_to_centroid_vs_attacks.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main():
    for dataset in DATASETS:
        run(dataset)


if __name__ == "__main__":
    main()
