"""Centroid-to-FIXED-Normal distance over training epochs: pseudo-anomaly
sources (ctgan_raw, ctgan_noise, noise_pseudo) vs. ALL real attack types
present in each dataset's test set (seed 42) -- not just the 2 rare/hard
ones used elsewhere in this project.

This is a DIFFERENT metric from visualize_distance_to_centroid.py /
visualize_distance_to_centroid_vs_attacks.py, which averaged PER-SAMPLE
distances to a single fixed real-normal centroid. It is also a variant of
this script's own earlier version, which compared each epoch's source
centroid to that SAME epoch's real_normal snapshot-batch centroid (a
moving reference). Here, the reference is fixed once and for all:

    centroid_normal = mean over the ENTIRE test-set's normal rows
                       (X_test[y_test == 0]), computed ONCE.

For every recorded epoch:
    1. A batch is taken for each generated source, sized to match exactly
       how many rows generated_data_snapshots.csv recorded for that source
       at that epoch (ctgan_raw/ctgan_noise/noise_pseudo/real_normal are
       always saved with equal row counts per epoch -- see
       MAE01Scale._save_generated_snapshot).
    2. The batch's CENTROID (feature-wise mean) is computed -- one point,
       not a per-sample distribution.
    3. distance(epoch) = || (centroid_source(epoch) - centroid_normal) / std_ref ||_2
       i.e. Euclidean distance from that batch's centroid to the FIXED
       test-normal centroid, each feature scaled by std_ref (also computed
       once from the full test-normal population, the same population
       centroid_normal comes from, so mean/std are internally consistent).
       Because it's a centroid-vs-centroid distance, there is no mean+/-std
       band to plot -- one scalar value per epoch per source.

Attacks plotted: EVERY distinct attack label present in the dataset's test
set (via ATTACK_LOADERS, same `sorted(labels.unique() - {normal_tag})` idiom
used in val_perattack_mae01scale_max.py) -- UNSW-NB15 has 9, CIC-IoT 2023
has 7, NSL-KDD and WSN-DS have 4 each. At each epoch a fresh random batch of
the SAME size as that epoch's generated batch is drawn from each attack's
test rows (with replacement if the attack has fewer rows than the batch
size) and its centroid is compared to the SAME fixed centroid_normal -- so
the attack lines carry the same batch-sampling noise character as the
generated-source lines, rather than being perfectly flat.

Usage: python visualize_centroid_distance_vs_attacks.py
Output: res/figures_v2/centroid_distance_vs_attacks/<dataset>_seed42_centroid_distance.png
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
OUT_DIR = RES_DIR / "figures_v2" / "centroid_distance_vs_attacks"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
SEED = 42
TIME_TAG = "t_03"

# NSL-KDD's loader returns numeric-coded labels ("attack_<code>"); standard
# NSL-KDD taxonomy: 1=DoS, 2=Probe, 3=R2L, 4=U2R. Cosmetic display only.
DISPLAY_NAME = {"attack_1": "DoS", "attack_2": "Probe", "attack_3": "R2L", "attack_4": "U2R"}

SOURCES = ("ctgan_raw", "ctgan_noise", "noise_pseudo")
COLORS = {
    "ctgan_raw":    "#33A54C",  # green
    "ctgan_noise":  "#8C3FBF",  # purple
    "noise_pseudo": "#E58C19",  # orange
}
LABELS = {
    "ctgan_raw":    "CTGAN raw",
    "ctgan_noise":  "CTGAN + noise",
    "noise_pseudo": "Real-normal + noise",
}
# Up to 9 distinct attack types (UNSW-NB15) -- avoids green/purple/orange
# (already used for the 3 pseudo-anomaly sources above) and blue/cyan (the
# "Normal" convention used elsewhere in this project).
ATTACK_COLORS = [
    "#d62728",  # red
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#7f7f7f",  # gray
    "#bcbd22",  # olive/gold
    "#000000",  # black
    "#800000",  # maroon
    "#fa8072",  # salmon
    "#4d4d4d",  # dark gray
]


def find_snapshot_csv(dataset):
    pattern = str(RES_DIR / f"mae01scale_max_se_{SEED}_{TIME_TAG}" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*" / "generated_data_snapshots.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def numeric_columns(df):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c != "epoch"]


def select_informative_columns(normal_df, feat_cols, rel_threshold=1e-3, abs_std_floor=1e-2):
    """Drop features whose real-normal std is negligible -- either relative
    to their own mean magnitude, OR in absolute terms -- these are
    near-constant-for-normal columns that would otherwise dominate the
    z-normalized distance via a near-zero denominator (a real attack sample
    differing even slightly on such a column would otherwise blow up to an
    astronomical z-score and swamp every other feature -- see
    visualize_distance_distribution_vs_attacks.py for the same fix applied
    to the per-sample distance metric).

    The relative check alone (std < rel_threshold * |mean|) misses columns
    whose mean is itself near 0 (e.g. a rare-event indicator that's ~0 for
    normal traffic, like WSN-DS's JOIN_R: mean=0.0003, std=0.0027 -- both
    tiny, so their ratio looks "fine" relatively, but the absolute std is
    still small enough to blow up when an attack's mean sits far away in
    absolute terms). The absolute floor catches that case directly.
    """
    X = normal_df[feat_cols].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    scale = np.maximum(np.abs(mean), 1.0)
    keep = (std >= rel_threshold * scale) & (std >= abs_std_floor)
    dropped = [c for c, k in zip(feat_cols, keep) if not k]
    if dropped:
        print(f"  [INFO] excluding {len(dropped)} near-constant-for-normal "
              f"column(s) from distance metric: {dropped}")
    kept_cols = [c for c, k in zip(feat_cols, keep) if k]
    return kept_cols, std[keep]


def centroid_distance_to_fixed(df_a, centroid_fixed, feat_cols, std_ref):
    c_a = df_a[feat_cols].mean(axis=0).to_numpy(dtype=np.float64)
    return float(np.linalg.norm((c_a - centroid_fixed) / std_ref))


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
    all_feat_cols = numeric_columns(df)
    epochs = sorted(df["epoch"].unique())

    # ── Fixed reference: centroid + std from the ENTIRE test-normal set ────
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    normal_test_df = X_test[y_test.to_numpy() == 0].reset_index(drop=True)
    all_feat_cols = [c for c in all_feat_cols if c in normal_test_df.columns]
    feat_cols, std_ref = select_informative_columns(normal_test_df, all_feat_cols)
    std_ref[std_ref < 1e-6] = 1e-6  # residual guard after the exclusion step above
    centroid_normal = normal_test_df[feat_cols].mean(axis=0).to_numpy(dtype=np.float64)
    print(f"  centroid_normal computed from full test-normal set (n={len(normal_test_df)})")

    # ── Real attacks: ALL attack types present in the test set ────────────
    attack_labels, normal_tag = vp.ATTACK_LOADERS[dataset](SEED)
    if len(attack_labels) != len(X_test):
        print(f"  [ERROR] attack labels ({len(attack_labels)}) != X_test rows "
              f"({len(X_test)}) -- skip.")
        return
    y_from_attack = (attack_labels != normal_tag).astype(int).to_numpy()
    if int((y_from_attack != y_test.to_numpy()).sum()) > 0:
        print(f"  [ERROR] label mismatch -- alignment broken, skip.")
        return

    all_attacks = sorted(a for a in attack_labels.unique() if a != normal_tag)
    print(f"  {len(all_attacks)} attack types: {all_attacks}")

    attack_pools = {}
    for name in all_attacks:
        mask = (attack_labels == name).to_numpy()
        n = int(mask.sum())
        if n == 0:
            print(f"  [WARN] no rows for attack '{name}' in {dataset} -- skip")
            continue
        attack_pools[name] = X_test[mask].reset_index(drop=True)
        print(f"  attack pool '{DISPLAY_NAME.get(name, name)}': n={n}")

    rng = np.random.default_rng(SEED)

    # ── Per-epoch centroid distances ────────────────────────────────────
    stats = {src: [] for src in SOURCES}
    attack_dist = {name: [] for name in attack_pools}
    for epoch in epochs:
        epoch_df = df[df["epoch"] == epoch]
        batch_size = len(epoch_df[epoch_df["source"] == "real_normal"])

        for src in SOURCES:
            src_batch = epoch_df[epoch_df["source"] == src]
            stats[src].append(
                centroid_distance_to_fixed(src_batch, centroid_normal, feat_cols, std_ref)
            )

        for name, pool in attack_pools.items():
            replace = len(pool) < batch_size
            idx = rng.choice(len(pool), size=batch_size, replace=replace)
            attack_batch = pool.iloc[idx].reset_index(drop=True)
            attack_dist[name].append(
                centroid_distance_to_fixed(attack_batch, centroid_normal, feat_cols, std_ref)
            )

    # ── Plot ────────────────────────────────────────────────────────────
    # Wider figure + legend outside the axes: with up to 9 attack types plus
    # 3 generated sources, an inside legend would cover half the plot.
    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=150)

    for src in SOURCES:
        ax.plot(epochs, stats[src], marker="o", markersize=4,
               color=COLORS[src], label=LABELS[src])

    for i, name in enumerate(attack_pools):
        color = ATTACK_COLORS[i % len(ATTACK_COLORS)]
        disp = DISPLAY_NAME.get(name, name)
        ax.plot(epochs, attack_dist[name], marker="^", markersize=4,
               linestyle="-.", color=color, label=f"{disp} (real attack)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Centroid distance to Normal\n(z-normalized)")
    ax.set_title(f"{dataset.upper()} (seed {SEED})")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_seed{SEED}_centroid_distance.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main():
    for dataset in DATASETS:
        run(dataset)


if __name__ == "__main__":
    main()
