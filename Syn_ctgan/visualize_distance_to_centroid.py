"""Distance-to-real-normal-centroid over training epochs, per dataset (seed 42).

For each dataset, reads generated_data_snapshots.csv (written every
SNAPSHOT_EVERY epochs by MAE01Scale._save_generated_snapshot(), once the
CTGAN gate has opened -- see mae01scale.py) and tracks how close each
pseudo-anomaly source -- ctgan_raw, ctgan_noise, and noise_pseudo (real
normal rows + Gaussian noise, no CTGAN involved) -- gets to the real-normal
distribution as training progresses:

  1. Centroid + per-feature std computed ONCE from the real_normal rows at
     the first recorded epoch (real_normal is a fixed reference set, not
     model-dependent -- it's the same underlying normal training data every
     snapshot, just re-saved alongside the generated data).
  2. Every feature is z-scored using that centroid/std (avoids a few
     high-variance raw columns dominating the Euclidean distance). Only
     numeric feature columns are used; categorical (string) columns left
     un-encoded by inverse_transform are excluded.
  3. For each snapshot epoch, mean +/- std of the per-sample Euclidean
     distance to the centroid is computed for all three sources.
  4. A horizontal reference line shows real_normal's own average distance
     to its centroid -- the "floor" generated data should approach. This is
     NOT zero: real data has natural spread, so a generated distance
     collapsing to near-zero is a mode-collapse warning sign, not a sign of
     quality -- generated distance should approach the reference line, not
     necessarily undercut it.

Datasets/paths are machine-independent (glob-based), so any dataset whose
generated_data_snapshots.csv isn't present on this machine is skipped with
a message instead of erroring.

Usage: python visualize_distance_to_centroid.py
Output: res/figures_v2/distance_to_centroid/<dataset>_seed42_distance_to_centroid.png
"""

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "distance_to_centroid"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
SEED = 42
TIME_TAG = "t_03"

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

    sources = ("ctgan_raw", "ctgan_noise", "noise_pseudo")
    stats = {src: {"mean": [], "std": []} for src in sources}
    for epoch in epochs:
        epoch_df = df[df["epoch"] == epoch]
        for src in sources:
            sub = epoch_df[epoch_df["source"] == src]
            d = compute_distances(sub, feat_cols, centroid, std)
            stats[src]["mean"].append(d.mean())
            stats[src]["std"].append(d.std())

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for src in sources:
        mean = np.array(stats[src]["mean"])
        std_arr = np.array(stats[src]["std"])
        color = COLORS[src]
        ax.plot(epochs, mean, marker="o", markersize=4, color=color, label=LABELS[src])
        ax.fill_between(epochs, mean - std_arr, mean + std_arr, color=color, alpha=0.15)

    ax.axhline(baseline_mean, color="#555555", linestyle="--", linewidth=1.2,
              label="Real-normal intrinsic spread")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean z-normalized distance to\nreal-normal centroid")
    ax.set_title(f"{dataset.upper()} (seed {SEED})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_seed{SEED}_distance_to_centroid.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main():
    for dataset in DATASETS:
        run(dataset)


if __name__ == "__main__":
    main()
