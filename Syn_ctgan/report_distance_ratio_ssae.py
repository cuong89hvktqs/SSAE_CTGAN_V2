"""Fair, scale-free distance-ratio report for SSAE (MAE01ScaleMax) --
companion to visualize_distance_to_centroid.py and to CLAN-main's
report_distance_ratio_clan.py (the CLAN counterpart).

Raw z-scored Euclidean distance-to-real-normal-centroid is NOT directly
comparable between SSAE and CLAN: different embedding/feature spaces
(CTGAN's mode-specific DataTransformer vs CLAN's plain MinMax), different
latent dimensionality, different training objectives -- there is no shared
"unit" between the two numbers. What IS comparable is a SCALE-FREE RATIO:

    ratio = mean_distance(group) / mean_distance(real_normal to its own centroid)

A ratio of 1.0 means "this group is, on average, exactly as far from the
normal centroid as real normal data naturally is from its own centroid" (i.e.
indistinguishable spread from real-normal's intrinsic variability). Dividing
by real-normal's own spread cancels out whatever arbitrary scale this
model's transformed/latent space happens to have, so the ratio CAN be placed
side by side with CLAN's equivalent ratio (report_distance_ratio_clan.py)
for a fair cross-model comparison, even though the raw distances cannot.

Reuses visualize_distance_to_centroid.py's exact centroid/std computation
(real_normal rows from generated_data_snapshots.csv, near-constant columns
guarded the same way) but, instead of tracking the ratio over every
snapshot epoch, reports it ONCE per (dataset, seed) at the LAST recorded
epoch -- the model's converged state, consistent with how
visualize_scalemax_gendata.py picks "last epoch" for its own plots.

In addition to the 3 raw sources (ctgan_raw, ctgan_noise, noise_pseudo),
also reports "generated_anomaly" = noise_pseudo + ctgan_noise concatenated
-- the SAME merged group visualize_scalemax_gendata.py's image 4 and
visualize_tsne_gendata.py use, and the one that actually matches what
MAE01Scale trains against by default (type_train="noise_std": 50% real+noise,
50% ctgan+noise -- see ctgan/synthesizers/m01aectgan.py::_make_pseudo_anomaly).
This merged group is the fairest single counterpart to CLAN's one and only
"clan_augmented" source.

This script CANNOT run on a machine where the t_03 runs are incomplete (see
visualize_tsne_gendata.py's docstring for the same caveat) -- run it on the
server where res/mae01scale_max_se_<seed>_t_03/per_1.0/<dataset>_...
/generated_data_snapshots.csv actually exists for all 4 datasets x 3 seeds.

Usage: python report_distance_ratio_ssae.py [--seeds 42,43,44]
Output: res/ssae_distance_ratio.xlsx
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
RES_DIR = BASE_DIR / "res"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
TIME_TAG = "t_03"

RAW_SOURCES = ("ctgan_raw", "ctgan_noise", "noise_pseudo")


def find_snapshot_csv(dataset, seed):
    pattern = str(RES_DIR / f"mae01scale_max_se_{seed}_{TIME_TAG}" / "per_1.0"
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


def _last_complete_epoch(df):
    """Some runs get killed mid-write, leaving the LAST snapshot with one
    source truncated (seen on nsl_kdd seed=42 t_03: ctgan_noise had 11436
    rows instead of 53874 at epoch 65). Pick the last epoch where every
    source's row count matches real_normal's, instead of blindly using
    epochs[-1]."""
    counts = df.groupby(["epoch", "source"]).size().unstack()
    complete = counts.eq(counts["real_normal"], axis=0).all(axis=1)
    complete_epochs = counts.index[complete]
    return int(complete_epochs.max()) if len(complete_epochs) else int(df["epoch"].max())


def run(dataset, seed, rows):
    csv_path = find_snapshot_csv(dataset, seed)
    if csv_path is None:
        print(f"[skip] {dataset} seed={seed}: no generated_data_snapshots.csv found")
        return

    print(f"[{dataset} seed={seed}] loading {csv_path}")
    df = pd.read_csv(csv_path)
    feat_cols = numeric_columns(df)
    epochs = sorted(df["epoch"].unique())
    last_epoch = _last_complete_epoch(df)
    if last_epoch != epochs[-1]:
        print(f"  [note] last snapshot (epoch {epochs[-1]}) looked truncated "
              f"(mismatched row counts across sources) -- using last COMPLETE "
              f"epoch instead: {last_epoch}")
    print(f"  using epoch: {last_epoch} (of {len(epochs)} snapshots)")

    epoch0_normal = df[(df["epoch"] == epochs[0]) & (df["source"] == "real_normal")]
    centroid = epoch0_normal[feat_cols].mean(axis=0).to_numpy(dtype=np.float64)
    std = epoch0_normal[feat_cols].std(axis=0).to_numpy(dtype=np.float64)
    std[std < 1e-8] = 1e-8

    real_dist = compute_distances(epoch0_normal, feat_cols, centroid, std)
    real_mean = float(real_dist.mean())

    last_df = df[df["epoch"] == last_epoch]

    def group_stats(name, sub_df):
        d = compute_distances(sub_df, feat_cols, centroid, std)
        mean = float(d.mean())
        print(f"  {name}: distance={mean:.4f}  ratio={mean / real_mean:.3f}  n={len(sub_df)}")
        rows.append(dict(Dataset=dataset, seed=seed, Group=name,
                          Distance=mean, Real_normal_spread=real_mean,
                          Ratio=mean / real_mean, N=len(sub_df), Epoch=int(last_epoch)))

    print(f"  real_normal_spread={real_mean:.4f}")
    for src in RAW_SOURCES:
        sub = last_df[last_df["source"] == src]
        if len(sub) == 0:
            print(f"  [warn] source '{src}' missing at epoch {last_epoch} for {dataset} seed={seed}")
            continue
        group_stats(src, sub)

    # fairest single counterpart to CLAN's one "clan_augmented" source --
    # matches type_train="noise_std"'s actual pseudo-anomaly composition.
    noise_pseudo = last_df[last_df["source"] == "noise_pseudo"]
    ctgan_noise = last_df[last_df["source"] == "ctgan_noise"]
    if len(noise_pseudo) and len(ctgan_noise):
        merged = pd.concat([noise_pseudo, ctgan_noise], ignore_index=True)
        group_stats("generated_anomaly", merged)


def main():
    parser = argparse.ArgumentParser("SSAE distance-ratio report (fair cross-model metric)")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for dataset in DATASETS:
        for seed in seeds:
            run(dataset, seed, rows)

    if not rows:
        print("\nno results -- generated_data_snapshots.csv not found for any dataset/seed "
              "(run this on the machine that has the t_03 runs)")
        return

    out = pd.DataFrame(rows)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = RES_DIR / "ssae_distance_ratio.xlsx"
    out.to_excel(xlsx_path, index=False)
    print(f"\nsaved -> {xlsx_path}")


if __name__ == "__main__":
    main()
