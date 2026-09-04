"""Distribution (KDE) of distance-to-real-normal-centroid: pseudo-anomaly
sources (P/G) vs. real rare attacks vs. real normal, at the final training
epoch, per dataset (seed 42).

Why this replaces the epoch-line version
(visualize_distance_to_centroid_vs_attacks.py): a real attack's raw feature
values don't change across epochs, so plotting "mean distance" as a single
flat epoch-line next to P/G's per-epoch curves collapses away exactly the
information we actually want -- how WIDE each population's spread is, and
whether the generated (P/G) spread overlaps the region real attacks occupy.
A static distribution plot (this script) answers that directly: if the P/G
density has enough mass covering the same distance range as a real attack,
that's evidence the pseudo-anomaly training signal is diverse enough to
plausibly help the model separate that attack from normal.

Distance metric: same z-normalized Euclidean distance to the real-normal
centroid as visualize_distance_to_centroid.py (raw/inverse-transformed
feature space, centroid + std fixed from the first snapshot epoch's
real_normal rows) -- see that script's docstring for the full derivation.

Robustness fix vs. the earlier script: the old per-feature std floor
(1e-8) is numerically fine for avoiding division-by-zero, but it means any
feature that is *almost* constant across real-normal (std ~ 1e-6, e.g. a
near-constant flag/port column) turns even a tiny raw deviation in a real
attack sample into an astronomically large z-score, which then dominates
the L2 norm and swamps every other feature. Such near-constant columns
carry ~no information about real-normal's own spread, so instead of
dividing by a near-zero std, this script EXCLUDES any feature whose
real-normal std falls below a small relative threshold from the distance
computation entirely (see `select_informative_columns`).

Usage: python visualize_distance_distribution_vs_attacks.py
Output: res/figures_v2/distance_distribution_vs_attacks/<dataset>_seed42_distance_distribution.png
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
from scipy.stats import gaussian_kde

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

import val_perattack_mae01scale_max as vp  # reuses ATTACK_LOADERS

RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "distance_distribution_vs_attacks"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
SEED = 42
TIME_TAG = "t_03"

# Same 2 rare/hard attacks per dataset as visualize_tsne_gendata.py.
RARE_ATTACKS = {
    "unsw":    ["Shellcode", "Worms"],
    "nsl_kdd": ["attack_3", "attack_4"],
    "cic_iot": ["Recon", "Spoofing"],
    "wsn_ds":  ["Blackhole", "Grayhole"],
}
DISPLAY_NAME = {"attack_3": "R2L", "attack_4": "U2R"}  # cosmetic only

SOURCES = ("ctgan_noise", "noise_pseudo")
COLORS = {
    "test_normal":  "#1f77b4",  # blue
    "ctgan_noise":  "#8C3FBF",  # purple
    "noise_pseudo": "#E58C19",  # orange
}
LABELS = {
    "test_normal":  "Normal (test)",
    "ctgan_noise":  "CTGAN + noise (G)",
    "noise_pseudo": "Real-normal + noise (P)",
}
ATTACK_COLORS = ["#d62728", "#e377c2"]  # red, pink -- same as t-SNE figures

MAX_PER_GROUP = 4000  # cap for KDE fit / plot speed


def find_snapshot_csv(dataset):
    pattern = str(RES_DIR / f"mae01scale_max_se_{SEED}_{TIME_TAG}" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*" / "generated_data_snapshots.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def numeric_columns(df):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c != "epoch"]


def select_informative_columns(normal_df, feat_cols, rel_threshold=1e-3):
    """Drop features whose real-normal std is negligible relative to their
    own mean magnitude -- these are near-constant-for-normal columns that
    would otherwise dominate the L2 distance via a near-zero denominator
    (see module docstring)."""
    X = normal_df[feat_cols].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    scale = np.maximum(np.abs(mean), 1.0)  # avoid rel-threshold blowing up near mean=0
    keep = std >= rel_threshold * scale
    dropped = [c for c, k in zip(feat_cols, keep) if not k]
    if dropped:
        print(f"  [INFO] excluding {len(dropped)} near-constant-for-normal "
              f"column(s) from distance metric: {dropped}")
    kept_cols = [c for c, k in zip(feat_cols, keep) if k]
    return kept_cols, mean[keep], std[keep]


def compute_distances(df, feat_cols, centroid, std):
    X = df[feat_cols].to_numpy(dtype=np.float64)
    z = (X - centroid) / std
    return np.linalg.norm(z, axis=1)


def _subsample(d, n, rng):
    if len(d) <= n:
        return d
    idx = rng.choice(len(d), size=n, replace=False)
    return d[idx]


def _kde_line(d, x_grid):
    if len(d) < 2 or np.std(d) < 1e-9:
        return None
    kde = gaussian_kde(d)
    return kde(x_grid)


def coverage_pct(attack_d, ref_d, lo_pct=5, hi_pct=95):
    """% of attack distances falling inside [lo_pct, hi_pct] percentile
    range of the generated (P+G) reference distribution -- a cheap,
    interpretable "does the training signal's spread cover this attack?"
    number to accompany the KDE plot."""
    lo, hi = np.percentile(ref_d, [lo_pct, hi_pct])
    inside = (attack_d >= lo) & (attack_d <= hi)
    return 100.0 * inside.mean(), lo, hi


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

    first_epoch_normal = df[(df["epoch"] == epochs[0]) & (df["source"] == "real_normal")]
    feat_cols, centroid, std = select_informative_columns(first_epoch_normal, all_feat_cols)
    std[std < 1e-6] = 1e-6  # residual guard after the exclusion step above

    last_epoch = epochs[-1]
    last_df = df[df["epoch"] == last_epoch]

    rng = np.random.default_rng(SEED)
    dists = {}
    for src in SOURCES:
        sub = last_df[last_df["source"] == src]
        d = compute_distances(sub, feat_cols, centroid, std)
        dists[src] = _subsample(d, MAX_PER_GROUP, rng)

    # ── Real test data: normal + 2 rare attacks ────────────────────────────
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

    real_feat_cols = [c for c in feat_cols if c in X_test.columns]
    normal_mask = (y_test.to_numpy() == 0)
    d_normal = compute_distances(X_test[normal_mask].reset_index(drop=True),
                                 real_feat_cols, centroid, std)
    dists["test_normal"] = _subsample(d_normal, MAX_PER_GROUP, rng)

    attack_dists = {}
    for name in RARE_ATTACKS[dataset]:
        mask = (attack_labels == name).to_numpy()
        n = int(mask.sum())
        if n == 0:
            print(f"  [WARN] no rows for attack '{name}' in {dataset} -- skip")
            continue
        d = compute_distances(X_test[mask].reset_index(drop=True), real_feat_cols, centroid, std)
        attack_dists[name] = d

    # ── Coverage stat: does P+G spread cover each attack's distances? ─────
    ref_d = np.concatenate([dists["ctgan_noise"], dists["noise_pseudo"]])
    for name, d in attack_dists.items():
        pct, lo, hi = coverage_pct(d, ref_d)
        print(f"  coverage: {DISPLAY_NAME.get(name, name)} -- {pct:.1f}% of "
              f"attack samples fall within P+G's [P5,P95]=[{lo:.2f},{hi:.2f}] range "
              f"(attack mean={d.mean():.2f})")

    # ── Plot: overlaid KDEs ────────────────────────────────────────────────
    all_d = np.concatenate(
        [dists["test_normal"], dists["ctgan_noise"], dists["noise_pseudo"]]
        + list(attack_dists.values())
    )
    x_hi = np.percentile(all_d, 99.5)
    x_grid = np.linspace(0, max(x_hi, 1e-3), 400)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    for src in ("test_normal",) + SOURCES:
        y = _kde_line(dists[src], x_grid)
        if y is None:
            continue
        ax.plot(x_grid, y, color=COLORS[src], linewidth=2, label=LABELS[src])
        ax.fill_between(x_grid, y, color=COLORS[src], alpha=0.12)

    for i, name in enumerate(RARE_ATTACKS[dataset]):
        if name not in attack_dists:
            continue
        d = attack_dists[name]
        color = ATTACK_COLORS[i % len(ATTACK_COLORS)]
        disp = DISPLAY_NAME.get(name, name)
        y = _kde_line(d, x_grid)
        if y is not None:
            ax.plot(x_grid, y, color=color, linewidth=2, linestyle="-.",
                    label=f"{disp} (real attack, n={len(d)})")
            ax.fill_between(x_grid, y, color=color, alpha=0.08)

    ax.set_xlim(0, x_hi)
    ax.set_xlabel("Z-normalized distance to real-normal centroid")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset.upper()} (seed {SEED}, epoch {last_epoch}) -- "
                 f"distance distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_seed{SEED}_distance_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main():
    for dataset in DATASETS:
        run(dataset)


if __name__ == "__main__":
    main()
