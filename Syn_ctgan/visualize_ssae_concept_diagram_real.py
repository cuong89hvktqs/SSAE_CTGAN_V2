"""SSAE concept diagram, built from REAL data/REAL trained checkpoints. NO
training happens here.

Plain scatter of four groups in M01AE latent space (get_latent_z),
PCA(2)-projected, centered at the TRUE latent origin (project(z) =
z @ components.T, no mean subtraction):
    Normal        (blue)   -- real test-normal rows (X_test.csv, y=0)
    Attack        (red)    -- real test-attack rows (X_test.csv, y=1)
    CTGAN + noise (purple) -- G: CTGAN-generated samples + noise
                               (generated_data_snapshots.csv, source=ctgan_noise)
    Real + noise  (orange) -- P: real-normal samples + noise
                               (generated_data_snapshots.csv, source=noise_pseudo)

No boundary/threshold circle is drawn (percentile circles and KDE
density-level contours were both tried and dropped per feedback -- the KDE
version in particular collapsed to a near-invisible ring at the 90%/95%
peak-density levels given how tightly the Normal cluster concentrates).

Requires a run whose generated_data_snapshots.csv exists (only saved once
the CTGAN gate opens -- see mae01scale.py::_save_generated_snapshot);
searches t_03 (main snapshot run) then t_02 (ablation runs) as a fallback,
preferring the default m=2.0/lambda=0.1 config when available.

Usage: python visualize_ssae_concept_diagram_real.py [--dataset all] [--seed 42]
Output: res/figures_v2/ssae_concept_diagram/<dataset>_seed<N>_concept_diagram.png
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from visualize_scalemax import load_model, get_latent_z

RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "ssae_concept_diagram"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
TIME_TAGS_TRY = ["t_03", "t_02", "t_04"]  # t_03 = main snapshot run, others = fallback

MAX_PER_GROUP = 2000

COLORS = {
    "normal":  "#3878B4",  # blue
    "attack":  "#BF1F1F",  # red
    "ctgan_noise":  "#8C3FBF",  # purple  (G)
    "noise_pseudo": "#E58C19",  # orange  (P)
}
LABELS = {
    "normal": "Normal", "attack": "Attack",
    "ctgan_noise": "CTGAN + noise", "noise_pseudo": "Real + noise",
}


def find_run_dir(dataset, seed):
    candidates = []
    for tag in TIME_TAGS_TRY:
        pattern = str(RES_DIR / f"mae01scale_max_se_{seed}_{tag}" / "per_1.0"
                      / f"{dataset}_*scalemax-noise_std_*")
        candidates += sorted(glob.glob(pattern))
    default = [c for c in candidates if "_m2.0_l0.1_" in c]
    ordered = default + [c for c in candidates if c not in default]
    for c in ordered:
        if os.path.exists(os.path.join(c, "generated_data_snapshots.csv")):
            return c
    return None


def load_last_epoch_sources(run_dir):
    path = os.path.join(run_dir, "generated_data_snapshots.csv")
    df = pd.read_csv(path)
    last_epoch = df["epoch"].max()
    df_last = df[df["epoch"] == last_epoch].drop(columns=["epoch"])
    sources = {src: g.drop(columns=["source"]).reset_index(drop=True)
              for src, g in df_last.groupby("source")}
    return sources, int(last_epoch)


def _cap(df, n, rng):
    if len(df) <= n:
        return df
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def robust_extent(coords_list, lo_pct=1, hi_pct=99, pad_frac=0.1):
    """Plot extent from ALL groups, percentile-based so a handful of extreme
    outliers can't blow up the axis range."""
    all_pts = np.concatenate(coords_list, axis=0)
    x_lo, x_hi = np.percentile(all_pts[:, 0], [lo_pct, hi_pct])
    y_lo, y_hi = np.percentile(all_pts[:, 1], [lo_pct, hi_pct])
    x_pad = pad_frac * (x_hi - x_lo + 1e-8)
    y_pad = pad_frac * (y_hi - y_lo + 1e-8)
    return (x_lo - x_pad, x_hi + x_pad, y_lo - y_pad, y_hi + y_pad)


def run(dataset, seed, device="cpu"):
    run_dir = find_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] {dataset} seed={seed}: no run with generated_data_snapshots.csv found")
        return False

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    model_path  = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")
    for p in [x_test_path, y_test_path, model_path]:
        if not os.path.exists(p):
            print(f"  [skip] {dataset} seed={seed}: missing {p}")
            return False

    print(f"[{dataset}] run_dir={run_dir}")
    print("  loading model (no training)...")
    model = load_model(model_path, device)

    sources, epoch = load_last_epoch_sources(run_dir)
    needed = {"ctgan_noise", "noise_pseudo"}
    missing = needed - set(sources)
    if missing:
        print(f"  [skip] generated_data_snapshots.csv missing sources: {missing}")
        return False
    print(f"  using generated-data snapshot at epoch {epoch}")

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"].to_numpy()

    rng = np.random.default_rng(seed)
    df_normal = _cap(X_test[y_test == 0].reset_index(drop=True), MAX_PER_GROUP, rng)
    df_attack = _cap(X_test[y_test == 1].reset_index(drop=True), MAX_PER_GROUP, rng)
    df_ctgan_noise  = _cap(sources["ctgan_noise"],  MAX_PER_GROUP, rng)
    df_noise_pseudo = _cap(sources["noise_pseudo"], MAX_PER_GROUP, rng)

    print("  encoding to M01AE latent space...")
    z_normal = get_latent_z(model, df_normal)
    z_attack = get_latent_z(model, df_attack)
    z_ctgan_noise = get_latent_z(model, df_ctgan_noise)
    z_noise_pseudo = get_latent_z(model, df_noise_pseudo)

    # PCA(2) fit on the pooled latent vectors, projected relative to the
    # true latent origin (no mean subtraction).
    fit_pool = np.concatenate([z_normal, z_attack, z_ctgan_noise, z_noise_pseudo], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(fit_pool)

    def project(z):
        return z @ pca.components_.T

    coords = {
        "normal": project(z_normal),
        "attack": project(z_attack),
        "ctgan_noise": project(z_ctgan_noise),
        "noise_pseudo": project(z_noise_pseudo),
    }

    extent = robust_extent(list(coords.values()))
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=150)

    # draw order: back-to-front so Normal (the reference cluster) stays visible on top
    for key in ("ctgan_noise", "noise_pseudo", "attack", "normal"):
        pts = coords[key]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=COLORS[key], alpha=0.55,
                  edgecolor="none", label=LABELS[key], zorder=2)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ssae_{dataset}_seed{seed}_concept_diagram.png"
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser("SSAE concept diagram (real data, loaded checkpoint)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"\n[{name}  seed={args.seed}]")
        try:
            run(name, args.seed)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
