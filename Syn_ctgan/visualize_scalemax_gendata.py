"""MAE01ScaleMax generated-data visualisation (companion to visualize_scalemax.py).

Compares real test data against the periodic generated-data snapshots
written by MAE01Scale._save_generated_snapshot() every SNAPSHOT_EVERY
epochs, once the CTGAN gate has opened (see mae01scale.py):
    real_normal   -- real training-normal rows, as-is
    noise_pseudo  -- real_normal + noise            (the "noise" pseudo-anomaly)
    ctgan_raw     -- fresh CTGAN-generated rows, unperturbed
    ctgan_noise   -- CTGAN-generated rows + noise   (the "std" pseudo-anomaly)

For each (dataset, seed), uses the LAST epoch recorded in that run's
generated_data_snapshots.csv. All generated rows are already saved in the
same HyperTransformer-space format as X_test.csv, so get_latent_z() (from
visualize_scalemax.py) encodes them unchanged, alongside real test data.

Polar-ized the same way as CLAN's visualize_clan_gen.py, for a fair,
score-faithful comparison between the two models' figures:
    angle  = direction of the M01AE latent vector z in the shared PCA(2)
             plane (PCA fit once per run, centered at the real-normal
             centroid, so all images use the same axes)
    radius = the model's OWN real anomaly score, s(x) = RE + lambda*||z||^2
             (the "total95" score -- see get_score(), the exact function
             every reported result is computed from), NOT the raw PCA
             magnitude.
This differs from CLAN's recipe in one deliberate way: CLAN L2-normalizes
its embeddings before PCA because its embedding space is trained purely by
cosine similarity (magnitude carries no information there), whereas
MAE01ScaleMax's z is trained by a Euclidean margin loss where ||z|| IS
directly meaningful (normal -> ||z||->0, pseudo-anomaly -> ||z||->margin).
So z is PCA'd in its native (un-normalized) space to pick the 2 most
informative directions -- but exactly like CLAN, only the resulting ANGLE
is kept from that projection; the actual radius plotted always comes from
the real score formula, never from the PCA magnitude itself.

Images produced per (dataset, seed), all in a single flat output folder:
  ssae_<dataset>_seed<N>_img1_test_normal_vs_ctgan_raw.png
      test-normal vs ctgan_raw                                  (2 groups)
  ssae_<dataset>_seed<N>_img2_test_normal_vs_noise_pseudo_vs_ctgan_noise.png
      test-normal vs noise_pseudo vs ctgan_noise                 (3 groups)
  ssae_<dataset>_seed<N>_img3_real_normal_vs_noise_pseudo_vs_ctgan_noise.png
      real_normal (train) vs noise_pseudo vs ctgan_noise         (3 groups)
  ssae_<dataset>_seed<N>_img4_test_normal_vs_test_attack_vs_generated_anomaly.png
      test-normal vs test-attack (real) vs
      generated-anomaly (noise_pseudo + ctgan_noise merged)      (3 groups)

Output: res/figures_v2/gendata/ssae_<dataset>_seed<N>_imgN_....png
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as T
from pathlib import Path
from sklearn.decomposition import PCA

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
import val_perattack_mae01scale_max as vp        # reuses ATTACK_LOADERS
from visualize_scalemax import get_latent_z, load_model, _subsample

RE_SCALE_MAX = BASE_DIR / "res"
OUT_DIR      = RE_SCALE_MAX / "figures_v2" / "gendata"


def find_run_dir(dataset, seed):
    """Like visualize_scalemax.find_run_dir, but skips candidates that lack
    <dataset>_X_test.csv. Some t_02 directories hold leftover m/lambda
    ablation-sweep runs (see aggregate_ablation_m_lambda.py) whose folder
    names still glob-match "*scalemax-noise_std_*" but were never carried
    through to the final X_test/y_test export -- picking the alphabetically
    first match (plain find_run_dir's behaviour) silently grabs one of those
    instead of the canonical m2.0_l0.1 run under t_03/t_04."""
    pattern = str(RE_SCALE_MAX / f"mae01scale_max_se_{seed}_t_03" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*")
    for candidate in sorted(glob.glob(pattern)):
        if os.path.exists(os.path.join(candidate, f"{dataset}_X_test.csv")):
            return candidate
    return None

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
SEEDS    = [42]

MAX_PER_GROUP = 2000   # cap per group in plots (and for the shared PCA fit)
N_CIRCLES     = 4
DPI           = 150
FIG_SIZE      = (5, 5)

# One consistent color per role across all 4 images.
COLORS = {
    "test_normal":       np.array([0.22, 0.47, 0.69, 0.65]),  # blue
    "real_normal":       np.array([0.13, 0.55, 0.55, 0.65]),  # teal
    "ctgan_raw":         np.array([0.20, 0.65, 0.30, 0.55]),  # green
    "noise_pseudo":      np.array([0.90, 0.55, 0.10, 0.55]),  # orange
    "ctgan_noise":       np.array([0.55, 0.25, 0.75, 0.55]),  # purple
    "test_attack":       np.array([0.75, 0.12, 0.12, 0.55]),  # red
}


# =============================================================================
# Drawing
# =============================================================================

def _draw_multi(groups, out_path):
    """groups: list of (points_2d, color) tuples, drawn in the given order
    (first = background, last = foreground). Centroid already at origin."""
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_pts = np.concatenate([pts for pts, _ in groups if len(pts)], axis=0)
    max_r = np.percentile(np.linalg.norm(all_pts, axis=1), 95) if len(all_pts) else 1.0
    if max_r < 1e-6:
        max_r = 1.0
    for k in range(1, N_CIRCLES + 1):
        r = max_r * k / N_CIRCLES
        circ = plt.Circle((0, 0), r, fill=False,
                          linestyle="--", linewidth=0.8,
                          color="#888888", zorder=1)
        ax.add_patch(circ)

    for i, (pts, color) in enumerate(groups):
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1],
                      color=color, s=10, linewidths=0, zorder=2 + i, rasterized=True)

    lim = max_r * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02,
               facecolor="white", dpi=DPI)
    plt.close(fig)


# =============================================================================
# Per (dataset, seed)
# =============================================================================

def load_generated_snapshot(run_dir):
    """Return {source: DataFrame} for the LAST epoch recorded in
    generated_data_snapshots.csv, or None if the file doesn't exist yet
    (e.g. the run predates the snapshot feature, or the gate never opened)."""
    path = os.path.join(run_dir, "generated_data_snapshots.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if df.empty:
        return None, None
    last_epoch = df["epoch"].max()
    df_last = df[df["epoch"] == last_epoch].drop(columns=["epoch"])
    return {src: g.drop(columns=["source"]).reset_index(drop=True)
           for src, g in df_last.groupby("source")}, int(last_epoch)


def get_total_score(model, df):
    """Real anomaly score s(x) = RE + lambda*||z||^2 for a raw-feature-space
    DataFrame (same format as X_test.csv -- true of every group here, since
    generated_data_snapshots.csv rows are saved via transformer.inverse_transform).
    The label argument is irrelevant to the score itself (only used by
    get_score to also return y_true unchanged), so a dummy all-zero vector
    is passed."""
    y_dummy = np.zeros(len(df))
    _, _, total_scores, _ = model.get_score(df, y_dummy)
    return np.asarray(total_scores, dtype=np.float64)


def run(dataset, seed, device):
    run_dir = find_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] no noise_std run found for {dataset} seed={seed}")
        return

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    model_path  = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best.pkl")
    for p in [x_test_path, y_test_path, model_path]:
        if not os.path.exists(p):
            print(f"  [skip] missing {p}")
            return

    sources, epoch = load_generated_snapshot(run_dir)
    if sources is None:
        print(f"  [skip] no generated_data_snapshots.csv (gate never opened, "
              f"or run predates the snapshot feature)")
        return
    needed = {"real_normal", "noise_pseudo", "ctgan_raw", "ctgan_noise"}
    missing = needed - set(sources)
    if missing:
        print(f"  [skip] generated_data_snapshots.csv missing sources: {missing}")
        return
    print(f"  using generated-data snapshot at epoch {epoch}")

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    attack_labels, normal_tag = vp.ATTACK_LOADERS[dataset](seed)
    if len(attack_labels) != len(X_test):
        print(f"  [ERROR] attack labels ({len(attack_labels)}) != X_test rows "
              f"({len(X_test)}) -- skip.")
        return
    y_from_attack = (attack_labels != normal_tag).astype(int).to_numpy()
    if int((y_from_attack != y_test.to_numpy()).sum()) > 0:
        print(f"  [ERROR] label mismatch -- alignment broken, skip.")
        return

    print(f"  loading model...")
    model = load_model(model_path, device)

    df_test_normal = X_test[y_test.to_numpy() == 0].reset_index(drop=True)
    df_test_attack = X_test[y_test.to_numpy() == 1].reset_index(drop=True)

    print(f"  extracting latent z (test={len(X_test)}, "
          f"generated sources: {', '.join(f'{k}={len(v)}' for k, v in sources.items())})...")
    z_test_normal = get_latent_z(model, df_test_normal)
    z_test_attack = get_latent_z(model, df_test_attack)
    z_real_normal  = get_latent_z(model, sources["real_normal"])
    z_noise_pseudo = get_latent_z(model, sources["noise_pseudo"])
    z_ctgan_raw    = get_latent_z(model, sources["ctgan_raw"])
    z_ctgan_noise  = get_latent_z(model, sources["ctgan_noise"])

    print(f"  scoring s(x) = RE + lambda*||z||^2 (get_score) for each group...")
    s_test_normal  = get_total_score(model, df_test_normal)
    s_test_attack  = get_total_score(model, df_test_attack)
    s_real_normal  = get_total_score(model, sources["real_normal"])
    s_noise_pseudo = get_total_score(model, sources["noise_pseudo"])
    s_ctgan_raw    = get_total_score(model, sources["ctgan_raw"])
    s_ctgan_noise  = get_total_score(model, sources["ctgan_noise"])

    # Centroid from real (train) normal -- consistent with visualize_scalemax.py's
    # use of normal embeddings as the geometric reference point.
    centroid = z_real_normal.mean(axis=0, keepdims=True)

    # One shared PCA fit (balanced, capped subsample per group) so all 4
    # images below share the same 2D axes and are directly comparable.
    # Fit on CENTERED z (matches CLAN's pca.fit(all_norm - centroid_norm)):
    # the direction PCA picks should describe how points vary AROUND the
    # normal centroid, not around the raw latent-space origin.
    rng = np.random.default_rng(seed)

    def cap_idx(n, k=MAX_PER_GROUP):
        return _subsample(np.arange(n), k, rng)

    fit_pool = np.concatenate([
        z_test_normal[cap_idx(len(z_test_normal))] - centroid,
        z_test_attack[cap_idx(len(z_test_attack))] - centroid,
        z_real_normal[cap_idx(len(z_real_normal))] - centroid,
        z_noise_pseudo[cap_idx(len(z_noise_pseudo))] - centroid,
        z_ctgan_raw[cap_idx(len(z_ctgan_raw))] - centroid,
        z_ctgan_noise[cap_idx(len(z_ctgan_noise))] - centroid,
    ], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(fit_pool)

    def to_polar(z, scores, n=MAX_PER_GROUP):
        """angle from the shared PCA direction, radius = the REAL score
        s(x) -- same polar-ization recipe as CLAN's visualize_clan_gen.py."""
        idx = _subsample(np.arange(len(z)), n, rng)
        dir2d = (z[idx] - centroid) @ pca.components_.T
        mag = np.linalg.norm(dir2d, axis=1, keepdims=True) + 1e-8
        unit = dir2d / mag
        radius = scores[idx].reshape(-1, 1)
        return unit * radius

    out_base = OUT_DIR
    out_base.mkdir(parents=True, exist_ok=True)
    tag = f"ssae_{dataset}_seed{seed}"

    # ── Image 1: test-normal vs ctgan_raw ────────────────────────────────────
    _draw_multi(
        [
            (to_polar(z_ctgan_raw, s_ctgan_raw),      COLORS["ctgan_raw"]),
            (to_polar(z_test_normal, s_test_normal),  COLORS["test_normal"]),
        ],
        out_base / f"{tag}_img1_test_normal_vs_ctgan_raw.png",
    )

    # ── Image 2: test-normal vs noise_pseudo vs ctgan_noise ──────────────────
    _draw_multi(
        [
            (to_polar(z_ctgan_noise, s_ctgan_noise),   COLORS["ctgan_noise"]),
            (to_polar(z_noise_pseudo, s_noise_pseudo), COLORS["noise_pseudo"]),
            (to_polar(z_test_normal, s_test_normal),   COLORS["test_normal"]),
        ],
        out_base / f"{tag}_img2_test_normal_vs_noise_pseudo_vs_ctgan_noise.png",
    )

    # ── Image 3: real_normal (train) vs noise_pseudo vs ctgan_noise ──────────
    _draw_multi(
        [
            (to_polar(z_ctgan_noise, s_ctgan_noise),   COLORS["ctgan_noise"]),
            (to_polar(z_noise_pseudo, s_noise_pseudo), COLORS["noise_pseudo"]),
            (to_polar(z_real_normal, s_real_normal),   COLORS["real_normal"]),
        ],
        out_base / f"{tag}_img3_real_normal_vs_noise_pseudo_vs_ctgan_noise.png",
    )

    # ── Image 4: test-normal vs test-attack(real) vs noise_pseudo vs ctgan_noise ──
    # Kept as 2 separate colors (orange/purple) rather than merged into one
    # "generated_anomaly" group, so the two pseudo-anomaly sources stay
    # visually distinguishable even alongside real attack data.
    _draw_multi(
        [
            (to_polar(z_ctgan_noise, s_ctgan_noise),   COLORS["ctgan_noise"]),
            (to_polar(z_noise_pseudo, s_noise_pseudo), COLORS["noise_pseudo"]),
            (to_polar(z_test_attack, s_test_attack),   COLORS["test_attack"]),
            (to_polar(z_test_normal, s_test_normal),   COLORS["test_normal"]),
        ],
        out_base / f"{tag}_img4_test_normal_vs_test_attack_vs_generated_anomaly.png",
    )

    print(f"  4 images saved -> {out_base}")


def main():
    import argparse
    parser = argparse.ArgumentParser("MAE01ScaleMax generated-data visualisation")
    parser.add_argument("--seed", type=int, default=None,
                       help="run a single seed only (default: all of SEEDS)")
    args = parser.parse_args()
    seeds = [args.seed] if args.seed is not None else SEEDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if T.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for dataset in DATASETS:
        for seed in seeds:
            print(f"\n[{dataset}  seed={seed}]")
            try:
                run(dataset, seed, device)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
