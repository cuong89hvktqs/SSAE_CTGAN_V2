"""t-SNE: rare real attack types vs CTGAN-generated data (companion to
visualize_scalemax_gendata.py, same res/ layout).

Like visualize_scalemax.py/visualize_scalemax_gendata.py, this script
t-SNEs the M01AE LATENT EMBEDDING (model._mae_enc output via
get_latent_z()), not the raw preprocessed feature space -- i.e. it shows
how the trained model itself separates normal/attack/generated samples in
the space it actually reasons in.

Only the RARE/minority real attack types listed below are used per
dataset (exact label strings/codes as returned by
val_perattack_mae01scale_max.ATTACK_LOADERS), compared against test-normal
(real) and the generated pseudo-anomaly snapshot -- sources 'ctgan_noise'
(CTGAN-generated + noise) and 'noise_pseudo' (real-normal + noise) in
generated_data_snapshots.csv, last epoch recorded, MERGED into a single
"generated anomaly" class/color (both are pseudo-anomaly training signals
for M01AE, not meaningfully different for this plot's purpose).

Each real attack type gets its OWN distinct color (not lumped into one
"red" class), so rare classes stay visually distinguishable from common
ones. Because attack sample counts vary wildly (e.g. UNSW: Fuzzers=6062 vs
Worms=44), each attack type is independently subsampled to MAX_PER_ATTACK
before the shared t-SNE fit -- otherwise a large class would dominate the
subsample budget and rare classes would nearly vanish.

Run selection: hardcoded to the
  mae01scale_max_se_<seed>_t_03 / per_1.0 / <dataset>_..._noise_std_...
run (explicit choice -- NOT the glob-any-t find_run_dir() from
visualize_scalemax.py/visualize_scalemax_gendata.py, since multiple t_*
runs coexist per seed). On this dev machine t_03 is missing
X_test.csv/checkpoints for some datasets -- this script is meant to run
on the server where that run is complete; it skips (with a message)
whatever's missing rather than failing the whole batch.

For each (dataset, seed), ONE shared t-SNE fit (all groups together, so
the two images below share the same 2D coordinates/axes), producing two
images:
  seed<N>_generated.png : test-normal (real) vs generated anomaly
  seed<N>_combined.png  : test-normal (real) vs each rare real attack type
                          (own color) vs generated anomaly

Colours: normal=blue, generated anomaly=orange, the two real attack types
per dataset=red and pink (in RARE_ATTACKS list order).

Output: res/figures_v2/gendata_tsne/<dataset>/seed<N>_{generated,combined}.png
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # prefer local ctgan/ and sdv/

import val_perattack_mae01scale_max as vp        # reuses ATTACK_LOADERS
from visualize_scalemax import load_model, get_latent_z

RES_DIR = BASE_DIR / "res"
OUT_DIR = RES_DIR / "figures_v2" / "gendata_tsne"

RUN_TAG = "t_03"   # hardcoded per instruction -- not auto-selected

# Only these rare/minority real attack types per dataset. NSL-KDD's loader
# returns numeric-coded labels ("attack_<code>"); 3=R2L, 4=U2R here
# (verified against all_datasets_all_seeds_per_attack_tpr_far.csv: code 3
# has 2887 seed-42 test rows, code 4 has 67 -- matches the counts given).
RARE_ATTACKS = {
    # "unsw":    ["Fuzzers", "Reconnaissance", "Shellcode", "Worms"],
    "unsw":    ["Shellcode", "Worms"],
    "nsl_kdd": ["attack_3", "attack_4"],
    "cic_iot": ["Recon", "Spoofing"],
    "wsn_ds":  ["Blackhole", "Grayhole"],
}
DISPLAY_NAME = {"attack_3": "R2L", "attack_4": "U2R"}   # cosmetic, for print() only

SEEDS = [42, 43, 44]

MAX_PER_GROUP  = 2000   # cap for normal / merged-generated pools
MAX_PER_ATTACK = 500    # cap PER real attack type, so large classes (e.g.
                         # UNSW Fuzzers=6062) don't drown out rare ones
                         # (e.g. UNSW Worms=44) in the shared t-SNE fit
DPI            = 150
FIG_SIZE       = (7, 5.5)

BLUE    = np.array([0.22, 0.47, 0.69, 0.65])   # normal -- blue is RESERVED for this
GEN_COL = np.array([1.00, 0.50, 0.05, 0.60])   # generated anomaly (ctgan_noise + noise_pseudo) -- orange

# Exactly 2 colors, one per real attack type in RARE_ATTACKS[dataset] order
# (each dataset lists exactly 2 rare attacks). Deliberately excludes any
# blue/cyan tone (reserved for "Normal") and the orange used by GEN_COL
# (reserved for "Generated anomaly"), so no attack color is confusable
# with either.
_ATTACK_COLORS = [
    "#d62728",  # red
    "#e377c2",  # pink
]


def attack_palette(dataset):
    return {
        name: np.array([*matplotlib.colors.to_rgb(_ATTACK_COLORS[i % len(_ATTACK_COLORS)]), 0.6])
        for i, name in enumerate(RARE_ATTACKS[dataset])
    }


# =============================================================================
# Helpers
# =============================================================================

def find_t03_run_dir(dataset, seed):
    pattern = str(RES_DIR / f"mae01scale_max_se_{seed}_{RUN_TAG}" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_generated_snapshot(run_dir):
    """Return {source: DataFrame} for the LAST epoch recorded in
    generated_data_snapshots.csv, or (None, None) if missing/empty."""
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


def _subsample(idx, n, rng):
    if len(idx) <= n:
        return idx
    return rng.choice(idx, size=n, replace=False)


def _draw_groups(groups, out_path, title=None):
    """groups: list of (points_2d, rgba, label) in back-to-front draw order."""
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (pts, color, label) in enumerate(groups):
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], color=color, s=10, linewidths=0,
                      zorder=2 + i, rasterized=True, label=label)

    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8, markerscale=1.5, framealpha=0.85)
    fig.tight_layout()

    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# =============================================================================
# Per (dataset, seed)
# =============================================================================

def run(dataset, seed):
    run_dir = find_t03_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] no {RUN_TAG} noise_std run found for {dataset} seed={seed}")
        return

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    model_path  = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best.pkl")
    for p in [x_test_path, y_test_path, model_path]:
        if not os.path.exists(p):
            print(f"  [skip] missing {p}")
            return

    sources, epoch = load_generated_snapshot(run_dir)
    gen_sources = [s for s in ("ctgan_noise", "noise_pseudo") if sources and s in sources]
    if not gen_sources:
        print(f"  [skip] no ctgan_noise/noise_pseudo in generated_data_snapshots.csv")
        return
    print(f"  using {'+'.join(gen_sources)} snapshot at epoch {epoch}")

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

    wanted = RARE_ATTACKS[dataset]
    normal_mask = (y_test.to_numpy() == 0)

    rng = np.random.default_rng(seed)

    def cap(df, n):
        idx = _subsample(np.arange(len(df)), n, rng)
        return df.iloc[idx].reset_index(drop=True)

    # Each attack type capped INDEPENDENTLY -- a shared cap over the union
    # would let large classes (e.g. UNSW Fuzzers=6062) crowd out rare ones
    # (e.g. UNSW Worms=44) in the random subsample.
    attack_dfs = {}
    for name in wanted:
        mask = (attack_labels == name).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        attack_dfs[name] = cap(X_test[mask].reset_index(drop=True), MAX_PER_ATTACK)
    if not attack_dfs:
        print(f"  [skip] none of {wanted} found in test attack labels")
        return

    print("  rare attacks in test set (capped to %d each): " % MAX_PER_ATTACK + ", ".join(
        f"{DISPLAY_NAME.get(k, k)}={len(v)}" for k, v in attack_dfs.items()))

    print("  loading model (transformer only, cpu)...")
    model = load_model(model_path, "cpu")

    X_normal_df = cap(X_test[normal_mask].reset_index(drop=True), MAX_PER_GROUP)
    X_gen_df = cap(
        pd.concat([sources[s] for s in gen_sources], ignore_index=True), MAX_PER_GROUP
    )

    print(f"  test_normal={len(X_normal_df)}  generated_anomaly={len(X_gen_df)}  "
          + ", ".join(f"{DISPLAY_NAME.get(k, k)}={len(v)}" for k, v in attack_dfs.items()))

    print("  encoding to M01AE latent space...")
    z_normal = get_latent_z(model, X_normal_df)
    z_gen    = get_latent_z(model, X_gen_df)
    z_attacks = {name: get_latent_z(model, df) for name, df in attack_dfs.items()}

    attack_names = list(z_attacks.keys())
    n_total = len(z_gen) + len(z_normal) + sum(len(z) for z in z_attacks.values())
    print(f"  t-SNE (single shared fit, n={n_total})...")
    pool = np.concatenate(
        [z_gen] + [z_attacks[n] for n in attack_names] + [z_normal], axis=0
    )
    perplexity = min(30, max(5, (len(pool) - 1) // 3))
    coords = TSNE(n_components=2, random_state=seed, init="pca",
                  perplexity=perplexity).fit_transform(pool)

    i = 0
    gen_2d = coords[i:i + len(z_gen)]; i += len(z_gen)
    attacks_2d = {}
    for name in attack_names:
        n = len(z_attacks[name])
        attacks_2d[name] = coords[i:i + n]
        i += n
    normal_2d = coords[i:]

    out_base = OUT_DIR / dataset
    out_base.mkdir(parents=True, exist_ok=True)
    tag = f"seed{seed}"

    palette = attack_palette(dataset)

    _draw_groups(
        [(gen_2d, GEN_COL, "Generated anomaly"), (normal_2d, BLUE, "Normal")],
        out_base / f"{tag}_generated.png",
    )
    _draw_groups(
        [(gen_2d, GEN_COL, "Generated anomaly")]
        + [(attacks_2d[name], palette[name], DISPLAY_NAME.get(name, name))
           for name in attack_names]
        + [(normal_2d, BLUE, "Normal")],
        out_base / f"{tag}_combined.png",
    )
    print(f"  2 images saved -> {out_base}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in RARE_ATTACKS:
        for seed in SEEDS:
            print(f"\n[{dataset}  seed={seed}]")
            try:
                run(dataset, seed)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
