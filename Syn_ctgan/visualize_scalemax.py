"""MAE01ScaleMax paper-style visualisation (mirrors CLAN's visualize_clan_v2.py).

- Embeddings = M01AE latent z (model._mae_enc output), NOT raw features.
- PCA to 2D, centered at the normal *validation* centroid (X_val_normal.csv
  stands in for "train normal" here, since raw train_normal isn't saved to
  disk -- only X_val_normal/X_test are).
- Concentric dashed circles as reference rings, "x" marker at centroid.
- Blue = normal, Red = attack(s). No text / axes / ticks on output images.

Reads directly from res/ (best val-loss checkpoint), and reuses
the exact same per-attack label reconstruction as val_perattack_mae01scale_max.py
(imported from that module) so alignment is guaranteed consistent.

IMPORTANT: res/ currently holds THREE type_train variants per
(dataset, seed) for most datasets (noise / noise_std / std, from an earlier
ablation study) -- this script filters explicitly to the "noise_std" run
(the standard/recommended configuration used throughout this project) to
avoid mixing embeddings from different training configurations.

Output: res/figures_v2/<dataset>/seed<N>_summary.png   (normal vs ALL attacks)
        res/figures_v2/<dataset>/seed<N>_<attack>.png  (normal vs one attack)
"""

import glob
import os
import sys

import cloudpickle
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
import val_perattack_mae01scale_max as vp  # reuses ATTACK_LOADERS

RE_SCALE_MAX = BASE_DIR / "res"
OUT_DIR      = RE_SCALE_MAX / "figures_v2"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds", "cic_ids", "nb_iot"]
SEEDS    = [42, 43, 44]

MAX_NORMAL = 2000   # cap for normal samples in plot
MAX_ATTACK = 1500   # cap per attack class in plot
N_CIRCLES  = 4       # concentric reference circles
DPI        = 150
FIG_SIZE   = (5, 5)

BLUE = np.array([0.22, 0.47, 0.69, 0.65])   # normal  (RGBA)
RED  = np.array([0.75, 0.12, 0.12, 0.55])   # attack  (RGBA)


# =============================================================================
# Helpers
# =============================================================================

def find_run_dir(dataset, seed):
    """Only the noise_std variant -- see module docstring."""
    pattern = str(RE_SCALE_MAX / f"mae01scale_max_se_{seed}_t_*" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


@T.no_grad()
def get_latent_z(model, X_df, chunk=4096):
    """M01AE latent z for already-HyperTransformer-processed data (X_test.csv-style)."""
    model._mae_enc.eval()
    X_t = model._transformer.transform(X_df)
    X_tensor = T.tensor(X_t, dtype=T.float32)
    zs = []
    for i in range(0, len(X_tensor), chunk):
        xb = X_tensor[i:i + chunk].to(model._device)
        z = model._mae_enc(xb)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


def load_model(model_path, device):
    """Load a pickled MAE01ScaleMax synthesizer's inner model, remapping
    tensors to whatever device is actually available here.

    Checkpoints are pinned to the training machine's device (e.g. 'cuda').
    Loading on a machine without a working CUDA device crashes deep inside
    torch's storage deserialization unless map_location is forced --
    cloudpickle.load() doesn't expose that parameter, so torch.load is
    monkey-patched for the duration of this call.
    """
    _orig_load = T.load
    T.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": device})
    try:
        with open(model_path, "rb") as f:
            synth = cloudpickle.load(f)
    finally:
        T.load = _orig_load
    model = synth._model
    model.set_device(device)
    return model


def _subsample(idx, n, rng):
    if len(idx) <= n:
        return idx
    return rng.choice(idx, size=n, replace=False)


def _draw(emb, attack_mask, out_path):
    """emb: (N,2) PCA coords, centroid already at origin. attack_mask: (N,) bool."""
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    max_r = np.percentile(np.linalg.norm(emb, axis=1), 95)
    if max_r < 1e-6:
        max_r = 1.0
    for k in range(1, N_CIRCLES + 1):
        r = max_r * k / N_CIRCLES
        circ = plt.Circle((0, 0), r, fill=False,
                          linestyle="--", linewidth=0.8,
                          color="#888888", zorder=1)
        ax.add_patch(circ)

    att = emb[attack_mask]
    if len(att):
        ax.scatter(att[:, 0], att[:, 1],
                   color=RED, s=10, linewidths=0, zorder=2, rasterized=True)

    nor = emb[~attack_mask]
    if len(nor):
        ax.scatter(nor[:, 0], nor[:, 1],
                   color=BLUE, s=10, linewidths=0, zorder=3, rasterized=True)

    ax.scatter([0], [0], marker="x", s=60, c="black", linewidths=1.5, zorder=5)

    lim = max_r * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02,
               facecolor="white", dpi=DPI)
    plt.close(fig)


# =============================================================================
# Per (dataset, seed)
# =============================================================================

def run(dataset, seed, device):
    run_dir = find_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] no noise_std run found for {dataset} seed={seed}")
        return

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    x_val_path  = os.path.join(run_dir, f"{dataset}_X_val_normal.csv")
    model_path  = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best.pkl")
    for p in [x_test_path, y_test_path, x_val_path, model_path]:
        if not os.path.exists(p):
            print(f"  [skip] missing {p}")
            return

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    X_val_normal = pd.read_csv(x_val_path)

    attack_labels, normal_tag = vp.ATTACK_LOADERS[dataset](seed)
    if len(attack_labels) != len(X_test):
        print(f"  [ERROR] attack labels ({len(attack_labels)}) != X_test rows "
              f"({len(X_test)}) -- skip.")
        return
    y_from_attack = (attack_labels != normal_tag).astype(int).to_numpy()
    mismatch = int((y_from_attack != y_test.to_numpy()).sum())
    if mismatch > 0:
        print(f"  [ERROR] label mismatch on {mismatch} rows -- alignment broken, skip.")
        return

    print(f"  loading model...")
    model = load_model(model_path, device)

    print(f"  extracting latent z (val_normal={len(X_val_normal)}, test={len(X_test)})...")
    z_val  = get_latent_z(model, X_val_normal)
    z_test = get_latent_z(model, X_test)

    centroid = z_val.mean(axis=0, keepdims=True)
    all_emb  = np.concatenate([z_val, z_test], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(all_emb)  # components_ are shift-invariant; do NOT let PCA's own
                       # mean-centering silently cancel our centroid shift below

    def project(X):
        # Manual projection instead of pca.transform(): guarantees
        # project(centroid) == (0, 0) exactly, regardless of how skewed
        # all_emb's own mean is (test is often attack-heavy).
        return (X - centroid) @ pca.components_.T

    rng = np.random.default_rng(seed)
    attack_arr  = attack_labels.to_numpy()
    y_test_arr  = y_test.to_numpy()

    val_idx  = _subsample(np.arange(len(z_val)), MAX_NORMAL, rng)
    test_normal_idx = _subsample(np.where(y_test_arr == 0)[0], MAX_NORMAL // 2, rng)

    normal_2d = project(
        np.concatenate([z_val[val_idx], z_test[test_normal_idx]], axis=0)
    )

    attack_names = sorted(a for a in pd.unique(attack_arr) if a != normal_tag)
    out_base = OUT_DIR / dataset
    out_base.mkdir(parents=True, exist_ok=True)

    # ── summary: normal vs ALL attacks ───────────────────────────────────────
    all_att_idx = np.where(attack_arr != normal_tag)[0]
    all_att_idx = _subsample(all_att_idx, MAX_ATTACK * max(len(attack_names), 1), rng)
    att_2d = project(z_test[all_att_idx])

    emb_plot = np.concatenate([normal_2d, att_2d], axis=0)
    att_mask = np.array([False] * len(normal_2d) + [True] * len(att_2d))
    _draw(emb_plot, att_mask, out_base / f"seed{seed}_summary.png")
    print(f"  summary saved")

    # ── per-attack ────────────────────────────────────────────────────────────
    for name in attack_names:
        safe_name = str(name).replace("/", "-").replace(" ", "_")
        idx = np.where(attack_arr == name)[0]
        idx = _subsample(idx, MAX_ATTACK, rng)
        if len(idx) == 0:
            continue
        att_2d_cls = project(z_test[idx])

        emb_plot = np.concatenate([normal_2d, att_2d_cls], axis=0)
        att_mask = np.array([False] * len(normal_2d) + [True] * len(att_2d_cls))
        _draw(emb_plot, att_mask, out_base / f"seed{seed}_{safe_name}.png")

    print(f"  {len(attack_names)} per-attack images saved")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if T.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"\n[{dataset}  seed={seed}]")
            try:
                run(dataset, seed, device)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
