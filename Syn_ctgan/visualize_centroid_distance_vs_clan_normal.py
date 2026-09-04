"""Centroid-to-Normal distance vs. CLAN's generated data -- the "mean_normal"
figure (res/figures_v2/centroid_distance_vs_clan/mean_normal/), rewritten
from scratch as its own script (visualize_centroid_distance_vs_clan.py has
since been repointed at attack centroids, not Normal). Same structure as
that file's original version, extends visualize_centroid_distance_vs_attacks.py
by adding CLAN's own generated (pseudo-anomaly) data as a 4th source, and
restricting the real-attack reference to the 2 "paper" rare/hard attacks per
dataset (RARE_ATTACKS: UNSW-NB15 Shellcode/Worms, NSL-KDD R2L/U2R, CIC-IoT
2023 Recon/Spoofing, WSN-DS Blackhole/Grayhole), BOTH ATTACKS TOGETHER IN
ONE IMAGE PER DATASET (not one image per attack).

Differences from the original mean_normal version: (1) the real-attack
reference is SSAE space ONLY now -- the "(real, CLAN space)" line has been
dropped; (2) both of a dataset's 2 rare attacks are drawn as separate lines
(red = 1st attack, pink = 2nd) in the SAME figure, so each image has 6 lines
total (3 SSAE generated sources + clan_augmented + 2 real-attack lines), and
there is exactly 1 output image per dataset (4 total), not 2 (8 total).

CLAN and SSAE use fundamentally different feature preprocessing: SSAE runs
its data through a HyperTransformer on the raw features, while CLAN uses
its own MinMaxScaler via data.load_data.get_data() -- with a slightly
different feature set (CLAN drops the proto/service categoricals SSAE
keeps, and keeps ct_state_ttl, which SSAE's export doesn't have). There is
therefore no single shared centroid both systems' distances can be measured
against fairly. Instead, exactly like this project's existing
independent-per-system convention (see
CLAN-main/visualize_distance_to_centroid_clan.py), TWO parallel
z-normalized distance-to-real-normal-centroid metrics are computed, each
entirely within its own system's native space:

  - SSAE space: centroid/std from SSAE's own X_test normal rows (same
    fixed-centroid convention as visualize_centroid_distance_vs_attacks.py).
    Sources: ctgan_raw, ctgan_noise, noise_pseudo, taken directly from that
    epoch's batch in generated_data_snapshots.csv; the real-attack
    reference line is a fresh random batch (same size as that epoch's
    generated batch) drawn from SSAE's own X_test attack rows of the given
    type.
  - CLAN space: centroid/std from CLAN's own reloaded test-normal rows, via
    data.load_data.get_data() with the exact same config
    CLAN-main/export_augmented_data.py and
    visualize_distance_to_centroid_clan.py use. The clan_augmented source
    comes from CLAN-main/results/generated/<dataset>_seed_<seed>_augmented.csv
    -- CLAN has no per-epoch generator (uniform_resample is a stateless
    perturbation, see that file's own docstring), so a fresh random batch
    (matching that epoch's batch size) is drawn from the single static
    clan_augmented pool at every epoch tick, purely so the line carries the
    same batch-sampling-noise character as the other sources.

Usage: python visualize_centroid_distance_vs_clan_normal.py
Output: res/figures_v2/centroid_distance_vs_clan/v4_normal/<dataset>_seed42_centroid_distance_vs_clan.png
"""

import glob
import importlib.util
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
OUT_DIR = RES_DIR / "figures_v2" / "centroid_distance_vs_clan" / "v4_normal"

CLAN_ROOT = Path(r"D:\Code_papper\CTGAN\CLGAN\CLAN-main\CLAN-main")
CLAN_GENERATED_DIR = CLAN_ROOT / "results" / "generated"

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
SEED = 42
TIME_TAG = "t_03"

# (display name, SSAE-side label, CLAN-side label) -- same pair per dataset
# used throughout this project (visualize_distance_to_centroid_vs_attacks.py
# / CLAN-main/visualize_distance_to_centroid_clan.py).
ATTACK_PAIRS = {
    "unsw":    [("Shellcode", "Shellcode", "shellcode"), ("Worms", "Worms", "worms")],
    "nsl_kdd": [("R2L", "attack_3", "3"), ("U2R", "attack_4", "4")],
    "cic_iot": [("Recon", "Recon", "recon"), ("Spoofing", "Spoofing", "spoofing")],
    "wsn_ds":  [("Blackhole", "Blackhole", "blackhole"), ("Grayhole", "Grayhole", "grayhole")],
}

# CLAN's own dataset reload config (must match CLAN-main/export_augmented_data.py
# and CLAN-main/visualize_distance_to_centroid_clan.py exactly, so the
# reloaded test split lines up with what results/generated/*.csv was built from).
CLAN_DATASETS = {
    "unsw": dict(
        data_path=CLAN_ROOT / "data/unsw/UNSW_NB15_training-set.csv",
        target="attack_cat", class_zero="Normal", drop_cols=["id"], total_rows=175341,
    ),
    "nsl_kdd": dict(
        data_path=CLAN_ROOT / "data/NSLKDD/NSLKDD_Train.csv",
        target="label", class_zero="0", drop_cols=[], total_rows=125973,
    ),
    "cic_iot": dict(
        data_path=CLAN_ROOT / "data/cic_iot/train.csv",
        target="label", class_zero="Benign", drop_cols=[], total_rows=418644,
    ),
    "wsn_ds": dict(
        data_path=CLAN_ROOT / "data/wsn-ds/WSN-DS.csv",
        target="Attack type", class_zero="Normal", drop_cols=["id", "Time"], total_rows=374661,
    ),
}
CLAN_SAMPLE_THRES = 100

SOURCES = ("ctgan_raw", "ctgan_noise", "noise_pseudo")
COLORS = {
    "ctgan_raw":    "#33A54C",  # green
    "ctgan_noise":  "#8C3FBF",  # purple
    "noise_pseudo": "#E58C19",  # orange
    "clan_augmented": "#17A398",  # teal
}
LABELS = {
    "ctgan_raw":    "CTGAN raw",
    "ctgan_noise":  "CTGAN + noise",
    "noise_pseudo": "Real-normal + noise",
    "clan_augmented": "CLAN augmented",
}
ATTACK_SSAE_COLORS = ["#d62728", "#e377c2"]  # red (1st attack), pink (2nd attack)


def _load_clan_get_data():
    """Import CLAN-main/data/load_data.py as a standalone module (by file
    path, not by package import) -- Syn_ctgan/data/ has no __init__.py, so
    it is an implicit Python 3 namespace package, and `import data...` here
    would risk silently resolving to the wrong "data" (this project's raw
    CSV folder) instead of CLAN's own data/load_data.py."""
    spec = importlib.util.spec_from_file_location("clan_load_data", CLAN_ROOT / "data" / "load_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_data


# =============================================================================
# Shared helpers
# =============================================================================

def find_snapshot_csv(dataset):
    pattern = str(RES_DIR / f"mae01scale_max_se_{SEED}_{TIME_TAG}" / "per_1.0"
                  / f"{dataset}_*scalemax-noise_std_*" / "generated_data_snapshots.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def numeric_columns(df):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c != "epoch"]


def select_informative_columns(normal_df, feat_cols, rel_threshold=1e-3, abs_std_floor=1e-2):
    """See visualize_centroid_distance_vs_attacks.py for the full rationale:
    excludes features whose real-normal std is negligible either relative to
    the feature's own mean magnitude, or in absolute terms, since such
    columns blow up the z-normalized distance via a near-zero denominator."""
    X = normal_df[feat_cols].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    scale = np.maximum(np.abs(mean), 1.0)
    keep = (std >= rel_threshold * scale) & (std >= abs_std_floor)
    dropped = [c for c, k in zip(feat_cols, keep) if not k]
    if dropped:
        print(f"    [INFO] SSAE space: excluding {len(dropped)} near-constant column(s): {dropped}")
    kept_cols = [c for c, k in zip(feat_cols, keep) if k]
    return kept_cols, std[keep]


def select_informative_columns_np(X_normal, rel_threshold=1e-3, abs_std_floor=1e-2):
    """Same exclusion rule as select_informative_columns(), for a plain
    numpy feature matrix (CLAN side has no column names, only positional
    feature_names from data.load_data.get_data())."""
    mean = X_normal.mean(axis=0)
    std = X_normal.std(axis=0)
    scale = np.maximum(np.abs(mean), 1.0)
    keep = (std >= rel_threshold * scale) & (std >= abs_std_floor)
    return np.where(keep)[0], std[keep]


def centroid_distance_to_fixed(X_a, centroid_fixed, std_ref):
    c_a = X_a.mean(axis=0)
    return float(np.linalg.norm((c_a - centroid_fixed) / std_ref))


def resample_batch(pool, batch_size, rng):
    replace = len(pool) < batch_size
    idx = rng.choice(len(pool), size=batch_size, replace=replace)
    return pool[idx]


# =============================================================================
# Per-dataset run
# =============================================================================

def run(dataset, get_data):
    csv_path = find_snapshot_csv(dataset)
    if csv_path is None:
        print(f"[skip] {dataset}: no generated_data_snapshots.csv found")
        return
    run_dir = os.path.dirname(csv_path)

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    y_test_path = os.path.join(run_dir, f"{dataset}_y_test.csv")
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print(f"[skip] {dataset}: missing X_test/y_test.csv in {run_dir}")
        return

    clan_csv_path = CLAN_GENERATED_DIR / f"{dataset}_seed_{SEED}_augmented.csv"
    if not clan_csv_path.exists():
        print(f"[skip] {dataset}: missing {clan_csv_path}")
        return

    print(f"[{dataset}]")
    print(f"  SSAE snapshot: {csv_path}")
    df = pd.read_csv(csv_path)
    all_feat_cols = numeric_columns(df)
    epochs = sorted(df["epoch"].unique())

    # ---- SSAE space: fixed centroid/std from the full test-normal set ----
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]
    normal_test_df = X_test[y_test.to_numpy() == 0].reset_index(drop=True)
    all_feat_cols = [c for c in all_feat_cols if c in normal_test_df.columns]
    feat_cols, std_ref = select_informative_columns(normal_test_df, all_feat_cols)
    std_ref[std_ref < 1e-6] = 1e-6
    centroid_ssae = normal_test_df[feat_cols].mean(axis=0).to_numpy(dtype=np.float64)
    print(f"  SSAE centroid_normal from full test-normal set (n={len(normal_test_df)})")

    attack_labels_ssae, normal_tag = vp.ATTACK_LOADERS[dataset](SEED)
    if len(attack_labels_ssae) != len(X_test):
        print(f"  [ERROR] SSAE attack labels ({len(attack_labels_ssae)}) != X_test rows "
              f"({len(X_test)}) -- skip.")
        return
    y_from_attack = (attack_labels_ssae != normal_tag).astype(int).to_numpy()
    if int((y_from_attack != y_test.to_numpy()).sum()) > 0:
        print(f"  [ERROR] SSAE label mismatch -- alignment broken, skip.")
        return

    # ---- CLAN space: reload CLAN's own test split, same config as
    # export_augmented_data.py / visualize_distance_to_centroid_clan.py ----
    cfg = CLAN_DATASETS[dataset]
    x_train_c, y_train_c, x_val_c, y_val_c, x_te_c, y_te_c, x_zd_c, y_zd_c, label_mapping, feature_names_c = get_data(
        data_path=str(cfg["data_path"]), target=cfg["target"], drop=cfg["drop_cols"],
        class_zero=cfg["class_zero"], sample_thres=CLAN_SAMPLE_THRES, split_seed=SEED,
        test_ratio=0.5, val_ratio=0.2, anomaly_detection=True, n_samples=cfg["total_rows"],
        sample_seed=SEED, return_label_mapping=True, return_feature_names=True,
    )
    if x_zd_c is not None:
        x_te_c = np.concatenate([x_te_c, x_zd_c])
        y_te_c = np.concatenate([y_te_c, y_zd_c])
    label_names_c = {v: k for k, v in label_mapping.items()}
    attack_labels_clan = pd.Series([label_names_c[y] for y in y_te_c])

    normal_mask_c = (y_te_c == 0)
    feat_idx_c, std_ref_c = select_informative_columns_np(x_te_c[normal_mask_c])
    std_ref_c[std_ref_c < 1e-6] = 1e-6
    centroid_clan = x_te_c[normal_mask_c][:, feat_idx_c].mean(axis=0)
    print(f"  CLAN centroid_normal from reloaded test-normal set (n={int(normal_mask_c.sum())})")

    # Select by position, not by name: feature_names_c may be plain integer
    # indices (e.g. one-hot-expanded categorical columns for nsl_kdd), which
    # become string column headers ("0", "1", ...) once round-tripped
    # through CSV -- but column ORDER is guaranteed to match, since
    # export_augmented_data.py wrote the CSV with these same feature_names.
    clan_df = pd.read_csv(clan_csv_path)
    clan_gen_pool = (clan_df[clan_df["source"] == "clan_augmented"]
                      .drop(columns=["source"]).to_numpy(dtype=np.float64))
    print(f"  CLAN generated pool: n={len(clan_gen_pool)}")

    # ---- Per-epoch centroid distances (SSAE sources use that epoch's real
    # batch; CLAN/attack sources are freshly resampled per tick, matching
    # that epoch's batch size, purely for comparable sampling-noise look) ----
    rng = np.random.default_rng(SEED)
    stats = {src: [] for src in SOURCES}
    clan_dist = []
    for epoch in epochs:
        epoch_df = df[df["epoch"] == epoch]
        batch_size = len(epoch_df[epoch_df["source"] == "real_normal"])

        for src in SOURCES:
            src_batch = epoch_df[epoch_df["source"] == src][feat_cols].to_numpy(dtype=np.float64)
            stats[src].append(centroid_distance_to_fixed(src_batch, centroid_ssae, std_ref))

        clan_batch = resample_batch(clan_gen_pool, batch_size, rng)[:, feat_idx_c]
        clan_dist.append(centroid_distance_to_fixed(clan_batch, centroid_clan, std_ref_c))

    # ---- Both attacks together in ONE image per dataset ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attack_lines = []  # (disp_name, color, dist_list)
    for i, (disp_name, ssae_label, clan_label) in enumerate(ATTACK_PAIRS[dataset]):
        mask_ssae = (attack_labels_ssae == ssae_label).to_numpy()
        pool_ssae = X_test[mask_ssae][feat_cols].to_numpy(dtype=np.float64)
        if len(pool_ssae) == 0:
            print(f"  [warn] {dataset}: no SSAE-space rows for attack '{ssae_label}' -- skip")
            continue

        rng_atk = np.random.default_rng(SEED)
        atk_dist_ssae = []
        for epoch in epochs:
            epoch_df = df[df["epoch"] == epoch]
            batch_size = len(epoch_df[epoch_df["source"] == "real_normal"])
            atk_dist_ssae.append(centroid_distance_to_fixed(
                resample_batch(pool_ssae, batch_size, rng_atk), centroid_ssae, std_ref))
        attack_lines.append((disp_name, ATTACK_SSAE_COLORS[i % len(ATTACK_SSAE_COLORS)], atk_dist_ssae))

    if not attack_lines:
        print(f"  [warn] {dataset}: no attack lines available -- skip image")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for src in SOURCES:
        ax.plot(epochs, stats[src], marker="o", markersize=4, color=COLORS[src], label=LABELS[src])
    ax.plot(epochs, clan_dist, marker="s", markersize=4,
           color=COLORS["clan_augmented"], label=LABELS["clan_augmented"])
    for disp_name, color, atk_dist_ssae in attack_lines:
        ax.plot(epochs, atk_dist_ssae, marker="^", markersize=4, linestyle="-.",
               color=color, label=f"{disp_name} (real)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Centroid distance to Normal")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = OUT_DIR / f"{dataset}_seed{SEED}_centroid_distance_vs_clan.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main():
    get_data = _load_clan_get_data()
    for dataset in DATASETS:
        run(dataset, get_data)


if __name__ == "__main__":
    main()
