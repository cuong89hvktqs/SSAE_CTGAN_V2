"""Does SSAE's generated pseudo-anomaly data resemble/correlate with real
attack data? (reviewer concern)

Everything is written under res/compare_ge_attack/.

Data source: res/t_03_SSAE_se_42/ (the only t_03 run available on this
machine -- seed 42 only, 4 core datasets). The per-epoch
generated_data_snapshots.csv that the OLDER visualize_tsne_gendata.py /
visualize_centroid_distance_vs_attacks.py / visualize_distance_distribution_
vs_attacks.py scripts expect is NOT present here (checked: absent from
every t_03 run dir) and those scripts also target the pre-reorganization
folder layout (mae01scale_max_se_<seed>_t_03/.../checkpoints/MAE_*_best.pkl)
which no longer exists -- so they cannot be run as-is on this machine.

Instead, fresh pseudo-anomaly / CTGAN samples are regenerated directly from
the final best_auc checkpoint, using the EXACT same calls the model itself
used during training (model._generate_ctgan_fakeact, model._add_noise_
transformed -- see ctgan/synthesizers/m01aectgan.py::_make_pseudo_anomaly):
    ctgan_raw     = fresh CTGAN sample, unperturbed
    ctgan_noise   = ctgan_raw + noise                 (the "std" pseudo-anomaly)
    noise_pseudo  = real val-normal rows + noise      (the "noise" pseudo-anomaly)
pseudo_anomaly (= 50/50 ctgan_noise + noise_pseudo) is the actual signal
M01AE is trained against under type_train="noise_std".

Part (a) -- reuses this project's existing visual/quantitative conventions
(adapted to the data available here, single final checkpoint, no per-epoch
curve):
  - t-SNE of M01AE latent z: real test-normal vs the dataset's 2 rare real
    attack types vs generated pseudo-anomaly (style of
    visualize_tsne_gendata.py)
  - Centroid-distance-to-real-normal bar chart: ctgan_raw / ctgan_noise /
    noise_pseudo vs EVERY real attack type in the test set (style of
    visualize_centroid_distance_vs_attacks.py, collapsed from an epoch
    curve to a single bar per source since only the final checkpoint is
    available)
  - KDE of per-sample distance-to-real-normal-centroid: generated sources
    vs the 2 rare real attacks vs real normal itself (style of
    visualize_distance_distribution_vs_attacks.py)

Part (b) -- new metrics, not previously in this repo:
  - Correlation-matrix similarity: Pearson correlation-of-correlations
    (vectorized upper triangle) and Frobenius distance between the
    feature-correlation matrix of generated data and of each real attack
    type / real normal.
  - Classifier Two-Sample Test (C2ST): 5-fold CV ROC-AUC of a
    RandomForest trying to tell (generated vs that real attack) apart, and
    (generated vs real normal) apart. AUC near 50% = statistically
    indistinguishable (strong evidence of similarity); AUC near 100% =
    easily told apart.
  Both computed for 2 "generated" definitions: ctgan_raw (pure synthesis)
  and pseudo_anomaly (the actual training signal).

All numeric work happens in the SDV HyperTransformer-processed space (same
convention as this project's X_test.csv -- see get_latent_z() in
visualize_scalemax.py), with the 1-2 remaining object/categorical columns
(e.g. unsw's proto/service) one-hot encoded using categories pooled across
every group so columns line up.

Usage:
    python compare_gen_vs_attack.py
    python compare_gen_vs_attack.py --datasets unsw,nsl_kdd
"""
import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from backfill_ssae_old_seeds import load_synthesizer
from reeval_ssae_runs import rebuild_val_test_split
from val_perattack_mae01scale_max import ATTACK_LOADERS

RES_DIR = os.path.join(SCRIPT_DIR, "res")
OUT_DIR = os.path.join(RES_DIR, "compare_ge_attack")
RUN_TAG = "t_03"
SEED = 42

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]

RARE_ATTACKS = {
    "unsw":    ["Shellcode", "Worms"],
    "nsl_kdd": ["attack_3", "attack_4"],
    "cic_iot": ["Recon", "Spoofing"],
    "wsn_ds":  ["Blackhole", "Grayhole"],
}
DISPLAY_NAME = {"attack_3": "R2L", "attack_4": "U2R"}

MAX_N = 2000           # cap per group everywhere (tSNE / corr / C2ST / plots)
RNG_SEED = 42
BLUE    = np.array([0.22, 0.47, 0.69, 0.65])
GEN_COL = np.array([1.00, 0.50, 0.05, 0.60])
ATTACK_COLORS = ["#d62728", "#e377c2"]


# =============================================================================
# Data loading / fresh sample generation
# =============================================================================

def find_run_dir(dataset, seed):
    pattern = os.path.join(RES_DIR, f"{RUN_TAG}_SSAE_se_{seed}", "per_1.0",
                            f"{dataset}_150e_m2.0_l0.1_renormal_noise_std_*_catra0.1")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def cap_df(df, n, rng):
    if len(df) <= n:
        return df.reset_index(drop=True)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


@torch.no_grad()
def encode_processed(model, df):
    """Processed-space DataFrame -> M01AE latent z (model._mae_enc output)."""
    model._mae_enc.eval()
    X_t = model._transformer.transform(df)
    X_tensor = torch.tensor(X_t, dtype=torch.float32).to(model._device)
    return model._mae_enc(X_tensor).cpu().numpy()


@torch.no_grad()
def generate_fresh_samples(model, val_normal_processed_df, n):
    """Fresh ctgan_raw / ctgan_noise / noise_pseudo samples straight from the
    trained model, mirroring _make_pseudo_anomaly exactly. Returns
    (dict[name, DataFrame in processed space], dict[name, latent z array])."""
    model._generator.eval()
    model._mae_enc.eval()

    ctgan_raw_t = model._generate_ctgan_fakeact(n)
    ctgan_noise_t = model._add_noise_transformed(ctgan_raw_t)

    real_t = torch.tensor(
        model._transformer.transform(val_normal_processed_df), dtype=torch.float32
    ).to(model._device)
    noise_pseudo_t = model._add_noise_transformed(real_t)

    tensors = {"ctgan_raw": ctgan_raw_t, "ctgan_noise": ctgan_noise_t, "noise_pseudo": noise_pseudo_t}
    dfs = {name: model._transformer.inverse_transform(t.cpu().numpy()) for name, t in tensors.items()}
    zs = {name: model._mae_enc(t).cpu().numpy() for name, t in tensors.items()}
    return dfs, zs


def prepare_dataset(dataset, seed=SEED):
    """Loads checkpoint + real test data, generates fresh synthetic samples.
    Returns a dict with everything downstream steps need, or None if the
    t_03 run/checkpoint is missing for this dataset."""
    run_dir = find_run_dir(dataset, seed)
    if run_dir is None:
        print(f"  [skip] no {RUN_TAG} run found for {dataset}")
        return None
    model_path = os.path.join(run_dir, f"MAE_{dataset}_best_auc.pkl")
    if not os.path.isfile(model_path):
        print(f"  [skip] missing checkpoint {model_path}")
        return None

    rng = np.random.default_rng(RNG_SEED)

    X_val_raw, y_val, X_test_raw, y_test = rebuild_val_test_split(dataset, seed, 1.0)
    synth = load_synthesizer(model_path)
    model = synth._model

    X_val_processed = synth._data_processor.transform(X_val_raw)
    X_test_processed = synth._data_processor.transform(X_test_raw)

    attack_labels, normal_tag = ATTACK_LOADERS[dataset](seed)
    if len(attack_labels) != len(X_test_processed):
        print(f"  [ERROR] attack labels ({len(attack_labels)}) != X_test rows "
              f"({len(X_test_processed)}) -- skip.")
        return None

    normal_mask = (attack_labels == normal_tag).to_numpy()
    real_normal_df = cap_df(X_test_processed[normal_mask].reset_index(drop=True), MAX_N, rng)

    attack_names = sorted(a for a in attack_labels.unique() if a != normal_tag)
    real_attacks = {}
    for name in attack_names:
        mask = (attack_labels == name).to_numpy()
        df = X_test_processed[mask].reset_index(drop=True)
        if len(df) == 0:
            continue
        real_attacks[name] = cap_df(df, MAX_N, rng)

    n_gen = min(MAX_N, len(X_val_processed))
    val_normal_sub = cap_df(X_val_processed, n_gen, rng)
    gen_dfs, gen_zs = generate_fresh_samples(model, val_normal_sub, n_gen)

    print(f"  real_normal={len(real_normal_df)}  attacks={{{', '.join(f'{k}:{len(v)}' for k, v in real_attacks.items())}}}  "
          f"generated(each)={n_gen}  model.noise_std={model.noise_std:.4f}  type_train={model.type_train}")

    z_normal = encode_processed(model, real_normal_df)
    z_attacks = {name: encode_processed(model, df) for name, df in real_attacks.items()}

    return {
        "dataset": dataset, "model": model,
        "real_normal_df": real_normal_df, "real_attacks": real_attacks,
        "gen_dfs": gen_dfs, "z_normal": z_normal, "z_attacks": z_attacks, "gen_zs": gen_zs,
    }


def save_data_csvs(bundle, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    bundle["real_normal_df"].to_csv(os.path.join(data_dir, "real_normal.csv"), index=False)
    for name, df in bundle["real_attacks"].items():
        safe = str(name).replace("/", "-").replace(" ", "_")
        df.to_csv(os.path.join(data_dir, f"real_attack_{safe}.csv"), index=False)
    for name, df in bundle["gen_dfs"].items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)


# =============================================================================
# Shared numeric-matrix helper (one-hot the few leftover object columns)
# =============================================================================

def build_numeric_frames(dfs):
    """dfs: dict[name, DataFrame], same columns. One-hot encodes object
    columns using categories pooled across ALL groups so columns line up.
    Returns dict[name, DataFrame] (all-numeric, identical columns)."""
    keys = list(dfs.keys())
    sizes = [len(dfs[k]) for k in keys]
    all_df = pd.concat([dfs[k] for k in keys], ignore_index=True)

    obj_cols = all_df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in all_df.columns if c not in obj_cols]

    parts = [all_df[num_cols].astype(float).reset_index(drop=True)]
    if obj_cols:
        dummies = pd.get_dummies(all_df[obj_cols].astype(str), columns=obj_cols)
        parts.append(dummies.astype(float).reset_index(drop=True))
    combined = pd.concat(parts, axis=1)

    out, start = {}, 0
    for k, n in zip(keys, sizes):
        out[k] = combined.iloc[start:start + n].reset_index(drop=True)
        start += n
    return out


def select_informative_columns(normal_df, feat_cols, rel_threshold=1e-3):
    """Drop features whose real-normal std is negligible relative to their
    own mean magnitude (near-constant-for-normal columns that would
    dominate z-normalized distances via a near-zero denominator)."""
    X = normal_df[feat_cols].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    scale = np.maximum(np.abs(mean), 1.0)
    keep = std >= rel_threshold * scale
    kept_cols = [c for c, k in zip(feat_cols, keep) if k]
    return kept_cols, mean[keep], std[keep]


def zscore_distance(df, feat_cols, centroid, std):
    X = df[feat_cols].to_numpy(dtype=np.float64)
    std_safe = np.where(std < 1e-8, 1e-8, std)
    return np.linalg.norm((X - centroid) / std_safe, axis=1)


# =============================================================================
# Part (a) -- visual / distance-based evidence
# =============================================================================

def viz_tsne(bundle, fig_dir, dataset):
    rare = RARE_ATTACKS[dataset]
    z_attacks = {n: bundle["z_attacks"][n] for n in rare if n in bundle["z_attacks"]}
    if not z_attacks:
        print("  [tsne] no rare attacks found, skip")
        return
    z_gen = np.concatenate([bundle["gen_zs"]["ctgan_noise"], bundle["gen_zs"]["noise_pseudo"]], axis=0)
    rng = np.random.default_rng(RNG_SEED)
    if len(z_gen) > MAX_N:
        z_gen = z_gen[rng.choice(len(z_gen), MAX_N, replace=False)]
    z_normal = bundle["z_normal"]

    names = list(z_attacks.keys())
    pool = np.concatenate([z_gen] + [z_attacks[n] for n in names] + [z_normal], axis=0)
    perplexity = min(30, max(5, (len(pool) - 1) // 3))
    coords = TSNE(n_components=2, random_state=RNG_SEED, init="pca", perplexity=perplexity).fit_transform(pool)

    i = 0
    gen_2d = coords[i:i + len(z_gen)]; i += len(z_gen)
    att_2d = {}
    for n in names:
        k = len(z_attacks[n]); att_2d[n] = coords[i:i + k]; i += k
    normal_2d = coords[i:]

    def draw(groups, out_path):
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        for pts, color, label in groups:
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], color=color, s=10, linewidths=0, rasterized=True, label=label)
        ax.legend(loc="best", fontsize=8, markerscale=1.5, framealpha=0.85)
        fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

    draw([(gen_2d, GEN_COL, "Generated anomaly"), (normal_2d, BLUE, "Normal")],
         os.path.join(fig_dir, "tsne_generated.png"))
    draw([(gen_2d, GEN_COL, "Generated anomaly")]
         + [(att_2d[n], np.array([*matplotlib.colors.to_rgb(ATTACK_COLORS[i % 2]), 0.6]), DISPLAY_NAME.get(n, n))
            for i, n in enumerate(names)]
         + [(normal_2d, BLUE, "Normal")],
         os.path.join(fig_dir, "tsne_combined.png"))
    print("  [tsne] 2 images saved")


def viz_centroid_distance(bundle, fig_dir, dataset, numeric_frames, feat_cols, centroid, std):
    sources = ["ctgan_raw", "ctgan_noise", "noise_pseudo"]
    rows = []
    for s in sources:
        d = zscore_distance(numeric_frames[s], feat_cols, centroid, std).mean()
        rows.append({"Group": "generated", "Name": s, "Distance": d})
    for name in bundle["real_attacks"]:
        key = f"attack::{name}"
        d = zscore_distance(numeric_frames[key], feat_cols, centroid, std).mean()
        rows.append({"Group": "real_attack", "Name": DISPLAY_NAME.get(name, name), "Distance": d})
    df = pd.DataFrame(rows).sort_values("Distance")
    df.to_csv(os.path.join(fig_dir, "centroid_distance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(df))), dpi=150)
    colors = df["Group"].map({"generated": "#ff7f0e", "real_attack": "#d62728"})
    ax.barh(df["Name"], df["Distance"], color=colors)
    ax.set_xlabel("z-normalized distance to real-normal centroid")
    ax.set_title(f"{dataset}: generated sources vs real attack types")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#ff7f0e"), plt.Rectangle((0, 0), 1, 1, color="#d62728")]
    ax.legend(handles, ["generated", "real attack"], loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "centroid_distance.png")); plt.close(fig)
    print(f"  [centroid_distance] saved, {len(df)} rows")


def viz_kde_distribution(bundle, fig_dir, dataset, numeric_frames, feat_cols, centroid, std):
    rare = RARE_ATTACKS[dataset]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    series = [("real_normal", "Normal (real)", "#1f77b4"),
              ("ctgan_raw", "ctgan_raw (generated)", "#ff7f0e"),
              ("ctgan_noise", "ctgan_noise (generated)", "#9467bd"),
              ("noise_pseudo", "noise_pseudo (generated)", "#8c564b")]
    for name in rare:
        key = f"attack::{name}"
        if key in numeric_frames:
            series.append((key, f"{DISPLAY_NAME.get(name, name)} (real attack)", None))

    palette_extra = iter(["#d62728", "#e377c2"])
    for key, label, color in series:
        if key not in numeric_frames:
            continue
        d = zscore_distance(numeric_frames[key], feat_cols, centroid, std)
        d = d[np.isfinite(d)]
        if len(d) < 2 or d.std() < 1e-9:
            continue
        c = color or next(palette_extra, "#333333")
        xs = np.linspace(0, max(d.max(), 1e-3) * 1.05, 300)
        try:
            kde = gaussian_kde(d)
            ax.plot(xs, kde(xs), label=label, color=c)
        except Exception:
            continue
    ax.set_xlabel("z-normalized distance to real-normal centroid")
    ax.set_ylabel("density")
    ax.set_title(f"{dataset}: distance-to-normal distribution")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "distance_distribution.png")); plt.close(fig)
    print("  [kde] saved")


# =============================================================================
# Part (b) -- correlation similarity + C2ST
# =============================================================================

def correlation_similarity(df_a, df_b, feat_cols):
    ca = np.nan_to_num(df_a[feat_cols].corr().to_numpy())
    cb = np.nan_to_num(df_b[feat_cols].corr().to_numpy())
    iu = np.triu_indices_from(ca, k=1)
    va, vb = ca[iu], cb[iu]
    corr_of_corr = np.nan
    if va.std() > 1e-12 and vb.std() > 1e-12:
        corr_of_corr = float(np.corrcoef(va, vb)[0, 1])
    frob = float(np.linalg.norm(ca - cb, ord="fro"))
    return corr_of_corr, frob


def c2st_auc(df_a, df_b, feat_cols):
    Xa = df_a[feat_cols].to_numpy(dtype=np.float64)
    Xb = df_b[feat_cols].to_numpy(dtype=np.float64)
    if len(Xa) < 10 or len(Xb) < 10:
        return np.nan
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RNG_SEED, n_jobs=-1)
    n_splits = min(5, int(min(np.bincount(y.astype(int)))))
    if n_splits < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    except Exception:
        return np.nan
    return float(scores.mean()) * 100


def compute_metrics_table(bundle, numeric_frames, feat_cols, dataset):
    rows = []
    gen_defs = {"ctgan_raw": numeric_frames["ctgan_raw"],
                "pseudo_anomaly": pd.concat(
                    [numeric_frames["ctgan_noise"], numeric_frames["noise_pseudo"]], ignore_index=True)}
    normal_df = numeric_frames["real_normal"]

    for gen_name, gen_df in gen_defs.items():
        corr_norm, frob_norm = correlation_similarity(gen_df, normal_df, feat_cols)
        c2st_norm = c2st_auc(gen_df, normal_df, feat_cols)
        for attack_name in bundle["real_attacks"]:
            key = f"attack::{attack_name}"
            atk_df = numeric_frames[key]
            corr_atk, frob_atk = correlation_similarity(gen_df, atk_df, feat_cols)
            c2st_atk = c2st_auc(gen_df, atk_df, feat_cols)
            rows.append({
                "Dataset": dataset, "Generated": gen_name,
                "Attack": DISPLAY_NAME.get(attack_name, attack_name), "N_attack": len(atk_df),
                "CorrOfCorr_vs_attack": corr_atk, "CorrOfCorr_vs_normal": corr_norm,
                "FrobDist_vs_attack": frob_atk, "FrobDist_vs_normal": frob_norm,
                "C2ST_AUC_vs_attack": c2st_atk, "C2ST_AUC_vs_normal": c2st_norm,
                "CloserToAttack_corr": bool(corr_atk > corr_norm) if not (np.isnan(corr_atk) or np.isnan(corr_norm)) else None,
                "IndistinguishableFromAttack_c2st": bool(c2st_atk < 60) if not np.isnan(c2st_atk) else None,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Orchestration
# =============================================================================

def run_dataset(dataset):
    print(f"\n[{dataset}]")
    bundle = prepare_dataset(dataset)
    if bundle is None:
        return None

    ds_dir = os.path.join(OUT_DIR, dataset)
    data_dir = os.path.join(ds_dir, "data")
    fig_dir = os.path.join(ds_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    save_data_csvs(bundle, data_dir)

    all_groups = {"real_normal": bundle["real_normal_df"],
                  **{f"attack::{k}": v for k, v in bundle["real_attacks"].items()},
                  **bundle["gen_dfs"]}
    numeric_frames = build_numeric_frames(all_groups)
    feat_cols, centroid, std = select_informative_columns(
        numeric_frames["real_normal"], numeric_frames["real_normal"].columns.tolist())
    print(f"  {len(feat_cols)}/{numeric_frames['real_normal'].shape[1]} informative feature columns kept")

    print("  -- part (a): visual / distance evidence --")
    viz_tsne(bundle, fig_dir, dataset)
    viz_centroid_distance(bundle, fig_dir, dataset, numeric_frames, feat_cols, centroid, std)
    viz_kde_distribution(bundle, fig_dir, dataset, numeric_frames, feat_cols, centroid, std)

    print("  -- part (b): correlation similarity + C2ST --")
    metrics_df = compute_metrics_table(bundle, numeric_frames, feat_cols, dataset)
    metrics_df.to_csv(os.path.join(ds_dir, "metrics.csv"), index=False)
    print(metrics_df.drop(columns=["Dataset"]).to_string(index=False))
    return metrics_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()
    datasets = args.datasets.split(",")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_metrics = []
    for ds in datasets:
        try:
            m = run_dataset(ds)
            if m is not None:
                all_metrics.append(m)
        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            traceback.print_exc()

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)
        print(f"\nsaved combined summary: {os.path.join(OUT_DIR, 'summary_metrics.csv')} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
