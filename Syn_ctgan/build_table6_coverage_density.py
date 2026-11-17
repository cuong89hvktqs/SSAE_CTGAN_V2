# -*- coding: utf-8 -*-
"""Reviewer 1 Comment 7 (Over-interpretation of the visualisations), parts
(i) and (ii): Figure 3 (score-informed polar PCA) and Figure 4 (centroid
distance) are not independent evidence that pseudo-intrusions resemble or
cover real attacks. The response commits to two new distribution-sensitive
measures, computed in the M01AE latent z space:

  (i)  Table [T6] + Figure [F1]: for every real attack test sample, the
       latent-space distance to its NEAREST pseudo-intrusion sample,
       reported as median and 90th percentile, separately for source P
       (real-normal + noise) and source Q (CTGAN-sample + noise). Figure
       [F1] plots the CDF of these nearest-neighbour distances per dataset.
  (ii) Coverage and density of the real-attack set with respect to each
       pseudo-intrusion source, using the k-NN manifold estimator of Naeem
       et al. (ICML 2020, "Reliable Fidelity and Diversity Metrics for
       Generative Models") with the real attacks as the REFERENCE set
       (manifold balls are built around real-attack points; P/Q play the
       role of the "generated" set being tested for coverage/density).

NO RETRAINING / NO NEW DATA: reuses compare_gen_vs_attack.py's
prepare_dataset() (t_03 seed 42 checkpoints, already-trained; real test
attacks pooled across EVERY attack type of each dataset, not only the 2
rare ones; P/Q freshly regenerated the same way the model itself builds
them during training). CLAN's augmented pseudo-anomaly pool is NOT
included here (it lives in a different, non-SSAE-latent space -- CLAN-main
sibling repo -- and would need its own encoder pass; left as a follow-up).

Output:
  res/compare_ge_attack/table_t6_coverage_density.csv
  res/compare_ge_attack/<dataset>/figures/f1_nn_distance_cdf.png

Usage:
    python build_table6_coverage_density.py [--k 5]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from compare_gen_vs_attack import DATASETS, OUT_DIR, prepare_dataset

SOURCE_KEYS = {"P": "noise_pseudo", "Q": "ctgan_noise"}
SOURCE_COLORS = {"P": "#1f77b4", "Q": "#ff7f0e"}


def nn_distance_to_source(attack_pool, source_z):
    """For every row in attack_pool, its distance to the NEAREST row in
    source_z. Returns a 1D array, one value per attack sample."""
    d = cdist(attack_pool, source_z)
    return d.min(axis=1)


def coverage_density(real_features, fake_features, k=5):
    """Naeem et al. (ICML 2020) k-NN manifold coverage/density, with
    real_features as the reference manifold.
    coverage: fraction of real points whose k-NN ball (radius = distance to
      their own k-th nearest REAL neighbour) contains >=1 fake point.
    density: average, over fake points, of how many real k-NN balls
      contain that fake point, normalized by k (can exceed 1 if fake points
      cluster densely inside real manifolds -- that is expected/informative,
      not a bug)."""
    n_real = len(real_features)
    if n_real <= k:
        return np.nan, np.nan
    real_real_d = cdist(real_features, real_features)
    np.fill_diagonal(real_real_d, np.inf)
    # k-th nearest neighbour distance (k=5 -> index 4 after sorting/partition)
    real_nn_radius = np.partition(real_real_d, k - 1, axis=1)[:, k - 1]

    real_fake_d = cdist(real_features, fake_features)  # (n_real, n_fake)
    within = real_fake_d < real_nn_radius[:, None]      # (n_real, n_fake) bool

    coverage = within.any(axis=0).mean()  # fraction of FAKE... wait see note below
    # NOTE: Naeem et al.'s coverage is defined per REAL point (fraction of
    # real manifolds that contain >=1 fake sample), not per fake point.
    coverage = within.any(axis=1).mean()
    density = (1.0 / k) * within.sum(axis=0).mean()
    return float(coverage), float(density)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    rows = []
    cdf_data = {}  # dataset -> {source: distances array}

    for dataset in DATASETS:
        print(f"=== {dataset} ===")
        bundle = prepare_dataset(dataset)
        if bundle is None:
            print(f"  [skip] no t_03 checkpoint for {dataset}")
            continue

        attack_pool = np.concatenate(list(bundle["z_attacks"].values()), axis=0)
        n_attack_total = len(attack_pool)
        cdf_data[dataset] = {}

        for src_label, gen_key in SOURCE_KEYS.items():
            source_z = bundle["gen_zs"][gen_key]
            dists = nn_distance_to_source(attack_pool, source_z)
            cdf_data[dataset][src_label] = dists
            cov, dens = coverage_density(attack_pool, source_z, k=args.k)

            rows.append({
                "Dataset": dataset, "Source": src_label,
                "N_attack_samples": n_attack_total, "N_source_samples": len(source_z),
                "NN_dist_median": float(np.median(dists)),
                "NN_dist_p90": float(np.percentile(dists, 90)),
                "Coverage": cov, "Density": dens,
            })
            print(f"  {src_label}: median_dist={np.median(dists):.4f}  "
                  f"p90_dist={np.percentile(dists, 90):.4f}  "
                  f"coverage={cov:.4f}  density={dens:.4f}")

        # Combined P+Q source (union pool) -- matches the response's
        # "coverage of P+Q exceeds that of either source alone" claim.
        pq_pool = np.concatenate(
            [bundle["gen_zs"][SOURCE_KEYS["P"]], bundle["gen_zs"][SOURCE_KEYS["Q"]]], axis=0)
        dists_pq = nn_distance_to_source(attack_pool, pq_pool)
        cov_pq, dens_pq = coverage_density(attack_pool, pq_pool, k=args.k)
        cdf_data[dataset]["P+Q"] = dists_pq
        rows.append({
            "Dataset": dataset, "Source": "P+Q",
            "N_attack_samples": n_attack_total, "N_source_samples": len(pq_pool),
            "NN_dist_median": float(np.median(dists_pq)),
            "NN_dist_p90": float(np.percentile(dists_pq, 90)),
            "Coverage": cov_pq, "Density": dens_pq,
        })
        print(f"  P+Q: median_dist={np.median(dists_pq):.4f}  "
              f"p90_dist={np.percentile(dists_pq, 90):.4f}  "
              f"coverage={cov_pq:.4f}  density={dens_pq:.4f}")

        # Figure [F1]: CDF of nearest-neighbour distances, P vs Q vs P+Q
        fig_dir = os.path.join(OUT_DIR, dataset, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, color in [("P", SOURCE_COLORS["P"]), ("Q", SOURCE_COLORS["Q"]), ("P+Q", "#2ca02c")]:
            d = np.sort(cdf_data[dataset][label])
            y = np.arange(1, len(d) + 1) / len(d)
            ax.plot(d, y, label=label, color=color if label != "P+Q" else "#2ca02c", lw=1.8)
        ax.set_xlabel("Latent distance to nearest pseudo-intrusion")
        ax.set_ylabel("CDF over real attack samples")
        ax.set_title(f"{dataset}: attack -> nearest pseudo-intrusion distance")
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(fig_dir, "f1_nn_distance_cdf.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  figure -> {fig_path}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "table_t6_coverage_density.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nTable [T6]: {len(df)} rows -> {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
