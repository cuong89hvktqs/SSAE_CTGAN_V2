# -*- coding: utf-8 -*-
"""Per-attack-class latent coverage/density on UNSW-NB15 (reviewer follow-up).

Table 12 of the manuscript (build_table6_coverage_density.py) pools every
attack class of a dataset. This script breaks the same quantities down per
attack class of UNSW-NB15, to show WHICH classes the pseudo-intrusions reach
(and which -- e.g. Generic / Exploits -- they do not), using the identical
protocol and checkpoint (compare_gen_vs_attack.prepare_dataset: t_03 seed 42,
<=2000 samples per group, fresh P / Q samples regenerated from the model).

Per class: N in test set, N used, mean latent radius ||z|| (vs. margin m=2
and vs. the pseudo-intrusion radius), cosine between the class-centroid
direction (relative to the normal centroid) and the P+Q centroid direction,
nearest-pseudo-intrusion distance (median / p90) for P, Q, P+Q, k-NN coverage
(Naeem et al., 2020; k=5) for P, Q, P+Q and density for P+Q. A "Normal (test)"
reference row shows how much of the real NORMAL manifold the same pseudo
samples cover. SSAE's per-class MCC at tau_99 (t_04 seed 42 and the 10-seed
mean) is appended for cross-reference.

Output: res/compare_ge_attack/unsw_coverage_per_class.csv
Usage:  python build_coverage_per_class_unsw.py [--k 5] [--dataset unsw]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from build_table6_coverage_density import coverage_density, nn_distance_to_source
from compare_gen_vs_attack import OUT_DIR, RES_DIR, prepare_dataset

SOURCE_KEYS = {"P": "noise_pseudo", "Q": "ctgan_noise"}


def ssae_mcc_per_class(dataset):
    """seed-42 MCC_99 from t_04 and the 10-seed mean/std from summary xlsx."""
    out = {}
    f = [p for p in glob.glob(os.path.join(RES_DIR, "t_04_SSAE_se_42", "per_1.0",
                                           f"{dataset}_*noise_std_*catra0.1", "best_auc",
                                           "per_attack_seeds.csv"))
         if "_gf" not in p and "gatefix" not in p]
    if f:
        df = pd.read_csv(f[0])
        for a, v in zip(df["Attack"], df["MCC_99"]):
            out.setdefault(str(a), {})["MCC_99_seed42"] = float(v)
    xlsx = os.path.join(RES_DIR, "summary_seeds42-51.xlsx")
    if os.path.exists(xlsx):
        import openpyxl
        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        rows = list(wb["5_PerAttack_t04"].iter_rows(values_only=True))
        ds = atk = m = None
        display = {"unsw": "UNSW-NB15"}.get(dataset, dataset)
        for r in rows[3:]:
            if r[0]: ds = r[0]
            if r[1]: atk = r[1]
            if r[2]: m = r[2]
            if ds == display and m == "SSAE" and r[3] == "MCC_99":
                out.setdefault(str(atk), {})["MCC_99_mean10"] = float(r[15])
                out[str(atk)]["MCC_99_std10"] = float(r[16])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dataset", default="unsw")
    args = ap.parse_args()

    bundle = prepare_dataset(args.dataset)
    if bundle is None:
        raise SystemExit("no checkpoint")
    model = bundle["model"]
    z_normal = bundle["z_normal"]
    zP, zQ = bundle["gen_zs"][SOURCE_KEYS["P"]], bundle["gen_zs"][SOURCE_KEYS["Q"]]
    zPQ = np.concatenate([zP, zQ])
    mu_n = z_normal.mean(0)
    dir_pq = zPQ.mean(0) - mu_n
    r_pq = np.linalg.norm(zPQ, axis=1).mean()

    # full test counts (before the 2000 cap)
    from val_perattack_mae01scale_max import ATTACK_LOADERS
    labels, normal_tag = ATTACK_LOADERS[args.dataset](42)
    counts = labels.value_counts().to_dict()
    mcc = ssae_mcc_per_class(args.dataset)

    def row(name, z_cls, n_full):
        mu_c = z_cls.mean(0) - mu_n
        cos = float(np.dot(mu_c, dir_pq) / (np.linalg.norm(mu_c) * np.linalg.norm(dir_pq) + 1e-12))
        r = {"Class": name, "N_test": int(n_full), "N_used": len(z_cls),
             "Mean_radius_||z||": float(np.linalg.norm(z_cls, axis=1).mean()),
             "Radius_ratio_vs_pseudo": float(np.linalg.norm(z_cls, axis=1).mean() / r_pq),
             "Cosine_to_PQ_direction": cos}
        for src, z_src in (("P", zP), ("Q", zQ), ("P+Q", zPQ)):
            d = nn_distance_to_source(z_cls, z_src)
            cov, dens = coverage_density(z_cls, z_src, k=args.k)
            r[f"NNdist_median_{src}"] = float(np.median(d))
            r[f"NNdist_p90_{src}"] = float(np.percentile(d, 90))
            r[f"Coverage_{src}"] = cov
            if src == "P+Q":
                r["Density_P+Q"] = dens
        m = mcc.get(name, {})
        r["SSAE_MCC99_seed42"] = m.get("MCC_99_seed42", np.nan)
        r["SSAE_MCC99_mean10"] = m.get("MCC_99_mean10", np.nan)
        r["SSAE_MCC99_std10"] = m.get("MCC_99_std10", np.nan)
        return r

    rows = [row("Normal (test, reference)", z_normal, counts.get(normal_tag, len(z_normal)))]
    for name, z_cls in bundle["z_attacks"].items():
        rows.append(row(name, z_cls, counts.get(name, len(z_cls))))
    all_attack = np.concatenate(list(bundle["z_attacks"].values()))
    rows.append(row("All attacks (pooled)", all_attack,
                    sum(v for k_, v in counts.items() if k_ != normal_tag)))

    df = pd.DataFrame(rows)
    df.attrs["pseudo_radius"] = r_pq
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.dataset}_coverage_per_class.csv")
    df.to_csv(out, index=False)
    print(f"mean ||z|| of P+Q pseudo-intrusions = {r_pq:.4f}; normal = "
          f"{np.linalg.norm(z_normal, axis=1).mean():.4f}; model.noise_std={model.noise_std:.4f}")
    print(df.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
