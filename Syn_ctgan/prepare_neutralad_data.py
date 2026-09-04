"""Data adapter for NeuTraL-AD (ICML 2021) -- builds the train_data.npy /
val_data.npy / test_data.npy / test_labels.npy / test_attack_types.npy
quintuplet consumed by run_neutralad.py (a standalone driver that never
imports or edits anything under NeuTraL-AD/'s own model/loss/trainer code --
see run_neutralad.py's docstring for why the whole integration lives outside
that repo).

Row split: uses THIS project's own process_data.py loaders directly (the
exact same functions train_mae01scale_max.py itself calls) and replicates
its split code line-for-line:
    n_test      = len(df_test)
    X_train_raw = df_full.iloc[:-n_test]              # SSAE's train pool
    X_test_raw  = df_full.iloc[-n_test:]               # SSAE's real test file
    normal-only subset -> .sample(frac=percentage, seed) -> train_test_split(
        test_size=0.2, random_state=seed)              # SSAE's val split
so train/val/test row membership is byte-identical to whatever SSAE run used
the same (dataset, seed, --per). Unlike DROCC's data prep (which reimplements
CLAN's own MinMax/drop-categorical pipeline for CLAN-independence reasons),
there is no such constraint here, so this script just calls process_data.py
directly -- simpler and guarantees exact row alignment with
val_perattack_mae01scale_max.py's ATTACK_LOADERS (used below to attach a
per-row attack-type string to the test split, for TPR@FAR-per-attack eval).

Feature encoding (fit on the normal TRAIN split only, mirroring NeuTraL-AD's
own tabular convention in loader/LoadTabular.py's KDD_preprocessing /
norm_kdd_data: one-hot the symbolic/categorical columns, z-normalize the
continuous ones):
    num_cols = numeric dtype columns  -> (x - mu) / std   (mu, std from train)
    cat_cols = non-numeric columns    -> pd.get_dummies, columns fixed from
                                          train's own categories; a category
                                          never seen in train maps to an
                                          all-zero one-hot row (same "unseen
                                          category -> zeros" convention this
                                          project already uses for CLAN/RDT).

percentage (--per, default 1.0): fraction of the normal-only TRAINING pool
to keep, matching SSAE's own `--per` flag. Applied BEFORE the train/val
split, same order-of-operations as train_mae01scale_max.py. Only
train_data.npy/val_data.npy shrink; test is unaffected.

Usage: python prepare_neutralad_data.py [--dataset all] [--seeds 42,43,44] [--per 1.0]
Output: ../NeuTraL-AD/DATA/ssae/per_<per>/<dataset>/seed<N>/
            {train_data,val_data,test_data,test_labels}.npy
            test_attack_types.npy  (object array, per-row attack-type string)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)  # process_data.py's loaders use "./data/..." relative paths
sys.path.insert(0, str(BASE_DIR))

from process_data import (
    load_data_cic_iot,
    load_data_nsl_kdd,
    load_data_unsw,
    load_data_wsn_ds,
)
import val_perattack_mae01scale_max as vp  # reuses ATTACK_LOADERS

NEUTRALAD_ROOT = BASE_DIR.parent / "NeuTraL-AD"
OUT_ROOT = NEUTRALAD_ROOT / "DATA" / "ssae"

DATASET_LOADERS = {
    "unsw":    lambda seed: load_data_unsw(),
    "nsl_kdd": lambda seed: load_data_nsl_kdd(),
    "cic_iot": lambda seed: load_data_cic_iot(),
    "wsn_ds":  lambda seed: load_data_wsn_ds(seed=seed, test_size=0.2),
}


def _encode(df, num_cols, cat_cols, mu, std, dummy_cols):
    num_part = ((df[num_cols] - mu) / std).to_numpy(dtype=np.float64)
    if not cat_cols:
        return num_part
    cat_part = pd.get_dummies(df[cat_cols].astype(str))
    cat_part = cat_part.reindex(columns=dummy_cols, fill_value=0).to_numpy(dtype=np.float64)
    return np.concatenate([num_part, cat_part], axis=1)


def build_one(dataset, seed, percentage):
    df_full, y_full, df_test, y_test = DATASET_LOADERS[dataset](seed)
    df_full.columns = df_full.columns.map(str)
    df_test.columns = df_test.columns.map(str)

    n_test = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)
    X_test_raw = df_full.iloc[-n_test:].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    normal_mask = y_train == 0
    X_train_normal_full = X_train_raw.loc[normal_mask].copy()
    y_train_normal_full = y_train.loc[normal_mask].copy()

    sampled_idx = X_train_normal_full.sample(frac=percentage, random_state=seed).index
    X_normal_sub = X_train_normal_full.loc[sampled_idx].reset_index(drop=True)
    y_normal_sub = y_train_normal_full.loc[sampled_idx].reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    X_train_normal, X_val_normal = train_test_split(
        X_normal_sub, test_size=0.2, random_state=seed
    )
    X_train_normal = X_train_normal.reset_index(drop=True)
    X_val_normal = X_val_normal.reset_index(drop=True)

    # ---- fit encoding on the normal TRAIN split only ----
    num_cols = X_train_normal.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train_normal.columns if c not in num_cols]
    mu = X_train_normal[num_cols].mean(axis=0)
    std = X_train_normal[num_cols].std(axis=0)
    std[std == 0] = 1.0
    dummy_cols = pd.get_dummies(X_train_normal[cat_cols].astype(str)).columns if cat_cols else []

    train_arr = _encode(X_train_normal, num_cols, cat_cols, mu, std, dummy_cols)
    val_arr = _encode(X_val_normal, num_cols, cat_cols, mu, std, dummy_cols)
    test_arr = _encode(X_test_raw, num_cols, cat_cols, mu, std, dummy_cols)
    test_labels = y_test.to_numpy(dtype=np.float64)  # 0=normal, 1=attack -- matches NeuTraL-AD's own convention

    # ---- per-row attack-type string for the test split (TPR@FAR-per-attack) ----
    attack_labels, normal_tag = vp.ATTACK_LOADERS[dataset](seed)
    if len(attack_labels) != len(X_test_raw):
        raise RuntimeError(
            f"[{dataset} seed={seed}] attack labels ({len(attack_labels)}) != "
            f"X_test rows ({len(X_test_raw)}) -- alignment broken."
        )
    y_from_attack = (attack_labels != normal_tag).astype(int).to_numpy()
    mismatch = int((y_from_attack != test_labels.astype(int)).sum())
    if mismatch > 0:
        raise RuntimeError(f"[{dataset} seed={seed}] label mismatch on {mismatch} rows -- alignment broken.")

    out_dir = OUT_ROOT / f"per_{percentage}" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_data.npy", train_arr.astype(np.float32))
    np.save(out_dir / "val_data.npy", val_arr.astype(np.float32))
    np.save(out_dir / "test_data.npy", test_arr.astype(np.float32))
    np.save(out_dir / "test_labels.npy", test_labels)
    np.save(out_dir / "test_attack_types.npy", attack_labels.to_numpy(dtype=object))
    with open(out_dir / "normal_tag.txt", "w") as f:
        f.write(normal_tag)

    n_normal = int((test_labels == 0).sum())
    n_attack = int((test_labels == 1).sum())
    print(f"  [{dataset} seed={seed}] d={train_arr.shape[1]} "
          f"(num={len(num_cols)}, cat_onehot={train_arr.shape[1]-len(num_cols)})  "
          f"train_normal={len(train_arr)}  val_normal={len(val_arr)}  "
          f"test_normal={n_normal}  test_attack={n_attack}  -> {out_dir}")


def main():
    parser = argparse.ArgumentParser("Prepare NeuTraL-AD .npy inputs, split exactly like SSAE (train_mae01scale_max.py)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASET_LOADERS.keys()])
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0",
                        help="fraction of the normal-only training pool to keep (matches SSAE's --per)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    names = list(DATASET_LOADERS.keys()) if args.dataset == "all" else [args.dataset]
    percentage = float(args.per)

    for name in names:
        for seed in seeds:
            build_one(name, seed, percentage)


if __name__ == "__main__":
    main()
