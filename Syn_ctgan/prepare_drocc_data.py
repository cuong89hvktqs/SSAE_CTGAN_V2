"""Data adapter for DROCC (ICML 2020) -- builds the train_data.npy /
test_data.npy / test_labels.npy triplet DROCC_icml2020/tabular's own
unmodified one_class_main_tabular.py expects via its `-d root_data` flag.

This file lives entirely outside DROCC_icml2020/ and never imports or edits
anything under tabular/ -- DROCC's own core algorithm code (models.py,
train_tabular.py, loss_tabular.py, one_class_main_tabular.py, dataset.py,
utils.py, base/) is untouched. All this script does is reshape our data into
the exact .npy layout their own README documents for the Abalone / Arrhythmia
/ Thyroid examples.

Feature preprocessing: clan_style_get_data.get_data() -- a standalone
reimplementation (in this project, reading this project's own Syn_ctgan/data/
raw CSVs) of CLAN's data.load_data.get_data() methodology, so DROCC sees
features encoded exactly like the CLAN baseline does: MinMax-scaled to
[0, 1], non-numeric categorical columns (proto/service, ...) dropped rather
than encoded. No dependency on the CLAN-main repo (code or data) at all --
see clan_style_get_data.py's docstring for why. DROCC's own load_data()
(inside one_class_main_tabular.py) re-standardizes with its own train-set
mean/std on top of this, so the incoming numeric scale doesn't matter --
only "purely numeric, categoricals resolved somehow" does.

train_data.npy = ONLY the real normal TRAINING rows (get_data()'s
anomaly_detection=True mode already restricts x_train to the normal class;
DROCC is one-class and only ever trains on this).

val_data.npy = a held-out slice of normal rows get_data() ALREADY carves out
of x_train via val_ratio=0.2 (anomaly_detection=True restricts to normal
BEFORE the val split, so x_val is normal-only too, and the MinMaxScaler is
fit only on the post-val-split x_train -- no leakage). DROCC's own training
script never sees this file; it exists purely so eval_drocc.py can calibrate
a classification threshold (95th percentile of anomaly score on val_data,
matching this project's "total95" convention -- see train_mae01scale_max.py)
on data the model never trained on, instead of reusing train_data itself.

test_data.npy / test_labels.npy = the full reloaded test split (normal +
every attack type). Label convention: DROCC's own code (train_lab =
np.ones(...) in one_class_main_tabular.py, target=0 for adversarial negatives
in loss_tabular.py's one_class_adv_loss, and the worked Abalone example in
process_dataset.py) uses 1 = normal, 0 = anomaly -- the OPPOSITE of this
project's own y_test.csv convention (0 = normal, 1 = attack). This script
flips the label when writing test_labels.npy so DROCC's own internal
AUC/F1 tracking in train_tabular.py's test() stays meaningful without
touching that file.

percentage (--per, default 1.0): fraction of the normal-only TRAINING pool
to keep, matching SSAE's own `--per` flag / `per_<percentage>` result-folder
convention (run_mae01scale_max.sh / train_mae01scale_max.py). Applied BEFORE
the train/val split -- see clan_style_get_data.py's docstring for the exact
order-of-operations match. Only train_data.npy/val_data.npy shrink;
test_data.npy/test_labels.npy are unaffected (percentage never touches test).

Usage: python prepare_drocc_data.py [--dataset all] [--seeds 42,43,44] [--per 1.0]
Output: DROCC_icml2020/tabular/data_ssae/per_<per>/<dataset>/seed<N>/{train_data,test_data,test_labels}.npy
"""

import argparse
from pathlib import Path

import numpy as np

from clan_style_get_data import DATASETS, SAMPLE_THRES, get_data

DROCC_ROOT = Path(__file__).parent.parent / "DROCC_icml2020"
OUT_ROOT = DROCC_ROOT / "tabular" / "data_ssae"


def build_one(dataset, seed, percentage):
    cfg = DATASETS[dataset]
    x_train, y_train, x_val, y_val, x_te, y_te, x_zd, y_zd, label_mapping, feature_names = get_data(
        data_path=str(cfg["data_path"]), target=cfg["target"], drop=cfg["drop_cols"],
        class_zero=cfg["class_zero"], sample_thres=SAMPLE_THRES, split_seed=seed,
        test_ratio=0.2, val_ratio=0.2, anomaly_detection=True, n_samples=cfg["total_rows"],
        sample_seed=seed, percentage=percentage,
    )
    if x_zd is not None:
        x_te = np.concatenate([x_te, x_zd])
        y_te = np.concatenate([y_te, y_zd])

    # get_data()'s own convention is 0=normal (class_zero), 1..K=attack types
    # -- flip to DROCC's 1=normal, 0=anomaly convention (see module docstring).
    test_labels_drocc = np.where(y_te == 0, 1, 0).astype(np.float64)

    out_dir = OUT_ROOT / f"per_{percentage}" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_data.npy", x_train.astype(np.float64))
    np.save(out_dir / "val_data.npy", x_val.astype(np.float64))
    np.save(out_dir / "test_data.npy", x_te.astype(np.float64))
    np.save(out_dir / "test_labels.npy", test_labels_drocc)

    n_normal = int((test_labels_drocc == 1).sum())
    n_anom = int((test_labels_drocc == 0).sum())
    print(f"  [{dataset} seed={seed}] d={len(feature_names)}  "
          f"train_normal={len(x_train)}  val_normal={len(x_val)}  "
          f"test_normal={n_normal}  test_anomaly={n_anom}  -> {out_dir}")


def main():
    parser = argparse.ArgumentParser("Prepare DROCC .npy inputs from clan_style_get_data's preprocessing pipeline")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS.keys()])
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0",
                        help="fraction of the normal-only training pool to keep (matches SSAE's --per)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    percentage = float(args.per)

    for name in names:
        for seed in seeds:
            build_one(name, seed, percentage)


if __name__ == "__main__":
    main()
