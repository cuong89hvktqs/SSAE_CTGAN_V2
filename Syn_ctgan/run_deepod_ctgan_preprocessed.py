"""Run a DeepOD algorithm (default: ICL) on the CTGAN-preprocessed CSVs
built for CLAN (see TH/build_ctgan_preprocessed_for_clan.py ->
CLAN-main/data_ctgan_preprocessed/<dataset>.csv), instead of NeuTraL-AD's
own one-hot+zscore .npy encoding that run_deepod.py normally uses.

Reuses CLAN's own get_data() (CLAN-main/data/load_data.py, unmodified) to
load + split + scale each CSV -- same test_ratio=0.5/val_ratio=0.2 split and
MinMax-on-top-of-CTGAN-space scaling that CLAN itself would receive from
these files -- so the algorithm trains on data through the identical
pipeline the CSVs were built for. Only feature scaling/splitting is CLAN's;
the actual detector (ICL et al.) is DeepOD's own unmodified class, reused
via run_deepod.py's get_algo_class()/metrics_at_threshold().

Usage: python run_deepod_ctgan_preprocessed.py --algo icl [--datasets unsw,nsl_kdd,cic_iot,wsn_ds] [--seeds 42,43,44]
Output: DeepOD/results_ssae_ctgan_preprocessed/<algo>/<dataset>/seed<N>/{test_scores.npy,test_labels_out.npy,val_scores.npy,metrics.json}
        DeepOD/results_ssae_ctgan_preprocessed/<algo>_summary.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
DEEPOD_ROOT = BASE_DIR.parent / "DeepOD"
if str(DEEPOD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPOD_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CLAN_ROOT = Path(r"D:\Code_papper\CTGAN\CLGAN\CLAN-main\CLAN-main")
if str(CLAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLAN_ROOT))
from data.load_data import get_data  # noqa: E402  (CLAN's own, unmodified)

from run_deepod import ALGO_NAMES, get_algo_class, metrics_at_threshold  # noqa: E402

RESULTS_ROOT = DEEPOD_ROOT / "results_ssae_ctgan_preprocessed"
DATA_DIR = CLAN_ROOT / "data_ctgan_preprocessed"
SAMPLE_THRES = 100


# train_csv: the file passed as get_data()'s data_path. For unsw/nsl_kdd/
# cic_iot this is the TRAIN half of a train/test pair (matching filename
# convention "training-set"/"Train"/"train" so CLAN's own _infer_test_path()
# auto-detects and loads the paired test file too -- see
# TH/build_ctgan_preprocessed_for_clan.py). wsn_ds has no separate test file
# in the original dataset, so it stays a single combined file split
# internally by get_data()'s own test_ratio.
DATASETS = {
    "unsw": dict(target="attack_cat", class_zero="Normal", train_csv="unsw_training-set.csv"),
    "nsl_kdd": dict(target="label", class_zero="0", train_csv="nsl_kdd_Train.csv"),
    "cic_iot": dict(target="label", class_zero="Benign", train_csv="cic_iot_train.csv"),
    "wsn_ds": dict(target="Attack type", class_zero="Normal", train_csv="wsn_ds.csv"),
}


def load_ctgan_preprocessed(dataset, seed):
    cfg = DATASETS[dataset]
    csv_path = DATA_DIR / cfg["train_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- run build_ctgan_preprocessed_for_clan.py first")

    x_train, y_train, x_val, y_val, x_test, y_test, x_zd, y_zd = get_data(
        data_path=str(csv_path), target=cfg["target"], drop=[], class_zero=cfg["class_zero"],
        sample_thres=SAMPLE_THRES, split_seed=seed, test_ratio=0.5, val_ratio=0.2,
        anomaly_detection=True, sample_seed=seed,
    )
    if x_zd is not None:
        x_test = np.concatenate([x_test, x_zd])
        y_test = np.concatenate([y_test, y_zd])

    return (
        x_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32),
        (y_test != 0).astype(int),
    )


def run_one(algo, dataset, seed, device):
    algo_cls = get_algo_class(algo)
    train, val, test, test_labels = load_ctgan_preprocessed(dataset, seed)

    out_dir = RESULTS_ROOT / algo / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(device=device, random_state=seed, verbose=0)
    if algo == "dif":
        kwargs["n_jobs"] = 4
    clf = algo_cls(**kwargs)

    t0 = time.time()
    clf.fit(train)
    fit_seconds = time.time() - t0

    test_scores = clf.decision_function(test)
    val_scores = clf.decision_function(val)

    np.save(out_dir / "test_scores.npy", test_scores)
    np.save(out_dir / "test_labels_out.npy", test_labels)
    np.save(out_dir / "val_scores.npy", val_scores)

    m = metrics_at_threshold(test_labels, test_scores, clf.threshold_)
    metrics = dict(
        threshold=float(clf.threshold_), fit_seconds=fit_seconds,
        n_train=len(train), n_val=len(val), n_test=len(test), **m,
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{algo} {dataset} seed={seed}] AUC={m['AUC']:.2f}  F1={m['F1']:.2f}  "
          f"MCC={m['MCC']:.2f}  Precision={m['Precision']:.2f}  Recall={m['Recall']:.2f}  "
          f"n_train={len(train)} n_test={len(test)}  ({fit_seconds:.1f}s)")
    return metrics


def main():
    parser = argparse.ArgumentParser("Run a DeepOD algorithm on CTGAN-preprocessed CLAN CSVs")
    parser.add_argument("--algo", type=str, default="icl", choices=ALGO_NAMES)
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for dataset in datasets:
        for seed in seeds:
            print(f"\n=== {args.algo} | {dataset} seed={seed} ===")
            try:
                m = run_one(args.algo, dataset, seed, device)
                rows.append(dict(Dataset=dataset, Seed=seed, **m))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [WARN] {dataset} seed={seed}: {e}")

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        out_csv = RESULTS_ROOT / f"{args.algo}_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nsaved: {out_csv}")
        print(df[["Dataset", "Seed", "AUC", "F1", "MCC", "Precision", "Recall", "n_train", "n_test"]].to_string(index=False))

        summary = df.groupby("Dataset")[["AUC", "F1", "MCC"]].agg(["mean", "std"])
        print("\nmean +/- std per dataset:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
