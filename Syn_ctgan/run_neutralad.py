"""Trains NeuTraL-AD (unmodified NeuTraL-AD/models/*.py, evaluation/*.py,
loader/*.py -- see module docstring below for why NOTHING under NeuTraL-AD/
is edited) on this project's 4 comparison datasets, controlling seeds
explicitly (42/43/44) to match SSAE's own convention.

Why this bypasses Launch_Exps.py / evaluation.Kvariants_Eval.KVariantEval:
those classes hardcode their run-to-run seed as `i + 41` (i = the 0-indexed
repeat counter inside a `num_repeat` loop) -- there is no way to request
seeds 42/43/44 specifically through that entry point, and num_cls/cls_type
machinery exists only for the one-vs-rest image/time-series datasets (all of
NeuTraL-AD's own tabular datasets already have num_cls=1 and ignore cls/
cls_type entirely -- see config/Dataset_Class.py and loader/LoadData.py's
load_data()). Rather than editing Kvariants_Eval.py's seed formula, this
script calls the same public building blocks Kvariants_Eval/Experiments
themselves call (config.base.Config, evaluation.Experiments.runExperiment,
loader.utils.CustomDataset) directly, with an explicit seed and a dataset
tuple built from prepare_neutralad_data.py's .npy output -- zero lines of
NeuTraL-AD/'s own code touched, only its documented extension points
(a new config yml under config_files/, exactly as its README instructs).

Hyperparameters: config_files/config_ssae.yml is an unmodified copy of the
paper's own config_kdd.yml (its closest tabular network-intrusion example)
-- no hyperparameters invented for this integration, matching this
project's DROCC/CLAN precedent of reusing each baseline's own paper
defaults rather than re-tuning them.

Checkpoint selection: unmodified from the paper -- Patience(patience=10,
use_train_loss=True) selects the epoch with the lowest TRAINING loss (not a
labeled validation metric), only checked from epoch 6 onward. This is
NeuTraL-AD's own genuinely-unsupervised model-selection criterion; this
script does not override it. val_data.npy (a real held-out normal split,
matching SSAE's own 80/20 val convention -- NOT the same rows as test,
unlike the paper's own thyroid/arrhythmia/kdd/kddrev examples which reuse
testset as "valset") is still threaded through so val_auc is tracked/logged
for reference, exactly like the paper's own datasets do.

Usage: python run_neutralad.py [--datasets unsw,nsl_kdd,cic_iot,wsn_ds]
                                [--seeds 42,43,44] [--per 1.0] [--device auto]
Output: ../NeuTraL-AD/results_ssae/per_<per>/<dataset>/seed<N>/
            {test_scores.npy, test_labels_out.npy, metrics.json, experiment.log}
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

NEUTRALAD_ROOT = Path(__file__).parent.parent / "NeuTraL-AD"
if str(NEUTRALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUTRALAD_ROOT))

from config.base import Config          # noqa: E402  (NeuTraL-AD's own, unmodified)
from evaluation.Experiments import runExperiment  # noqa: E402
from loader.utils import CustomDataset  # noqa: E402
from utils import Logger, read_config_file  # noqa: E402

DATA_ROOT = NEUTRALAD_ROOT / "DATA" / "ssae"
RESULTS_ROOT = NEUTRALAD_ROOT / "results_ssae"
DEFAULT_CONFIG = NEUTRALAD_ROOT / "config_files" / "config_ssae.yml"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_config(config_path, device):
    """Grid-yml values are all length-1 lists (config.base.Grid's own
    convention for hyperparameter sweeps); unwrap them since we only ever
    want the single value, not a sweep. scheduler/early_stopper entries are
    already {class, args} dicts inside their length-1 list."""
    raw = read_config_file(str(config_path))
    flat = {}
    for k, v in raw.items():
        flat[k] = v[0] if isinstance(v, list) and len(v) == 1 else v
    flat["device"] = device
    flat["num_repeat"] = 1
    return flat


def mcc_at_paper_threshold(labels, scores):
    """MCC at the exact same threshold NeuTraL-AD's own utils.compute_pre_recall_f1()
    uses for its reported test_f1 (not reimplemented there -- that function only
    returns f1 and discards precision/recall/pred -- so the threshold is
    recomputed here from the same saved (scores, labels) pair, giving an MCC
    directly comparable to test_f1 since both are evaluated at the same cut):
        threshold = percentile(scores, 100 * normal_ratio)
        pred = 1 if score > threshold else 0
    """
    normal_ratio = (labels == 0).sum() / len(labels)
    threshold = np.percentile(scores, 100 * normal_ratio)
    pred = (scores > threshold).astype(int)
    return matthews_corrcoef(labels, pred)


def load_dataset_tuple(dataset, seed, per):
    d = DATA_ROOT / f"per_{per}" / dataset / f"seed{seed}"
    for name in ("train_data.npy", "val_data.npy", "test_data.npy", "test_labels.npy"):
        if not (d / name).exists():
            raise FileNotFoundError(
                f"missing {d / name} -- run prepare_neutralad_data.py first "
                f"(--dataset {dataset} --seeds {seed} --per {per})"
            )
    train = np.load(d / "train_data.npy")
    val = np.load(d / "val_data.npy")
    test = np.load(d / "test_data.npy")
    test_labels = np.load(d / "test_labels.npy")

    trainset = CustomDataset(train, np.zeros(len(train)))
    valset = CustomDataset(val, np.zeros(len(val)))
    testset = CustomDataset(test, test_labels)
    return trainset, valset, testset


def run_one(dataset, seed, per, config_path, device):
    flat_config = flatten_config(config_path, device)
    set_seed(seed)
    dataset_tuple = load_dataset_tuple(dataset, seed, per)

    out_dir = RESULTS_ROOT / f"per_{per}" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(out_dir / "experiment.log"), mode="w")

    experiment = runExperiment(flat_config, str(out_dir))
    t0 = time.time()
    val_auc, test_auc, test_ap, test_f1, scores, labels = experiment.run_test(dataset_tuple, logger)
    elapsed = time.time() - t0

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    np.save(out_dir / "test_scores.npy", scores)
    np.save(out_dir / "test_labels_out.npy", labels)  # sanity check vs test_labels.npy (same order)

    test_mcc = mcc_at_paper_threshold(labels, scores)

    metrics = dict(
        val_auc=float(val_auc), test_auc=float(test_auc),
        test_ap=float(test_ap), test_f1=float(test_f1), test_mcc=float(test_mcc),
        train_seconds=elapsed,
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{dataset} seed={seed}] val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  "
          f"test_ap={test_ap:.4f}  test_f1={test_f1:.4f}  test_mcc={test_mcc:.4f}  "
          f"({elapsed:.1f}s)  -> {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        "Train NeuTraL-AD on SSAE-split data (paper's own unmodified tabular model/trainer)"
    )
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    failed = []
    for dataset in datasets:
        for seed in seeds:
            print(f"\n=== {dataset} seed={seed} per={args.per} ===")
            try:
                run_one(dataset, seed, args.per, args.config, device)
            except Exception as e:
                import traceback
                traceback.print_exc()
                failed.append((dataset, seed, str(e)))

    if failed:
        print(f"\n[WARN] {len(failed)} run(s) failed:")
        for dataset, seed, err in failed:
            print(f"    {dataset} seed={seed}: {err}")


if __name__ == "__main__":
    main()
