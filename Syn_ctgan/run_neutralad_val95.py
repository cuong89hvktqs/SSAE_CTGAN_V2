"""NeuTraL-AD, evaluated at a validation-calibrated threshold instead of the
paper's own oracle-ratio one -- separate output tree from run_neutralad.py's
(NeuTraL-AD/results_ssae_val95/, not results_ssae/), same training run
otherwise (identical model/loss/trainer, identical seed control, identical
data from prepare_neutralad_data.py).

Why this exists: the paper's own utils.compute_pre_recall_f1() (used for its
reported test_f1, and for the MCC aggregate_neutralad_overall.py added at
the same cut) sets its threshold from
    normal_ratio = (y_test == 0).sum() / len(y_test)
    threshold = percentile(scores, 100 * normal_ratio)
i.e. it needs the TRUE normal/anomaly ratio of the test set to pick an
operating point -- an oracle assumption no real deployment has (you don't
know in advance what fraction of live traffic is an attack). SSAE's own
reported F1/MCC/etc. (test_mode_total95.csv) instead calibrate purely from a
held-out NORMAL validation split's 95th-percentile score -- realistic, since
it only assumes access to traffic you already trust as clean, not the test
set's contamination rate. This script reproduces that SAME convention for
NeuTraL-AD, using val_data.npy (prepare_neutralad_data.py's genuine
normal-only held-out split, analogous to SSAE's own 80/20 val).

How the val-calibrated threshold is obtained without editing NeuTraL-AD's
own code: NeutralAD_trainer.train() (models/NeutralAD_trainer.py, untouched)
never exposes raw per-sample validation scores -- only aggregate val_auc/
val_f1 -- so its own train() method cannot be reused as-is here. Instead,
train_with_val_scores() below reimplements train()'s epoch loop 1:1 (same
calls in the same order: _train -> scheduler.step -> detect_outliers(test)
-> detect_outliers(val) -> epoch>5 Patience check) using ONLY its public,
unmodified _train()/detect_outliers() methods and its own unmodified
Patience class for model-selection -- the loop's control flow and every
number it produces (train loss, test/val AUC/F1, which epoch is "best") are
identical to calling trainer.train() directly. The only addition is
snapshotting the validation set's raw scores at whichever epoch Patience
selects as best, since that is the one piece train() itself discards.

Usage: python run_neutralad_val95.py [--datasets ...] [--seeds 42,43,44] [--per 1.0]
Output: NeuTraL-AD/results_ssae_val95/per_<per>/<dataset>/seed<N>/
            test_scores.npy, test_labels_out.npy, val_scores.npy,
            metrics_val95.json   (Threshold, AUC, F1, MCC, Precision, Recall, ACC
                                   -- same column set as SSAE's test_mode_total95.csv)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from run_neutralad import DEFAULT_CONFIG, flatten_config, load_dataset_tuple, set_seed  # noqa: E402

NEUTRALAD_ROOT = BASE_DIR.parent / "NeuTraL-AD"
if str(NEUTRALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUTRALAD_ROOT))
from config.base import Config              # noqa: E402  (NeuTraL-AD's own, unmodified)
from models.NeutralAD_trainer import NeutralAD_trainer  # noqa: E402
from utils import Logger                    # noqa: E402

RESULTS_ROOT = NEUTRALAD_ROOT / "results_ssae_val95"


def train_with_val_scores(flat_config, dataset_tuple):
    """See module docstring -- 1:1 reimplementation of
    NeutralAD_trainer.train()'s epoch loop via its own public
    _train()/detect_outliers() methods and its own Patience class, plus a
    parallel snapshot of the validation set's raw scores at the best epoch."""
    train_data, val_data, test_data = dataset_tuple
    cfg = Config.from_dict(flat_config)

    batch_size = cfg["batch_size"]
    drop = len(train_data) > batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=cfg["shuffle"], drop_last=drop)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    x_dim = train_data.dim_features
    model = cfg.model(cfg.network(), x_dim, config=cfg)
    optimizer = cfg.optimizer(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["l2"])
    scheduler = cfg.scheduler(optimizer) if cfg.scheduler is not None else None
    trainer = NeutralAD_trainer(model, loss_function=cfg.loss(cfg["loss_temp"]), device=cfg["device"])
    early_stopper = cfg.early_stopper() if cfg.early_stopper is not None else None

    best_val_scores, best_val_labels = None, None
    train_loss = val_auc = testin_loss = test_auc = test_ap = test_f1 = None
    score = target = val_score = val_target = None

    for epoch in range(1, cfg["training_epochs"] + 1):
        train_loss = trainer._train(train_loader, optimizer)
        if scheduler is not None:
            scheduler.step()

        test_auc, test_ap, test_f1, testin_loss, testout_loss, score, target = \
            trainer.detect_outliers(test_loader, None)
        val_auc, val_ap, val_f1, valin_loss, valout_loss, val_score, val_target = \
            trainer.detect_outliers(val_loader, None)

        if epoch > 5 and early_stopper is not None:
            stop = early_stopper.stop(epoch, valin_loss, val_auc, testin_loss, test_auc,
                                       test_ap, test_f1, train_loss, score, target)
            if early_stopper.best_epoch == epoch:  # this epoch just became the new best
                best_val_scores, best_val_labels = val_score, val_target
            if stop:
                break

    if early_stopper is not None and early_stopper.best_epoch != -1:
        (_, _, val_auc, _, test_auc, test_ap, test_f1, best_epoch, score, target) = \
            early_stopper.get_best_vl_metrics()
    else:
        best_epoch = cfg["training_epochs"]
        best_val_scores, best_val_labels = val_score, val_target

    return dict(
        best_epoch=best_epoch, val_auc=val_auc, test_auc=test_auc, test_ap=test_ap, test_f1=test_f1,
        test_scores=np.asarray(score), test_labels=np.asarray(target),
        val_scores=np.asarray(best_val_scores), val_labels=np.asarray(best_val_labels),
    )


def metrics_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Threshold": float(threshold),
        "AUC": roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "MCC": matthews_corrcoef(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "ACC": accuracy_score(y_true, y_pred) * 100,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def run_one(dataset, seed, per, config_path, device):
    flat_config = flatten_config(config_path, device)
    set_seed(seed)
    dataset_tuple = load_dataset_tuple(dataset, seed, per)

    out_dir = RESULTS_ROOT / f"per_{per}" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = train_with_val_scores(flat_config, dataset_tuple)
    elapsed = time.time() - t0

    test_scores, test_labels = result["test_scores"], result["test_labels"]
    val_scores, val_labels = result["val_scores"], result["val_labels"]
    assert (val_labels == 0).all(), "val split should be normal-only"

    val_threshold = float(np.percentile(val_scores, 95))
    m = metrics_at_threshold(test_labels, test_scores, val_threshold)

    np.save(out_dir / "test_scores.npy", test_scores)
    np.save(out_dir / "test_labels_out.npy", test_labels)
    np.save(out_dir / "val_scores.npy", val_scores)

    metrics = dict(
        best_epoch=int(result["best_epoch"]), val_auc=float(result["val_auc"]) if result["val_auc"] is not None else None,
        paper_test_auc=float(result["test_auc"]), paper_test_f1=float(result["test_f1"]),
        train_seconds=elapsed, **m,
    )
    with open(out_dir / "metrics_val95.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{dataset} seed={seed}] best_epoch={metrics['best_epoch']}  "
          f"val95_threshold={val_threshold:.4f}  AUC={m['AUC']:.2f}  F1={m['F1']:.2f}  "
          f"MCC={m['MCC']:.2f}  Precision={m['Precision']:.2f}  Recall={m['Recall']:.2f}  "
          f"ACC={m['ACC']:.2f}  ({elapsed:.1f}s)  -> {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        "Train NeuTraL-AD (paper's own unmodified model/trainer) and evaluate at a "
        "validation-95th-percentile threshold (SSAE's own total95 convention) instead "
        "of the paper's oracle test-ratio threshold."
    )
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    failed = []
    for dataset in datasets:
        for seed in seeds:
            print(f"\n=== {dataset} seed={seed} per={args.per} (val95) ===")
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
