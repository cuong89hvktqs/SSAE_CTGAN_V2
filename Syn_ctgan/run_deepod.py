"""Trains DeepOD (github.com/xuhongzuo/DeepOD, unmodified detection algorithms
-- see module note below on the 2 files patched, both pure import-time fixes,
zero behavior change) on this project's 4 comparison datasets, using the
SAME pre-built .npy data NeuTraL-AD's own pipeline already produced
(NeuTraL-AD/DATA/ssae/per_<per>/<dataset>/seed<seed>/{train,val,test}_data.npy
+ test_labels.npy + test_attack_types.npy) -- the encoding (one-hot
categoricals + z-normalized numerics, fit on the normal train split, exactly
SSAE's own train/val/test row split) is algorithm-agnostic, so there is no
need for a separate prepare_deepod_data.py.

Algorithms run (DeepOD's own sklearn-style API: model.fit(X_train) ->
model.decision_function(X) -> raw scores, higher = more anomalous):
    dsvdd    Deep SVDD          (Ruff et al., ICML 2018)
    goad     GOAD               (Bergman & Hoshen, ICLR 2020)
    neutral  NeuTraL (DeepOD's own reimplementation, distinct from the
             standalone NeuTraL-AD/ repo already integrated separately)
    icl      ICL                (Shenkar & Wolf, ICLR 2022)
    dif      Deep Isolation Forest (Xu et al., TKDE 2023)
DeepSAD is intentionally excluded: it hard-requires >=1 labeled anomaly in
training (its sample-weighting divides by the count of known anomalies --
ZeroDivisionError on normal-only data), incompatible with this project's
normal-only training convention shared by every other baseline (SSAE, CLAN,
DROCC, NeuTraL-AD, and the other 4 DeepOD algorithms here). Per explicit
user decision, DeepSAD is skipped rather than given labeled attacks.

Hyperparameters: every algorithm's own class-level defaults are used
unmodified (DeepOD's own maintained "paper default" settings per algorithm,
e.g. DeepSVDD epochs=100/batch=64/rep_dim=128; GOAD n_trans=256/trans_dim=32;
NeuTraL n_trans=11/trans_type=residual; ICL n_ensemble=2) -- nothing tuned
for this project, matching the DROCC/NeuTraL-AD precedent of reusing each
baseline's own defaults rather than re-tuning them.

Threshold / F1 / MCC (the "paper default" reported here, separate from the
val95 variant in run_deepod_val95.py): DeepOD's own unsupervised convention
-- threshold_ = percentile(decision_scores_ON_TRAINING_DATA, 100*(1-contamination))
with contamination=0.1 (BaseDeepAD's own default) -- NOT derived from test
labels at all (unlike NeuTraL-AD's oracle test-ratio threshold), so there is
no label-leakage concern here; it just assumes ~10% of the (actually fully
clean) training pool is contaminated, per the library's own default.

Patched files (deepod/core/base_model.py, deepod/models/tabular/dsvdd.py,
deepod/models/time_series/{couta,tcned}.py): DeepOD's package __init__.py
eagerly imports every model (including time-series ones never used here),
several of which import Ray Tune (`from ray import tune`, `ray.air`) at
module level purely for an optional hyperparameter-search feature
(fit_auto_hyper/_training_ray/set_tuned_params) this project never calls.
The `ray` version resolvable in this environment has its own unrelated
pkg_resources/pydantic incompatibility that makes the package fail to
import at all. Fix: those 4 `from ray import ...` module-level imports were
moved to be local to the specific method that uses them (a pure import-time
relocation -- the training/scoring/network code paths actually exercised by
this project are byte-identical to upstream; only .fit_auto_hyper() /
._training_ray() / .set_tuned_params(), none of which this project calls,
would now additionally need `ray` importable if ever invoked).

Usage: python run_deepod.py --algo dsvdd [--datasets ...] [--seeds 42,43,44] [--per 1.0]
Output: DeepOD/results_ssae/<algo>/per_<per>/<dataset>/seed<N>/
            test_scores.npy, test_labels_out.npy, val_scores.npy, metrics.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import gc

import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)

BASE_DIR = Path(__file__).parent
DEEPOD_ROOT = BASE_DIR.parent / "DeepOD"
if str(DEEPOD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPOD_ROOT))

NEUTRALAD_ROOT = BASE_DIR.parent / "NeuTraL-AD"
DATA_ROOT = NEUTRALAD_ROOT / "DATA" / "ssae"  # reused as-is, algorithm-agnostic encoding
RESULTS_ROOT = DEEPOD_ROOT / "results_ssae"

ALGO_NAMES = ["dsvdd", "goad", "neutral", "icl", "dif"]


def _make_goad32_class():
    """GOAD's own training_prepare()/inference_prepare() (deepod/models/tabular/
    goad.py, untouched) build a full (n_samples, trans_dim, n_trans) array via
    `X.dot(rot)` before batching. `self.affine_weights = np.random.randn(...)`
    returns float64 regardless of X's dtype, so numpy silently promotes the
    WHOLE array to float64 even when X is float32 -- doubling memory and
    reliably OOM-crashing on cic_iot/wsn_ds's 200k+-row training pools
    (`_ArrayMemoryError: Unable to allocate 15.6 GiB ...`). Rather than
    editing GOAD's own file, this subclass only downcasts affine_weights to
    float32 right after GOAD's own training_prepare() builds them, so the
    matmul stays float32 throughout -- identical random transforms, identical
    math, half the memory. No override of any actual detection logic."""
    from deepod.models.tabular import GOAD as _GOAD

    class GOAD32(_GOAD):
        def training_prepare(self, X, y):
            train_loader, net, criterion = super().training_prepare(X, y)
            self.affine_weights = self.affine_weights.astype(np.float32)
            return train_loader, net, criterion

    return GOAD32


def get_algo_class(name):
    from deepod.models.tabular import DeepSVDD, NeuTraL, ICL, DeepIsolationForest
    return {
        "dsvdd": DeepSVDD, "goad": _make_goad32_class(), "neutral": NeuTraL,
        "icl": ICL, "dif": DeepIsolationForest,
    }[name]


def load_npy(dataset, seed, per):
    d = DATA_ROOT / f"per_{per}" / dataset / f"seed{seed}"
    for name in ("train_data.npy", "val_data.npy", "test_data.npy", "test_labels.npy"):
        if not (d / name).exists():
            raise FileNotFoundError(
                f"missing {d / name} -- run prepare_neutralad_data.py first "
                f"(--dataset {dataset} --seeds {seed} --per {per})"
            )
    # float32 (not float64): GOAD's own training_prepare() materializes a full
    # (n_samples, trans_dim, n_trans) array BEFORE batching -- float64 doubled
    # that to 15.6 GiB on cic_iot's 255728-row train pool, which reliably blew
    # past available RAM (see run log: numpy._core._exceptions._ArrayMemoryError
    # on cic_iot seed44 and all 3 wsn_ds seeds). float32 is standard precision
    # for neural net training anyway; this is a driver-side dtype choice, not
    # a change to any algorithm's own code.
    train = np.load(d / "train_data.npy").astype(np.float32)
    val = np.load(d / "val_data.npy").astype(np.float32)
    test = np.load(d / "test_data.npy").astype(np.float32)
    test_labels = np.load(d / "test_labels.npy").astype(int)
    return train, val, test, test_labels


def mcc_at_threshold(labels, scores, threshold):
    pred = (scores > threshold).astype(int)
    return matthews_corrcoef(labels, pred)


def metrics_at_threshold(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "MCC": matthews_corrcoef(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def run_one(algo, dataset, seed, per, device):
    algo_cls = get_algo_class(algo)
    train, val, test, test_labels = load_npy(dataset, seed, per)

    out_dir = RESULTS_ROOT / algo / f"per_{per}" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(device=device, random_state=seed, verbose=0)
    if algo == "dif":
        kwargs["n_jobs"] = 4  # isolation-forest fit only, safe to parallelize on CPU
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
    metrics = dict(threshold=float(clf.threshold_), fit_seconds=fit_seconds, **m)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{algo} {dataset} seed={seed} per={per}] AUC={m['AUC']:.2f}  F1={m['F1']:.2f}  "
          f"MCC={m['MCC']:.2f}  Precision={m['Precision']:.2f}  Recall={m['Recall']:.2f}  "
          f"({fit_seconds:.1f}s)  -> {out_dir}")

    # This process runs dozens of fit()s back-to-back without exiting; free the
    # model/arrays and clear the CUDA cache each time to keep RAM/VRAM from
    # accumulating across runs (see GOAD's cic_iot/wsn_ds OOM crashes).
    del clf, train, val, test, test_scores, val_scores
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser("Train a DeepOD algorithm on SSAE-split data")
    parser.add_argument("--algo", type=str, required=True, choices=ALGO_NAMES + ["all"])
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    algos = ALGO_NAMES if args.algo == "all" else [args.algo]
    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    failed = []
    for algo in algos:
        for dataset in datasets:
            for seed in seeds:
                print(f"\n=== {algo} | {dataset} seed={seed} per={args.per} ===")
                try:
                    run_one(algo, dataset, seed, args.per, device)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    failed.append((algo, dataset, seed, str(e)))

    if failed:
        print(f"\n[WARN] {len(failed)} run(s) failed:")
        for algo, dataset, seed, err in failed:
            print(f"    {algo} {dataset} seed={seed}: {err}")


if __name__ == "__main__":
    main()
