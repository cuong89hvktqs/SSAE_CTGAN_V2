"""Measure each DeepOD algorithm's average inference time per test sample
(ms), CPU, batch_size=1 -- same protocol as measure_drocc_inference_time.py
/ CLAN-main/measure_inference_time.py (this project's existing "Table 4"
latency comparison), so the numbers are directly comparable:
  - CPU only, sequential batch_size=1 (no batching amortizes the per-sample
    cost away).
  - WARMUP=50 samples run first and excluded from the timed average.
  - Up to n_time=5000 random test samples timed (bounded runtime).
  - time.perf_counter() around each single-sample decision_function() call.
  - Data preprocessing is NOT part of the timed cost -- train/val/test.npy
    are already one-hot/z-normalized (prepare_neutralad_data.py's output,
    reused as-is); only `model.decision_function(x) -> score` is timed.

Trains a fresh model per (algo, dataset) on GPU (fast), then moves it to
CPU before timing, since run_deepod.py never persisted the fitted models to
disk (only scores/metrics) -- the exact checkpoint doesn't matter for a
latency measurement, only the architecture/hyperparameters, which are
identical to what was actually evaluated (same defaults, same --per/seed).

KNOWN CAVEAT -- GOAD only: DeepOD's own GOAD.inference_prepare()
(deepod/models/tabular/goad.py, unmodified) recomputes `rep_means` by
re-running a forward pass over the ENTIRE training set on every single
decision_function() call, regardless of how many rows are being scored --
this is a real inefficiency in the vendored implementation, not something
this script works around. Timed under the same batch_size=1 sequential
protocol as the other 4 algorithms, GOAD's reported per-sample time
therefore also includes a full training-set forward pass on every call and
will look wildly inflated / not representative of a properly cached
deployment (where rep_means would be computed once after training). Do not
directly compare GOAD's number to the other algorithms' without reading
this note.

Usage: python measure_deepod_inference_time.py [--algos dsvdd,goad,neutral,icl,dif]
                                                 [--datasets unsw,nsl_kdd,cic_iot,wsn_ds]
                                                 [--seed 42] [--per 1.0] [--n_time 5000]
Output: DeepOD/results_ssae/inference_time_ms.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as T

BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from run_deepod import DEEPOD_ROOT, get_algo_class, load_npy  # noqa: E402

DATASET_LABELS = {"unsw": "UNSW-NB15", "nsl_kdd": "NSL-KDD", "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS"}
ALGO_NAMES = ["dsvdd", "goad", "neutral", "icl", "dif"]


def _move_to_cpu(clf):
    """Best-effort move of every torch-resident attribute to CPU so timing
    reflects CPU-only inference (matches this project's DROCC/CLAN protocol).
    Purely a device transfer -- no algorithm logic touched."""
    clf.device = "cpu"
    net = getattr(clf, "net", None)
    if isinstance(net, T.nn.Module):
        net.to("cpu")
    c = getattr(clf, "c", None)
    if isinstance(c, T.Tensor):
        clf.c = c.to("cpu")
    net_lst = getattr(clf, "net_lst", None)
    if net_lst:
        clf.net_lst = [n.to("cpu") if isinstance(n, T.nn.Module) else n for n in net_lst]


def measure(algo, dataset, seed, per, n_time, warmup):
    train, val, test, test_labels = load_npy(dataset, seed, per)

    algo_cls = get_algo_class(algo)
    device = "cuda" if T.cuda.is_available() else "cpu"
    kwargs = dict(device=device, random_state=seed, verbose=0)
    if algo == "dif":
        kwargs["n_jobs"] = 4
    clf = algo_cls(**kwargs)

    print("  fitting (gpu, for speed)...")
    clf.fit(train)

    print("  moving model to cpu for timing...")
    _move_to_cpu(clf)

    rng = np.random.default_rng(seed)
    n_test = len(test)
    n_time_eff = min(n_time, n_test)
    idx = rng.choice(n_test, size=n_time_eff, replace=False)

    print(f"  warm-up ({warmup} samples)...")
    for i in idx[:warmup]:
        clf.decision_function(test[i:i + 1])

    print(f"  timing ({n_time_eff} samples, batch_size=1, cpu)...")
    times_ms = []
    for i in idx:
        t0 = time.perf_counter()
        clf.decision_function(test[i:i + 1])
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.asarray(times_ms)
    print(f"  mean={times_ms.mean():.4f} ms  std={times_ms.std():.4f} ms  "
          f"(over {n_time_eff}/{n_test} test samples)")
    return dict(
        Algo=algo, Dataset=DATASET_LABELS.get(dataset, dataset), seed=seed,
        n_test=n_test, n_timed=n_time_eff,
        mean_ms=round(float(times_ms.mean()), 4),
        std_ms=round(float(times_ms.std()), 4),
        times_ms=times_ms,
    )


def main():
    parser = argparse.ArgumentParser("Measure DeepOD per-sample inference time (ms, cpu, batch=1)")
    parser.add_argument("--algos", type=str, default=",".join(ALGO_NAMES))
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--n_time", type=int, default=5000,
                        help="max number of test samples to time per (algo, dataset)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="warm-up calls before timing starts (excluded from the average)")
    parser.add_argument("--out", type=str, default="inference_time_ms.csv")
    args = parser.parse_args()

    algos = args.algos.split(",")
    datasets = args.datasets.split(",")

    rows = []
    for algo in algos:
        for dataset in datasets:
            print(f"\n[{algo}  {dataset}  seed={args.seed}]")
            try:
                row = measure(algo, dataset, args.seed, args.per, args.n_time, args.warmup)
                rows.append(row)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [WARN] {algo} {dataset}: {e}")

    if not rows:
        print("\nnothing measured")
        return

    times_by_algo = {}
    for r in rows:
        times_by_algo.setdefault(r["Algo"], []).append(r.pop("times_ms"))

    df = pd.DataFrame(rows)[["Algo", "Dataset", "mean_ms", "std_ms", "seed", "n_test", "n_timed"]]
    out_path = DEEPOD_ROOT / "results_ssae" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        prev = pd.read_csv(out_path)
        prev = prev[~(prev["Algo"].isin(algos) & prev["Dataset"].isin(df["Dataset"]))]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(df.to_string(index=False))
    print("\nNote: GOAD's numbers include a full training-set forward pass on every "
          "call (see script docstring) -- not directly comparable to the other 4.")

    print("\n[POOLED per algo, device=cpu] (all datasets' timed samples concatenated)")
    for algo, chunks in times_by_algo.items():
        pooled = np.concatenate(chunks)
        print(f"  {algo:8s} n_total={len(pooled):5d}  mean={pooled.mean():.4f} ms  std={pooled.std():.4f} ms")


if __name__ == "__main__":
    main()
