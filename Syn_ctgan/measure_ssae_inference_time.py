"""Measure SSAE's (MAE01ScaleMax) average inference time per test sample
(ms), CPU, batch_size=1 -- same protocol as measure_drocc_inference_time.py
/ measure_deepod_inference_time.py / CLAN-main/measure_inference_time.py
(this project's "Table 4" latency comparison), so all 3 are directly
comparable:
  - CPU only, sequential batch_size=1 (no batching amortizes the per-sample
    cost away).
  - WARMUP=50 samples run first and excluded from the timed average.
  - Up to n_time=5000 random test samples timed (bounded runtime).
  - time.perf_counter() around each single-sample forward pass.
  - Preprocessing excluded from the timed loop, done ONCE beforehand.

Checkpoint: MAE_<dataset>_best_auc.pkl (best val-AUC), matching the
checkpoint-selection convention used everywhere else in this project
(val_perattack_mae01scale_max.py) -- this is also exactly the model behind
SSAE's own val95/total95 threshold (threshold_95_total, the 95th-percentile-
of-validation-normal-scores convention already used for every reported SSAE
result), so "SSAE with val95" here just means: the one checkpoint/scoring
path SSAE has always used -- inference latency itself does not depend on
which threshold is applied afterward (scoring and thresholding are separate
steps; only the score computation is timed here).

"Preprocessing" for SSAE happens in TWO stages, both excluded from the
timed loop by doing them once, up front, exactly like the SDV wrapper's own
get_score(X_processed, y) does internally (ctgan/synthesizers/m01aectgan.py):
  1. SDV's HyperTransformer (raw dataframe -> SDV-numeric space) -- already
     baked into the saved <dataset>_X_test.csv (see train_mae01scale_max.py:
     `X_test = synthesizer._data_processor.transform(X_test_raw)`), so
     simply loading that CSV already skips this stage.
  2. CTGAN's own internal DataTransformer (SDV-numeric -> CTGAN's
     mode-specific-normalized / one-hot space) -- done once via
     `model._transformer.transform(X_test)` OUTSIDE the timed loop, so only
     the actual neural forward pass + score arithmetic
     (`_mae_forward` -> reconstruction error + lambda*||z||^2`) is timed,
     replicating get_score()'s own per-batch body but skipping its
     transform() call.

Usage: python measure_ssae_inference_time.py [--datasets unsw,nsl_kdd,cic_iot,wsn_ds]
                                               [--seed 42] [--per 1.0] [--n_time 5000]
Output: Syn_ctgan/res/inference_time_ms_ssae.csv
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))  # prefer local ctgan/ and sdv/
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from val_perattack_mae01scale_max import load_synthesizer  # noqa: E402

RES_DIR = BASE_DIR / "res"
WARMUP = 50
TIME_TAG = "04"  # main default-config run (m2.0/l0.1/noise_std) -- same as
                 # visualize_anomaly_score_distribution.py's TIME_TAG, NOT the
                 # generic find_run_dirs() glob, which also matches leftover
                 # ablation-sweep (t_02/t_03) directories under other margins.


def find_run_dir(dataset, seed, per):
    pattern = str(RES_DIR / f"mae01scale_max_se_{seed}_t_{TIME_TAG}" / f"per_{per}"
                  / f"{dataset}_*scalemax-noise_std_*")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None
DATASET_LABELS = {"unsw": "UNSW-NB15", "nsl_kdd": "NSL-KDD", "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS"}


@torch.no_grad()
def _score_one(model, x_row):
    """Replicates get_score()'s per-batch body (ctgan/synthesizers/
    m01aectgan.py) for a single already-CTGAN-transformed row, skipping its
    internal self._transformer.transform(X) call (done once, outside)."""
    z, x_hat = model._mae_forward(x_row)
    re = torch.nn.functional.mse_loss(x_hat, x_row, reduction="none").mean(dim=1)
    z_norm = torch.norm(z, p=2, dim=1)
    total = re + model.lamda * z_norm ** 2
    return total


def measure(dataset, seed, per, n_time, device):
    run_dir = find_run_dir(dataset, seed, per)
    if run_dir is None:
        print(f"  [skip] no run found for {dataset} seed={seed} per={per}")
        return None

    x_test_path = os.path.join(run_dir, f"{dataset}_X_test.csv")
    model_path = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")
    if not os.path.exists(x_test_path) or not os.path.exists(model_path):
        print(f"  [skip] missing X_test.csv or checkpoint under {run_dir}")
        return None

    print("  loading model + test data (excluded from timing)...")
    synthesizer = load_synthesizer(model_path)
    model = synthesizer._model
    model._mae_enc.eval()
    model._mae_dec.eval()
    model.set_device(device)

    X_test = pd.read_csv(x_test_path)  # already SDV-HyperTransformer-space

    # CTGAN's own internal DataTransformer -- done ONCE, excluded from timing,
    # exactly like get_score()'s internal transform() call, just hoisted out.
    X_t = model._transformer.transform(X_test)
    X_t = torch.tensor(X_t, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    n_test = X_t.shape[0]
    n_time_eff = min(n_time, n_test)
    idx = rng.choice(n_test, size=n_time_eff, replace=False)

    print(f"  warm-up ({WARMUP} samples)...")
    for i in idx[:WARMUP]:
        _score_one(model, X_t[i:i + 1])

    print(f"  timing ({n_time_eff} samples, batch_size=1, {device})...")
    times_ms = []
    for i in idx:
        t0 = time.perf_counter()
        _score_one(model, X_t[i:i + 1])
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.asarray(times_ms)
    print(f"  mean={times_ms.mean():.4f} ms  std={times_ms.std():.4f} ms  "
          f"(over {n_time_eff}/{n_test} test samples)")
    return dict(
        Dataset=DATASET_LABELS.get(dataset, dataset), seed=seed, n_test=n_test, n_timed=n_time_eff,
        mean_ms=round(float(times_ms.mean()), 4),
        std_ms=round(float(times_ms.std()), 4),
        times_ms=times_ms,
    )


def main():
    parser = argparse.ArgumentParser("Measure SSAE per-sample inference time (ms, cpu, batch=1)")
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per", type=str, default="1.0")
    parser.add_argument("--n_time", type=int, default=5000,
                        help="max number of test samples to time per dataset")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="CPU only, to match the DROCC/DeepOD/CLAN latency protocol")
    parser.add_argument("--out", type=str, default="inference_time_ms_ssae.csv")
    args = parser.parse_args()

    names = args.datasets.split(",")
    rows = []
    for name in names:
        print(f"\n[{name}  seed={args.seed}]")
        row = measure(name, args.seed, args.per, args.n_time, args.device)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nnothing measured - no checkpoints found")
        return

    df = pd.DataFrame(rows)[["Dataset", "mean_ms", "std_ms", "seed", "n_test", "n_timed"]]
    out_path = RES_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(df.to_string(index=False))

    # Pooled overall average across ALL datasets' individual timed samples
    # (not an average-of-per-dataset-means): concatenate every per-sample
    # timing from every dataset processed in this run into one array.
    pooled = np.concatenate([r["times_ms"] for r in rows])
    print(f"\n[POOLED, device={args.device}] n_total={len(pooled)}  "
          f"mean={pooled.mean():.4f} ms  std={pooled.std():.4f} ms")


if __name__ == "__main__":
    main()
