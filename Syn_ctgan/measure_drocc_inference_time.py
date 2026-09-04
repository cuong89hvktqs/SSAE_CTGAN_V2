"""Measure DROCC's average inference time per test sample (ms), CPU,
batch_size=1 -- same protocol as CLAN-main/measure_inference_time.py (the
"Table 4" latency comparison this project already uses for
LOF/DIF/AE/SAE/VAE/SSAE/CLAN), so the numbers are directly comparable:
  - CPU only, sequential batch_size=1 (no batching amortizes the per-sample
    cost away).
  - WARMUP=50 samples run first and excluded from the timed average.
  - Up to n_time=5000 random test samples timed (bounded runtime).
  - time.perf_counter() around each single-sample scoring call.
  - Data preprocessing/standardization is NOT part of the timed cost --
    exactly like CLAN's script pre-scales the whole test set via
    get_data()'s MinMaxScaler before the timing loop and only times
    `model(x) -> cosine similarity`, this script pre-standardizes the whole
    test set with DROCC's own (train-mean, train-std) ONCE beforehand and
    only times `model(x) -> 1 - sigmoid(logit)` (the exact scoring formula
    from eval_drocc.py's anomaly_score(), minus its chunking loop -- here
    each call is a true batch_size=1 forward pass).

Loads the --per 1.0 checkpoint (log_ssae/per_1.0/<dataset>/seed<seed>/
model_epilepsy.pt) by default, matching the main/full-data comparison point
against SSAE/CLAN. Does not touch DROCC_icml2020/ -- same read-only
checkpoint-loading approach as eval_drocc.py.

Usage: python measure_drocc_inference_time.py [--datasets unsw,nsl_kdd,cic_iot,wsn_ds]
                                                [--seed 42] [--per 1.0] [--n_time 5000]
Output: DROCC_icml2020/tabular/results_ssae/drocc_inference_time.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as T

from clan_style_get_data import DATASETS, SAMPLE_THRES, get_data

DROCC_ROOT = Path(__file__).parent.parent / "DROCC_icml2020"
DROCC_TABULAR = DROCC_ROOT / "tabular"

if str(DROCC_TABULAR) not in sys.path:
    sys.path.insert(0, str(DROCC_TABULAR))
from models import TwoLayer  # noqa: E402  (DROCC's own, unmodified)

HD = 128
WARMUP = 50

DATASET_LABELS = {"unsw": "UNSW-NB15", "nsl_kdd": "NSL-KDD", "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS"}


@T.no_grad()
def _score_one(model, x_row):
    logit = model(x_row)
    return 1.0 - T.sigmoid(logit)


def measure(name, seed, per, n_time, device):
    data_dir = DROCC_TABULAR / "data_ssae" / f"per_{per}" / name / f"seed{seed}"
    ckpt = DROCC_TABULAR / "log_ssae" / f"per_{per}" / name / f"seed{seed}" / "model_epilepsy.pt"
    train_path = data_dir / "train_data.npy"
    if not ckpt.exists() or not train_path.exists():
        print(f"  [skip] missing checkpoint or train_data.npy under per_{per}/{name}/seed{seed}")
        return None

    train_data = np.load(train_path)
    mean, std = train_data.mean(0), train_data.std(0)

    cfg = DATASETS[name]
    print("  loading test data (excluded from timing)...")
    _, _, _, _, x_te, y_te, x_zd, y_zd, _, _ = get_data(
        data_path=str(cfg["data_path"]), target=cfg["target"], drop=cfg["drop_cols"],
        class_zero=cfg["class_zero"], sample_thres=SAMPLE_THRES, split_seed=seed,
        test_ratio=0.2, val_ratio=0.2, anomaly_detection=True, n_samples=cfg["total_rows"],
        sample_seed=seed,
    )
    if x_zd is not None:
        x_te = np.concatenate([x_te, x_zd])

    # Standardization done ONCE for the whole test set, same as CLAN's script
    # pre-scaling via get_data() before timing starts -- not part of the
    # per-sample cost being measured (see module docstring).
    x_te_std = (x_te - mean) / (std + 1e-4)
    x_te_t = T.tensor(x_te_std, dtype=T.float32, device=device)

    print("  loading model (cpu)...")
    model = TwoLayer(input_dim=x_te_t.shape[1], num_classes=1, num_hidden_nodes=HD)
    state = T.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    rng = np.random.default_rng(seed)
    n_test = x_te_t.shape[0]
    n_time_eff = min(n_time, n_test)
    idx = rng.choice(n_test, size=n_time_eff, replace=False)

    print(f"  warm-up ({WARMUP} samples)...")
    for i in idx[:WARMUP]:
        _score_one(model, x_te_t[i:i + 1])

    print(f"  timing ({n_time_eff} samples, batch_size=1, cpu)...")
    times_ms = []
    for i in idx:
        t0 = time.perf_counter()
        _score_one(model, x_te_t[i:i + 1])
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.asarray(times_ms)
    print(f"  mean={times_ms.mean():.4f} ms  std={times_ms.std():.4f} ms  "
          f"(over {n_time_eff}/{n_test} test samples)")
    return dict(
        Dataset=DATASET_LABELS.get(name, name), seed=seed, n_test=n_test, n_timed=n_time_eff,
        DROCC=round(float(times_ms.mean()), 3),
        DROCC_std=round(float(times_ms.std()), 3),
        times_ms=times_ms,
    )


def main():
    parser = argparse.ArgumentParser("measure DROCC per-sample inference time (ms, batch=1)")
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per", type=str, default="1.0", help="must match the --per used to train the checkpoint")
    parser.add_argument("--n_time", type=int, default=5000,
                        help="max number of test samples to time per dataset (random subsample, bounded runtime)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default="results_ssae/drocc_inference_time.csv")
    args = parser.parse_args()

    T.set_num_threads(T.get_num_threads())  # explicit: default intra-op thread count, matches typical CPU deployment
    device = args.device

    names = args.datasets.split(",")
    rows = []
    for name in names:
        print(f"\n[{name}  seed={args.seed}]")
        row = measure(name, args.seed, args.per, args.n_time, device)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nnothing measured - no checkpoints found")
        return

    df = pd.DataFrame(rows)[["Dataset", "DROCC", "DROCC_std", "seed", "n_test", "n_timed"]]
    out_path = DROCC_TABULAR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(df.to_string(index=False))

    pooled = np.concatenate([r["times_ms"] for r in rows])
    print(f"\n[POOLED, device={device}] n_total={len(pooled)}  "
          f"mean={pooled.mean():.4f} ms  std={pooled.std():.4f} ms")


if __name__ == "__main__":
    main()
