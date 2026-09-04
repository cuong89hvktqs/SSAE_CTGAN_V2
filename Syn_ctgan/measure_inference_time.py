"""Measure average CPU inference time (ms/sample) for the SSAE (MAE01ScaleMax,
type_train=noise_std) model, seed=42 only, on unsw / nsl_kdd / cic_iot / wsn_ds.

Loads each run's best-val-AUC checkpoint, forces it onto CPU. The CTGAN
DataTransformer.transform() step (VGM mode-specific normalization for
continuous columns) is run ONCE outside the timed region -- it dominates
get_score()'s wall time but isn't part of "model inference" in the sense
comparable to LOF/DIF/AE/VAE (simpler baselines with no such transform
step). What's timed is only the encoder/decoder forward pass +
reconstruction-error/‖z‖ scoring, over the full saved test set, repeated
several times after a warm-up pass.

Output: TH/inference_time_seed42.xlsx  (Dataset | Samples | SSAE (ms/sample))
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cloudpickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "res")

SEED = 42
DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
N_REPEATS = 5


def load_synthesizer_cpu(model_path):
    """Load a pickled synthesizer forcing every tensor onto CPU."""
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": "cpu"})
    try:
        with open(model_path, "rb") as f:
            synthesizer = cloudpickle.load(f)
    finally:
        torch.load = _orig_load
    synthesizer._model.set_device("cpu")
    return synthesizer


def find_run_dir(dataset):
    pattern = os.path.join(
        RES_DIR, f"mae01scale_max_se_{SEED}_t_*", "per_1.0",
        f"{dataset}_*scalemax-noise_std_*",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No seed={SEED} noise_std run found for {dataset}: {pattern}")
    return matches[0]


def _score_forward_only(model, X_tensor, y_tensor):
    """Reproduces get_score()'s loop body exactly, minus the DataTransformer
    step -- X_tensor is already CTGAN-transformed, computed once beforehand."""
    model._mae_enc.eval()
    model._mae_dec.eval()
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=model._batch_size, shuffle=False,
    )
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(model._device)
            z, xb_hat = model._mae_forward(xb)
            re = F.mse_loss(xb_hat, xb, reduction="none").mean(dim=1)
            z_norm = torch.norm(z, p=2, dim=1)
            total = re + model.lamda * z_norm ** 2
            _ = total.cpu().numpy()


def measure_dataset(dataset):
    run_dir = find_run_dir(dataset)
    model_path = os.path.join(run_dir, "checkpoints", f"MAE_{dataset}_best_auc.pkl")
    synthesizer = load_synthesizer_cpu(model_path)
    model = synthesizer._model

    X_test = pd.read_csv(os.path.join(run_dir, f"{dataset}_X_test.csv"))
    y_test = pd.read_csv(os.path.join(run_dir, f"{dataset}_y_test.csv"))["label"]
    n_samples = len(X_test)

    # DataTransformer step run ONCE, outside the timed region.
    X_t = model._transformer.transform(X_test)
    X_tensor = torch.tensor(X_t, dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y_test, dtype=np.float32), dtype=torch.float32)

    # warm-up (excluded from timing)
    _score_forward_only(model, X_tensor, y_tensor)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _score_forward_only(model, X_tensor, y_tensor)
        times.append(time.perf_counter() - t0)

    total_s = float(np.mean(times))
    ms_per_sample = total_s / n_samples * 1000.0
    print(f"{dataset}: n={n_samples}  mean_total={total_s:.4f}s "
          f"(over {N_REPEATS} runs)  ->  {ms_per_sample:.4f} ms/sample")
    return n_samples, ms_per_sample


def main():
    torch.set_grad_enabled(False)
    rows = []
    for dataset in DATASETS:
        print(f"\n=== {dataset} ===")
        n_samples, ms = measure_dataset(dataset)
        rows.append({"Dataset": dataset.upper(), "Samples": n_samples, "SSAE (ms/sample)": round(ms, 3)})

    df = pd.DataFrame(rows)
    out_path = os.path.join(BASE_DIR, "TH", "inference_time_seed42.xlsx")
    df.to_excel(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(df)


if __name__ == "__main__":
    main()
