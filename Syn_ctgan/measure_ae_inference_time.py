"""Measure the AE baseline's (D:\\Code_papper\\CTGAN\\TestDT_VChuan\\models\\ae.py)
average inference time per test sample (ms), CPU, batch_size=1 -- same
protocol as measure_drocc_inference_time.py / measure_deepod_inference_time.py
/ measure_ssae_inference_time.py, so all are directly comparable:
  - CPU only, sequential batch_size=1.
  - WARMUP=50 samples run first and excluded from the timed average.
  - n_time=5000 samples timed.
  - time.perf_counter() around each single-sample forward pass.
  - Preprocessing excluded from the timed loop entirely.

Loads REAL trained weights (ae_model.pt, {'model_state_dict': ...}) from
this project's already-run AE checkpoints in TestDT_VChuan/results/ -- but
that tree is a messy pile of ad-hoc experiment folders (different
optimizers, epoch counts, percentage subfolders) and does NOT have a clean
per_1.0/<dataset>/seed42 checkpoint for all 4 datasets the way SSAE/DROCC/
DeepOD do (e.g. no cic_iot checkpoint anywhere under percentage_1.0/ at
all, no unsw/nsl_kdd seed42 under percentage_1.0/ either). CHECKPOINTS
below are simply the best/only available match per dataset across the
whole results/ tree (see the CHECKPOINTS dict) -- not necessarily the
seed42/per_1.0 run used for any accuracy comparison elsewhere.

Input data: AE's own n_features per dataset is recovered directly from the
checkpoint's enc.0.weight shape (196/122/46/16 for unsw/nsl_kdd/cic_iot/
wsn_ds -- each matching that dataset's own established one-hot-expanded
feature count). Rather than reproducing AE's own separate load_data.py
preprocessing pipeline just to get correctly-shaped real rows, this script
times the model on random CORRECTLY-SHAPED float tensors instead: a dense
MLP's forward-pass cost (matrix multiplies of fixed layer sizes) depends
only on tensor shape, never on the actual values inside it, so this does
not change the timing result at all while avoiding a second preprocessing
pipeline -- consistent with the instruction to exclude preprocessing from
the timed cost (there is none left to exclude here; the "preprocessing" is
simply never performed, which is equivalent for latency purposes).

Usage: python measure_ae_inference_time.py [--datasets unsw,nsl_kdd,cic_iot,wsn_ds]
                                             [--n_time 5000] [--seed 42]
Output: Syn_ctgan/res/inference_time_ms_ae.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AE_ROOT = Path(r"D:\Code_papper\CTGAN\TestDT_VChuan")
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))
from models.ae import AE  # noqa: E402  (TestDT_VChuan's own, unmodified)

BASE_DIR = Path(__file__).parent
RES_DIR = BASE_DIR / "res"
WARMUP = 50
DATASET_LABELS = {"unsw": "UNSW-NB15", "nsl_kdd": "NSL-KDD", "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS"}

# Best/only available checkpoint per dataset across TestDT_VChuan/results/
# (see module docstring -- no clean per_1.0/seed42-for-all-4-datasets set exists).
CHECKPOINTS = {
    "unsw":    AE_ROOT / "results" / "percentage_1.0" / "ae_unsw_epochs200_seed43" / "ae_model.pt",
    "nsl_kdd": AE_ROOT / "results" / "ada" / "ae_nsl_kdd_epochs200" / "ae_model.pt",
    "cic_iot": AE_ROOT / "results" / "ada" / "ae_cic_iot_epochs200" / "ae_model.pt",
    "wsn_ds":  AE_ROOT / "results" / "percentage_1.0" / "ae_wsn_ds_epochs100_seed42" / "ae_model.pt",
}


@torch.no_grad()
def _score_one(model, x_row):
    _, x_recon = model.forward(x_row)
    return torch.nn.functional.mse_loss(x_recon, x_row, reduction="none").mean(dim=1)


def measure(dataset, seed, n_time, device="cpu"):
    ckpt_path = CHECKPOINTS.get(dataset)
    if ckpt_path is None or not ckpt_path.exists():
        print(f"  [skip] no checkpoint found for {dataset} ({ckpt_path})")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    n_features = state_dict["enc.0.weight"].shape[1]

    model = AE(n_features).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  loaded {ckpt_path.relative_to(AE_ROOT)}  (n_features={n_features})")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x_all = torch.randn(n_time + WARMUP, n_features, device=device)  # see docstring: shape-only cost

    print(f"  warm-up ({WARMUP} samples)...")
    for i in range(WARMUP):
        _score_one(model, x_all[i:i + 1])

    print(f"  timing ({n_time} samples, batch_size=1, {device})...")
    times_ms = []
    for i in range(WARMUP, WARMUP + n_time):
        t0 = time.perf_counter()
        _score_one(model, x_all[i:i + 1])
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.asarray(times_ms)
    print(f"  mean={times_ms.mean():.4f} ms  std={times_ms.std():.4f} ms  (over {n_time} random samples)")
    return dict(
        Dataset=DATASET_LABELS.get(dataset, dataset), n_features=n_features,
        mean_ms=round(float(times_ms.mean()), 4), std_ms=round(float(times_ms.std()), 4),
        seed=seed, n_timed=n_time,
    )


def main():
    parser = argparse.ArgumentParser("Measure AE (TestDT_VChuan) per-sample inference time (ms, cpu, batch=1)")
    parser.add_argument("--datasets", type=str, default="unsw,nsl_kdd,cic_iot,wsn_ds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_time", type=int, default=5000)
    parser.add_argument("--out", type=str, default="inference_time_ms_ae.csv")
    args = parser.parse_args()

    names = args.datasets.split(",")
    rows = []
    for name in names:
        print(f"\n[{name}]")
        row = measure(name, args.seed, args.n_time)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nnothing measured - no checkpoints found")
        return

    df = pd.DataFrame(rows)[["Dataset", "mean_ms", "std_ms", "n_features", "seed", "n_timed"]]
    out_path = RES_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
