"""Stress-test pseudo-validation AUC at validation-time noise levels SMALLER
than each checkpoint's own (adaptive) training-time noise_std, to see which
m/lambda config's latent margin degrades first as the pseudo-anomaly task
gets harder (noise shrinks toward the real normal distribution).

Why: plain "Validation AUC" (last logged epoch, see build_val_auc_report.py)
saturates near 100% for almost every m/lambda, because it's validated with
the SAME noise magnitude/distribution the model was just trained to
separate -- not a generalization test. "Validation Separation" is circular
(tracks the margin target m almost by construction). "Validation RE Loss"
only rules out extreme values (see append_re_loss_val_auc.py), it can't
discriminate within e.g. m in [1,5].

This script reloads each already-trained MAE_<dataset>_best_auc.pkl
checkpoint (NO retraining, same load_synthesizer()/rebuild_val_test_split()
pipeline as eval_realistic_ratio_t04_seed42.py) and recomputes the SAME
pseudo-validation AUC metric (re-implementing
m01aectgan.py::_evaluate_pseudo_validation_auc's exact scoring logic against
the loaded model), but at noise_std = fraction * (checkpoint's own saved
noise_std), for fraction in NOISE_FRACTIONS. A checkpoint whose AUC survives
down to small fractions has a more robust/generalizing latent margin than
one that only separates well at its own (possibly oversized) native scale.

Data source (same convention as build_val_auc_report.py): t_04 (m=2.0,
lambda=0.1 baseline) supplies the shared baseline point for both sweeps;
t_02 is the margin sweep (lambda=0.1 fixed); t_01 is the lambda sweep
(m=2.0 fixed).

Output: a NEW standalone file res/stress_test_val_auc.xlsx (does not touch
val_auc.xlsx or summary_seeds42-51.xlsx), with sheets:
  raw      -- one row per (dataset, seed, sweep, value): native_noise_std +
              AUC at every tested fraction
  m        -- mean+-std across seeds, Dataset x m, one column per fraction
  lambda   -- mean+-std across seeds, Dataset x lambda, one column per fraction
  summary  -- 2 comparison tables (Dataset rows x m/lambda columns) at the
              HARDEST fraction only, to pick the most robust m / lambda

Usage:
    python stress_test_val_auc.py                        # full: seeds 42-51
    python stress_test_val_auc.py --seeds 42              # quick check
    python stress_test_val_auc.py --seeds 42,43 --out res/stress_test_quick.xlsx
"""
import argparse
import glob
import os
import sys
import zlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import build_seeds_summary_excel as B
from reeval_ssae_runs import rebuild_val_test_split
from val_perattack_mae01scale_max import load_synthesizer

RES_DIR = os.path.join(SCRIPT_DIR, "res")
MARGINS = ["1.0", "2.0", "3.0", "4.0", "5.0", "9.0"]
LAMBDAS = ["0.0", "0.01", "0.05", "0.1", "0.5", "1.0"]
NOISE_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]

_val_split_cache = {}


def first_match(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def get_cached_val_split(dataset, seed):
    key = (dataset, seed)
    if key not in _val_split_cache:
        X_val_raw, y_val, _, _ = rebuild_val_test_split(dataset, seed, 1.0)
        _val_split_cache[key] = (X_val_raw, y_val)
    return _val_split_cache[key]


def find_run_dir(time_tag, dataset, seed, m, lam):
    return first_match(os.path.join(
        RES_DIR, f"t_{time_tag}_SSAE_se_{seed}", "per_1.0",
        f"{dataset}_150e_m{m}_l{lam}_renormal_noise_std_*_catra0.1"))


def stress_test_one_checkpoint(time_tag, dataset, seed, m, lam):
    run_dir = find_run_dir(time_tag, dataset, seed, m, lam)
    if run_dir is None:
        return None
    model_path = os.path.join(run_dir, f"MAE_{dataset}_best_auc.pkl")
    if not os.path.isfile(model_path):
        return None

    X_val_raw, y_val = get_cached_val_split(dataset, seed)
    synth = load_synthesizer(model_path)
    model = synth._model

    X_val_processed = synth._data_processor.transform(X_val_raw)
    X_t = model._transformer.transform(X_val_processed)
    X_tensor = torch.tensor(X_t, dtype=torch.float32)
    y_zeros = torch.zeros(X_tensor.size(0))
    loader = DataLoader(TensorDataset(X_tensor, y_zeros),
                         batch_size=model._batch_size, shuffle=False)

    model._mae_enc.eval()
    model._mae_dec.eval()
    native_noise_std = float(model.noise_std)

    out = {"native_noise_std": native_noise_std}
    with torch.no_grad():
        for frac in NOISE_FRACTIONS:
            # Fixed seed per (dataset, training-seed, fraction) so the noise/CTGAN
            # draws are reproducible across repeated runs and comparable across
            # different m/lambda checkpoints (same RNG stream base). Python's
            # built-in hash() is randomized per-process for strings, so use a
            # stable crc32-based hash instead.
            seed_key = f"{dataset}|{seed}|{frac}".encode()
            torch.manual_seed(zlib.crc32(seed_key))
            model.noise_std = native_noise_std * frac
            normal_scores, pseudo_scores = [], []
            for xb, yb in loader:
                xb = xb.to(model._device)
                ctgan_sample = model._generate_ctgan_fakeact(xb.size(0))
                xb_pseudo = model._make_pseudo_anomaly(xb, ctgan_sample, True)
                z_n, _ = model._mae_forward(xb)
                z_p, _ = model._mae_forward(xb_pseudo)
                normal_scores.append(torch.norm(z_n, p=2, dim=1).cpu().numpy())
                pseudo_scores.append(torch.norm(z_p, p=2, dim=1).cpu().numpy())
            normal_scores = np.concatenate(normal_scores)
            pseudo_scores = np.concatenate(pseudo_scores)
            y = np.concatenate([np.zeros_like(normal_scores), np.ones_like(pseudo_scores)])
            scores = np.concatenate([normal_scores, pseudo_scores])
            auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) >= 2 else float("nan")
            out[f"AUC_frac_{frac}"] = auc

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(42, 52)))
    ap.add_argument("--out", default=os.path.join(RES_DIR, "stress_test_val_auc.xlsx"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    m_for_m = lambda v: v
    lam_for_m = lambda v: "0.1"
    tag_for_m = lambda v: "04" if v == "2.0" else "02"

    m_for_l = lambda v: "2.0"
    lam_for_l = lambda v: v
    tag_for_l = lambda v: "04" if v == "0.1" else "01"

    # Resumable checkpoint CSV: a full sweep can take hours, so every
    # successfully computed row is appended immediately. On restart, rows
    # already present in this file are skipped instead of recomputed.
    ckpt_path = os.path.join(RES_DIR, "stress_test_val_auc_raw_checkpoint.csv")
    done_keys = set()
    rows = []
    if os.path.isfile(ckpt_path):
        prev = pd.read_csv(ckpt_path)
        rows = prev.to_dict("records")
        done_keys = set(zip(prev["Sweep"], prev["Value"].astype(str), prev["Dataset"], prev["Seed"]))
        print(f"Resuming: {len(rows)} row(s) already in {ckpt_path}")
    missing = []

    def append_checkpoint_row(row):
        df1 = pd.DataFrame([row])
        write_header = not os.path.isfile(ckpt_path)
        df1.to_csv(ckpt_path, mode="a", header=write_header, index=False)

    def run_sweep(values, m_for, lam_for, tag_for, sweep_name):
        for ds in B.DATASETS:
            for seed in seeds:
                for v in values:
                    key = (sweep_name, str(v), B.DATASET_DISPLAY[ds], seed)
                    if key in done_keys:
                        continue
                    m, lam, tag = m_for(v), lam_for(v), tag_for(v)
                    try:
                        res = stress_test_one_checkpoint(tag, ds, seed, m, lam)
                    except Exception as e:
                        print(f"[ERROR] {sweep_name}={v}/{ds}/seed{seed}: {e}")
                        missing.append(f"{sweep_name}={v}/{ds}/seed{seed} (ERROR: {e})")
                        continue
                    if res is None:
                        missing.append(f"{sweep_name}={v}/{ds}/seed{seed}")
                        continue
                    row = {"Dataset": B.DATASET_DISPLAY[ds], "Seed": seed,
                           "Sweep": sweep_name, "Value": v, **res}
                    rows.append(row)
                    append_checkpoint_row(row)
                    print(f"[{sweep_name}={v}] {ds} seed={seed}: "
                          f"native_noise_std={res['native_noise_std']:.4f}  "
                          + "  ".join(f"f={f}:{res[f'AUC_frac_{f}']:.4f}" for f in NOISE_FRACTIONS))

    run_sweep(MARGINS, m_for_m, lam_for_m, tag_for_m, "m")
    run_sweep(LAMBDAS, m_for_l, lam_for_l, tag_for_l, "lambda")

    print(f"\nTotal rows: {len(rows)}, missing: {len(missing)}")
    for mi in missing:
        print("  missing:", mi)

    df_raw = pd.DataFrame(rows)

    def build_sweep_sheet(sweep_name, values):
        sub = df_raw[df_raw["Sweep"] == sweep_name]
        out_rows = []
        for ds in B.DATASETS:
            ds_disp = B.DATASET_DISPLAY[ds]
            for v in values:
                cell = sub[(sub["Dataset"] == ds_disp) & (sub["Value"] == v)]
                row = {"Dataset": ds_disp, "Value": v, "N": len(cell)}
                for f in NOISE_FRACTIONS:
                    col = f"AUC_frac_{f}"
                    vals = cell[col].dropna().values
                    mean = vals.mean() if len(vals) else np.nan
                    std = vals.std(ddof=1) if len(vals) > 1 else np.nan
                    row[f"AUC@frac={f}"] = f"{mean*100:.2f}±{std*100:.2f}" if len(vals) > 1 else (
                        f"{mean*100:.2f}" if len(vals) else "-")
                out_rows.append(row)
        return pd.DataFrame(out_rows)

    df_m = build_sweep_sheet("m", MARGINS)
    df_l = build_sweep_sheet("lambda", LAMBDAS)

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="raw", index=False)
        df_m.to_excel(writer, sheet_name="m", index=False)
        df_l.to_excel(writer, sheet_name="lambda", index=False)

        from openpyxl.styles import Font

        ws = writer.book.create_sheet("summary")
        ws.cell(row=1, column=1,
                value="So sanh AUC gia-lap o muc nhieu KHO NHAT (frac=0.02 x noise_std goc cua tung checkpoint) "
                      "-- model nao con tach duoc tot o day co bien latent 'khoe' hon, khong chi dat dung target m."
                ).font = Font(bold=True, size=12)
        r = 3
        hardest = NOISE_FRACTIONS[-1]

        def gather_hardest(sweep_name, values):
            value_headers = [f"{sweep_name}={v}" for v in values]
            out_rows = []
            for ds in B.DATASETS:
                ds_disp = B.DATASET_DISPLAY[ds]
                cells = []
                for v in values:
                    sub = df_raw[(df_raw["Sweep"] == sweep_name) & (df_raw["Dataset"] == ds_disp) & (df_raw["Value"] == v)]
                    vals = sub[f"AUC_frac_{hardest}"].dropna().values * 100
                    if len(vals) == 0:
                        cells.append(("-", np.nan))
                    else:
                        mean, std = vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)
                        cells.append((f"{mean:.2f}±{std:.2f}", mean))
                out_rows.append(([ds_disp], cells))
            return value_headers, out_rows

        vh, rows_m = gather_hardest("m", MARGINS)
        r = B.write_labeled_table(ws, r, f"Bang 1 - AUC gia-lap @ frac={hardest} theo m (lambda=0.1 co dinh)",
                                   ["Dataset"], vh, rows_m, best="max")
        vh, rows_l = gather_hardest("lambda", LAMBDAS)
        r = B.write_labeled_table(ws, r, f"Bang 2 - AUC gia-lap @ frac={hardest} theo lambda (m=2.0 co dinh)",
                                   ["Dataset"], vh, rows_l, best="max")

    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
