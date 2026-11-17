"""Backfill best_auc/test_mode_total.csv and best_auc/per_attack_seeds.csv
for the OLD seed 42/43/44 SSAE runs (t_02 margin ablation, t_04 main
config, t_06 fixed-gate ablation) so they match the layout the NEW
train_mae01scale_max.py already produces for seeds 45-51's t_04 runs --
making it easy to aggregate all 10 seeds the same way later.

Scope (per explicit decision, not the full set of everything on disk):
  - per_1.0 only (seeds 45-51 only ever ran per_1.0).
  - t_02, t_06: already scoped to the 4 core datasets
    (unsw, cic_iot, nsl_kdd, wsn_ds), single (noise_std) variant -- no
    extra filtering needed.
  - t_04: restricted to those same 4 datasets and the noise_std ("P+Q",
    full-model) variant only -- t_04 for these seeds also has cic_ids/
    nb_iot and the noise-only/std-only ablation variants, which are
    intentionally left untouched here.

For each matched run folder:
  1. best_auc/test_mode_total.csv is built directly from the run's own
     ALREADY-COMPUTED best_auc/test_mode_total95.csv and
     best_auc/test_mode_total99.csv (same AUC/MCC/Recall numbers, just
     merged/relabeled into the new schema -- no model reload, no
     ambiguity, since it reads only this exact run folder's own files).
  2. best_auc/per_attack_seeds.csv is recomputed from scratch: loads this
     run's own MAE_<dataset>_best_auc.pkl checkpoint, reproduces the
     EXACT val/test split train_mae01scale_max.py used at training time
     (same seed, same loader, same 80/20 split), scores it, then applies
     the same per-attack FAR-based evaluation as
     evaluate_per_attack()/ATTACK_LOADERS.
  3. Every file that was already directly inside best_auc/ (before this
     script's own writes) is MOVED into a new best_auc/old/ subfolder --
     best_auc/ keeps only the new test_mode_total.csv/per_attack_seeds.csv
     at its top level, the old files survive only under old/.

Dry-run by default -- only prints the plan (including a quick read of the
existing test_mode_total95/99.csv values, no model loading). Pass --apply
to actually recompute per_attack_seeds.csv (loads each checkpoint -- this
is the slow part) and write everything to disk.

Usage:
    python backfill_ssae_old_seeds.py                   # dry run
    python backfill_ssae_old_seeds.py --apply            # actually do it
    python backfill_ssae_old_seeds.py --seeds 42,43 --apply
"""

import argparse
import csv
import glob
import os
import shutil
import sys

import cloudpickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from process_data import load_data_cic_iot, load_data_nsl_kdd, load_data_unsw, load_data_wsn_ds
from val_perattack_mae01scale_max import ATTACK_LOADERS, FAR_LEVELS, evaluate_ids_metrics

CORE_DATASETS = ("unsw", "cic_iot", "nsl_kdd", "wsn_ds")

DATASET_LOADERS = {
    "unsw":    lambda seed: load_data_unsw(),
    "cic_iot": lambda seed: load_data_cic_iot(),
    "nsl_kdd": lambda seed: load_data_nsl_kdd(),
    "wsn_ds":  lambda seed: load_data_wsn_ds(seed=seed, test_size=0.2),
}

# t_01 = lambda-ablation runs for seeds 42/43/44, moved out of t_02 (which
# originally held both the margin and lambda sweeps) into their own folder.
TIME_TAGS = ("t_01", "t_02", "t_04", "t_06")


def find_target_run_dirs(root, seeds):
    """(seed, time_tag, dataset, run_dir) tuples for every matched folder,
    scoped to per_1.0 / CORE_DATASETS / the noise_std variant only."""
    targets = []
    for seed in seeds:
        for time_tag in TIME_TAGS:
            seed_dir = os.path.join(root, f"{time_tag}_SSAE_se_{seed}")
            per_dir = os.path.join(seed_dir, "per_1.0")
            if not os.path.isdir(per_dir):
                continue
            for dataset in CORE_DATASETS:
                pattern = os.path.join(per_dir, f"{dataset}_*noise_std_*")
                for run_dir in sorted(glob.glob(pattern)):
                    if os.path.isdir(run_dir):
                        targets.append((seed, time_tag, dataset, run_dir))
    return targets


def read_metric_csv(path):
    """Read a Metric,Value CSV (test_mode_total95.csv / ..._total99.csv)
    into a dict, ignoring the trailing confusion-matrix block."""
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header row: Metric,Value
        for row in reader:
            if len(row) < 2:
                continue
            key, val = row[0], row[1]
            try:
                values[key] = float(val)
            except ValueError:
                continue
    return values


def backfill_test_mode_total_from_old_csvs(best_auc_dir, apply):
    """Build best_auc/test_mode_total.csv from the run's own
    test_mode_total95.csv + test_mode_total99.csv (already computed by the
    old pipeline) -- same schema as evaluate_best_auc_checkpoint(). Cheap:
    no model reload. Returns True if it actually had the source files to
    work from, False otherwise (caller should fall back to a fresh
    checkpoint-based computation)."""
    # Prefer the top-level (pre-archive) location; fall back to old/ in
    # case a previous run of this script already archived them there
    # before this function got a chance to read them.
    path95 = os.path.join(best_auc_dir, "test_mode_total95.csv")
    path99 = os.path.join(best_auc_dir, "test_mode_total99.csv")
    if not (os.path.exists(path95) and os.path.exists(path99)):
        path95 = os.path.join(best_auc_dir, "old", "test_mode_total95.csv")
        path99 = os.path.join(best_auc_dir, "old", "test_mode_total99.csv")
    if not (os.path.exists(path95) and os.path.exists(path99)):
        return False

    m95 = read_metric_csv(path95)
    m99 = read_metric_csv(path99)

    save_path = os.path.join(best_auc_dir, "test_mode_total.csv")
    print(f"    write {save_path}  (from test_mode_total95.csv + test_mode_total99.csv)")
    if apply:
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Value"])
            w.writerow(["Threshold_95", f"{m95['Threshold']:.6f}"])
            w.writerow(["TPR@FAR=5%",   f"{m95['Recall']:.6f}"])
            w.writerow(["AUC_95",       f"{m95['AUC']:.6f}"])
            w.writerow(["MCC_95",       f"{m95['MCC']:.6f}"])
            w.writerow(["Threshold_99", f"{m99['Threshold']:.6f}"])
            w.writerow(["TPR@FAR=1%",   f"{m99['Recall']:.6f}"])
            w.writerow(["AUC_99",       f"{m99['AUC']:.6f}"])
            w.writerow(["MCC_99",       f"{m99['MCC']:.6f}"])
    return True


def write_test_mode_total_from_scores(save_path, val_total, test_total, y_true_bin, apply):
    """Fresh checkpoint-based computation of test_mode_total.csv -- same
    val95/val99-threshold logic as evaluate_best_auc_checkpoint() in
    train_mae01scale_max.py. Used when the old test_mode_total95/99.csv
    files never existed in the first place (old pipeline's separate
    evaluation script was apparently never run for this specific config)."""
    print(f"    write {save_path}  (fresh, from reloaded checkpoint -- old "
          f"test_mode_total95/99.csv were missing)")
    if not apply:
        return

    val_total = np.asarray(val_total)
    test_total = np.asarray(test_total)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for percentile, far_label in ((95, "TPR@FAR=5%"), (99, "TPR@FAR=1%")):
            threshold = float(np.percentile(val_total, percentile))
            y_pred = (test_total > threshold).astype(int)
            auc = roc_auc_score(y_true_bin, test_total) * 100 if len(np.unique(y_true_bin)) >= 2 else 0.0
            mcc = matthews_corrcoef(y_true_bin, y_pred) * 100
            tpr = recall_score(y_true_bin, y_pred, zero_division=0) * 100
            w.writerow([f"Threshold_{percentile}", f"{threshold:.6f}"])
            w.writerow([far_label,                 f"{tpr:.6f}"])
            w.writerow([f"AUC_{percentile}",        f"{auc:.6f}"])
            w.writerow([f"MCC_{percentile}",        f"{mcc:.6f}"])


def load_synthesizer(model_path):
    """Same device-safe loader as val_perattack_mae01scale_max.py."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": device})
    try:
        with open(model_path, "rb") as f:
            synthesizer = cloudpickle.load(f)
    finally:
        torch.load = _orig_load
    synthesizer._model.set_device(device)
    return synthesizer


def rebuild_val_test_split(dataset, seed):
    """Reproduce the EXACT val/test raw split train_mae01scale_max.py used
    at training time (per=1.0 -> frac=1.0 sample is a no-op, so this only
    depends on seed): 80/20 split of normal-only training rows, seeded the
    same way, plus the loader's own official test set."""
    df_full, y_full, df_test, y_test = DATASET_LOADERS[dataset](seed)

    n_test = len(df_test)
    X_train_raw = df_full.iloc[:-n_test].reset_index(drop=True)
    X_train_raw.columns = X_train_raw.columns.map(str)
    y_train = y_full.iloc[:-n_test].reset_index(drop=True)

    normal_mask = y_train == 0
    X_train_normal_full = X_train_raw.loc[normal_mask].copy()
    y_train_normal_full = y_train.loc[normal_mask].copy()

    # per=1.0 -> frac=1.0, sample() still reindexes/shuffles but keeps all rows
    sampled_idx = X_train_normal_full.sample(frac=1.0, random_state=seed).index
    X_normal_sub = X_train_normal_full.loc[sampled_idx].reset_index(drop=True)
    y_normal_sub = y_train_normal_full.loc[sampled_idx].reset_index(drop=True)

    _, X_val_raw, _, y_val = train_test_split(
        X_normal_sub, y_normal_sub, test_size=0.2, random_state=seed
    )
    X_test_raw = df_test.reset_index(drop=True)
    X_test_raw.columns = X_test_raw.columns.map(str)
    y_test = y_test.reset_index(drop=True)

    return X_val_raw, y_val.reset_index(drop=True), X_test_raw, y_test


def load_and_score(run_dir, dataset, seed):
    """Load this run's own MAE_<dataset>_best_auc.pkl checkpoint and score
    the EXACT val/test split train_mae01scale_max.py used at training
    time. Returns (val_total, test_total, y_true_bin) as numpy arrays, or
    None if the checkpoint file is missing."""
    model_path = os.path.join(run_dir, f"MAE_{dataset}_best_auc.pkl")
    if not os.path.exists(model_path):
        return None

    X_val_raw, y_val, X_test_raw, y_test = rebuild_val_test_split(dataset, seed)

    best_synth = load_synthesizer(model_path)
    X_val = best_synth._data_processor.transform(X_val_raw)
    X_test = best_synth._data_processor.transform(X_test_raw)

    _, _, val_total, _ = best_synth.get_score(X_val, y_val)
    _, _, test_total, y_true = best_synth.get_score(X_test, y_test)
    y_true_bin = (np.asarray(y_true) != 0).astype(int)
    return np.asarray(val_total), np.asarray(test_total), y_true_bin


def write_per_attack_from_scores(save_path, dataset, seed, test_total, apply):
    """FAR-based per-attack breakdown -- same convention as
    evaluate_per_attack()/ATTACK_LOADERS in train_mae01scale_max.py."""
    print(f"    write {save_path}  (per-attack FAR metrics)")
    if not apply:
        return

    attack_labels, normal_tag = ATTACK_LOADERS[dataset](seed)
    if len(attack_labels) != len(test_total):
        print(f"    [ERROR] attack labels ({len(attack_labels)}) != test scores "
              f"({len(test_total)}) -- alignment broken, skip.")
        return

    normal_mask = (attack_labels == normal_tag).to_numpy()
    n_normal = int(normal_mask.sum())
    attacks = sorted(a for a in attack_labels.unique() if a != normal_tag)

    scores_normal = test_total[normal_mask]
    normal_thresholds = {
        far: np.percentile(scores_normal, 100 * (1 - far)) for far in FAR_LEVELS
    }

    rows = []
    for attack in attacks:
        atk_mask = (attack_labels == attack).to_numpy()
        sub_scores = np.concatenate([scores_normal, test_total[atk_mask]])
        sub_y = np.concatenate([
            np.zeros(n_normal, dtype=int),
            np.ones(int(atk_mask.sum()), dtype=int),
        ])
        metrics = evaluate_ids_metrics(sub_y, sub_scores, normal_thresholds)
        rows.append({
            "Dataset": dataset, "Seed": seed, "Attack": attack,
            "N": int(atk_mask.sum()), "Score": "total", **metrics,
        })

    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"    -> saved {len(rows)} attack rows")


NEW_OUTPUT_FILES = {"test_mode_total.csv", "per_attack_seeds.csv"}


def archive_old_files(best_auc_dir, apply):
    """Move every pre-existing OLD-format file directly inside
    best_auc_dir into a new best_auc_dir/old/ subfolder -- best_auc_dir
    keeps only the NEW-format outputs (test_mode_total.csv,
    per_attack_seeds.csv) at its top level; the old files survive only
    under old/. Excludes this script's own output files
    (NEW_OUTPUT_FILES). Idempotent: if a file was already archived by an
    earlier (copy-only) run of this script -- i.e. it exists in BOTH
    best_auc_dir and old_dir -- the leftover top-level copy is simply
    removed rather than re-copied."""
    entries = sorted(
        f for f in os.listdir(best_auc_dir)
        if os.path.isfile(os.path.join(best_auc_dir, f)) and f not in NEW_OUTPUT_FILES
    )
    if not entries:
        return

    old_dir = os.path.join(best_auc_dir, "old")
    for entry in entries:
        src = os.path.join(best_auc_dir, entry)
        dest = os.path.join(old_dir, entry)
        if os.path.exists(dest):
            print(f"    delete leftover top-level copy (already archived)  {src}")
            if apply:
                os.remove(src)
            continue
        print(f"    move  {src}  ->  {dest}")
        if apply:
            os.makedirs(old_dir, exist_ok=True)
            shutil.move(src, dest)


def process_run(seed, time_tag, dataset, run_dir, apply):
    best_auc_dir = os.path.join(run_dir, "best_auc")
    print(f"\n[{time_tag}] seed={seed} dataset={dataset}\n  {run_dir}")
    if not os.path.isdir(best_auc_dir):
        print(f"    [WARN] no best_auc/ folder -- skip.")
        return

    total_path = os.path.join(best_auc_dir, "test_mode_total.csv")
    per_attack_path = os.path.join(best_auc_dir, "per_attack_seeds.csv")
    need_total = not os.path.exists(total_path)
    need_per_attack = not os.path.exists(per_attack_path)

    if not need_total and not need_per_attack:
        print(f"    (test_mode_total.csv + per_attack_seeds.csv already exist -- "
              f"skip recompute, only cleaned up old/ below)")
    else:
        # Try the cheap path (reads the run's own already-computed
        # test_mode_total95/99.csv) FIRST, before archive_old_files moves
        # those source files out of best_auc_dir.
        if need_total and backfill_test_mode_total_from_old_csvs(best_auc_dir, apply):
            need_total = False

        if need_total or need_per_attack:
            scores = load_and_score(run_dir, dataset, seed)
            if scores is None:
                print(f"    [WARN] missing MAE_{dataset}_best_auc.pkl -- "
                      f"cannot backfill remaining output(s).")
            else:
                val_total, test_total, y_true_bin = scores
                if need_total:
                    write_test_mode_total_from_scores(
                        total_path, val_total, test_total, y_true_bin, apply
                    )
                if need_per_attack:
                    write_per_attack_from_scores(
                        per_attack_path, dataset, seed, test_total, apply
                    )

    # Archive old-format leftovers now that everything needed has been read.
    archive_old_files(best_auc_dir, apply)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=str, default=None,
                         help="folder containing t_XX_SSAE_se_* run dirs "
                              "(default: <this script's dir>/res)")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                         help="comma-separated list of seeds (default: 42,43,44)")
    parser.add_argument("--apply", action="store_true",
                         help="actually recompute + write everything "
                              "(default: dry-run, only prints the plan)")
    args = parser.parse_args()

    root = args.root or os.path.join(SCRIPT_DIR, "res")
    if not os.path.isdir(root):
        sys.exit(f"Root folder not found: {root}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    targets = find_target_run_dirs(root, seeds)
    if not targets:
        print(f"No matching run folders found under {root} for seeds {seeds} "
              f"(time tags: {TIME_TAGS}, datasets: {CORE_DATASETS}, per_1.0, noise_std).")
        return

    print(f"{'[DRY RUN] ' if not args.apply else ''}"
          f"Backfilling {len(targets)} run folder(s) (seeds={seeds}, "
          f"time_tags={TIME_TAGS}, datasets={CORE_DATASETS}, per_1.0)")
    print("=" * 70)

    failures = []
    for seed, time_tag, dataset, run_dir in targets:
        try:
            process_run(seed, time_tag, dataset, run_dir, args.apply)
        except Exception as e:
            print(f"    [ERROR] {type(e).__name__}: {e} -- skipping this run, continuing.")
            failures.append((seed, time_tag, dataset, run_dir))

    if failures:
        print(f"\n{len(failures)} run(s) FAILED and were skipped:")
        for seed, time_tag, dataset, run_dir in failures:
            print(f"  [{time_tag}] seed={seed} dataset={dataset}  {run_dir}")

    print("\n" + "=" * 70)
    if args.apply:
        print("DONE: backfill applied.")
    else:
        print("DRY RUN complete -- nothing was changed. Re-run with --apply to execute.")


if __name__ == "__main__":
    main()
