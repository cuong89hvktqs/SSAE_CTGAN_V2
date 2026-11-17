"""Reviewer scenario: real-world IDS deployments have far more normal traffic
than attack traffic, the OPPOSITE of this project's test sets (where attacks
are a large fraction, sometimes the majority -- see Table A, sheet GiaiTrinh).

No retraining: reuses the already-trained t_04 seed=42 MAE_<ds>_best_auc.pkl
checkpoint (same rebuild_val_test_split()/load_synthesizer() pipeline as
reeval_ssae_runs.py) and the SAME val-normal thresholds. Only the TEST set's
class ratio is modified: every normal row is kept, attack rows are randomly
downsampled to ~1/10 of their original count (random_state=42), so the
realistic case the reviewer asks for (
    few attacks relative to normal, instead of the original >=30-70% attack
    share) can be measured directly, with no new training.

Reports, original vs downsampled-attack test set:
  - AUC, TPR@FAR=5%/1% -- expected near-IDENTICAL (threshold-independent /
    per-class-conditional metrics, invariant to class ratio by construction)
  - MCC, Precision, F1 @ both thresholds -- these mix both classes through
    the confusion matrix, so they ARE sensitive to class ratio; expected to
    drop once attacks become rare, since false positives (on the now much
    larger normal pool) weigh more heavily against the now much smaller
    true-positive count. This is the actual "base-rate" effect the reviewer
    is asking the paper to confront.

Usage: python eval_realistic_ratio_t04_seed42.py
"""
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from reeval_ssae_runs import rebuild_val_test_split
from val_perattack_mae01scale_max import load_synthesizer

RES_DIR = os.path.join(SCRIPT_DIR, "res")
SEED = 42
TIME_TAG = "04"
DOWNSAMPLE_FRACTION = 0.10  # keep ~1/10 of attack rows
DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]


def find_run_dir(dataset, seed):
    import glob
    pattern = os.path.join(
        RES_DIR, f"t_{TIME_TAG}_SSAE_se_{seed}", "per_1.0",
        f"{dataset}_150e_m2.0_l0.1_renormal_noise_std_*_catra0.1")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def score_dataset(dataset, seed):
    run_dir = find_run_dir(dataset, seed)
    model_path = os.path.join(run_dir, f"MAE_{dataset}_best_auc.pkl")
    X_val_raw, y_val, X_test_raw, y_test = rebuild_val_test_split(dataset, seed, 1.0)

    synth = load_synthesizer(model_path)
    X_val = synth._data_processor.transform(X_val_raw)
    X_test = synth._data_processor.transform(X_test_raw)

    _, _, val_total, _ = synth.get_score(X_val, y_val)
    _, _, test_total, y_true = synth.get_score(X_test, y_test)
    y_true_bin = (np.asarray(y_true) != 0).astype(int)
    return np.asarray(val_total), np.asarray(test_total), y_true_bin


def compute_metrics(test_total, y_true_bin, val_total):
    out = {}
    auc = roc_auc_score(y_true_bin, test_total) * 100 if len(np.unique(y_true_bin)) >= 2 else float("nan")
    out["AUC"] = auc
    out["N_normal"] = int((y_true_bin == 0).sum())
    out["N_attack"] = int((y_true_bin == 1).sum())
    for percentile, far_label in ((95, "TPR@FAR=5%"), (99, "TPR@FAR=1%")):
        threshold = float(np.percentile(val_total, percentile))
        y_pred = (test_total > threshold).astype(int)
        out[far_label] = recall_score(y_true_bin, y_pred, zero_division=0) * 100
        out[f"MCC_{percentile}"] = matthews_corrcoef(y_true_bin, y_pred) * 100
        out[f"Precision_{percentile}"] = precision_score(y_true_bin, y_pred, zero_division=0) * 100
        out[f"F1_{percentile}"] = f1_score(y_true_bin, y_pred, zero_division=0) * 100
    return out


def downsample_attacks(test_total, y_true_bin, fraction, seed):
    rng = np.random.RandomState(seed)
    normal_idx = np.where(y_true_bin == 0)[0]
    attack_idx = np.where(y_true_bin == 1)[0]
    n_keep = max(1, int(round(len(attack_idx) * fraction)))
    keep_attack_idx = rng.choice(attack_idx, size=n_keep, replace=False)
    keep_idx = np.concatenate([normal_idx, keep_attack_idx])
    rng.shuffle(keep_idx)
    return test_total[keep_idx], y_true_bin[keep_idx]


def main():
    lines = []
    lines.append(f"t_{TIME_TAG} SSAE, seed={SEED} -- realistic test-ratio scenario "
                  f"(attacks downsampled to ~{DOWNSAMPLE_FRACTION*100:.0f}% of original count, "
                  f"all normal rows kept, no retraining)")
    lines.append("=" * 100)

    for ds in DATASETS:
        val_total, test_total, y_true_bin = score_dataset(ds, SEED)

        orig = compute_metrics(test_total, y_true_bin, val_total)
        test_total_ds, y_true_bin_ds = downsample_attacks(test_total, y_true_bin, DOWNSAMPLE_FRACTION, SEED)
        down = compute_metrics(test_total_ds, y_true_bin_ds, val_total)

        lines.append(f"\n--- {ds} ---")
        lines.append(f"{'Metric':<16}{'Original (N=' + str(orig['N_normal']) + ' normal / ' + str(orig['N_attack']) + ' attack)':<42}"
                      f"{'Downsampled (N=' + str(down['N_normal']) + ' normal / ' + str(down['N_attack']) + ' attack)':<42}")
        for key in ["AUC", "TPR@FAR=5%", "TPR@FAR=1%",
                    "MCC_95", "MCC_99", "Precision_95", "Precision_99", "F1_95", "F1_99"]:
            lines.append(f"{key:<16}{orig[key]:<42.2f}{down[key]:<42.2f}")

    out_path = os.path.join(SCRIPT_DIR, "res", "realistic_ratio_t04_seed42.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
