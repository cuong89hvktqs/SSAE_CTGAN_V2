"""Build res/summary_seeds42-51.xlsx -- per-seed overall results for the 4 core
datasets, one sheet per experiment:

  1_Overall_t04       SSAE (t_04 main config) vs DeepSVDD / DROCC / ICL / DIF / CLAN
  2_Lambda_t01        SSAE lambda ablation (t_01, m=2.0) + lambda=0.1 baseline from t_04
  3_Margin_t02        SSAE margin ablation (t_02, lambda=0.1) + m=2.0 baseline from t_04
  4_CTGAN_preproc     AE / ICL / CLAN before / after CTGAN preprocessing (+ SSAE)
  07_CTGAN_pre        DeepSVDD/GOAD/NeuTraL/DIF/DROCC/ICL on CTGAN-preprocessed
                      input, seeds 45-51 only (not trained for 42-44)
  8_PseudoSource_t05  Pseudo-intrusion source ablation (t_05): P-only /
                      Q-only vs P+Q (t_04 baseline) -- seeds 42-44 ONLY
                      (t_05 was never trained for seeds 45-51 on this
                      machine)
  5_PerAttack_t04     Per-attack breakdown, t_04 main config, all 9 methods
  06_fixgate          Gate-opening ablation (t_06), AUC only: fixed epoch
                      0/20/40 vs the adaptive gate (t_04)
  TongHopChung        Paper Tables 2/3/4/5/7/8/9/10/11, recomputed from all
                      seeds (not the paper's original 3); best value
                      bold+red. Table 9 (P-only/Q-only/P+Q) always uses
                      seeds 42-44 (see 8_PseudoSource_t05), independently of
                      whatever seeds the rest of the sheet uses. Table 5 is
                      followed by Table 5b (SSAE win/tie/loss vs each
                      baseline on TPR@FAR=1%, 8 challenging attacks, tie
                      margin 0.5pp) -- added for Reviewer 1 Comment 3 (Table
                      5 must be descriptive, not an inferential test over
                      nested attack classes).

Every value comes from the binary normal-vs-attack result file of each run,
all on the same convention (Threshold = 95th/99th percentile of the
validation-normal score):
  SSAE      res/t_<tag>_SSAE_se_<seed>/per_1.0/<ds>_*/best_auc/test_mode_total.csv
  DeepOD    DeepOD/results_ssae[_ctgan_preprocessed]/<algo>/[per_1.0/]<ds>/seed<N>/best_auc/test_mode_total.csv
  DROCC     DROCC_icml2020/tabular/results_ssae/per_1.0/<ds>/seed<N>/best_auc/test_mode_total.csv
  CLAN      CLAN-main/results/[ctgan_preprocessed/]clan_[ctgan_]<ds>_seed_<N>_eval_binary.xlsx

Sheet 5 reads the matching per-attack file instead (per_attack_seeds.csv for
SSAE/DeepOD/DROCC/AE, clan_<ds>_seed_<N>_eval.xlsx -- the non-binary variant
-- for CLAN), plus:
  AE        <repo-parent>/TestDT_VChuan/results/percentage_1.0/ae_<ds>_epochs150_seed<N>/best_auc/per_attack_seeds.csv

Layout of each sheet: one row per (Dataset, Method, Metric) -- sheet 5 adds
an Attack grouping level (Dataset, Attack, Method, Metric) -- one column per
seed, then Mean / Std (sample std, ddof=1) / Mean±Std.

Usage:
    python build_seeds_summary_excel.py
    python build_seeds_summary_excel.py --seeds 42,43,44 --out res/summary.xlsx
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RES_DIR = os.path.join(SCRIPT_DIR, "res")
DEEPOD_DIR = os.path.join(PROJECT_DIR, "DeepOD")
DROCC_DIR = os.path.join(PROJECT_DIR, "DROCC_icml2020", "tabular")
CLAN_DIR = r"D:\Code_papper\CTGAN\CLGAN\CLAN-main\CLAN-main"
AE_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "TestDT_VChuan")

DATASETS = ["unsw", "nsl_kdd", "cic_iot", "wsn_ds"]
DATASET_DISPLAY = {"unsw": "UNSW-NB15", "nsl_kdd": "NSL-KDD", "cic_iot": "CIC-IoT 2023", "wsn_ds": "WSN-DS"}
CLAN_DATASET_NAME = {"unsw": "unsw", "nsl_kdd": "nslkdd", "cic_iot": "cic_iot", "wsn_ds": "wsn_ds"}

METRICS = ["AUC", "TPR@FAR=1%", "MCC_99", "TPR@FAR=5%", "MCC_95"]

LAMBDAS = ["0.0", "0.01", "0.05", "0.1", "0.5", "1.0"]
MARGINS = ["1.0", "2.0", "3.0", "4.0", "5.0", "9.0"]

# Per-attack files use 3 different naming conventions for the SAME attacks:
#   SSAE/DeepOD/AE  -- Title/CamelCase from the raw attack_cat column (e.g. "Shellcode", "BruteForce"),
#                      "attack_1".."attack_4" for nsl_kdd (process_data.py's own numeric codes)
#   DROCC/CLAN      -- all-lowercase ("shellcode", "bruteforce"), bare digits for nsl_kdd ("1".."4")
# normalize_attack_key() maps all 3 to one canonical lowercase key per dataset;
# ATTACK_DISPLAY supplies the display name shown in the sheet.
ATTACK_DISPLAY = {
    "unsw": {
        "analysis": "Analysis", "backdoor": "Backdoor", "dos": "DoS",
        "exploits": "Exploits", "fuzzers": "Fuzzers", "generic": "Generic",
        "reconnaissance": "Reconnaissance", "shellcode": "Shellcode", "worms": "Worms",
    },
    "nsl_kdd": {
        "attack_1": "attack_1", "attack_2": "attack_2",
        "attack_3": "attack_3 (R2L)", "attack_4": "attack_4 (U2R)",
    },
    "cic_iot": {
        "bruteforce": "BruteForce", "ddos": "DDoS", "dos": "DoS",
        "mirai": "Mirai", "recon": "Recon", "spoofing": "Spoofing", "web": "Web",
    },
    "wsn_ds": {
        "blackhole": "Blackhole", "flooding": "Flooding",
        "grayhole": "Grayhole", "tdma": "TDMA",
    },
}


def normalize_attack_key(dataset, raw):
    s = str(raw).strip().lower()
    if dataset == "nsl_kdd":
        m = re.match(r"^attack[_ ]?(\d+)$", s) or re.match(r"^(\d+)$", s)
        if m:
            return f"attack_{m.group(1)}"
    return s


# ── Readers: each returns {metric: value} or None ─────────────────────────────

def read_test_mode_total(path):
    if not os.path.isfile(path):
        return None
    s = pd.read_csv(path).set_index("Metric")["Value"].astype(float)
    return {
        "AUC": s["AUC_95"],  # AUC is threshold-independent (AUC_95 == AUC_99)
        "TPR@FAR=1%": s["TPR@FAR=1%"],
        "MCC_99": s["MCC_99"],
        "TPR@FAR=5%": s["TPR@FAR=5%"],
        "MCC_95": s["MCC_95"],
    }


def first_match(pattern):
    matches = sorted(glob.glob(pattern))
    if len(matches) > 1:
        print(f"[WARN] {len(matches)} matches, using first: {pattern}")
    return matches[0] if matches else None


def ssae(time_tag, dataset, seed, m="2.0", lam="0.1"):
    run_dir = first_match(os.path.join(
        RES_DIR, f"t_{time_tag}_SSAE_se_{seed}", "per_1.0",
        f"{dataset}_150e_m{m}_l{lam}_renormal_noise_std_*_catra0.1"))
    if run_dir is None:
        return None
    return read_test_mode_total(os.path.join(run_dir, "best_auc", "test_mode_total.csv"))


PSEUDO_SOURCE_SEEDS = list(range(42, 52))  # t_05 (P-only/Q-only): seeds 45-51 extracted from t_05.tar.gz on 2026-11-09


def ssae_pseudo_source(variant, dataset, seed):
    """t_05 ablation: variant='noise' is P-only (real-normal+noise), variant='std'
    is Q-only (CTGAN-sample+noise). The P+Q baseline is t_04's ssae()."""
    run_dir = first_match(os.path.join(
        RES_DIR, f"t_05_SSAE_se_{seed}", "per_1.0",
        f"{dataset}_150e_m2.0_l0.1_renormal_{variant}_*_catra0.1"))
    if run_dir is None:
        return None
    return read_test_mode_total(os.path.join(run_dir, "best_auc", "test_mode_total.csv"))


def ssae_gatefix(gate_epoch, dataset, seed, time_tag="06", m="2.0", lam="0.1"):
    """t_06 ablation: --gate_fixed_epoch {0,20,40} bypasses the adaptive
    activation gate and opens it at a fixed epoch instead."""
    run_dir = first_match(os.path.join(
        RES_DIR, f"t_{time_tag}_SSAE_se_{seed}", "per_1.0",
        f"{dataset}_150e_m{m}_l{lam}_renormal_noise_std_*_catra0.1_gf{gate_epoch}"))
    if run_dir is None:
        return None
    return read_test_mode_total(os.path.join(run_dir, "best_auc", "test_mode_total.csv"))


def deepod(algo, dataset, seed, ctgan=False):
    if ctgan:
        base = os.path.join(DEEPOD_DIR, "results_ssae_ctgan_preprocessed", algo, dataset)
    else:
        base = os.path.join(DEEPOD_DIR, "results_ssae", algo, "per_1.0", dataset)
    return read_test_mode_total(os.path.join(base, f"seed{seed}", "best_auc", "test_mode_total.csv"))


def drocc(dataset, seed):
    return read_test_mode_total(os.path.join(
        DROCC_DIR, "results_ssae", "per_1.0", dataset, f"seed{seed}", "best_auc", "test_mode_total.csv"))


def drocc_ctgan(dataset, seed):
    return read_test_mode_total(os.path.join(
        DROCC_DIR, "results_ssae_ctgan_preprocessed", dataset, f"seed{seed}", "best_auc", "test_mode_total.csv"))


def clan(dataset, seed, ctgan=False):
    name = CLAN_DATASET_NAME[dataset]
    if ctgan:
        path = os.path.join(CLAN_DIR, "results", "ctgan_preprocessed",
                            f"clan_ctgan_{name}_seed_{seed}_eval_binary.xlsx")
    else:
        path = os.path.join(CLAN_DIR, "results", f"clan_{name}_seed_{seed}_eval_binary.xlsx")
    if not os.path.isfile(path):
        return None
    row = pd.read_excel(path).iloc[0]
    return {k: float(row[k]) for k in METRICS}


def ae(dataset, seed, ctgan=False):
    sub = "ctgan_preprocessed" if ctgan else "percentage_1.0"
    base = os.path.join(AE_DIR, "results", sub, f"ae_{dataset}_epochs150_seed{seed}")
    return read_test_mode_total(os.path.join(base, "best_auc", "test_mode_total.csv"))


# ── Per-attack readers: each returns {attack_key: {metric: value}} or None ───

def read_per_attack_csv(path, dataset):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        key = normalize_attack_key(dataset, row["Attack"])
        out[key] = {m: float(row[m]) for m in METRICS}
    return out


def read_clan_per_attack_xlsx(path, dataset):
    if not os.path.isfile(path):
        return None
    df = pd.read_excel(path)
    df = df[df["Attack"].astype(str).str.upper() != "MEAN"]
    out = {}
    for _, row in df.iterrows():
        key = normalize_attack_key(dataset, row["Attack"])
        out[key] = {m: float(row[m]) for m in METRICS}
    return out


def ssae_perattack(time_tag, dataset, seed, m="2.0", lam="0.1"):
    run_dir = first_match(os.path.join(
        RES_DIR, f"t_{time_tag}_SSAE_se_{seed}", "per_1.0",
        f"{dataset}_150e_m{m}_l{lam}_renormal_noise_std_*_catra0.1"))
    if run_dir is None:
        return None
    return read_per_attack_csv(os.path.join(run_dir, "best_auc", "per_attack_seeds.csv"), dataset)


def deepod_perattack(algo, dataset, seed, ctgan=False):
    if ctgan:
        base = os.path.join(DEEPOD_DIR, "results_ssae_ctgan_preprocessed", algo, dataset)
    else:
        base = os.path.join(DEEPOD_DIR, "results_ssae", algo, "per_1.0", dataset)
    return read_per_attack_csv(os.path.join(base, f"seed{seed}", "best_auc", "per_attack_seeds.csv"), dataset)


def drocc_perattack(dataset, seed):
    return read_per_attack_csv(os.path.join(
        DROCC_DIR, "results_ssae", "per_1.0", dataset, f"seed{seed}", "best_auc", "per_attack_seeds.csv"), dataset)


def drocc_ctgan_perattack(dataset, seed):
    return read_per_attack_csv(os.path.join(
        DROCC_DIR, "results_ssae_ctgan_preprocessed", dataset, f"seed{seed}", "best_auc", "per_attack_seeds.csv"), dataset)


def clan_perattack(dataset, seed, ctgan=False):
    name = CLAN_DATASET_NAME[dataset]
    if ctgan:
        path = os.path.join(CLAN_DIR, "results", "ctgan_preprocessed", f"clan_ctgan_{name}_seed_{seed}_eval.xlsx")
    else:
        path = os.path.join(CLAN_DIR, "results", f"clan_{name}_seed_{seed}_eval.xlsx")
    return read_clan_per_attack_xlsx(path, dataset)


def ae_perattack(dataset, seed, ctgan=False):
    sub = "ctgan_preprocessed" if ctgan else "percentage_1.0"
    base = os.path.join(AE_DIR, "results", sub, f"ae_{dataset}_epochs150_seed{seed}")
    return read_per_attack_csv(os.path.join(base, "best_auc", "per_attack_seeds.csv"), dataset)


# ── Table building ────────────────────────────────────────────────────────────

def build_table(methods, seeds, metrics=None):
    """methods: list of (label, fn(dataset, seed) -> dict|None)."""
    metrics = METRICS if metrics is None else metrics
    rows, missing = [], []
    for ds in DATASETS:
        for label, fn in methods:
            per_seed = {seed: fn(ds, seed) for seed in seeds}
            missing += [f"{ds}/{label}/seed{s}" for s, v in per_seed.items() if v is None]
            for metric in metrics:
                vals = {s: (v[metric] if v is not None else np.nan) for s, v in per_seed.items()}
                arr = np.array([x for x in vals.values() if not np.isnan(x)])
                mean = arr.mean() if len(arr) else np.nan
                std = arr.std(ddof=1) if len(arr) > 1 else np.nan
                rows.append({
                    "Dataset": DATASET_DISPLAY[ds], "Method": label, "Metric": metric,
                    **{str(s): vals[s] for s in seeds},
                    "N": len(arr), "Mean": mean, "Std": std,
                    "Mean±Std": f"{mean:.2f}±{std:.2f}" if len(arr) > 1 else "",
                })
    return pd.DataFrame(rows), missing


def build_per_attack_table(methods, seeds):
    """methods: list of (label, fn(dataset, seed) -> {attack_key: {metric: value}} | None).
    One row per (Dataset, Attack, Method, Metric)."""
    rows, missing = [], []
    for ds in DATASETS:
        per_method = {}
        for label, fn in methods:
            per_seed = {seed: fn(ds, seed) for seed in seeds}
            missing += [f"{ds}/{label}/seed{s}" for s, v in per_seed.items() if v is None]
            per_method[label] = per_seed
        for attack_key, display_attack in ATTACK_DISPLAY[ds].items():
            for label, _ in methods:
                per_seed = per_method[label]
                for metric in METRICS:
                    vals = {
                        s: (v[attack_key][metric] if (v is not None and attack_key in v) else np.nan)
                        for s, v in per_seed.items()
                    }
                    arr = np.array([x for x in vals.values() if not np.isnan(x)])
                    mean = arr.mean() if len(arr) else np.nan
                    std = arr.std(ddof=1) if len(arr) > 1 else np.nan
                    rows.append({
                        "Dataset": DATASET_DISPLAY[ds], "Attack": display_attack,
                        "Method": label, "Metric": metric,
                        **{str(s): vals[s] for s in seeds},
                        "N": len(arr), "Mean": mean, "Std": std,
                        "Mean±Std": f"{mean:.2f}±{std:.2f}" if len(arr) > 1 else "",
                    })
    return pd.DataFrame(rows), missing


def write_sheet(writer, df, sheet_name, title):
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
    ws = writer.sheets[sheet_name]
    ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=12)

    thin = Side(style="thin", color="BFBFBF")
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    band_fill = PatternFill("solid", fgColor="F2F2F2")
    n_cols = df.shape[1]
    header_row = 3

    for c in range(1, n_cols + 1):
        cell = ws.cell(row=header_row, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)

    # Shade alternate (Dataset, Method) groups; blank repeated Dataset/Method labels.
    prev_key, band = None, False
    for i, (_, r) in enumerate(df.iterrows()):
        excel_row = header_row + 1 + i
        key = (r["Dataset"], r["Method"])
        if key != prev_key:
            band = not band
        else:
            ws.cell(row=excel_row, column=2).value = None
            if r["Dataset"] == prev_key[0]:
                ws.cell(row=excel_row, column=1).value = None
        if i > 0 and key[0] != prev_key[0]:
            for c in range(1, n_cols + 1):
                ws.cell(row=excel_row, column=c).border = Border(top=Side(style="medium"))
        prev_key = key
        for c in range(1, n_cols + 1):
            cell = ws.cell(row=excel_row, column=c)
            if band:
                cell.fill = band_fill
            if isinstance(cell.value, float):
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="center")
        ws.cell(row=excel_row, column=n_cols).font = Font(bold=True)

    widths = {"Dataset": 14, "Method": 24, "Metric": 22, "Mean±Std": 14}
    for c, name in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(c)].width = widths.get(name, 9)
    ws.freeze_panes = ws.cell(row=header_row + 1, column=4)


def write_sheet_perattack(writer, df, sheet_name, title):
    """Like write_sheet(), but groups/bands on (Dataset, Attack) and blanks
    repeated (Dataset, Attack, Method) labels across each method's Metric rows."""
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
    ws = writer.sheets[sheet_name]
    ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=12)

    thin = Side(style="thin", color="BFBFBF")
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    band_fill = PatternFill("solid", fgColor="F2F2F2")
    n_cols = df.shape[1]
    header_row = 3

    for c in range(1, n_cols + 1):
        cell = ws.cell(row=header_row, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)

    prev_band_key, prev_full_key, band = None, None, False
    for i, (_, r) in enumerate(df.iterrows()):
        excel_row = header_row + 1 + i
        band_key = (r["Dataset"], r["Attack"])
        full_key = (r["Dataset"], r["Attack"], r["Method"])
        if band_key != prev_band_key:
            band = not band
            if prev_band_key is not None and r["Dataset"] != prev_band_key[0]:
                for c in range(1, n_cols + 1):
                    ws.cell(row=excel_row, column=c).border = Border(top=Side(style="medium"))
            elif prev_band_key is not None:
                for c in range(1, n_cols + 1):
                    ws.cell(row=excel_row, column=c).border = Border(top=Side(style="thin", color="999999"))
        if full_key == prev_full_key:
            ws.cell(row=excel_row, column=1).value = None
            ws.cell(row=excel_row, column=2).value = None
            ws.cell(row=excel_row, column=3).value = None
        prev_band_key, prev_full_key = band_key, full_key
        for c in range(1, n_cols + 1):
            cell = ws.cell(row=excel_row, column=c)
            if band:
                cell.fill = band_fill
            if isinstance(cell.value, float):
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="center")
        ws.cell(row=excel_row, column=n_cols).font = Font(bold=True)

    widths = {"Dataset": 14, "Attack": 16, "Method": 14, "Metric": 13, "Mean±Std": 14}
    for c, name in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(c)].width = widths.get(name, 9)
    ws.freeze_panes = ws.cell(row=header_row + 1, column=5)


def write_labeled_table(ws, start_row, caption, label_headers, value_headers, rows, best="max"):
    """Generic writer for a 'paper table' block: label columns + method/value
    columns, each data cell as (display_str, raw_mean). The best raw_mean per
    row (max or min) is bolded and colored red. Returns the next free row."""
    thin = Side(style="thin", color="BFBFBF")
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    band_fill = PatternFill("solid", fgColor="F2F2F2")

    r = start_row
    ws.cell(row=r, column=1, value=caption).font = Font(bold=True, size=11)
    r += 1
    header_row = r
    all_headers = label_headers + value_headers
    for c, h in enumerate(all_headers, start=1):
        cell = ws.cell(row=header_row, column=c, value=h)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)
    r += 1

    n_label_cols = len(label_headers)
    band, prev_label0 = False, None
    for label_vals, cells in rows:
        if label_vals[0] != prev_label0:
            band = not band
            prev_label0 = label_vals[0]
        for c, lv in enumerate(label_vals, start=1):
            cell = ws.cell(row=r, column=c, value=lv)
            cell.alignment = Alignment(horizontal="left" if c == 1 else "center")
            if band:
                cell.fill = band_fill

        raw_vals = [rv for (_, rv) in cells]
        finite = [v for v in raw_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        best_val = ((max(finite) if best == "max" else min(finite)) if finite else None) if best else None

        for i, (disp, raw) in enumerate(cells):
            col = n_label_cols + 1 + i
            cell = ws.cell(row=r, column=col, value=disp)
            cell.alignment = Alignment(horizontal="center")
            if band:
                cell.fill = band_fill
            if (best_val is not None and raw is not None
                    and not (isinstance(raw, float) and np.isnan(raw))
                    and abs(raw - best_val) < 1e-9):
                cell.font = Font(bold=True, color="FF0000")
        r += 1
    return r + 1


def _collect(fn, ds, seeds, metric_key):
    """Returns (display 'mean±std' string, raw mean) across seeds for one (method, dataset, metric)."""
    means = [
        r[metric_key] for s in seeds for r in [fn(ds, s)]
        if r is not None and metric_key in r and not (isinstance(r[metric_key], float) and np.isnan(r[metric_key]))
    ]
    if not means:
        return "-", np.nan
    arr = np.array(means)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return f"{mean:.2f}±{std:.2f}", mean


def _collect_perattack(fn, ds, attack_key, seeds, metric_key):
    means = [
        r[attack_key][metric_key] for s in seeds for r in [fn(ds, s)]
        if r is not None and attack_key in r and metric_key in r[attack_key]
    ]
    if not means:
        return "-", np.nan
    arr = np.array(means)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return f"{mean:.2f}±{std:.2f}", mean


# Methods/metrics mirroring the paper's Table 2/4/7 columns. "MCC" in the
# paper pairs with the primary metric TPR@FAR=1%, which in this project's
# own schema is derived from the SAME p99 threshold as MCC_99 -- so MCC_99
# is used as "MCC" throughout this sheet.
OVERALL_METHODS = [
    ("AE",       lambda d, s: ae(d, s)),
    ("DeepSVDD", lambda d, s: deepod("dsvdd", d, s)),
    ("DROCC",    lambda d, s: drocc(d, s)),
    ("ICL",      lambda d, s: deepod("icl", d, s)),
    ("DIF",      lambda d, s: deepod("dif", d, s)),
    ("CLAN",     lambda d, s: clan(d, s)),
    ("SSAE",     lambda d, s: ssae("04", d, s)),
]
PERATTACK_METHODS = [
    ("AE",       lambda d, s: ae_perattack(d, s)),
    ("DeepSVDD", lambda d, s: deepod_perattack("dsvdd", d, s)),
    ("DROCC",    lambda d, s: drocc_perattack(d, s)),
    ("ICL",      lambda d, s: deepod_perattack("icl", d, s)),
    ("DIF",      lambda d, s: deepod_perattack("dif", d, s)),
    ("CLAN",     lambda d, s: clan_perattack(d, s)),
    ("SSAE",     lambda d, s: ssae_perattack("04", d, s)),
]
PAPER_METRICS = [("TPR@FAR=1%", "TPR@FAR=1%"), ("AUC", "AUC"), ("MCC_99", "MCC")]

PSEUDO_SOURCE_METHODS = [
    ("P-only (noise)", lambda d, s: ssae_pseudo_source("noise", d, s)),
    ("Q-only (std)",   lambda d, s: ssae_pseudo_source("std", d, s)),
    ("P+Q (t_04)",     lambda d, s: ssae("04", d, s)),
]

# The 2 paper-highlighted challenging attacks per dataset (Table 4/5),
# in the paper's own order. nsl_kdd: R2L=attack_3, U2R=attack_4 (see CLAUDE.md).
CHALLENGING_ATTACKS = {
    "unsw":    [("shellcode", "Shellcode"), ("worms", "Worms")],
    "nsl_kdd": [("attack_3", "R2L"), ("attack_4", "U2R")],
    "cic_iot": [("recon", "Recon"), ("spoofing", "Spoofing")],
    "wsn_ds":  [("blackhole", "Blackhole"), ("grayhole", "Grayhole")],
}


# ── Table [T11] -- Precision/F1/MCC at realistic attack prevalence (1/5/10%) ──
# Reviewer 5 Comment 1. Threshold = 99th percentile of VAL-normal scores (the
# "deployment" convention, NOT a test-set-derived FAR threshold -- see
# [[feedback_drop_test_far_metric]] in project memory). No retraining: reads
# raw per-sample val/test scores cached by t11_extract_core.py (SSAE/DeepOD/
# DROCC) and the sibling t11_extract_scores.py in TestDT_VChuan/ (AE) and
# CLAN-main/ (CLAN) into res/t11_cache/<method>_<dataset>_seed<seed>.npz.
T11_CACHE_DIR = os.path.join(RES_DIR, "t11_cache")
T11_METHOD_LABELS = ["AE", "DeepSVDD", "DROCC", "ICL", "DIF", "CLAN", "SSAE"]
T11_PREVALENCES = [0.01, 0.05, 0.10]
T11_N_DRAWS = 20

_T11_RAW_CACHE = {}
_T11_METRIC_CACHE = {}


def _t11_load_raw(method, dataset, seed):
    key = (method, dataset, seed)
    if key in _T11_RAW_CACHE:
        return _T11_RAW_CACHE[key]
    path = os.path.join(T11_CACHE_DIR, f"{method}_{dataset}_seed{seed}.npz")
    result = None
    if os.path.isfile(path):
        d = np.load(path)
        result = (d["val"], d["test"], d["y_test"].astype(int))
    _T11_RAW_CACHE[key] = result
    return result


def _t11_subsample_metrics(test_scores, y_true_bin, threshold, prevalence, n_draws=T11_N_DRAWS):
    normal_idx = np.where(y_true_bin == 0)[0]
    attack_idx = np.where(y_true_bin == 1)[0]
    n_normal = len(normal_idx)
    n_attack_target = int(round(prevalence * n_normal / (1 - prevalence)))
    n_attack_target = max(1, min(n_attack_target, len(attack_idx)))
    precs, f1s, mccs, aucs, tprs = [], [], [], [], []
    for draw in range(n_draws):
        rng = np.random.RandomState(1000 * int(round(prevalence * 100)) + draw)
        keep_attack = rng.choice(attack_idx, size=n_attack_target, replace=False)
        keep_idx = np.concatenate([normal_idx, keep_attack])
        sub_scores, sub_y = test_scores[keep_idx], y_true_bin[keep_idx]
        y_pred = (sub_scores > threshold).astype(int)
        precs.append(precision_score(sub_y, y_pred, zero_division=0) * 100)
        f1s.append(f1_score(sub_y, y_pred, zero_division=0) * 100)
        mccs.append(matthews_corrcoef(sub_y, y_pred) * 100 if len(np.unique(sub_y)) >= 2 else np.nan)
        # AUC is threshold-free (rank-based), evaluated on the same resample.
        aucs.append(roc_auc_score(sub_y, sub_scores) * 100 if len(np.unique(sub_y)) >= 2 else np.nan)
        # TPR@FAR=1% uses the SAME fixed val-p99 threshold as MCC (deployment convention),
        # never a threshold re-derived from the resampled set.
        tprs.append(recall_score(sub_y, y_pred, zero_division=0) * 100)
    return (float(np.mean(precs)), float(np.mean(f1s)), float(np.nanmean(mccs)),
            float(np.nanmean(aucs)), float(np.mean(tprs)))


def t11_reader(method, dataset, seed, prevalence):
    key = (method, dataset, seed, prevalence)
    if key in _T11_METRIC_CACHE:
        return _T11_METRIC_CACHE[key]
    loaded = _t11_load_raw(method, dataset, seed)
    if loaded is None:
        _T11_METRIC_CACHE[key] = None
        return None
    val, test, y = loaded
    threshold = float(np.percentile(val, 99))
    prec, f1, mcc, auc, tpr = _t11_subsample_metrics(test, y, threshold, prevalence)
    result = {"Precision": prec, "F1": f1, "MCC": mcc, "AUC": auc, "TPR@FAR=1%": tpr}
    _T11_METRIC_CACHE[key] = result
    return result


def gather_table11_prevalence(seeds, prevalence):
    """Table [T11] sub-table for one prevalence level -- Precision/F1/AUC/
    TPR@FAR=1%/MCC, all 7 methods, mean+/-std across seeds (each seed's value
    already averaged over T11_N_DRAWS random prevalence-resamples). MCC is
    evaluated at the same fixed val-p99 threshold as TPR@FAR=1%; AUC is
    threshold-free (rank-based) on the same resample."""
    value_headers = T11_METHOD_LABELS
    rows = []
    for ds in DATASETS:
        for metric in ["Precision", "F1", "AUC", "TPR@FAR=1%", "MCC"]:
            cells = [
                _collect((lambda d, s, m=m: t11_reader(m, d, s, prevalence)), ds, seeds, metric)
                for m in T11_METHOD_LABELS
            ]
            rows.append(([DATASET_DISPLAY[ds], metric], cells))
    return value_headers, rows


def build_t11_prevalence_sheet(writer, seeds):
    """Standalone sheet with the full Table [T11] breakdown (3 sub-tables,
    one per prevalence level)."""
    ws = writer.book.create_sheet("9_Prevalence_T11")
    ws.cell(row=1, column=1,
            value=f"Table [T11] -- Precision/F1/AUC/TPR@FAR=1%/MCC at realistic attack prevalence "
                  f"(val-normal p99 threshold, {T11_N_DRAWS} resamples/seed) -- seeds {seeds[0]}-{seeds[-1]}"
            ).font = Font(bold=True, size=13)
    r = 3
    for prevalence in T11_PREVALENCES:
        vh, rows = gather_table11_prevalence(seeds, prevalence)
        r = write_labeled_table(ws, r, f"Prevalence = {int(round(prevalence * 100))}% attack traffic",
                                 ["Dataset", "Metric"], vh, rows, best="max")
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 13
    for c in range(3, 3 + len(T11_METHOD_LABELS)):
        ws.column_dimensions[get_column_letter(c)].width = 13
    ws.freeze_panes = "A4"


def gather_table2(seeds):
    """Paper Table 2 -- overall performance across 4 datasets."""
    value_headers = [label for label, _ in OVERALL_METHODS]
    rows = []
    for ds in DATASETS:
        for mkey, mdisp in PAPER_METRICS:
            cells = [_collect(fn, ds, seeds, mkey) for _, fn in OVERALL_METHODS]
            rows.append(([DATASET_DISPLAY[ds], mdisp], cells))
    return value_headers, rows


def _average_ranks(methods, metric_keys, per_group_means):
    """per_group_means: list of {method_label: mean|nan} dicts, one per group
    (dataset, or (dataset, attack)). Returns {metric_key: {method_label: avg_rank}}."""
    labels = [label for label, _ in methods]
    raw = {mkey: {m: [] for m in labels} for mkey in metric_keys}
    for mkey, group_vals in per_group_means:
        s = pd.Series(group_vals)
        ranks = s.rank(ascending=False, na_option="bottom")
        for label in labels:
            raw[mkey][label].append(ranks[label])
    return raw


def gather_table3(seeds):
    """Paper Table 3 -- average ranks across 4 datasets (lower = better)."""
    labels = [label for label, _ in OVERALL_METHODS]
    methods_by_label = dict(OVERALL_METHODS)
    metric_keys = [mkey for mkey, _ in PAPER_METRICS]
    metric_disp = dict(PAPER_METRICS)

    def mean_val(label, mkey, ds):
        return _collect(methods_by_label[label], ds, seeds, mkey)[1]

    per_group = []
    for ds in DATASETS:
        for mkey in metric_keys:
            vals = {label: mean_val(label, mkey, ds) for label in labels}
            per_group.append((mkey, vals))
    raw = _average_ranks(OVERALL_METHODS, metric_keys, per_group)
    rows = []
    for mkey in metric_keys:
        cells = []
        for label in labels:
            mean_rank = float(np.mean(raw[mkey][label]))
            cells.append((f"{mean_rank:.2f}", mean_rank))
        rows.append(([metric_disp[mkey]], cells))
    return labels, rows


def gather_table4(seeds):
    """Paper Table 4 -- performance on the 8 challenging attacks (2 per dataset)."""
    value_headers = [label for label, _ in PERATTACK_METHODS]
    rows = []
    for ds in DATASETS:
        for attack_key, attack_disp in CHALLENGING_ATTACKS[ds]:
            for mkey, mdisp in PAPER_METRICS:
                cells = [_collect_perattack(fn, ds, attack_key, seeds, mkey) for _, fn in PERATTACK_METHODS]
                rows.append(([DATASET_DISPLAY[ds], attack_disp, mdisp], cells))
    return value_headers, rows


def gather_table5(seeds):
    """Paper Table 5 -- average ranks across the 8 challenging attacks (lower = better)."""
    labels = [label for label, _ in PERATTACK_METHODS]
    metric_keys = [mkey for mkey, _ in PAPER_METRICS]
    metric_disp = dict(PAPER_METRICS)
    per_group = []
    for ds in DATASETS:
        for attack_key, _ in CHALLENGING_ATTACKS[ds]:
            for mkey in metric_keys:
                vals = {label: _collect_perattack(fn, ds, attack_key, seeds, mkey)[1]
                        for label, fn in PERATTACK_METHODS}
                per_group.append((mkey, vals))
    raw = _average_ranks(PERATTACK_METHODS, metric_keys, per_group)
    rows = []
    for mkey in metric_keys:
        cells = []
        for label in labels:
            mean_rank = float(np.mean(raw[mkey][label]))
            cells.append((f"{mean_rank:.2f}", mean_rank))
        rows.append(([metric_disp[mkey]], cells))
    return labels, rows


def gather_table9():
    """Paper Table 9 -- SSAE AUC/TPR@FAR=1%/MCC vs pseudo-intrusion source
    (P-only / Q-only / P+Q = full model). ALWAYS uses PSEUDO_SOURCE_SEEDS
    (42-51, full 10 seeds) regardless of the seeds argument elsewhere in
    this sheet."""
    seeds = PSEUDO_SOURCE_SEEDS
    value_headers = [label for label, _ in PSEUDO_SOURCE_METHODS]
    rows = []
    for ds in DATASETS:
        for mkey, mdisp in PAPER_METRICS:
            cells = [_collect(fn, ds, seeds, mkey) for _, fn in PSEUDO_SOURCE_METHODS]
            rows.append(([DATASET_DISPLAY[ds], mdisp], cells))
    return value_headers, rows


def gather_table5_winloss(seeds, tie_margin=0.5):
    """Win/tie/loss counts of SSAE vs each baseline on TPR@FAR=1%, over the 8
    challenging attacks (CHALLENGING_ATTACKS, mean-across-seeds values, same
    source as gather_table4/5). A tie is |diff| <= tie_margin percentage
    points. Descriptive companion to gather_table5's average ranks (Reviewer
    1 Comment 3: Table 5 must be explicitly descriptive, not an inferential
    test, since the 8 attack classes are nested within 4 datasets)."""
    methods = dict(PERATTACK_METHODS)
    baseline_labels = [label for label in methods if label != "SSAE"]
    ssae_fn = methods["SSAE"]
    counts = {label: {"Win": 0, "Tie": 0, "Loss": 0} for label in baseline_labels}
    for ds in DATASETS:
        for attack_key, _ in CHALLENGING_ATTACKS[ds]:
            _, ssae_val = _collect_perattack(ssae_fn, ds, attack_key, seeds, "TPR@FAR=1%")
            for label in baseline_labels:
                _, base_val = _collect_perattack(methods[label], ds, attack_key, seeds, "TPR@FAR=1%")
                if np.isnan(ssae_val) or np.isnan(base_val):
                    continue
                diff = ssae_val - base_val
                if abs(diff) <= tie_margin:
                    counts[label]["Tie"] += 1
                elif diff > 0:
                    counts[label]["Win"] += 1
                else:
                    counts[label]["Loss"] += 1
    value_headers = baseline_labels
    rows = []
    for outcome in ("Win", "Tie", "Loss"):
        cells = [(str(counts[label][outcome]), counts[label][outcome]) for label in baseline_labels]
        rows.append(([outcome], cells))
    return value_headers, rows


# All 7 methods under the COMMON (CTGAN-preprocessed) representation used by
# Table T1/T2 (reviewer1.tex, Reviewer 1 Comment 2). AE/ICL/CLAN/SSAE use
# 10-seed means (42-51); DeepSVDD/DROCC/DIF use 7-seed means (45-51) since
# those three were only retrained on this representation for seeds 45-51,
# with library/paper-default hyperparameters (not yet retuned for this
# input -- same caveat as Table T1's own caption).
PERATTACK_METHODS_COMMON_REPR = [
    ("AE",       list(range(42, 52)), lambda d, s: ae_perattack(d, s, ctgan=True)),
    ("DeepSVDD", list(range(45, 52)), lambda d, s: deepod_perattack("dsvdd", d, s, ctgan=True)),
    ("DROCC",    list(range(45, 52)), lambda d, s: drocc_ctgan_perattack(d, s)),
    ("ICL",      list(range(42, 52)), lambda d, s: deepod_perattack("icl", d, s, ctgan=True)),
    ("DIF",      list(range(45, 52)), lambda d, s: deepod_perattack("dif", d, s, ctgan=True)),
    ("CLAN",     list(range(42, 52)), lambda d, s: clan_perattack(d, s, ctgan=True)),
    ("SSAE",     list(range(42, 52)), lambda d, s: ssae_perattack("04", d, s)),
]


def gather_best_or_tied_best_common_repr(tie_margin=0.5):
    """For each of the 8 challenging attacks, is SSAE best or tied-best
    (within tie_margin points) on TPR@FAR=1% under the COMMON representation
    (Table T1), against the single best of the other 6 methods? Mirrors the
    published-representation statistic already reported to Reviewer 1
    Comment 3 / Reviewer 5 Comment 6 ("best or tied-best on six of the eight
    challenging classes"), but recomputed with every method sharing the same
    CTGAN-preprocessed input. Returns (count, total, per-class detail list)."""
    methods = {label: (seeds, fn) for label, seeds, fn in PERATTACK_METHODS_COMMON_REPR}
    detail = []
    count = 0
    total = 0
    for ds in DATASETS:
        for attack_key, attack_disp in CHALLENGING_ATTACKS[ds]:
            vals = {}
            for label, (seeds, fn) in methods.items():
                _, v = _collect_perattack(fn, ds, attack_key, seeds, "TPR@FAR=1%")
                if not np.isnan(v):
                    vals[label] = v
            if "SSAE" not in vals or len(vals) < 2:
                detail.append((ds, attack_disp, vals, None))
                continue
            total += 1
            best_other = max(v for label, v in vals.items() if label != "SSAE")
            is_best_or_tied = vals["SSAE"] >= best_other - tie_margin
            if is_best_or_tied:
                count += 1
            detail.append((ds, attack_disp, vals, is_best_or_tied))
    return count, total, detail


def gather_table7(seeds):
    """Paper Table 7 -- AE/ICL/CLAN before vs after CTGAN preprocessing, vs SSAE."""
    methods = [
        ("AE (before)",   lambda d, s: ae(d, s)),
        ("AE (after)",    lambda d, s: ae(d, s, ctgan=True)),
        ("ICL (before)",  lambda d, s: deepod("icl", d, s)),
        ("ICL (after)",   lambda d, s: deepod("icl", d, s, ctgan=True)),
        ("CLAN (before)", lambda d, s: clan(d, s)),
        ("CLAN (after)",  lambda d, s: clan(d, s, ctgan=True)),
        ("SSAE",          lambda d, s: ssae("04", d, s)),
    ]
    value_headers = [label for label, _ in methods]
    rows = []
    for ds in DATASETS:
        for mkey, mdisp in PAPER_METRICS:
            cells = [_collect(fn, ds, seeds, mkey) for _, fn in methods]
            rows.append(([DATASET_DISPLAY[ds], mdisp], cells))
    return value_headers, rows


def gather_table8(seeds):
    """Paper Table 8 -- SSAE AUC vs gate-opening strategy (t_06 ablation:
    fixed epoch 0/20/40 vs the adaptive gate used everywhere else, t_04)."""
    value_headers = ["Fixed epoch 0", "Fixed epoch 20", "Fixed epoch 40", "Adaptive"]
    fns = [
        lambda d, s: ssae_gatefix(0, d, s),
        lambda d, s: ssae_gatefix(20, d, s),
        lambda d, s: ssae_gatefix(40, d, s),
        lambda d, s: ssae("04", d, s),
    ]
    rows = []
    for ds in DATASETS:
        cells = [_collect(fn, ds, seeds, "AUC") for fn in fns]
        rows.append(([DATASET_DISPLAY[ds]], cells))
    return value_headers, rows


def gather_table10(seeds):
    """Paper Table 10 -- SSAE AUC vs margin m, lambda=0.1 fixed."""
    value_headers = [f"m={m}" for m in MARGINS]
    rows = []
    for ds in DATASETS:
        cells = []
        for m in MARGINS:
            tag = "04" if m == "2.0" else "02"
            fn = (lambda m, tag: lambda d, s: ssae(tag, d, s, m=m, lam="0.1"))(m, tag)
            cells.append(_collect(fn, ds, seeds, "AUC"))
        rows.append(([DATASET_DISPLAY[ds]], cells))
    return value_headers, rows


def gather_table11(seeds):
    """Paper Table 11 -- SSAE AUC vs lambda, m=2.0 fixed."""
    value_headers = [f"lambda={lam}" for lam in LAMBDAS]
    rows = []
    for ds in DATASETS:
        cells = []
        for lam in LAMBDAS:
            tag = "04" if lam == "0.1" else "01"
            fn = (lambda lam, tag: lambda d, s: ssae(tag, d, s, m="2.0", lam=lam))(lam, tag)
            cells.append(_collect(fn, ds, seeds, "AUC"))
        rows.append(([DATASET_DISPLAY[ds]], cells))
    return value_headers, rows


def build_tonghopchung_sheet(writer, seeds):
    """New sheet mirroring the paper's Tables 2/3/4/5/7/8/10/11, recomputed
    from all available seeds (not just the paper's original 3) -- best value
    per row in bold red. Table 9 (pseudo-source ablation, t_05) is skipped
    (not finished)."""
    ws = writer.book.create_sheet("TongHopChung")
    ws.cell(row=1, column=1,
            value=f"Tong hop cac bang trong bai bao SSAE -- seeds {seeds[0]}-{seeds[-1]} (n={len(seeds)})"
            ).font = Font(bold=True, size=13)
    r = 3

    vh, rows = gather_table2(seeds)
    r = write_labeled_table(ws, r, "Table 2 - Overall performance across 4 datasets",
                             ["Dataset", "Metric"], vh, rows, best="max")

    vh, rows = gather_table3(seeds)
    r = write_labeled_table(ws, r, "Table 3 - Average ranks across 4 datasets (lower is better)",
                             ["Metric"], vh, rows, best="min")

    vh, rows = gather_table4(seeds)
    r = write_labeled_table(ws, r, "Table 4 - Performance on the 8 challenging attacks",
                             ["Dataset", "Attack", "Metric"], vh, rows, best="max")

    vh, rows = gather_table5(seeds)
    r = write_labeled_table(ws, r, "Table 5 - Average ranks across the 8 challenging attacks (lower is better)",
                             ["Metric"], vh, rows, best="min")

    vh, rows = gather_table5_winloss(seeds)
    r = write_labeled_table(ws, r, "Table 5b - SSAE win/tie/loss vs baselines on TPR@FAR=1% "
                             "(8 challenging attacks, tie margin = 0.5pp) -- descriptive only, no inferential test",
                             ["Outcome"], vh, rows, best=None)

    vh, rows = gather_table7(seeds)
    r = write_labeled_table(ws, r, "Table 7 - Effect of CTGAN preprocessing (AE / ICL / CLAN) vs SSAE",
                             ["Dataset", "Metric"], vh, rows, best="max")

    vh, rows = gather_table8(seeds)
    r = write_labeled_table(ws, r, "Table 8 - Effect of the gate-opening strategy in SSAE (AUC %)",
                             ["Dataset"], vh, rows, best="max")

    vh, rows = gather_table9()
    r = write_labeled_table(ws, r, "Table 9 - SSAE vs pseudo-intrusion source (P-only / Q-only / "
                             "P+Q=full model) -- seeds 42-51 (mean +/- std, 10 seeds)",
                             ["Dataset", "Metric"], vh, rows, best="max")

    vh, rows = gather_table10(seeds)
    r = write_labeled_table(ws, r, "Table 10 - SSAE AUC vs margin m (lambda=0.1)",
                             ["Dataset"], vh, rows, best="max")

    vh, rows = gather_table11(seeds)
    r = write_labeled_table(ws, r, "Table 11 - SSAE AUC vs lambda (m=2.0)",
                             ["Dataset"], vh, rows, best="max")

    for prevalence in T11_PREVALENCES:
        vh, rows = gather_table11_prevalence(seeds, prevalence)
        r = write_labeled_table(
            ws, r,
            f"Table [T11] - Precision/F1/AUC/TPR@FAR=1%/MCC at {int(round(prevalence * 100))}% attack prevalence "
            f"(val-normal p99 threshold, {T11_N_DRAWS} resamples/seed, Reviewer 5 Comment 1)",
            ["Dataset", "Metric"], vh, rows, best="max")

    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 13
    n_val_cols = max(len(OVERALL_METHODS), len(PERATTACK_METHODS), 7)
    for c in range(4, 4 + n_val_cols):
        ws.column_dimensions[get_column_letter(c)].width = 13
    ws.freeze_panes = "A4"


def main():
    parser = argparse.ArgumentParser("Per-seed summary Excel (overall / lambda / margin / CTGAN-preprocessing)")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in range(42, 52)))
    parser.add_argument("--out", type=str, default=os.path.join(RES_DIR, "summary_seeds42-51.xlsx"))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    sheets = [
        ("1_Overall_t04",
         "Overall results -- SSAE (t_04, m=2.0, lambda=0.1) vs baselines",
         [
             ("AE",       lambda d, s: ae(d, s)),
             ("DeepSVDD", lambda d, s: deepod("dsvdd", d, s)),
             ("DROCC",    lambda d, s: drocc(d, s)),
             ("ICL",      lambda d, s: deepod("icl", d, s)),
             ("DIF",      lambda d, s: deepod("dif", d, s)),
             ("CLAN",     lambda d, s: clan(d, s)),
             ("SSAE",     lambda d, s: ssae("04", d, s)),
         ],
         METRICS),
        ("2_Lambda_t01",
         "SSAE lambda ablation (t_01, m=2.0); lambda=0.1 = main config from t_04",
         [(f"lambda={lam}" + (" (t_04)" if lam == "0.1" else ""),
           (lambda lam: lambda d, s: ssae("04" if lam == "0.1" else "01", d, s, lam=lam))(lam))
          for lam in LAMBDAS], None),
        ("3_Margin_t02",
         "SSAE margin ablation (t_02, lambda=0.1); m=2.0 = main config from t_04",
         [(f"m={m}" + (" (t_04)" if m == "2.0" else ""),
           (lambda m: lambda d, s: ssae("04" if m == "2.0" else "02", d, s, m=m))(m))
          for m in MARGINS], None),
        ("4_CTGAN_preproc",
         "AE, ICL and CLAN before and after applying CTGAN preprocessing (SSAE shown for reference)",
         [
             ("AE (original)",             lambda d, s: ae(d, s)),
             ("AE (CTGAN-preprocessed)",   lambda d, s: ae(d, s, ctgan=True)),
             ("ICL (original)",           lambda d, s: deepod("icl", d, s)),
             ("ICL (CTGAN-preprocessed)", lambda d, s: deepod("icl", d, s, ctgan=True)),
             ("CLAN (original)",           lambda d, s: clan(d, s)),
             ("CLAN (CTGAN-preprocessed)", lambda d, s: clan(d, s, ctgan=True)),
             ("SSAE",                      lambda d, s: ssae("04", d, s)),
         ], None),
        ("06_fixgate",
         "Gate-opening ablation (t_06) -- AUC only, fixed epoch 0/20/40 vs the adaptive gate (t_04)",
         [
             ("Fixed epoch 0",  lambda d, s: ssae_gatefix(0, d, s)),
             ("Fixed epoch 20", lambda d, s: ssae_gatefix(20, d, s)),
             ("Fixed epoch 40", lambda d, s: ssae_gatefix(40, d, s)),
             ("Adaptive (t_04)", lambda d, s: ssae("04", d, s)),
         ], ["AUC"]),
    ]

    perattack_sheet = (
        "5_PerAttack_t04",
        "Per-attack breakdown -- SSAE (t_04, m=2.0, lambda=0.1) vs baselines",
        [
            ("SSAE",     lambda d, s: ssae_perattack("04", d, s)),
            ("DeepSVDD", lambda d, s: deepod_perattack("dsvdd", d, s)),
            ("GOAD",     lambda d, s: deepod_perattack("goad", d, s)),
            ("NeuTraL",  lambda d, s: deepod_perattack("neutral", d, s)),
            ("ICL",      lambda d, s: deepod_perattack("icl", d, s)),
            ("DIF",      lambda d, s: deepod_perattack("dif", d, s)),
            ("DROCC",    lambda d, s: drocc_perattack(d, s)),
            ("CLAN",     lambda d, s: clan_perattack(d, s)),
            ("AE",       lambda d, s: ae_perattack(d, s)),
        ],
    )

    # Seeds 45-51 only: DeepSVDD/GOAD/NeuTraL/DIF/ICL (DeepOD) and DROCC on the
    # CTGAN-preprocessed input only have this subset trained (seeds 42-44 were
    # never run for this variant, except ICL and a stray dif/unsw/seed42).
    # Reported as its own sheet, on its own seed range, rather than folded
    # into "4_CTGAN_preproc" (whose other rows cover seeds 42-51).
    ctgan45_51_sheet = (
        "07_CTGAN_pre",
        "CTGAN-preprocessed baselines, seeds 45-51 only (DeepSVDD/GOAD/NeuTraL/DIF/DROCC/ICL "
        "not trained on this variant for seeds 42-44) -- all baselines use default "
        "hyperparameters on this input (DROCC reuses the per-dataset inp_radius/lr tuned "
        "for the ORIGINAL input, not retuned for CTGAN-preprocessed; its cic_iot run "
        "diverges to NaN under this reused lr for all 7 seeds -- see per-seed train_log.txt "
        "in DROCC_icml2020/tabular/log_ssae_ctgan_preprocessed/cic_iot/, best_auc checkpoint "
        "is from before the crash, not a fully-trained model)",
        [
            ("DeepSVDD", lambda d, s: deepod("dsvdd", d, s, ctgan=True)),
            ("GOAD",     lambda d, s: deepod("goad", d, s, ctgan=True)),
            ("NeuTraL",  lambda d, s: deepod("neutral", d, s, ctgan=True)),
            ("DIF",      lambda d, s: deepod("dif", d, s, ctgan=True)),
            ("ICL",      lambda d, s: deepod("icl", d, s, ctgan=True)),
            ("DROCC",    lambda d, s: drocc_ctgan(d, s)),
            ("SSAE",     lambda d, s: ssae("04", d, s)),
        ],
        None,
    )
    ctgan45_51_seeds = list(range(45, 52))

    # Pseudo-intrusion source ablation (t_05): P-only / Q-only, vs the P+Q
    # full model (t_04). Full 10 seeds (42-51) as of 2026-11-09; this sheet
    # ALWAYS uses PSEUDO_SOURCE_SEEDS regardless of --seeds.
    pseudo_source_sheet = (
        "8_PseudoSource_t05",
        f"Pseudo-intrusion source ablation (t_05): P-only (real-normal+noise) / "
        f"Q-only (CTGAN-sample+noise) vs the P+Q full model (t_04) -- seeds 42-51 "
        f"(mean +/- std, 10 seeds)",
        PSEUDO_SOURCE_METHODS,
        None,
    )

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for sheet_name, title, methods, sheet_metrics in sheets:
            df, missing = build_table(methods, seeds, metrics=sheet_metrics)
            write_sheet(writer, df, sheet_name, title)
            print(f"{sheet_name}: {len(df)} rows, {len(missing)} missing run(s)")
            for m in missing:
                print(f"    missing: {m}")

        sheet_name, title, methods, sheet_metrics = ctgan45_51_sheet
        df, missing = build_table(methods, ctgan45_51_seeds, metrics=sheet_metrics)
        write_sheet(writer, df, sheet_name, title)
        print(f"{sheet_name}: {len(df)} rows, {len(missing)} missing run(s)")
        for m in missing:
            print(f"    missing: {m}")

        sheet_name, title, methods, sheet_metrics = pseudo_source_sheet
        df, missing = build_table(methods, PSEUDO_SOURCE_SEEDS, metrics=sheet_metrics)
        write_sheet(writer, df, sheet_name, title)
        print(f"{sheet_name}: {len(df)} rows, {len(missing)} missing run(s)")
        for m in missing:
            print(f"    missing: {m}")

        sheet_name, title, methods = perattack_sheet
        df, missing = build_per_attack_table(methods, seeds)
        write_sheet_perattack(writer, df, sheet_name, title)
        print(f"{sheet_name}: {len(df)} rows, {len(missing)} missing run(s)")
        for m in missing:
            print(f"    missing: {m}")

        build_tonghopchung_sheet(writer, seeds)
        print(f"TongHopChung: built (Tables 2,3,4,5,5b,7,8,9,10,11,[T11]; {len(seeds)} seeds)")

        build_t11_prevalence_sheet(writer, seeds)
        print("9_Prevalence_T11: built")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
