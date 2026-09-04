"""Standalone reimplementation of CLAN's data.load_data.get_data() --
covers only what prepare_drocc_data.py / eval_drocc.py need (the 4
CLAN-comparison datasets: unsw, nsl_kdd, cic_iot, wsn_ds), reading from
THIS project's own Syn_ctgan/data/ raw CSVs instead of importing anything
from the sibling CLAN-main repo. Written to keep DROCC's data prep runnable
on machines that never had CLAN-main copied onto them (e.g. a separate
Ubuntu training box) -- no code or data dependency on that repo at all.

Faithfully mirrors the source logic (verified against
CLAN-main/data/load_data.py, byte-identical raw CSVs confirmed via `wc -l`
against Syn_ctgan/data/):
  - read CSV with header; if `target` isn't a resulting column, reread with
    header=None and use the last (integer-named) column as target -- this is
    how NSL-KDD and CIC-IoT (headerless, numeric-coded/last-column label)
    differ from UNSW-NB15 and WSN-DS (headed, named target column);
  - drop inf/nan rows; drop requested columns; drop non-numeric columns
    (rather than encoding them -- e.g. UNSW's proto/service are dropped, not
    one-hot'd);
  - multiclass label encoding: 0 = normal/class_zero (case-insensitive,
    stripped), other labels assigned incrementing ids as encountered;
  - if a same-shaped "testing"/"test" counterpart file exists next to
    data_path (checked via the same suffix-swap patterns CLAN uses), it is
    used as the REAL held-out test set (unsw, nsl_kdd, cic_iot all have
    one); otherwise a stratified test_ratio split of the single file is used
    (wsn_ds has no separate test file, so this applies there);
  - attack types with fewer than sample_thres rows in the training pool are
    pulled out into a separate x_zd/y_zd ("zero-day") bucket rather than
    used for training;
  - anomaly_detection=True restricts the training pool to normal rows only,
    THEN (if percentage < 1.0) subsamples that pool down to `percentage` of
    its rows, THEN carves out val_ratio of what's left as x_val -- exactly
    SSAE's own order of operations (train_mae01scale_max.py:
    `X_train_normal_full.sample(frac=percentage, random_state=seed)` runs
    BEFORE the 80/20 train/val split, not after -- so a smaller percentage
    shrinks both train and val together, matching the `per_<percentage>`
    result-folder convention used throughout this project);
  - a MinMaxScaler is fit ONLY on the final (post-val-split) normal training
    rows and applied to train/val/test/zero-day alike.

Split convention matches Syn_ctgan's own SSAE loaders (process_data.py /
train_mae01scale_max.py), NOT CLAN's own arbitrary defaults, wherever the two
differ -- verified by reading both:
  - unsw / nsl_kdd / cic_iot: SSAE's df_full is exactly
    concat(raw "training-set" file, raw "testing-set" file), then
    df_full.iloc[:-n_test] / iloc[-n_test:] recovers those two original
    files unchanged (inf/nan-drop is row-independent, so this holds exactly)
    -- i.e. SSAE's real train/test split for these 3 is byte-identical to
    just using the two original files directly, which is what
    _infer_test_path()'s branch below already does.
  - wsn_ds: SSAE's load_data_wsn_ds() has NO separate test file -- it does
    its own stratified train_test_split(..., test_size=0.2, random_state=seed)
    of the single WSN-DS.csv. CLAN's own DATASETS config instead falls back
    to a generic test_ratio=0.5 for any dataset without a real test file,
    which is wrong for wsn_ds specifically (was silently giving DROCC a
    50/50 split here instead of SSAE's 80/20 -- a ~2.5x larger test set than
    SSAE's own evaluation uses). test_ratio is now 0.2 everywhere it's
    called from (only wsn_ds actually exercises this branch).
  - Random seeding: SSAE reuses the SAME raw `seed` for both the wsn_ds
    train/test split and the normal-only train/val split (no derived
    secondary seed). This module now does the same -- `split_seed` is
    passed straight through as random_state to both splits below, instead
    of CLAN's original random.Random(split_seed)-derived two-seed scheme.
  - wsn_ds's fallback test split also now stratifies by the BINARY label
    (normal vs. any-attack), matching SSAE's `stratify=y` where y is already
    binary at that point in load_data_wsn_ds() -- not by the multiclass
    per-attack-type label this module otherwise carries internally.

Usage: from clan_style_get_data import get_data, DATASETS
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).parent
DATA_ROOT = BASE_DIR / "data"

DATASETS = {
    "unsw": dict(
        data_path=DATA_ROOT / "unsw/UNSW_NB15_training-set.csv",
        target="attack_cat", class_zero="Normal", drop_cols=["id"], total_rows=175341,
    ),
    "nsl_kdd": dict(
        data_path=DATA_ROOT / "NSLKDD/NSLKDD_Train.csv",
        target="label", class_zero="0", drop_cols=[], total_rows=125973,
    ),
    "cic_iot": dict(
        data_path=DATA_ROOT / "cic_iot/train.csv",
        target="label", class_zero="Benign", drop_cols=[], total_rows=418644,
    ),
    "wsn_ds": dict(
        data_path=DATA_ROOT / "wsn-ds/WSN-DS.csv",
        target="Attack type", class_zero="Normal", drop_cols=["id", "Time"], total_rows=374661,
    ),
}
SAMPLE_THRES = 100


def _infer_test_path(data_path):
    path = Path(data_path)
    for old, new in (
        ("training-set", "testing-set"), ("Training-set", "Testing-set"),
        ("TRAINING-set", "TESTING-set"), ("Train", "Test"), ("train", "test"), ("TRAIN", "TEST"),
    ):
        if old in path.name:
            candidate = path.with_name(path.name.replace(old, new))
            if candidate.exists():
                return str(candidate)
    return None


def _read_csv(data_path, target):
    data_df = pd.read_csv(data_path)
    data_df.columns = data_df.columns.str.strip()
    if target in data_df.columns:
        return data_df, target
    data_df = pd.read_csv(data_path, header=None)
    return data_df, data_df.columns[-1]


def _clean_frame(data_df):
    data_df = data_df.replace([np.inf, -np.inf], np.nan)
    return data_df.dropna(axis=0).reset_index(drop=True)


def _expanded_drop_columns(drop):
    drop_cols = []
    for col in drop or []:
        if col is None or col == "":
            continue
        drop_cols.append(col)
        if isinstance(col, str) and col.lstrip("-").isdigit():
            drop_cols.append(int(col))
    return drop_cols


def _prepare_xy(data_path, target, drop, n_samples=None, sample_seed=42):
    data_df, target_col = _read_csv(data_path, target)
    data_df = _clean_frame(data_df)
    if n_samples is not None and n_samples < len(data_df):
        data_df = data_df.sample(n=n_samples, random_state=sample_seed).reset_index(drop=True)
    y_data = data_df[target_col].to_numpy()
    x_df = data_df.drop(columns=[target_col])
    drop_cols = _expanded_drop_columns(drop)
    if drop_cols:
        x_df = x_df.drop(columns=drop_cols, errors="ignore")
    if target_col != "label" and "label" in x_df.columns:
        x_df = x_df.drop(columns=["label"])
    non_numeric = x_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        x_df = x_df.drop(columns=non_numeric)
    return x_df.to_numpy(dtype=np.float32), y_data, list(x_df.columns)


def _label_key(value):
    return str(value).strip().lower()


def _is_normal_label(value, class_zero):
    keys = {"0", "0.0", "normal", "benign"}
    if class_zero is not None:
        keys.add(str(class_zero).strip().lower())
    return _label_key(value) in keys


def _make_multiclass_labels(y_data, class_zero, label_mapping=None):
    label_mapping = {} if label_mapping is None else dict(label_mapping)
    for v in y_data:
        if _is_normal_label(v, class_zero):
            label_mapping[_label_key(v)] = 0
    for v in np.unique(y_data):
        k = _label_key(v)
        if k not in label_mapping:
            label_mapping[k] = len(set(label_mapping.values()))
    y_enc = np.asarray([label_mapping[_label_key(v)] for v in y_data], dtype=np.int64)
    return y_enc, label_mapping


def _filter_small_classes(x_data, y_data, sample_thres):
    sample_thres = sample_thres or 0
    if sample_thres <= 0:
        return x_data, y_data, None, None
    vals, counts = np.unique(y_data, return_counts=True)
    drop_vals = vals[(counts < sample_thres) & (vals != 0)]
    drop_idx = np.where(np.isin(y_data, drop_vals))[0]
    if len(drop_idx) == 0:
        return x_data, y_data, None, None
    x_zd, y_zd = x_data[drop_idx], y_data[drop_idx]
    return np.delete(x_data, drop_idx, 0), np.delete(y_data, drop_idx), x_zd, y_zd


def _subsample_frac(x_train, y_train, percentage, seed):
    """Matches SSAE's X_train_normal_full.sample(frac=percentage,
    random_state=seed).index exactly -- same pandas RNG, same seed, applied
    to a fresh 0..N-1 RangeIndex, so the same fraction/mechanism is used
    even though the specific rows differ (the two pipelines' upstream row
    order already diverges before this point)."""
    idx = pd.DataFrame(index=range(len(x_train))).sample(frac=percentage, random_state=seed).index.to_numpy()
    return x_train[idx], y_train[idx]


def _split_val(x_train, y_train, val_ratio, random_state):
    if val_ratio <= 0.0:
        return x_train, y_train, None, None
    stratify = y_train if len(np.unique(y_train)) > 1 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_ratio, random_state=random_state,
        shuffle=True, stratify=stratify,
    )
    return x_train, y_train, x_val, y_val


def get_data(data_path, target, drop, class_zero, sample_thres, split_seed,
             test_ratio, val_ratio, anomaly_detection, n_samples, sample_seed,
             percentage=1.0):
    """Returns (x_train, y_train, x_val, y_val, x_test, y_test, x_zd, y_zd,
    label_mapping, feature_names). y_* convention: 0 = normal, 1..K = attack
    types (multiclass; binary is just `y != 0`).

    percentage: fraction of the normal-only training pool to keep, applied
    BEFORE the train/val split -- see module docstring."""
    x_data, y_data, feature_names = _prepare_xy(
        data_path=data_path, target=target, drop=drop,
        n_samples=n_samples, sample_seed=sample_seed,
    )
    y_data, label_mapping = _make_multiclass_labels(y_data, class_zero=class_zero)
    x_data, y_data, x_zd, y_zd = _filter_small_classes(x_data, y_data, sample_thres)

    # SSAE reuses the same raw seed for both the (wsn_ds-only) test split and
    # the normal-only val split below -- see module docstring.
    test_data_path = _infer_test_path(data_path)
    if test_data_path is not None:
        x_train, y_train = x_data, y_data
        x_test, y_test, _ = _prepare_xy(data_path=test_data_path, target=target, drop=drop)
        y_test, label_mapping = _make_multiclass_labels(y_test, class_zero=class_zero, label_mapping=label_mapping)
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_ratio, random_state=split_seed,
            shuffle=True, stratify=(y_data != 0),
        )

    if anomaly_detection:
        normal_mask = y_train == 0
        if not np.any(normal_mask):
            raise ValueError("No normal samples found in training set.")
        x_train, y_train = x_train[normal_mask], y_train[normal_mask]

    if percentage < 1.0:
        x_train, y_train = _subsample_frac(x_train, y_train, percentage, split_seed)

    x_train, y_train, x_val, y_val = _split_val(x_train, y_train, val_ratio, split_seed)
    normal_mask = y_train == 0
    if not np.any(normal_mask):
        raise ValueError("No normal samples found in training set to fit MinMaxScaler.")

    scaler = MinMaxScaler()
    scaler.fit(x_train[normal_mask])
    x_val = scaler.transform(x_val) if x_val is not None else None
    x_test = scaler.transform(x_test) if x_test is not None else None
    x_zd = scaler.transform(x_zd) if x_zd is not None else None
    x_train = scaler.transform(x_train)

    return x_train, y_train, x_val, y_val, x_test, y_test, x_zd, y_zd, label_mapping, feature_names
