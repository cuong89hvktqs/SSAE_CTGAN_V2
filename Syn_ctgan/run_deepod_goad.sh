#!/bin/bash
# Trains + evaluates GOAD only (dsvdd/neutral/icl/dif have their own runs
# already, or use run_deepod_datasets.sh for all 5) on this project's 4
# datasets, 3 seeds, at both the paper-default threshold and the val95
# threshold (separate folder).
#
# Needs the GOAD32 float32 fix in run_deepod.py (get_algo_class's
# _make_goad32_class()): GOAD's own training_prepare()/inference_prepare()
# (deepod/models/tabular/goad.py, untouched) build a full
# (n_samples, trans_dim=32, n_trans=256) array via `X.dot(rot)` BEFORE
# batching; `self.affine_weights = np.random.randn(...)` is float64
# regardless of X's dtype, so numpy silently promotes the WHOLE array to
# float64 -- doubling memory. On this project's larger datasets that is
# ~15.6 GiB for cic_iot and ~13.3 GiB for wsn_ds (per=1.0), which OOM-crashed
# on the dev machine (32 GB RAM, ~16 GB free at the time). GOAD32 (defined
# in run_deepod.py, NOT in deepod/models/tabular/goad.py) downcasts
# affine_weights to float32 right after GOAD's own training_prepare()
# builds them -- identical random transforms/math, half the memory. Still
# needs ~7-8 GiB free RAM for cic_iot/wsn_ds at per=1.0; if your Ubuntu box
# has less than that free, run with a smaller --per first (per=0.1 uses
# ~10x fewer training rows, well under 1 GiB for this array).
#
# Usage: bash run_deepod_goad.sh [per]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PER=${1:-1.0}
SEEDS="42,43,44"
DATASETS="unsw,nsl_kdd,cic_iot,wsn_ds"

echo "========================================"
echo "  GOAD | train + score (per=$PER)"
echo "========================================"
python "$SCRIPT_DIR/run_deepod.py" --algo goad --datasets "$DATASETS" --seeds "$SEEDS" --per "$PER"

echo ""
echo "  GOAD | per-attack eval (default threshold)"
python "$SCRIPT_DIR/eval_deepod_perattack.py" --algo goad --d all --seeds "$SEEDS" --per "$PER"

echo ""
echo "  GOAD | overall AUC/F1/MCC (default threshold)"
python "$SCRIPT_DIR/aggregate_deepod_overall.py" --algo goad --per "$PER"

echo ""
echo "  GOAD | val95 threshold (separate folder)"
python "$SCRIPT_DIR/run_deepod_val95.py" --algo goad --datasets "$DATASETS" --seeds "$SEEDS" --per "$PER"
python "$SCRIPT_DIR/eval_deepod_perattack_val95.py" --algo goad --d all --seeds "$SEEDS" --per "$PER"

echo ""
echo "========================================"
echo "  DONE: algo=goad  seeds=$SEEDS  per=$PER"
echo "  Results -> DeepOD/results_ssae/goad/  and  DeepOD/results_ssae_val95/goad/"
echo "========================================"
