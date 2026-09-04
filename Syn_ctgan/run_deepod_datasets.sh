#!/bin/bash
# Trains + evaluates all 5 DeepOD algorithms (dsvdd/goad/neutral/icl/dif --
# DeepSAD excluded, see run_deepod.py's docstring) on this project's 4
# datasets, 3 seeds, at BOTH the paper-default (contamination-based)
# threshold and the val-95th-percentile (SSAE total95) threshold in a
# separate results folder.
#
# Requires NeuTraL-AD/DATA/ssae/per_<per>/... to already exist (built by
# prepare_neutralad_data.py) -- DeepOD reuses that data as-is.
#
# Usage: bash run_deepod_datasets.sh [per]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PER=${1:-1.0}
ALGOS=(dsvdd goad neutral icl dif)
SEEDS="42,43,44"
DATASETS="unsw,nsl_kdd,cic_iot,wsn_ds"

for ALGO in "${ALGOS[@]}"; do
    echo ""
    echo "========================================"
    echo "  $ALGO | train + score (per=$PER)"
    echo "========================================"
    python "$SCRIPT_DIR/run_deepod.py" --algo "$ALGO" --datasets "$DATASETS" --seeds "$SEEDS" --per "$PER"

    echo ""
    echo "  $ALGO | per-attack eval (default threshold)"
    python "$SCRIPT_DIR/eval_deepod_perattack.py" --algo "$ALGO" --d all --seeds "$SEEDS" --per "$PER"

    echo ""
    echo "  $ALGO | overall AUC/F1/MCC (default threshold)"
    python "$SCRIPT_DIR/aggregate_deepod_overall.py" --algo "$ALGO" --per "$PER"

    echo ""
    echo "  $ALGO | val95 threshold (separate folder)"
    python "$SCRIPT_DIR/run_deepod_val95.py" --algo "$ALGO" --datasets "$DATASETS" --seeds "$SEEDS" --per "$PER"
    python "$SCRIPT_DIR/eval_deepod_perattack_val95.py" --algo "$ALGO" --d all --seeds "$SEEDS" --per "$PER"
done

echo ""
echo "========================================"
echo "  DONE: algos=${ALGOS[*]}  seeds=$SEEDS  per=$PER"
echo "  Results -> DeepOD/results_ssae/<algo>/  and  DeepOD/results_ssae_val95/<algo>/"
echo "========================================"
