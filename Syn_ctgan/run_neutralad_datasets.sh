#!/bin/bash
# Prepares data, trains, and evaluates NeuTraL-AD (unmodified NeuTraL-AD/
# models/loss/trainer/evaluation code -- see run_neutralad.py's own
# docstring for why the seed-controlled driver lives outside that repo
# entirely rather than going through Launch_Exps.py) on this project's 4
# comparison datasets, 3 seeds each, split exactly like SSAE
# (train_mae01scale_max.py) via prepare_neutralad_data.py.
#
# Usage: bash run_neutralad_datasets.sh [per]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PER=${1:-1.0}
DATASETS="unsw,nsl_kdd,cic_iot,wsn_ds"
SEEDS="42,43,44"

echo "========================================"
echo "  Step 1/3: prepare_neutralad_data.py (per=$PER)"
echo "========================================"
python "$SCRIPT_DIR/prepare_neutralad_data.py" --dataset all --seeds "$SEEDS" --per "$PER"

echo ""
echo "========================================"
echo "  Step 2/3: run_neutralad.py (train + score) | datasets: $DATASETS  seeds: $SEEDS  per=$PER"
echo "========================================"
python "$SCRIPT_DIR/run_neutralad.py" --datasets "$DATASETS" --seeds "$SEEDS" --per "$PER"

echo ""
echo "========================================"
echo "  Step 3/3: eval_neutralad_perattack.py (per-attack AUC / PR-AUC / TPR@FAR)"
echo "========================================"
python "$SCRIPT_DIR/eval_neutralad_perattack.py" --d all --seeds "$SEEDS" --per "$PER"

echo ""
echo "========================================"
echo "  DONE: datasets=$DATASETS  seeds=$SEEDS  per=$PER"
echo "  Results -> NeuTraL-AD/results_ssae/per_attack_results/per$PER/"
echo "========================================"
