#!/bin/bash
# Control A (Reviewer 4, Comment 2b): compactness-only MiAE, no pseudo-intrusions.
# Usage: bash run_control_a.sh [per]
set -e

DATASETS=(unsw cic_iot nsl_kdd wsn_ds)

PER=${1:-1.0}

SEEDS=(42)

COMMON_ARGS=(
    --e 150
    --b 128
    --lae 0.1
    --patience 50
    --per "$PER"
)

echo "========================================"
echo "  ControlA | datasets: ${DATASETS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  per: $PER"
echo "========================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "======== dataset=$DATASET ========"
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- $DATASET seed=$SEED ---"
        python train_control_a.py \
            --d "$DATASET" \
            --seed "$SEED" \
            --time 8 \
            "${COMMON_ARGS[@]}"
    done
done

echo ""
echo "========================================"
echo "  DONE: datasets=${DATASETS[*]}  seeds=${SEEDS[*]}"
echo "========================================"
