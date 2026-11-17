#!/bin/bash
# Pseudo-anomaly-source ablation for MAE01ScaleMax (m=2.0, lambda=0.1),
# P-only (type_train=noise) and Q-only (type_train=std) -- the noise_std
# ("P+Q", full model) variant is NOT re-run here since it's already
# covered by run_mae01scale_max_seeds45_51.sh under --time 4.
#
# For seeds 42/43/44 this ablation was stored mixed together with the
# main config under --time 4 (see res/t_04_SSAE_se_<42|43|44>/); for
# seeds 45-51 it gets its own --time 5 (t_05) folder instead, so it never
# collides with t_04's single main-config run.
#
# Usage: bash run_pseudo_source_seeds45_51.sh [per]
set -e

DATASETS=(unsw cic_iot nsl_kdd wsn_ds)
SEEDS=(45 46 47 48 49 50 51)
TYPE_TRAIN_LIST=(noise std)

PER=${1:-1.0}

COMMON_ARGS=(
    --e 150
    --b 128
    --lae 0.1
    --mae 2.0
    --mae_loss_mode renormal
    --noise_std 0.3
    --noise_type global
    --noise_ratio 0.5
    --cat_noise_type combined
    --cat_noise_ratio 0.1
    --patience 50
    --noise_scale 1.5
    --gate_stability 0.05
    --gate_window 10
    --cov_floor 0.05
    --eps_zero 0.1
    --tol 0.08
    --per "$PER"
)

echo "========================================"
echo "  MAE01ScaleMax pseudo-source ablation | datasets: ${DATASETS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  type_train: ${TYPE_TRAIN_LIST[*]}"
echo "  per: $PER"
echo "========================================"

for TYPE_TRAIN in "${TYPE_TRAIN_LIST[@]}"; do
    echo ""
    echo "==================== type_train=$TYPE_TRAIN ===================="
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "======== dataset=$DATASET ========"
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "--- $DATASET seed=$SEED type_train=$TYPE_TRAIN ---"
            python train_mae01scale_max.py \
                --d "$DATASET" \
                --seed "$SEED" \
                --time 5 \
                --type_train "$TYPE_TRAIN" \
                "${COMMON_ARGS[@]}"
        done
    done
done

echo ""
echo "========================================"
echo "  DONE: datasets=${DATASETS[*]}  seeds=${SEEDS[*]}  type_train=${TYPE_TRAIN_LIST[*]}"
echo "  Results -> res/t_05_SSAE_se_<seed>/per_$PER/"
echo "========================================"
