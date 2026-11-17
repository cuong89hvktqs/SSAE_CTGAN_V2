#!/bin/bash
# Pseudo-anomaly-source ablation for MAE01ScaleMax (m=2.0, lambda=0.1):
#   P-only -> --type_train noise  (Gaussian noise on real normal samples)
#   Q-only -> --type_train std    (noise on CTGAN samples)
# The P+Q full model (noise_std) is NOT run here -- it lives in t_04.
# Seeds 45..51, --time 5 (seeds 42/43/44 already exist in t_05).
#
# Results -> res/t_05_SSAE_se_<seed>/per_<per>/<dataset>_..._renormal_<noise|std>_.../
#
# Usage: bash run_pseudo_source_t05.sh [per]
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
echo "  MAE01ScaleMax pseudo-source ablation (t_05) | datasets: ${DATASETS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  type_train: ${TYPE_TRAIN_LIST[*]}  (noise = P only, std = Q only)"
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
