#!/bin/bash
# Re-trains SSAE (MAE01ScaleMax, standard m=2.0/lambda=0.1 config) for seed 42,
# 4 core datasets, with generated_data_snapshots.csv RESTORED (train_mae01scale_max.py
# normally disables this dump -- see train_mae01scale_max_gendata.py's docstring
# for why: the original t_03 snapshot files were deleted by
# reorganize_ssae_folders.py's cleanup and cannot be recovered).
#
# Output: res/t_03_gendata_SSAE_se_42/per_1.0/<dataset>_..._catra0.1/
#   generated_data_snapshots.csv   -- real_normal/noise_pseudo/ctgan_raw/ctgan_noise,
#                                      every 5 epochs once the CTGAN gate opens
#   MAE_<dataset>_best_auc.pkl, best_auc/test_mode_total.csv, per_attack_seeds.csv
#   (same as a normal t_03/t_04 run)
#
# This is a SEPARATE folder (t_03_gendata_SSAE_se_42) -- never touches/overwrites
# the existing res/t_03_SSAE_se_42/ or res/t_04_SSAE_se_*/ results.
#
# Usage: bash run_t03_gendata.sh
set -e

SEED=42
PER=1.0
DATASETS=(unsw cic_iot nsl_kdd wsn_ds)

MARGIN=2.0
LAMBDA=0.1

COMMON_ARGS=(
    --e 150
    --b 128
    --mae_loss_mode renormal
    --type_train noise_std
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
echo "  t_03_gendata | m=$MARGIN  lambda=$LAMBDA  seed=$SEED"
echo "  datasets: ${DATASETS[*]}"
echo "========================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "======== dataset=$DATASET  seed=$SEED ========"
    python train_mae01scale_max_gendata.py \
        --d "$DATASET" \
        --seed "$SEED" \
        --lae "$LAMBDA" \
        --mae "$MARGIN" \
        "${COMMON_ARGS[@]}"
done

echo ""
echo "========================================"
echo "  ALL DONE: t_03_gendata seed=$SEED for ${DATASETS[*]}"
echo "  -> res/t_03_gendata_SSAE_se_$SEED/per_$PER/<dataset>_..._catra0.1/generated_data_snapshots.csv"
echo "========================================"
