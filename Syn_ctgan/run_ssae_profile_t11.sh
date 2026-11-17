#!/usr/bin/env bash
# SSAE training-cost breakdown (Reviewer 2): 12-epoch profiled runs of the
# main config with the gate forced open at epoch 6 (so epochs 6-11 reflect
# the post-gate P+Q regime), SSAE_PROFILE=1 writes
# res/t_11_profile_se_42/per_1.0/<ds>_.../profile_components.csv
# (per-epoch seconds for CTGAN D+G, PAG, MiAE, validation, gate, checkpoint
# + peak GPU memory). Aggregate with build_training_cost_tables.py.
set -u
cd "$(dirname "$0")"
export SSAE_PROFILE=1
for ds in unsw nsl_kdd cic_iot wsn_ds; do
  python train_mae01scale_max.py --d "$ds" --seed 42 --time 11 \
    --out_tag t_11_profile_se_42 --gate_fixed_epoch 6 \
    --e 12 --b 128 --lae 0.1 --mae 2.0 --mae_loss_mode renormal \
    --type_train noise_std --noise_std 0.3 --noise_type global --noise_ratio 0.5 \
    --cat_noise_type combined --cat_noise_ratio 0.1 --patience 50 \
    --noise_scale 1.5 --gate_stability 0.05 --gate_window 10 \
    --cov_floor 0.05 --eps_zero 0.1 --tol 0.08 \
    > "res/ssae_profile_${ds}.log" 2>&1
  echo "$ds done $(date)"
done
