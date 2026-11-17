#!/usr/bin/env bash
# Input-representation ablation on UNSW-NB15 (reviewer follow-up): is the
# CTGAN one-hot representation of proto/service/state the main cost of SSAE
# on this dataset?  Same main (t_04) config, only --cat_mode differs:
#   drop   : proto/service/state removed
#   freq   : proto/service/state replaced by their normal-train frequency
#   allnum : freq + every column forced numerical (no one-hot at all)
# Results: res/t_10_UNSW_repr_se_<seed>/per_1.0/unsw_..._cat<mode>/
# The 3 modes of one seed run in parallel on the GPU; seeds run sequentially.
set -u
cd "$(dirname "$0")"
SEEDS="${1:-42 43 44}"
for seed in $SEEDS; do
  for mode in drop freq allnum; do
    python train_mae01scale_max.py --d unsw --seed "$seed" --time 10 \
      --out_tag "t_10_UNSW_repr_se_${seed}" --cat_mode "$mode" \
      --e 150 --b 128 --lae 0.1 --mae 2.0 --mae_loss_mode renormal \
      --type_train noise_std --noise_std 0.3 --noise_type global --noise_ratio 0.5 \
      --cat_noise_type combined --cat_noise_ratio 0.1 --patience 50 \
      --noise_scale 1.5 --gate_stability 0.05 --gate_window 10 \
      --cov_floor 0.05 --eps_zero 0.1 --tol 0.08 \
      > "res/unsw_repr_${mode}_se${seed}.log" 2>&1 &
  done
  wait
  echo "seed $seed done: $(date)"
done
