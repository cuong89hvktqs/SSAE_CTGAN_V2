@echo off
REM "SSAE with the gate always open" comparison run: trains the pre-gate
REM ancestor class M01AECTGAN (train_M01AECTGAN.py / M01AECTGANSynthesizer --
REM no CoV-based adaptive gate exists in this class, only a FIXED
REM mae_start_ratio warmup: mae_start_epoch = int(mae_start_ratio * epochs);
REM use_ctgan = (epoch >= mae_start_epoch)) with --mae_start_ratio 0.0 --
REM i.e. mae_start_epoch=0, CTGAN pseudo-anomalies used from epoch 0, no
REM warmup/gate delay at all.
REM
REM No code changes: train_M01AECTGAN.py already supports this via its own
REM --mae_start_ratio flag (default 0.25); this script just passes 0.0, and
REM otherwise matches run_mae01scale_max.bat's default hyperparameters
REM exactly so the only deliberate difference from the main SSAE
REM (MAE01ScaleMax) runs is: fixed warmup=0 vs. the adaptive CoV/near-zero
REM gate -- gate-only params don't exist on this ancestor class at all.
REM
REM Usage: run_m01aectgan_gate0.bat [per]

setlocal enabledelayedexpansion

rem set DATASETS=unsw cic_iot nsl_kdd wsn_ds
set DATASETS=wsn_ds
set PER=%1
if "%PER%"=="" set PER=1.0
set SEEDS=42 43 44

set COMMON_ARGS=--e 150 --b 128 --lae 0.1 --mae 2.0 --mae_loss_mode renormal --type_train noise_std --mae_start_ratio 0.0 --noise_std 0.3 --noise_type global --noise_ratio 0.5 --cat_noise_type combined --cat_noise_ratio 0.1 --patience 50 --per %PER%

echo ========================================
echo   M01AECTGAN (mae_start_ratio=0.0, gate always open) ^| datasets: %DATASETS%
echo   Seeds: %SEEDS%
echo   per: %PER%
echo ========================================

for %%D in (%DATASETS%) do (
    echo.
    echo ======== dataset=%%D ========
    for %%S in (%SEEDS%) do (
        echo.
        echo --- %%D seed=%%S ---
        python train_M01AECTGAN.py --d %%D --seed %%S --time 1 %COMMON_ARGS%
    )
)

echo.
echo ========================================
echo   DONE: datasets=%DATASETS%  seeds=%SEEDS%
echo   Results -^> res\m01ae_noise_std_se_^<seed^>_t_01\per_%PER%\
echo ========================================
