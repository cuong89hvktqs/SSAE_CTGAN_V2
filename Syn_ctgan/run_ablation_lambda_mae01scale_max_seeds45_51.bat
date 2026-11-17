@echo off
:: Extra-seed lambda ablation for MAE01ScaleMax (margin fixed at 2.0).
:: Seeds 42/43/44 already exist under res\mae01scale_max_se_<seed>_t_01\ --
:: this script adds 7 more seeds to the same ablation (--time 1), same
:: lambda grid and hyperparameters.
:: Usage: run_ablation_lambda_mae01scale_max_seeds45_51.bat
setlocal enabledelayedexpansion

set SEEDS=45 46 47 48 49 50 51
set TIME=2
set PER=1.0

set DATASETS=unsw cic_iot nsl_kdd wsn_ds

set LAMBDA_LIST=0.0 0.01 0.05 0.5 1.0
set FIXED_MARGIN=2.0

set COMMON_ARGS=^
 --e 150^
 --b 128^
 --mae_loss_mode renormal^
 --type_train noise_std^
 --noise_std 0.3^
 --noise_type global^
 --noise_ratio 0.5^
 --cat_noise_type combined^
 --cat_noise_ratio 0.1^
 --patience 50^
 --noise_scale 1.5^
 --gate_stability 0.05^
 --gate_window 10^
 --cov_floor 0.05^
 --eps_zero 0.1^
 --tol 0.08^
 --per %PER%

echo ========================================
echo   Lambda ablation ^(margin fixed at %FIXED_MARGIN%^) -- extra seeds
echo   lambda in: %LAMBDA_LIST%
echo   Seeds: %SEEDS%
echo ========================================

for %%S in (%SEEDS%) do (
    for %%D in (%DATASETS%) do (
        for %%L in (%LAMBDA_LIST%) do (
            echo.
            echo ======== seed=%%S  dataset=%%D  margin=%FIXED_MARGIN%  lambda=%%L ========
            python train_mae01scale_max.py --d %%D --seed %%S --time %TIME% --lae %%L --mae %FIXED_MARGIN% %COMMON_ARGS%
            if errorlevel 1 (
                echo ERROR at seed=%%S dataset=%%D lambda=%%L, exiting.
                exit /b 1
            )
        )
    )
)

echo.
echo ========================================
echo   ALL DONE: lambda ablation for %DATASETS% (seeds=%SEEDS%)
echo ========================================
endlocal
