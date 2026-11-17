@echo off
:: Extra-seed margin (m) ablation for MAE01ScaleMax (lambda fixed at 0.1).
:: Seeds 42/43/44 already exist under res\mae01scale_max_se_<seed>_t_02\ --
:: this script adds 7 more seeds to the same ablation (--time 2), same
:: margin grid and hyperparameters.
:: Usage: run_ablation_m_mae01scale_max_seeds45_51.bat
setlocal enabledelayedexpansion

set SEEDS=45 46 47 48 49 50 51
set TIME=2
set PER=1.0

set DATASETS=unsw cic_iot nsl_kdd wsn_ds

set M_LIST=1.0 3.0 4.0 5.0 9.0
set FIXED_LAMBDA=0.1

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
echo   Margin ablation ^(lambda fixed at %FIXED_LAMBDA%^) -- extra seeds
echo   m in: %M_LIST%
echo   Seeds: %SEEDS%
echo ========================================

for %%S in (%SEEDS%) do (
    for %%D in (%DATASETS%) do (
        for %%M in (%M_LIST%) do (
            echo.
            echo ======== seed=%%S  dataset=%%D  margin=%%M  lambda=%FIXED_LAMBDA% ========
            python train_mae01scale_max.py --d %%D --seed %%S --time %TIME% --lae %FIXED_LAMBDA% --mae %%M %COMMON_ARGS%
            if errorlevel 1 (
                echo ERROR at seed=%%S dataset=%%D margin=%%M, exiting.
                exit /b 1
            )
        )
    )
)

echo.
echo ========================================
echo   ALL DONE: margin ablation for %DATASETS% (seeds=%SEEDS%)
echo ========================================
endlocal
