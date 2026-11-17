@echo off
:: t_07: single extra config m=9.0, lambda=1.0 (seed 42 only), 4 core datasets.
:: Usage: run_t07_m9_l1.bat
setlocal enabledelayedexpansion

set DATASETS=unsw cic_iot nsl_kdd wsn_ds
set SEEDS=42
set TIME=7
set PER=1.0

set MARGIN=9.0
set LAMBDA=1.0

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
echo   t_07 ^| m=%MARGIN%  lambda=%LAMBDA%
echo   datasets: %DATASETS%
echo   seeds: %SEEDS%
echo ========================================

for %%S in (%SEEDS%) do (
    for %%D in (%DATASETS%) do (
        echo.
        echo ======== seed=%%S  dataset=%%D  margin=%MARGIN%  lambda=%LAMBDA% ========
        python train_mae01scale_max.py --d %%D --seed %%S --time %TIME% --lae %LAMBDA% --mae %MARGIN% %COMMON_ARGS%
        if errorlevel 1 (
            echo ERROR at dataset=%%D seed=%%S, exiting.
            exit /b 1
        )
    )
)

echo.
echo ========================================
echo   ALL DONE: t_07 m=%MARGIN% lambda=%LAMBDA% for %DATASETS% (seeds=%SEEDS%)
echo ========================================
endlocal
