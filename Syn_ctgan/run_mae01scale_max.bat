@echo off
:: Usage: run_mae01scale_max.bat [per]
setlocal enabledelayedexpansion

:: ── Datasets to run (edit this list to add/remove datasets) ─────────────────
set DATASETS=unsw cic_iot nsl_kdd wsn_ds nb_iot cic_ids

set PER=%~1
if "%PER%"=="" set PER=1.0

set SEEDS=42 43 44

set COMMON_ARGS=^
 --e 150^
 --b 128^
 --lae 0.1^
 --mae 2.0^
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
echo   MAE01ScaleMax ^| datasets: %DATASETS%
echo   Seeds: %SEEDS%
echo   per: %PER%
echo ========================================

for %%D in (%DATASETS%) do (
    echo.
    echo ======== dataset=%%D ========
    for %%S in (%SEEDS%) do (
        echo.
        echo --- %%D seed=%%S ---
        python train_mae01scale_max.py --d %%D --seed %%S --time 4 %COMMON_ARGS%
        if errorlevel 1 (
            echo ERROR at dataset=%%D seed=%%S, exiting.
            exit /b 1
        )
    )
)

echo.
echo ========================================
echo   DONE: datasets=%DATASETS%  seeds=%SEEDS%
echo ========================================
endlocal
