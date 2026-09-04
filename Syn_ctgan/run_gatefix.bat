@echo off
:: Fixed-gate ablation: opens the CTGAN gate at a FIXED epoch (50/80/0)
:: instead of the adaptive CoV/near-zero logic, via --gate_fixed_epoch (see
:: ctgan/synthesizers/mae01scale_max.py's _maybe_open_gate override). Every
:: other hyperparameter matches run_mae01scale_max.bat's own defaults
:: exactly, so the ONLY deliberate difference from the normal SSAE
:: (MAE01ScaleMax) runs is: fixed-epoch gate vs. the adaptive CoV/near-zero
:: gate.
::
:: Saved under --time 6 (t_06) so results never collide with the default
:: adaptive-gate run (t_04) or the lambda/margin ablation sweep (t_02).
::
:: Usage: run_gatefix.bat [per]
setlocal enabledelayedexpansion

set DATASETS=cic_iot nsl_kdd wsn_ds
set SEEDS=42 43 44
set GATE_EPOCHS=50

set PER=%~1
if "%PER%"=="" set PER=1.0

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
echo   MAE01ScaleMax fixed-gate ablation ^| datasets: %DATASETS%
echo   Seeds: %SEEDS%
echo   Gate-fixed epochs: %GATE_EPOCHS%
echo   per: %PER%
echo ========================================

for %%G in (%GATE_EPOCHS%) do (
    echo.
    echo ==================== gate_fixed_epoch=%%G ====================
    for %%D in (%DATASETS%) do (
        echo.
        echo ======== dataset=%%D ========
        for %%S in (%SEEDS%) do (
            echo.
            echo --- %%D seed=%%S gate_fixed_epoch=%%G ---
            python train_mae01scale_max.py --d %%D --seed %%S --time 6 --gate_fixed_epoch %%G %COMMON_ARGS%
            if errorlevel 1 (
                echo ERROR at dataset=%%D seed=%%S gate_fixed_epoch=%%G, exiting.
                exit /b 1
            )
        )
    )
)

echo.
echo ========================================
echo   DONE: datasets=%DATASETS%  seeds=%SEEDS%  gate_epochs=%GATE_EPOCHS%
echo   Results -^> res\mae01scale_max_se_^<seed^>_t_06\per_%PER%\
echo ========================================
endlocal
