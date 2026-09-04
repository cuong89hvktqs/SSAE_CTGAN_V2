@echo off
REM Trains + evaluates all 5 DeepOD algorithms (dsvdd/goad/neutral/icl/dif --
REM DeepSAD excluded, see run_deepod.py's docstring) on this project's 4
REM datasets, 3 seeds, at BOTH the paper-default threshold and the
REM val-95th-percentile (SSAE total95) threshold in a separate folder.
REM
REM Usage: run_deepod_datasets.bat [per]

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PER=%1
if "%PER%"=="" set PER=1.0

set ALGOS=dsvdd goad neutral icl dif
set SEEDS=42,43,44
set DATASETS=unsw,nsl_kdd,cic_iot,wsn_ds

for %%A in (%ALGOS%) do (
    echo.
    echo ========================================
    echo   %%A ^| train + score (per=%PER%)
    echo ========================================
    python "%SCRIPT_DIR%run_deepod.py" --algo %%A --datasets %DATASETS% --seeds %SEEDS% --per %PER%

    echo.
    echo   %%A ^| per-attack eval (default threshold)
    python "%SCRIPT_DIR%eval_deepod_perattack.py" --algo %%A --d all --seeds %SEEDS% --per %PER%

    echo.
    echo   %%A ^| overall AUC/F1/MCC (default threshold)
    python "%SCRIPT_DIR%aggregate_deepod_overall.py" --algo %%A --per %PER%

    echo.
    echo   %%A ^| val95 threshold (separate folder)
    python "%SCRIPT_DIR%run_deepod_val95.py" --algo %%A --datasets %DATASETS% --seeds %SEEDS% --per %PER%
    python "%SCRIPT_DIR%eval_deepod_perattack_val95.py" --algo %%A --d all --seeds %SEEDS% --per %PER%
)

echo.
echo ========================================
echo   DONE: algos=%ALGOS%  seeds=%SEEDS%  per=%PER%
echo   Results -^> DeepOD\results_ssae\^<algo^>\  and  DeepOD\results_ssae_val95\^<algo^>\
echo ========================================
