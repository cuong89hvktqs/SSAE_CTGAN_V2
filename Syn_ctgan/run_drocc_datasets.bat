@echo off
REM Trains DROCC (unmodified DROCC_icml2020\tabular\one_class_main_tabular.py)
REM on this project's 4 CLAN-comparison datasets, 3 seeds each, evaluating
REM each model IMMEDIATELY after it finishes training (via eval_drocc.py)
REM rather than waiting for the whole batch. See run_drocc_datasets.sh for
REM the full rationale behind the per-dataset --inp_radius / --lr choices,
REM including why cic_iot uses a much lower lr (2e-6) than the rest.
REM
REM %1 = per (fraction of the normal-only training pool to use, default 1.0)
REM -- matches SSAE's own --per convention; see clan_style_get_data.py's
REM docstring. data/checkpoints/results all land under a per_<per>\ subtree.
REM
REM Usage: run_drocc_datasets.bat [per]

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set DROCC_TABULAR=%SCRIPT_DIR%..\DROCC_icml2020\tabular

set PER=%1
if "%PER%"=="" set PER=1.0

set DATASETS=unsw nsl_kdd cic_iot wsn_ds
rem set DATASETS=wsn_ds
set SEEDS=42 43 44

if exist "%SCRIPT_DIR%drocc_failed_runs.txt" del "%SCRIPT_DIR%drocc_failed_runs.txt"

echo ========================================
echo   Step 1/2: prepare_drocc_data.py (per=%PER%)
echo ========================================
python "%SCRIPT_DIR%prepare_drocc_data.py" --seeds 42,43,44 --per %PER%

echo.
echo ========================================
echo   Step 2/2: train + eval per run ^| datasets: %DATASETS%  seeds: %SEEDS%  per=%PER%
echo ========================================

for %%D in (%DATASETS%) do (
    REM lr rationale (incl. why cic_iot is much lower) is in run_drocc_datasets.sh
    if "%%D"=="unsw"     ( set RADIUS=6.0  & set LR=0.00001 )
    if "%%D"=="nsl_kdd"  ( set RADIUS=11.0 & set LR=0.00001 )
    if "%%D"=="cic_iot"  ( set RADIUS=7.0  & set LR=0.000002 )
    if "%%D"=="wsn_ds"   ( set RADIUS=4.0  & set LR=0.00001 )

    echo.
    echo ======== dataset=%%D  inp_radius=!RADIUS!  lr=!LR! ========
    for %%S in (%SEEDS%) do (
        echo.
        echo --- %%D seed=%%S : train ---
        set DATA_DIR=data_ssae\per_%PER%\%%D\seed%%S
        set MODEL_DIR=log_ssae\per_%PER%\%%D\seed%%S
        pushd "%DROCC_TABULAR%"
        if not exist "!MODEL_DIR!" mkdir "!MODEL_DIR!"
        python one_class_main_tabular.py -d "!DATA_DIR!" --model_dir "!MODEL_DIR!" --lr !LR! --inp_radius !RADIUS! --hd 128 --batch_size 256 --epochs 200 --inp_lamda 1 --one_class_adv 1 --optim 0 --restore 0 > "!MODEL_DIR!\train_log.txt" 2>&1
        if errorlevel 1 (
            echo   [FAILED] %%D seed=%%S -- see !MODEL_DIR!\train_log.txt
            echo %%D seed=%%S>> "%SCRIPT_DIR%drocc_failed_runs.txt"
        )
        popd

        echo.
        echo --- %%D seed=%%S : eval ---
        python "%SCRIPT_DIR%eval_drocc.py" --dataset %%D --seeds %%S --per %PER%
    )
)

if exist "%SCRIPT_DIR%drocc_failed_runs.txt" (
    echo.
    echo ========================================
    echo   WARNING: some run(s) crashed mid-training (likely NaN) -- see:
    echo   %SCRIPT_DIR%drocc_failed_runs.txt
    echo   (still evaluated above if a checkpoint existed from before the crash)
    echo ========================================
)

echo.
echo ========================================
echo   DONE: datasets=%DATASETS%  seeds=%SEEDS%  per=%PER%
echo   Results -^> DROCC_icml2020\tabular\results_ssae\per_%PER%\
echo ========================================
