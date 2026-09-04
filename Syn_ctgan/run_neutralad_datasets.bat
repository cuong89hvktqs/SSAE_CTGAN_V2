@echo off
REM Prepares data, trains, and evaluates NeuTraL-AD (unmodified NeuTraL-AD\
REM models\loss\trainer\evaluation code) on this project's 4 comparison
REM datasets, 3 seeds each, split exactly like SSAE via
REM prepare_neutralad_data.py. See run_neutralad.py's docstring for why this
REM bypasses NeuTraL-AD's own Launch_Exps.py entry point.
REM
REM Usage: run_neutralad_datasets.bat [per]

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PER=%1
if "%PER%"=="" set PER=1.0

set DATASETS=unsw,nsl_kdd,cic_iot,wsn_ds
set SEEDS=42,43,44

echo ========================================
echo   Step 1/3: prepare_neutralad_data.py (per=%PER%)
echo ========================================
python "%SCRIPT_DIR%prepare_neutralad_data.py" --dataset all --seeds %SEEDS% --per %PER%

echo.
echo ========================================
echo   Step 2/3: run_neutralad.py (train + score) ^| datasets: %DATASETS%  seeds: %SEEDS%  per=%PER%
echo ========================================
python "%SCRIPT_DIR%run_neutralad.py" --datasets %DATASETS% --seeds %SEEDS% --per %PER%

echo.
echo ========================================
echo   Step 3/3: eval_neutralad_perattack.py (per-attack AUC / PR-AUC / TPR@FAR)
echo ========================================
python "%SCRIPT_DIR%eval_neutralad_perattack.py" --d all --seeds %SEEDS% --per %PER%

echo.
echo ========================================
echo   DONE: datasets=%DATASETS%  seeds=%SEEDS%  per=%PER%
echo   Results -^> NeuTraL-AD\results_ssae\per_attack_results\per%PER%\
echo ========================================
