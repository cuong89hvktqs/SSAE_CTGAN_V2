@echo off
setlocal

cd /d "%~dp0"

echo [1/6] nsl_kdd noise_3std mae_start_ratio=0.25
python train_MAE_CTGAN.py --d nsl_kdd --e 200 --b 128 --lae 0.1 --mae 1.0 --mae_loss_mode renormal --per 1.0 --type_train noise_3std --mae_start_ratio 0.25 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_start_ratio 1.0 --ctgan_noise_std 0.05 --ctgan_noise_feature_ratio 0.5 --seed 42 --time 1
if errorlevel 1 exit /b %errorlevel%

echo [2/6] nsl_kdd noise_3std mae_start_ratio=0.0
python train_MAE_CTGAN.py --d nsl_kdd --e 200 --b 128 --lae 0.1 --mae 1.0 --mae_loss_mode renormal --per 1.0 --type_train noise_3std --mae_start_ratio 0.0 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_start_ratio 1.0 --ctgan_noise_std 0.05 --ctgan_noise_feature_ratio 0.5 --seed 42 --time 1
if errorlevel 1 exit /b %errorlevel%

echo [3/6] nsl_kdd noise_std mae_start_ratio=0.25
python train_MAE_CTGAN.py --d nsl_kdd --e 200 --b 128 --lae 0.1 --mae 1.0 --mae_loss_mode renormal --per 1.0 --type_train noise_std --mae_start_ratio 0.25 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_start_ratio 0.75 --ctgan_noise_std 0.05 --ctgan_noise_feature_ratio 0.5 --seed 42 --time 1
if errorlevel 1 exit /b %errorlevel%

echo [4/6] nsl_kdd std mae_start_ratio=0.25
python train_MAE_CTGAN.py --d nsl_kdd --e 200 --b 128 --lae 0.1 --mae 1.0 --mae_loss_mode renormal --per 1.0 --type_train std --mae_start_ratio 0.25 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_start_ratio 0.5 --ctgan_noise_std 0.05 --ctgan_noise_feature_ratio 0.5 --seed 42 --time 1
if errorlevel 1 exit /b %errorlevel%

echo [5/6] nsl_kdd 3std mae_start_ratio=0.25
python train_MAE_CTGAN.py --d nsl_kdd --e 200 --b 128 --lae 0.1 --mae 1.0 --mae_loss_mode renormal --per 1.0 --type_train 3std --mae_start_ratio 0.25 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_start_ratio 1.0 --ctgan_noise_std 0.05 --ctgan_noise_feature_ratio 0.5 --seed 42 --time 1
if errorlevel 1 exit /b %errorlevel%

echo [6/6] nsl_kdd all mae_start_ratio=0.25
python train_MAE_CTGAN.py --d nsl_kdd --e 200 --b 128 --lae 0.1 --mae 1.0 --mae_loss_mode renormal --per 1.0 --type_train all --mae_start_ratio 0.25 --ctgan_pseudo_anomaly_mode origin --ctgan_noise_start_ratio 0.5 --ctgan_noise_std 0.05 --ctgan_noise_feature_ratio 0.5 --seed 42 --time 1
if errorlevel 1 exit /b %errorlevel%

echo All nsl_kdd ablation runs finished.
