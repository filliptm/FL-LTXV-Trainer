@echo off
echo Starting Unified LTXV Training Pipeline...

REM Check if config file is provided
if "%1"=="" (
    echo Usage: train.bat ^<config_file^>
    echo Example: train.bat configs/unified_training_config.yaml
    pause
    exit /b 1
)

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the training pipeline
python train.py %1

pause