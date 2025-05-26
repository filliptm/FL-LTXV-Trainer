@echo off
echo Setting up FL-LTXV-Trainer environment...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Install package in development mode
echo Installing FL-LTXV-Trainer in development mode...
pip install -e .

echo.
echo Setup complete! Virtual environment is ready.
echo To activate the environment in the future, run: venv\Scripts\activate.bat
pause