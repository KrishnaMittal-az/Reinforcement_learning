@echo off
:: This script provides a clean, reliable setup for the Personalized Learning RL project.
:: It creates a dedicated virtual environment, installs exact library versions, and prevents common conflicts.

echo =================================================
echo  Setting Up Personalized Learning RL Environment
echo =================================================
echo.

:: Ensure Python 3.9 is used
set PYTHON=
where python.exe | find "Python39" > nul
if %errorlevel% == 0 (
    for /f "delims=" %%i in ('where python.exe ^| find "Python39"') do set PYTHON=%%i
)

if not defined PYTHON (
    echo Error: Python 3.9 not found in PATH.
    echo Please install Python 3.9 and ensure it's in your system PATH.
    pause
    exit /b
)

echo Found Python 3.9 at: %PYTHON%
echo.

:: 1. Create a fresh virtual environment
echo [1/4] Creating virtual environment in '.\venv_rl'...
if exist venv_rl (
    echo    - Virtual environment 'venv_rl' already exists. Deleting for a clean setup.
    rmdir /s /q venv_rl
)
%PYTHON% -m venv venv_rl
echo    - Done.
echo.

:: 2. Activate the virtual environment
echo [2/4] Activating virtual environment...
call .\venv_rl\Scripts\activate
echo    - Done.
echo.

:: 3. Upgrade pip and clear cache
echo [3/4] Upgrading pip and clearing cache...
python -m pip install --upgrade pip
pip cache purge
echo    - Done.
echo.

:: 4. Install all dependencies from the curated requirements.txt
echo [4/4] Installing all required libraries...
pip install -r requirements.txt
echo.

echo =================================================
echo                  Setup Complete!
echo =================================================
echo.
echo Your environment is ready. To run the quiz:
echo.
echo   1. Activate the environment: call .\venv_rl\Scripts\activate
echo   2. Run the script:         python interactive_quiz.py
echo.
pause
