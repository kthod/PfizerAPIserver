@echo off
echo Setting up Quantum Optimization API Server...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.9 or later.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create logs directory if it doesn't exist
if not exist logs (
    echo Creating logs directory...
    mkdir logs
)

echo.
echo Setup complete! Starting server...
echo.
echo Server will be available at:
echo http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python app.py

REM Deactivate virtual environment when done
call venv\Scripts\deactivate.bat 