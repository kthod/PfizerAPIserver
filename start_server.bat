@echo off
echo Starting Quantum Optimization API Server...

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Start the server
python app.py

echo.
echo Server is running at http://localhost:8000
echo Press Ctrl+C to stop the server 