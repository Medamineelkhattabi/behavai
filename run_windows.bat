@echo off
echo ========================================
echo Smart Mobility Platform - Windows Setup
echo ========================================

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.11 from https://python.org
    pause
    exit /b 1
)

echo Checking pip installation...
pip --version
if %errorlevel% neq 0 (
    echo ERROR: pip not found! Please ensure Python is properly installed
    pause
    exit /b 1
)

echo Installing required packages...
pip install streamlit pandas numpy plotly python-dotenv requests

echo Starting the application...
echo.
echo ========================================
echo Access your dashboard at:
echo http://localhost:8501
echo ========================================
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py --server.port=8501 --server.address=localhost