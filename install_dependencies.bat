@echo off
echo ========================================
echo   Hybrid Book Recommender System Setup
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✓ Python is installed
python --version

:: Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo ✓ pip is available
echo.

:: Upgrade pip to latest version
echo Upgrading pip to latest version...
python -m pip install --upgrade pip

:: Install setuptools and wheel for better package installation
echo Installing setuptools and wheel...
pip install setuptools wheel

:: Install project dependencies
echo.
echo Installing project dependencies...
echo This may take a few minutes...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some dependencies
    echo Please check the error messages above
    echo You may need to install Visual Studio Build Tools for some packages
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Installation Complete! ✓
echo ========================================
echo.
echo To run the application:
echo    streamlit run app.py
echo.
echo Then open your browser to: http://localhost:8501
echo.
echo Press any key to exit...
pause >nul
