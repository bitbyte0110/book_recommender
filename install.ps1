# Hybrid Book Recommender System - PowerShell Installation Script
# Run this script as Administrator for best results

param(
    [switch]$Force,
    [switch]$SkipChecks
)

# Set console colors
$Host.UI.RawUI.ForegroundColor = "Cyan"
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Hybrid Book Recommender System Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check if Python is installed
if (-not $SkipChecks) {
    if (-not (Test-Command "python")) {
        Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
        Write-Host "Please install Python 3.8 or higher from https://python.org" -ForegroundColor Yellow
        Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host "✓ Python is installed" -ForegroundColor Green
    python --version
    
    # Check if pip is available
    if (-not (Test-Command "pip")) {
        Write-Host "ERROR: pip is not available" -ForegroundColor Red
        Write-Host "Please ensure pip is installed with Python" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host "✓ pip is available" -ForegroundColor Green
    Write-Host ""
}

# Check if requirements.txt exists
if (-not (Test-Path "requirements.txt")) {
    Write-Host "ERROR: requirements.txt not found in current directory" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Upgrade pip to latest version
Write-Host "Upgrading pip to latest version..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade pip
    Write-Host "✓ pip upgraded successfully" -ForegroundColor Green
} catch {
    Write-Host "Warning: Failed to upgrade pip, continuing..." -ForegroundColor Yellow
}

# Install setuptools and wheel for better package installation
Write-Host "Installing setuptools and wheel..." -ForegroundColor Yellow
try {
    pip install setuptools wheel
    Write-Host "✓ setuptools and wheel installed" -ForegroundColor Green
} catch {
    Write-Host "Warning: Failed to install setuptools/wheel, continuing..." -ForegroundColor Yellow
}

# Install project dependencies
Write-Host ""
Write-Host "Installing project dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Cyan

try {
    pip install -r requirements.txt
    Write-Host "✓ All dependencies installed successfully!" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to install some dependencies" -ForegroundColor Red
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
    Write-Host "You may need to install Visual Studio Build Tools for some packages" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Troubleshooting tips:" -ForegroundColor Cyan
    Write-Host "1. Run PowerShell as Administrator" -ForegroundColor White
    Write-Host "2. Install Visual Studio Build Tools" -ForegroundColor White
    Write-Host "3. Try: pip install --upgrade pip setuptools wheel" -ForegroundColor White
    Write-Host "4. Check your internet connection" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
try {
    python -c "import streamlit, pandas, numpy, sklearn, plotly, joblib; print('✓ All packages imported successfully')"
    Write-Host "✓ Installation verified!" -ForegroundColor Green
} catch {
    Write-Host "Warning: Some packages may not be properly installed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   Installation Complete! ✓" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:" -ForegroundColor Cyan
Write-Host "   streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "Then open your browser to: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Additional commands:" -ForegroundColor Cyan
Write-Host "   python scripts/evaluate_system.py  # Run system evaluation" -ForegroundColor White
Write-Host "   python scripts/retrain.py          # Retrain models" -ForegroundColor White
Write-Host "   streamlit run evaluation/user_survey.py  # User survey" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
