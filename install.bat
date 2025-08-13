@echo off
echo ========================================
echo   Hybrid Book Recommender - Auto Install
echo ========================================
echo.

:: Check if PowerShell is available
powershell -Command "Write-Host 'PowerShell available'" >nul 2>&1
if errorlevel 1 (
    echo PowerShell not available, using basic batch installer...
    call install_dependencies.bat
) else (
    echo Using PowerShell installer for better experience...
    powershell -ExecutionPolicy Bypass -File install_dependencies.ps1
)

pause
