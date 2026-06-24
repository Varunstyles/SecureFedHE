@echo off
setlocal enabledelayedexpansion
title SecureFedHE Setup

:: ============================================================
::  SecureFedHE — One-Click Windows Setup
::  Run ONCE on each hospital PC before deployment.
:: ============================================================

:: ── Parse arguments ───────────────────────────────────────────────────────
set NODE_ID=
set IS_MASTER=0
set DEV_MODE=0

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--id"     ( set NODE_ID=%~2 & shift & shift & goto parse_args )
if "%~1"=="--master" ( set IS_MASTER=1 & shift & goto parse_args )
if "%~1"=="--dev"    ( set DEV_MODE=1  & shift & goto parse_args )
shift
goto parse_args
:args_done

:: ── Banner ────────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   SecureFedHE ^| Real Deployment Setup
echo   5-Hospital Federated Learning ^| mTLS + CKKS + DP + ZKP
echo  ============================================================
echo.

:: ── Guided mode ──────────────────────────────────────────────────────────
if "%NODE_ID%"=="" (
    echo  Which hospital PC is this?
    echo.
    echo    [0] City General Hospital        ^(MASTER^)
    echo    [1] Suburban Medical Clinic
    echo    [2] Rural Health Centre
    echo    [3] University Research Hospital
    echo    [4] Private Wellness Clinic
    echo    [d] Dev mode ^(single PC, HTTP only^)
    echo.
    set /p NODE_ID="  Enter node ID (0-4) or d for dev: "
    if /i "!NODE_ID!"=="d" (
        set DEV_MODE=1
        set NODE_ID=0
    )
    if "!NODE_ID!"=="0" set IS_MASTER=1
)

echo.
echo  Node ID : %NODE_ID%
if "%DEV_MODE%"=="1" (
    echo  Mode    : DEV ^(HTTP, no TLS^)
) else (
    echo  Mode    : PRODUCTION ^(HTTPS + mTLS^)
)
echo.

:: ── Step 1: Check Python ──────────────────────────────────────────────────
echo [1/6] Checking Python installation...
py --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found.
    echo  Install Python 3.9+ from https://python.org
    echo  Tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('py --version 2^>^&1') do set PYVER=%%v
echo  OK  Python %PYVER%

:: ── Step 2: Virtual environment ───────────────────────────────────────────
echo.
echo [2/6] Setting up virtual environment...
if not exist ".venv" (
    py -m venv .venv
    if errorlevel 1 ( echo  ERROR: Could not create venv. & pause & exit /b 1 )
    echo  Created .venv\
) else (
    echo  OK  .venv already exists
)
call .venv\Scripts\activate.bat
echo  OK  Virtual environment activated

:: ── Step 3: Install dependencies ──────────────────────────────────────────
echo.
echo [3/6] Installing Python dependencies...
echo  This may take 2-5 minutes on first run.
echo.
py -m pip install --upgrade pip --quiet
py -m pip install -r "%~dp0requirements.txt" --quiet
if errorlevel 1 (
    echo.
    echo  ERROR: pip install failed. Check your internet connection.
    pause & exit /b 1
)
echo  OK  All dependencies installed

:: ── Step 4: Certificates ──────────────────────────────────────────────────
echo.
echo [4/6] Certificate setup...
if "%DEV_MODE%"=="1" (
    echo  SKIP  Dev mode -- no certificates needed
    goto certs_done
)
if "%IS_MASTER%"=="1" (
    echo  Generating mTLS certificates...
    py generate_certs.py
    if errorlevel 1 ( echo  ERROR: Cert generation failed. & pause & exit /b 1 )
    echo.
    echo  *** Copy the certs\ folder to all other PCs before continuing ***
    echo.
    pause
) else (
    if exist "certs\ca.crt" (
        echo  OK  Certificates found
    ) else (
        echo  WARNING: certs\ folder not found. Copy it from Node 0.
    )
)
:certs_done

:: ── Step 5: Firewall ──────────────────────────────────────────────────────
echo.
echo [5/6] Firewall configuration...
if "%DEV_MODE%"=="1" (
    echo  SKIP  Dev mode -- no firewall rules needed
) else (
    netsh advfirewall firewall add rule name="SecureFedHE Node" dir=in action=allow protocol=TCP localport=8000 >nul 2>&1
    echo  OK  Firewall rule added for port 8000
    if "%NODE_ID%"=="0" (
        netsh advfirewall firewall add rule name="SecureFedHE Dashboard" dir=in action=allow protocol=TCP localport=8080 >nul 2>&1
        echo  OK  Firewall rule added for port 8080
    )
)

:: ── Step 6: Desktop shortcuts ─────────────────────────────────────────────
echo.
echo [6/6] Creating desktop shortcuts...
for /f "usebackq tokens=*" %%D in (`powershell -command "[Environment]::GetFolderPath('Desktop')"`) do set DESKTOP=%%D
set LAUNCH_DIR=%CD%

if "%DEV_MODE%"=="1" (
    set LAUNCH_CMD=py launch.py --id %NODE_ID% --dev
    set DASH_CMD=py dashboard\dashboard.py --dev
) else (
    set LAUNCH_CMD=py launch.py --id %NODE_ID%
    set DASH_CMD=py dashboard\dashboard.py
)

:: Node shortcut
(
    echo @echo off
    echo title SecureFedHE Node %NODE_ID%
    echo cd /d "%LAUNCH_DIR%"
    echo call .venv\Scripts\activate.bat
    echo %LAUNCH_CMD%
    echo pause
) > "%DESKTOP%\Start SecureFedHE Node %NODE_ID%.bat"
echo  OK  "%DESKTOP%\Start SecureFedHE Node %NODE_ID%.bat"

:: Dashboard shortcut (Node 0 only)
if "%NODE_ID%"=="0" (
    (
        echo @echo off
        echo title SecureFedHE Dashboard
        echo cd /d "%LAUNCH_DIR%"
        echo call .venv\Scripts\activate.bat
        echo %DASH_CMD%
        echo pause
    ) > "%DESKTOP%\Start SecureFedHE Dashboard.bat"
    echo  OK  "%DESKTOP%\Start SecureFedHE Dashboard.bat"
)

:: ── Done ──────────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Setup complete!
echo  ============================================================
echo.
if "%DEV_MODE%"=="1" (
    echo  1. Double-click "Start SecureFedHE Node 0" on Desktop
    echo  2. Double-click "Start SecureFedHE Dashboard" on Desktop
    echo  3. Open browser: http://localhost:8080
    echo  4. Click "Run Demo" to simulate training
) else (
    echo  Start nodes 1-4 first, then Node 0 last.
    echo  Dashboard: http://%LAUNCH_DIR%:8080
)
echo.
pause
