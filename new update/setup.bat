@echo off
setlocal enabledelayedexpansion
title SecureFedHE Setup

:: ============================================================
::  SecureFedHE — One-Click Windows Setup
::  Run this ONCE on each hospital PC before deployment.
::
::  What it does:
::    1. Checks Python 3.9+
::    2. Creates a virtual environment (.venv)
::    3. Installs all Python dependencies
::    4. Generates mTLS certificates (on master PC only)
::    5. Creates per-node launch shortcuts on the desktop
::    6. Prints firewall and IP instructions
::
::  Usage:
::    Double-click setup.bat        — guided setup (asks your node ID)
::    setup.bat --id 0 --master     — Node 0 (also generates certs)
::    setup.bat --id 1              — Node 1 (certs already copied)
::    setup.bat --dev               — Single-PC dev mode (HTTP, no certs)
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
echo   5-Hospital Federated Learning ^| mTLS + CKKS + DP(e=3) + ZKP
echo  ============================================================
echo.

:: ── Guided mode: ask for node ID if not provided ─────────────────────────
if "%NODE_ID%"=="" (
    echo  Which hospital PC is this?
    echo.
    echo    [0] City General Hospital        ^(MASTER — run setup --master^)
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
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found.
    echo  Please install Python 3.9 or later from https://python.org
    echo  Make sure to tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo  ERROR: Python 3.9+ required. Found: %PYVER%
    pause & exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 9 (
    echo  ERROR: Python 3.9+ required. Found: %PYVER%
    pause & exit /b 1
)
echo  OK  Python %PYVER%

:: ── Step 2: Virtual environment ───────────────────────────────────────────
echo.
echo [2/6] Setting up virtual environment...
if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo  ERROR: Could not create virtual environment.
        pause & exit /b 1
    )
    echo  Created .venv\
) else (
    echo  OK  .venv already exists
)

:: Activate venv
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Could not activate virtual environment.
    pause & exit /b 1
)
echo  OK  Virtual environment activated

:: ── Step 3: Install dependencies ──────────────────────────────────────────
echo.
echo [3/6] Installing Python dependencies...
echo  This may take 2-5 minutes on first run.
echo.
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo  ERROR: pip install failed. Check your internet connection.
    echo  Try running manually: pip install -r requirements.txt
    pause & exit /b 1
)
echo  OK  All dependencies installed

:: ── Step 4: Certificates ──────────────────────────────────────────────────
echo.
echo [4/6] Certificate setup...
if "%DEV_MODE%"=="1" (
    echo  SKIP  Dev mode — no certificates needed
    goto certs_done
)

if "%IS_MASTER%"=="1" (
    echo  Generating mTLS certificates ^(master PC only^)...
    python generate_certs.py
    if errorlevel 1 (
        echo  ERROR: Certificate generation failed.
        pause & exit /b 1
    )
    echo.
    echo  *** IMPORTANT — MASTER STEP ***
    echo  Copy the entire certs\ folder to all other hospital PCs
    echo  before running setup.bat on them.
    echo  You can delete certs\ca.key after distributing.
    echo.
    pause
) else (
    if exist "certs\ca.crt" (
        echo  OK  Certificates found in certs\
    ) else (
        echo.
        echo  WARNING: certs\ folder not found.
        echo  You must copy the certs\ folder from the master PC ^(Node 0^)
        echo  to this PC before training will work.
        echo.
        echo  Path to copy to: %CD%\certs\
        echo.
    )
)
:certs_done

:: ── Step 5: Firewall rule ─────────────────────────────────────────────────
echo.
echo [5/6] Firewall configuration...
if "%DEV_MODE%"=="0" (
    echo  Adding Windows Firewall rule for port 8000 ^(node^)...
    netsh advfirewall firewall add rule ^
        name="SecureFedHE Node" ^
        dir=in ^
        action=allow ^
        protocol=TCP ^
        localport=8000 ^
        >nul 2>&1
    if errorlevel 1 (
        echo  WARNING: Could not add firewall rule automatically.
        echo  Run this manually as Administrator:
        echo    netsh advfirewall firewall add rule name="SecureFedHE Node" dir=in action=allow protocol=TCP localport=8000
    ) else (
        echo  OK  Firewall rule added for port 8000
    )

    if "%NODE_ID%"=="0" (
        echo  Adding Windows Firewall rule for port 8080 ^(dashboard^)...
        netsh advfirewall firewall add rule ^
            name="SecureFedHE Dashboard" ^
            dir=in ^
            action=allow ^
            protocol=TCP ^
            localport=8080 ^
            >nul 2>&1
        if errorlevel 1 (
            echo  WARNING: Could not add firewall rule for dashboard.
            echo  Run manually: netsh advfirewall firewall add rule name="SecureFedHE Dashboard" dir=in action=allow protocol=TCP localport=8080
        ) else (
            echo  OK  Firewall rule added for port 8080 ^(dashboard^)
        )
    )
) else (
    echo  SKIP  Dev mode — no firewall rules needed
)

:: ── Step 6: Desktop shortcuts ─────────────────────────────────────────────
echo.
echo [6/6] Creating desktop shortcuts...
set DESKTOP=%USERPROFILE%\Desktop
set VENV_PYTHON=%CD%\.venv\Scripts\python.exe
set LAUNCH_DIR=%CD%

:: Shortcut: Start This Node
set SHORTCUT_NODE=%DESKTOP%\Start SecureFedHE Node %NODE_ID%.bat
(
    echo @echo off
    echo title SecureFedHE Node %NODE_ID%
    echo cd /d "%LAUNCH_DIR%"
    echo call .venv\Scripts\activate.bat
    if "%DEV_MODE%"=="1" (
        echo python launch.py --id %NODE_ID% --dev
    ) else (
        echo python launch.py --id %NODE_ID%
    )
    echo pause
) > "%SHORTCUT_NODE%"
echo  OK  "%SHORTCUT_NODE%"

:: Shortcut: Dashboard (Node 0 only)
if "%NODE_ID%"=="0" (
    set SHORTCUT_DASH=%DESKTOP%\Start SecureFedHE Dashboard.bat
    (
        echo @echo off
        echo title SecureFedHE Dashboard
        echo cd /d "%LAUNCH_DIR%"
        echo call .venv\Scripts\activate.bat
        if "%DEV_MODE%"=="1" (
            echo python dashboard\dashboard.py --dev
        ) else (
            echo python dashboard\dashboard.py
        )
        echo pause
    ) > "%SHORTCUT_DASH%"
    echo  OK  "%SHORTCUT_DASH%"
)

:: ── Final instructions ────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Setup complete!
echo  ============================================================
echo.

if "%DEV_MODE%"=="1" (
    echo  DEV MODE — Single PC Test
    echo  ─────────────────────────────────────────────────────
    echo  1. Double-click "Start SecureFedHE Node 0" on desktop
    echo  2. Double-click "Start SecureFedHE Dashboard" on desktop
    echo  3. Open browser: http://localhost:8080
    echo  4. Click "Run Demo" to simulate training
) else (
    echo  PRODUCTION DEPLOYMENT — 5 Hospital PCs
    echo  ─────────────────────────────────────────────────────
    echo.

    :: Get this PC's IP
    for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /i "IPv4"') do (
        set MY_IP=%%i
        goto got_ip
    )
    :got_ip
    set MY_IP=%MY_IP: =%

    echo  This PC's IP address: %MY_IP%
    echo.
    echo  Edit config.json and set this IP for Node %NODE_ID%:
    echo    "ip": "%MY_IP%"
    echo.
    echo  Deployment order:
    echo    Step 1: Edit config.json on ALL PCs with correct IPs
    echo    Step 2: Run setup.bat --id 0 --master on Node 0 to generate certs
    echo    Step 3: Copy certs\ folder to all other PCs
    echo    Step 4: Run setup.bat --id 1 (2, 3, 4) on each other PC
    echo    Step 5: Start nodes 1-4 first (double-click desktop shortcut)
    echo    Step 6: Start node 0 last (it waits for all others, then fires)
    echo    Step 7: Open dashboard on Node 0: http://%MY_IP%:8080
    echo.
    echo  Node start order:  Nodes 1,2,3,4 first → then Node 0 (master)
)

echo.
echo  ─────────────────────────────────────────────────────────────
echo  Logs:     logs\node_%NODE_ID%.log
echo  Config:   config.json  ^(edit IPs and Claude API key here^)
echo  Docs:     README.md
echo  ─────────────────────────────────────────────────────────────
echo.
pause
