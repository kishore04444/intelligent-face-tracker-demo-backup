@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.10 and enable the Python Launcher.
    exit /b 1
)

py -3.10 --version >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 was not found. Install Python 3.10, then run this setup again.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    py -3.10 -m venv .venv
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if not exist "models\yolov8n-face.pt" (
    echo The face detection model is missing: models\yolov8n-face.pt
    echo Restore the model file before launching the tracker.
    exit /b 1
)

echo.
echo Setup completed. Run start_local.bat to start the local tracker.
exit /b 0
