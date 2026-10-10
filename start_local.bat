@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The local Python environment is not installed.
    echo Run setup_local.bat first.
    exit /b 1
)

if not exist "config.json" (
    echo config.json is missing from the project folder.
    exit /b 1
)

if not exist "models\yolov8n-face.pt" (
    echo The face detection model is missing: models\yolov8n-face.pt
    exit /b 1
)

".venv\Scripts\python.exe" src\main.py
set "exit_code=%errorlevel%"
if not "%exit_code%"=="0" (
    echo.
    echo Tracker stopped with an error. Review the message above and logs\events.log.
    pause
)
exit /b %exit_code%
