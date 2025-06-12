@echo off
cd /d "%~dp0"

REM Check if arguments are provided
if "%1"=="" (
    echo 📁 Creating folders...
    ..\python-3.10.11.amd64\python.exe main.py
    pause
    exit /b
)

REM Run the script with arguments
..\python-3.10.11.amd64\python.exe main.py %1 %2
pause
