@echo off
echo ===================================================
echo [!] Starting Exam Monitoring System
echo ===================================================

echo [1/2] Launching Backend Server...
start "AI Exam Backend" python run_system.py

echo Waiting for backend to initialize (5s)...
timeout /t 5 > nul

echo [2/2] Launching Camera Client...
echo Press 'q' in the camera window to quit.
python scripts/camera_client.py

echo.
echo ===================================================
echo Backend is still running in the other window.
echo Close it manually when done.
echo ===================================================
pause
