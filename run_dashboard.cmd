@echo off
setlocal
cd /d "%~dp0"
set "RADAR_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "RADAR_PYTHON=%CD%\.venv\Scripts\python.exe"
"%RADAR_PYTHON%" -m streamlit run src\radar_dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless false --browser.gatherUsageStats false --theme.base dark --theme.primaryColor "#5eead4"
if errorlevel 1 (
  echo.
  echo Dashboard could not start. Check that Python and the packages in requirements.txt are installed.
  pause
)
