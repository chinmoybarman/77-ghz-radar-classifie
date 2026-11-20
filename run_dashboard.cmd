@echo off
setlocal
cd /d "%~dp0"
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=1).read()" >nul 2>nul
if not errorlevel 1 (
  start "" "http://127.0.0.1:8501"
  echo The dashboard is already running. It has been opened in your browser.
  exit /b 0
)
python -m streamlit run src\radar_dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false --theme.base dark --theme.primaryColor "#5eead4"
if errorlevel 1 (
  echo.
  echo Dashboard could not start. Check that Python and the packages in requirements.txt are installed.
  pause
)

