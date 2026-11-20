@echo off
setlocal
cd /d "%~dp0"
set "RADAR_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "RADAR_PYTHON=%CD%\.venv\Scripts\python.exe"
set "RADAR_ARCHIVE=%~1"
if not defined RADAR_ARCHIVE set "RADAR_ARCHIVE=%USERPROFILE%\Downloads\data_SAAB_SIRS_77GHz_FMCW.npy"
"%RADAR_PYTHON%" src\view_kth77_dataset.py --input "%RADAR_ARCHIVE%" --measurement 0 --segment 0 --output results\kth77_view.png
pause
