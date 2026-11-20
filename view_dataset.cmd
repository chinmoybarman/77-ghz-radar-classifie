@echo off
setlocal
cd /d "%~dp0"
python src\view_kth77_dataset.py --input "C:\Users\chinm\Downloads\data_SAAB_SIRS_77GHz_FMCW.npy" --measurement 0 --segment 0 --output results\kth77_view.png
pause

