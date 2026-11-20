# Run from PowerShell: .\run_dashboard.ps1
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$radarPython = python -c 'import sys; print(sys.executable)'
& $radarPython -m streamlit run src/radar_dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false --theme.base dark --theme.primaryColor '#5eead4'

