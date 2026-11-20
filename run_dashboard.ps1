# Run from PowerShell: .\run_dashboard.ps1
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$radarPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $radarPython -PathType Leaf)) { $radarPython = 'python' }
& $radarPython -m streamlit run src/radar_dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless false --browser.gatherUsageStats false --theme.base dark --theme.primaryColor '#5eead4'
exit $LASTEXITCODE
