Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e arithmetic-gpu pytest "./src/test_broadcast1.py" --benchmark-autosave --benchmark-histogram=histogram/