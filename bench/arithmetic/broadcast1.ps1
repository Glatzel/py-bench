Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e arithmetic-gpu pytest "./src/arithmetic/test_broadcast1.py" --benchmark-autosave --benchmark-histogram=histogram/
