Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e arithmetic-gpu pytest "./src/test_datum_compense_loop.py" --benchmark-autosave --benchmark-histogram=histogram/