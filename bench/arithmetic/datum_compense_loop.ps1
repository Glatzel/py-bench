Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e arithmetic pytest "./src/arithmetic/test_datum_compense_loop.py" --benchmark-autosave --benchmark-histogram=histogram/
