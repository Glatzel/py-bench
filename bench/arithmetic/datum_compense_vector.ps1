Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e arithmetic-gpu pytest "./src/arithmetic/test_datum_compense_vector.py" --benchmark-autosave --benchmark-histogram=histogram/