Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e arithmetic pytest "./src/arithmetic/test_fft.py" --benchmark-autosave --benchmark-histogram=histogram/
