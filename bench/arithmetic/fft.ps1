Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e arithmetic-gpu pytest "./src/test_fft.py" --benchmark-autosave --benchmark-histogram=histogram/