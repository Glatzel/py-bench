Set-Location $PSScriptRoot
Set-Location ..
pixi run pytest "./src/test_stft" --benchmark-autosave --benchmark-histogram=histogram/audio