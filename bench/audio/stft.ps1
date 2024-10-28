Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e audio pytest "./src/audio/test_stft" --benchmark-autosave --benchmark-histogram=histogram/audio
