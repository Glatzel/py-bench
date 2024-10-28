Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e audio pytest "./src/test_channel_merge.py" --benchmark-autosave --benchmark-histogram=histogram/audio