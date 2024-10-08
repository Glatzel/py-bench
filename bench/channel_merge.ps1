Set-Location $PSScriptRoot
Set-Location ..
pixi run pytest "./src/test_channel_merge.py" --benchmark-autosave --benchmark-histogram=histogram/audio