Set-Location $PSScriptRoot
Set-Location ..
pixi run pytest "./src/test_amplitude_to_db.py" --benchmark-autosave --benchmark-histogram=histogram/audio