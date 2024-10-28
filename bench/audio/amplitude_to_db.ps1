Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e audio pytest "./src/audio/test_amplitude_to_db.py" --benchmark-autosave --benchmark-histogram=histogram/audio
