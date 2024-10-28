Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e audio pytest "./src/audio/audio_load*.ps1" --benchmark-autosave --benchmark-histogram=histogram/audio