Set-Location $PSScriptRoot
Set-Location ..
Set-Location ..
pixi run -e audio pytest "./src/audio_load*.ps1" --benchmark-autosave --benchmark-histogram=histogram/audio