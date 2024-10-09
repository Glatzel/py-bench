Set-Location $PSScriptRoot
Set-Location ..
pixi run pytest "./src/audio_load*.ps1" --benchmark-autosave --benchmark-histogram=histogram/audio