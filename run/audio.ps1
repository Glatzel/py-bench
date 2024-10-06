Set-Location $PSScriptRoot
Set-Location ..
pixi run -e audio pytest "./src" --benchmark-autosave --benchmark-histogram=histogram/audio