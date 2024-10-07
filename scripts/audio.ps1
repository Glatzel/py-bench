Set-Location $PSScriptRoot
Set-Location ..
pixi run pytest "./src" --benchmark-autosave --benchmark-histogram=histogram/audio