Set-Location $PSScriptRoot
Set-Location ..
pixi run pytest "./src" `
--benchmark-max-time=0.00005 `
--benchmark-min-rounds=1 `
--benchmark-histogram=histogram/audio `
--cov --cov-report=html