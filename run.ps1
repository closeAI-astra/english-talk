param(
    [ValidateSet('app','models','doctor','sync','test')]
    [string]$Action = 'app'
)
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'data\uv-cache'
$env:HF_HOME = Join-Path $PSScriptRoot 'data\model-cache'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $PSScriptRoot
switch ($Action) {
    'sync' { uv sync }
    'models' { uv run prepare_models.py }
    'doctor' { uv run doctor.py }
    'test' { uv run pytest -q }
    default { uv run app.py }
}
exit $LASTEXITCODE
