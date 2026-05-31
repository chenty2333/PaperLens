param(
  [int]$Count = 20,
  [string]$CorpusDir = "tmp\arxiv_corpus_100",
  [string]$OutputDir = "tmp\batch_validation_output",
  [string]$ProviderKind = "none",
  [switch]$EnableLlmStages,
  [string]$Config = "config\default_config.json",
  [string]$Worker = "src-tauri\binaries\paperlens-core-x86_64-pc-windows-msvc.exe"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$CorpusPath = Join-Path $Root $CorpusDir
if (-not (Test-Path -LiteralPath $CorpusPath)) {
  New-Item -ItemType Directory -Force -Path $CorpusPath | Out-Null
}

$pdfs = @(Get-ChildItem -LiteralPath $CorpusPath -File -Filter *.pdf | Sort-Object Name)
if ($pdfs.Count -lt $Count) {
  uv run python scripts/download_arxiv_corpus.py --output-dir $CorpusDir --max-results ([Math]::Max($Count, 100))
  $pdfs = @(Get-ChildItem -LiteralPath $CorpusPath -File -Filter *.pdf | Sort-Object Name)
}
if ($pdfs.Count -lt $Count) {
  throw "Only $($pdfs.Count) PDFs are available in $CorpusDir; need $Count."
}

$InputDir = Join-Path $Root ("tmp\batch_input_{0}" -f $Count)
$OutputPath = Join-Path $Root $OutputDir
if (Test-Path -LiteralPath $InputDir) {
  Remove-Item -LiteralPath $InputDir -Recurse -Force
}
if (Test-Path -LiteralPath $OutputPath) {
  Remove-Item -LiteralPath $OutputPath -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $InputDir | Out-Null
foreach ($pdf in ($pdfs | Select-Object -First $Count)) {
  Copy-Item -LiteralPath $pdf.FullName -Destination (Join-Path $InputDir $pdf.Name)
}

$args = @(
  "run",
  "--input-dir", $InputDir,
  "--output-dir", $OutputPath,
  "--config", (Join-Path $Root $Config),
  "--provider-kind", $ProviderKind,
  "--concurrency", "4"
)
if ($EnableLlmStages) {
  $args += "--enable-llm-stages"
}

if (Test-Path -LiteralPath (Join-Path $Root $Worker)) {
  & (Join-Path $Root $Worker) @args
} else {
  uv run python -m paperlens_core.main @args
}

$Manifest = Join-Path $OutputPath "manifest.json"
if (-not (Test-Path -LiteralPath $Manifest)) {
  throw "Batch validation failed: manifest.json missing."
}
Get-Content -LiteralPath $Manifest
