param(
  [switch]$SkipInstall,
  [switch]$SkipBuild,
  [switch]$NoZip
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$BuildRoot = Join-Path $Root "build"
$PortableRoot = Join-Path $BuildRoot "PaperLens"
$Sidecar = Join-Path $Root "src-tauri\binaries\paperlens-core-x86_64-pc-windows-msvc.exe"

function Read-PackageVersion {
  return (Get-Content -LiteralPath (Join-Path $Root "package.json") -Raw | ConvertFrom-Json).version
}

function Find-AppExe {
  $candidates = @(
    (Join-Path $Root "src-tauri\target\release\PaperLens.exe"),
    (Join-Path $Root "src-tauri\target\release\paperlens.exe"),
    (Join-Path $Root "src-tauri\target\x86_64-pc-windows-msvc\release\PaperLens.exe"),
    (Join-Path $Root "src-tauri\target\x86_64-pc-windows-msvc\release\paperlens.exe")
  )
  return $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

function Assert-FileSize {
  param(
    [string]$Path,
    [int64]$MinBytes,
    [string]$Label
  )
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "$Label was not found: $Path"
  }
  $size = (Get-Item -LiteralPath $Path).Length
  if ($size -lt $MinBytes) {
    throw "$Label is unexpectedly small ($size bytes): $Path"
  }
}

function Copy-RequiredPath {
  param(
    [string]$Source,
    [string]$Destination
  )
  if (-not (Test-Path -LiteralPath $Source)) {
    throw "Required package input is missing: $Source"
  }
  Copy-Item -LiteralPath $Source -Destination $Destination -Recurse
}

function File-Sha256 {
  param([string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

Set-Location $Root

$version = Read-PackageVersion
npm run version:check

if (-not $SkipInstall) {
  npm ci
}

if (-not $SkipBuild) {
  npm run core:build
  npm run tauri:build
}

$appExe = Find-AppExe
if (-not $appExe) {
  throw "PaperLens.exe was not found in Tauri release targets."
}

Assert-FileSize -Path $appExe -MinBytes 5242880 -Label "PaperLens desktop executable"
Assert-FileSize -Path $Sidecar -MinBytes 1048576 -Label "PaperLens Core sidecar"

if (Test-Path -LiteralPath $PortableRoot) {
  Remove-Item -LiteralPath $PortableRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $PortableRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PortableRoot "sidecars") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PortableRoot "resources") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PortableRoot "config") | Out-Null

Copy-Item -LiteralPath $appExe -Destination (Join-Path $PortableRoot "PaperLens.exe")
Copy-Item -LiteralPath $Sidecar -Destination (Join-Path $PortableRoot "sidecars\paperlens-core-x86_64-pc-windows-msvc.exe")
Copy-RequiredPath -Source (Join-Path $Root "README.txt") -Destination (Join-Path $PortableRoot "README.txt")
Copy-RequiredPath -Source (Join-Path $Root "RELEASE_NOTES.md") -Destination (Join-Path $PortableRoot "RELEASE_NOTES.md")
Copy-RequiredPath -Source (Join-Path $Root "config\default_config.json") -Destination (Join-Path $PortableRoot "config\default_config.json")
Copy-RequiredPath -Source (Join-Path $Root "resources\prompts") -Destination (Join-Path $PortableRoot "resources\prompts")
Copy-RequiredPath -Source (Join-Path $Root "resources\schemas") -Destination (Join-Path $PortableRoot "resources\schemas")
Copy-RequiredPath -Source (Join-Path $Root "resources\static") -Destination (Join-Path $PortableRoot "resources\static")

$metadata = [ordered]@{
  product = "PaperLens"
  version = $version
  package = "windows-x64-portable"
  appExe = "PaperLens.exe"
  sidecar = "sidecars/paperlens-core-x86_64-pc-windows-msvc.exe"
  builtAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  gitCommit = (git rev-parse --short HEAD 2>$null)
  files = [ordered]@{
    appSha256 = File-Sha256 (Join-Path $PortableRoot "PaperLens.exe")
    sidecarSha256 = File-Sha256 (Join-Path $PortableRoot "sidecars\paperlens-core-x86_64-pc-windows-msvc.exe")
  }
}
$metadataPath = Join-Path $PortableRoot "BUILD-METADATA.json"
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding utf8
"PaperLens $version" | Set-Content -LiteralPath (Join-Path $PortableRoot "VERSION.txt") -Encoding utf8

Write-Host "Portable folder ready: $PortableRoot"

if (-not $NoZip) {
  $zipPath = Join-Path $BuildRoot "PaperLens-$version-windows-x64-portable.zip"
  $hashPath = "$zipPath.sha256"
  $releaseMetadataPath = Join-Path $BuildRoot "PaperLens-$version-windows-x64-portable.json"
  if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
  }
  Compress-Archive -LiteralPath $PortableRoot -DestinationPath $zipPath -CompressionLevel Optimal
  $zipHash = File-Sha256 $zipPath
  "$zipHash  $(Split-Path -Leaf $zipPath)" | Set-Content -LiteralPath $hashPath -Encoding ascii
  $releaseMetadata = [ordered]@{
    product = "PaperLens"
    version = $version
    package = "windows-x64-portable"
    zip = (Split-Path -Leaf $zipPath)
    sha256 = $zipHash
    builtAtUtc = $metadata.builtAtUtc
    gitCommit = $metadata.gitCommit
  }
  $releaseMetadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $releaseMetadataPath -Encoding utf8
  Write-Host "Portable zip ready: $zipPath"
  Write-Host "Portable SHA256: $zipHash"
}
