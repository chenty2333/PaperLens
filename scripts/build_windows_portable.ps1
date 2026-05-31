$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PortableRoot = Join-Path $Root "build\PaperLens"
$Sidecar = Join-Path $Root "src-tauri\binaries\paperlens-core-x86_64-pc-windows-msvc.exe"

Set-Location $Root

uv run --extra build python scripts/build_core.py
npm install
npm run tauri:build

$Candidates = @(
  (Join-Path $Root "src-tauri\target\release\PaperLens.exe"),
  (Join-Path $Root "src-tauri\target\release\paperlens.exe"),
  (Join-Path $Root "src-tauri\target\x86_64-pc-windows-msvc\release\PaperLens.exe"),
  (Join-Path $Root "src-tauri\target\x86_64-pc-windows-msvc\release\paperlens.exe")
)
$AppExe = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $AppExe) {
  throw "PaperLens.exe was not found in Tauri release targets."
}
if (-not (Test-Path -LiteralPath $Sidecar)) {
  throw "Windows sidecar was not built: $Sidecar"
}
if ((Get-Item -LiteralPath $Sidecar).Length -lt 1048576) {
  throw "Windows sidecar is still too small; likely incomplete: $Sidecar"
}

if (Test-Path -LiteralPath $PortableRoot) {
  Remove-Item -LiteralPath $PortableRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $PortableRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PortableRoot "sidecars") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PortableRoot "resources") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PortableRoot "config") | Out-Null

Copy-Item -LiteralPath $AppExe -Destination (Join-Path $PortableRoot "PaperLens.exe")
Copy-Item -LiteralPath $Sidecar -Destination (Join-Path $PortableRoot "sidecars\paperlens-core-x86_64-pc-windows-msvc.exe")
Copy-Item -LiteralPath (Join-Path $Root "README.txt") -Destination (Join-Path $PortableRoot "README.txt")
Copy-Item -LiteralPath (Join-Path $Root "config\default_config.json") -Destination (Join-Path $PortableRoot "config\default_config.json")
Copy-Item -LiteralPath (Join-Path $Root "resources\prompts") -Destination (Join-Path $PortableRoot "resources\prompts") -Recurse
Copy-Item -LiteralPath (Join-Path $Root "resources\schemas") -Destination (Join-Path $PortableRoot "resources\schemas") -Recurse
Copy-Item -LiteralPath (Join-Path $Root "resources\static") -Destination (Join-Path $PortableRoot "resources\static") -Recurse

Write-Host "Portable folder ready: $PortableRoot"
