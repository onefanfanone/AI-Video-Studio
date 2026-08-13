param(
  [Parameter(Mandatory=$true)][string]$AssetsDirectory,
  [string]$OutputDirectory = "dist"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$lock = Get-Content -Raw -LiteralPath (Join-Path $repo "release\assets.lock.json") | ConvertFrom-Json
$destination = Join-Path $repo $OutputDirectory
$public = Join-Path $destination "AI-Video-Studio"
if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination -Recurse -Force }
New-Item -ItemType Directory -Path $destination | Out-Null
& (Join-Path $repo ".venv\Scripts\python.exe") (Join-Path $repo "tools\build_public_export.py") --destination $public
$verified = @{}
foreach ($entry in @($lock.python, $lock.ffmpeg)) {
  if (-not $entry.sha256) { throw "release/assets.lock.json has no verified SHA-256 for $($entry.filename)." }
  $asset = Join-Path $AssetsDirectory $entry.filename
  if (-not (Test-Path -LiteralPath $asset)) { throw "Missing release asset: $asset" }
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $asset).Hash.ToLowerInvariant()
  if ($actual -ne $entry.sha256.ToLowerInvariant()) { throw "Checksum mismatch: $($entry.filename)" }
  $verified[$entry.filename] = $asset
}
$bootstrap = Join-Path $public "bootstrap"
New-Item -ItemType Directory -Path $bootstrap -Force | Out-Null
Copy-Item -LiteralPath $verified[$lock.python.filename] -Destination (Join-Path $bootstrap $lock.python.filename)
$ffmpegStaging = Join-Path $destination "ffmpeg-staging"
Expand-Archive -LiteralPath $verified[$lock.ffmpeg.filename] -DestinationPath $ffmpegStaging -Force
$ffmpegExe = Get-ChildItem -LiteralPath $ffmpegStaging -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
$ffprobeExe = Get-ChildItem -LiteralPath $ffmpegStaging -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
if (-not $ffmpegExe -or -not $ffprobeExe) { throw "The verified FFmpeg archive does not contain ffmpeg.exe and ffprobe.exe." }
$ffmpegBin = Join-Path $bootstrap "ffmpeg\bin"
New-Item -ItemType Directory -Path $ffmpegBin -Force | Out-Null
Copy-Item -LiteralPath $ffmpegExe.FullName -Destination $ffmpegBin
Copy-Item -LiteralPath $ffprobeExe.FullName -Destination $ffmpegBin
$wheelhouse = Join-Path $public "wheelhouse"
New-Item -ItemType Directory -Path $wheelhouse -Force | Out-Null
& (Join-Path $repo ".venv\Scripts\python.exe") -m pip download --requirement (Join-Path $repo "requirements.txt") --dest $wheelhouse --only-binary=:all:
if ($LASTEXITCODE -ne 0) { throw "Unable to build the pinned offline wheelhouse." }
Remove-Item -LiteralPath $ffmpegStaging -Recurse -Force
Compress-Archive -LiteralPath $public -DestinationPath (Join-Path $destination "AI-Video-Studio-windows-x64.zip") -CompressionLevel Optimal
Write-Host "Release package created under $destination"
