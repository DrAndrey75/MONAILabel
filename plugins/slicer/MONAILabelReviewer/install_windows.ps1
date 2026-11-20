param(
    [string]$DestinationQtModules = ""
)

$ErrorActionPreference = "Stop"
$moduleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceModule = Join-Path $moduleRoot "MONAILabelReviewer.py"
$sourceLib = Join-Path $moduleRoot "MONAILabelReviewerLib"

if ([string]::IsNullOrWhiteSpace($DestinationQtModules)) {
    $DestinationQtModules = Join-Path $env:LOCALAPPDATA "slicer.org\3D Slicer 5.12.0\slicer.org\Extensions-34621\MONAILabel\lib\Slicer-5.12\qt-scripted-modules"
}

if (-not (Test-Path -LiteralPath $sourceModule)) {
    throw "Source module not found: $sourceModule"
}
if (-not (Test-Path -LiteralPath $DestinationQtModules)) {
    throw "Destination not found: $DestinationQtModules"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$destinationModule = Join-Path $DestinationQtModules "MONAILabelReviewer.py"
$destinationLib = Join-Path $DestinationQtModules "MONAILabelReviewerLib"

if (Test-Path -LiteralPath $destinationModule) {
    Copy-Item -LiteralPath $destinationModule -Destination "$destinationModule.before_custom_$stamp.bak" -Force
}
if (Test-Path -LiteralPath $destinationLib) {
    Copy-Item -LiteralPath $destinationLib -Destination "$destinationLib.before_custom_$stamp.bak" -Recurse -Force
}

Copy-Item -LiteralPath $sourceModule -Destination $destinationModule -Force
New-Item -ItemType Directory -Path $destinationLib -Force | Out-Null
Get-ChildItem -LiteralPath $sourceLib -Filter "*.py" | Copy-Item -Destination $destinationLib -Force

Write-Host "Installed MONAI Label Reviewer custom branch files to: $DestinationQtModules"
Write-Host "Restart 3D Slicer before using the module."
