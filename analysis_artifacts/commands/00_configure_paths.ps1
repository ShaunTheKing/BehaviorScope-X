param(
    [string]$ConfigYaml,
    [string]$RunRoot,
    [string]$MarsDataRoot,
    [string]$DataRoot,
    [string]$DlcWorkRoot,
    [string]$BehaviorScopeXGuiRoot,
    [string]$ManuscriptPackageRoot,
    [string]$Python = "python",
    [string]$OutputDir,
    [switch]$SkipValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-SimpleYaml {
    param([string]$Path)

    $values = @{}
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $values
    }

    $expanded = [Environment]::ExpandEnvironmentVariables($Path)
    if (-not (Test-Path -LiteralPath $expanded)) {
        throw "ConfigYaml does not exist: $expanded"
    }

    foreach ($line in Get-Content -LiteralPath $expanded) {
        $trimmed = $line.Trim()
        if ($trimmed.Length -eq 0 -or $trimmed.StartsWith("#")) {
            continue
        }

        $match = [regex]::Match($line, "^\s*([A-Za-z0-9_]+)\s*:\s*(.*)\s*$")
        if (-not $match.Success) {
            continue
        }

        $key = $match.Groups[1].Value
        $value = $match.Groups[2].Value.Trim()
        $commentIndex = $value.IndexOf(" #")
        if ($commentIndex -ge 0) {
            $value = $value.Substring(0, $commentIndex).Trim()
        }
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        $values[$key] = $value
    }

    return $values
}

function Use-ConfigValue {
    param(
        [string]$CurrentValue,
        [hashtable]$Config,
        [string]$Key
    )

    if (-not [string]::IsNullOrWhiteSpace($CurrentValue)) {
        return $CurrentValue
    }
    if ($Config.ContainsKey($Key)) {
        return $Config[$Key]
    }
    return $CurrentValue
}

function Get-ConfigBool {
    param(
        [hashtable]$Config,
        [string]$Key,
        [bool]$DefaultValue
    )

    if (-not $Config.ContainsKey($Key)) {
        return $DefaultValue
    }
    $value = $Config[$Key]
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $DefaultValue
    }
    return [System.Convert]::ToBoolean($value)
}

function Resolve-ProvidedPath {
    param(
        [string]$PathValue,
        [string]$Label,
        [switch]$RequireExisting
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $null
    }

    $expanded = [Environment]::ExpandEnvironmentVariables($PathValue)
    if ($RequireExisting -and -not $SkipValidation -and -not (Test-Path -LiteralPath $expanded)) {
        throw "$Label does not exist: $expanded"
    }

    return [System.IO.Path]::GetFullPath($expanded).TrimEnd('\')
}

$config = Read-SimpleYaml -Path $ConfigYaml
$RunRoot = Use-ConfigValue -CurrentValue $RunRoot -Config $config -Key "run_root"
$MarsDataRoot = Use-ConfigValue -CurrentValue $MarsDataRoot -Config $config -Key "mars_data_root"
$DataRoot = Use-ConfigValue -CurrentValue $DataRoot -Config $config -Key "data_root"
$DlcWorkRoot = Use-ConfigValue -CurrentValue $DlcWorkRoot -Config $config -Key "dlc_work_root"
$BehaviorScopeXGuiRoot = Use-ConfigValue -CurrentValue $BehaviorScopeXGuiRoot -Config $config -Key "behaviorscope_x_gui_root"
$ManuscriptPackageRoot = Use-ConfigValue -CurrentValue $ManuscriptPackageRoot -Config $config -Key "manuscript_package_root"
$Python = Use-ConfigValue -CurrentValue $Python -Config $config -Key "python"
$OutputDir = Use-ConfigValue -CurrentValue $OutputDir -Config $config -Key "output_dir"

if (-not $SkipValidation -and $config.ContainsKey("skip_validation")) {
    $SkipValidation = [System.Convert]::ToBoolean($config["skip_validation"])
}

$includeByTemplate = @{
    "01_mars_yolo_sppf_command_template.txt" = Get-ConfigBool -Config $config -Key "include_mars_yolo_sppf" -DefaultValue $true
    "03_mobilenetv3_backbone_command_template.txt" = Get-ConfigBool -Config $config -Key "include_mobilenetv3_backbone" -DefaultValue $true
    "04_dlc_hrnet_topdown_command_template.txt" = Get-ConfigBool -Config $config -Key "include_dlc_hrnet_topdown" -DefaultValue $true
    "06_fly_v_fly_command_template.txt" = Get-ConfigBool -Config $config -Key "include_fly_v_fly" -DefaultValue $true
    "00_render_all_manuscript_figures_from_upstream_outputs_template.ps1" = Get-ConfigBool -Config $config -Key "include_figure_rendering" -DefaultValue $true
}

$packageRootDefault = Split-Path -Parent $PSScriptRoot
$PackageRoot = Resolve-ProvidedPath -PathValue $(if ($ManuscriptPackageRoot) { $ManuscriptPackageRoot } else { $packageRootDefault }) -Label "ManuscriptPackageRoot" -RequireExisting
$RunRootResolved = Resolve-ProvidedPath -PathValue $RunRoot -Label "RunRoot" -RequireExisting
$MarsDataRootResolved = Resolve-ProvidedPath -PathValue $MarsDataRoot -Label "MarsDataRoot" -RequireExisting
$DataRootResolved = Resolve-ProvidedPath -PathValue $DataRoot -Label "DataRoot" -RequireExisting
$DlcWorkRootResolved = Resolve-ProvidedPath -PathValue $DlcWorkRoot -Label "DlcWorkRoot" -RequireExisting
$BehaviorScopeXGuiRootResolved = Resolve-ProvidedPath -PathValue $BehaviorScopeXGuiRoot -Label "BehaviorScopeXGuiRoot" -RequireExisting

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $PSScriptRoot "localized"
}
$OutputDirResolved = Resolve-ProvidedPath -PathValue $OutputDir -Label "OutputDir"
New-Item -ItemType Directory -Force -Path $OutputDirResolved | Out-Null

if ($MarsDataRootResolved -and -not $SkipValidation) {
    $expectedSplits = @("train", "validation", "test_1", "test_2")
    foreach ($split in $expectedSplits) {
        $splitPath = Join-Path $MarsDataRootResolved $split
        if (-not (Test-Path -LiteralPath $splitPath)) {
            Write-Warning "MARS split folder was not found: $splitPath"
        }
    }
}

$replacements = [ordered]@{
    "<PACKAGE_ROOT>" = $PackageRoot
    "<MANUSCRIPT_PACKAGE_ROOT>" = $PackageRoot
    "<RUN_ROOT>" = $RunRootResolved
    "<MARS_DATA_ROOT>" = $MarsDataRootResolved
    "<DATA_ROOT>" = $DataRootResolved
    "<DLC_WORK_ROOT>" = $DlcWorkRootResolved
    "<BEHAVIORSCOPE_X_GUI_ROOT>" = $BehaviorScopeXGuiRootResolved
    "<PYTHON>" = $Python
}

$templateFiles = Get-ChildItem -LiteralPath $PSScriptRoot -File |
    Where-Object { $_.Name -like "*_template.txt" -or $_.Name -like "*_template.ps1" }

$reportRows = @()
foreach ($template in $templateFiles) {
    $localizedName = $template.Name -replace "_template", ""
    if ($includeByTemplate.ContainsKey($template.Name) -and -not $includeByTemplate[$template.Name]) {
        $reportRows += [pscustomobject]@{
            File = $localizedName
            Status = "skipped"
            MissingPlaceholders = "disabled in reproducibility_paths.yml"
        }
        Write-Host "Skipped $localizedName"
        continue
    }

    $dest = Join-Path $OutputDirResolved $localizedName
    $text = Get-Content -LiteralPath $template.FullName -Raw

    foreach ($key in $replacements.Keys) {
        $value = $replacements[$key]
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            $text = $text.Replace($key, $value)
        }
    }

    Set-Content -LiteralPath $dest -Value $text -NoNewline -Encoding UTF8
    $activeText = (($text -split "\r?\n") | Where-Object { -not $_.TrimStart().StartsWith("#") }) -join [Environment]::NewLine
    $remaining = @([regex]::Matches($activeText, "<[A-Z_]+>") | ForEach-Object { $_.Value } | Sort-Object -Unique)
    $status = if ($remaining.Count -eq 0) { "runnable" } else { "not runnable" }
    $reportRows += [pscustomobject]@{
        File = $localizedName
        Status = $status
        MissingPlaceholders = ($remaining -join ", ")
    }
    Write-Host "Wrote $dest"
}

$reportPath = Join-Path $OutputDirResolved "RUNNABILITY_REPORT.md"
$report = @()
$report += "# Localized Command Runnability Report"
$report += ""
$report += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
$report += ""
$report += "| Command file | Status | Missing placeholders |"
$report += "| --- | --- | --- |"
foreach ($row in $reportRows) {
    $missing = if ($row.MissingPlaceholders) { $row.MissingPlaceholders } else { "" }
    $report += "| $($row.File) | $($row.Status) | $missing |"
}
$report += ""
$report += 'Blank paths in `reproducibility_paths.yml` intentionally leave placeholders unresolved. Treat those command files as not runnable until the corresponding path is set. Disabled experiment families are skipped and can be enabled by changing their `include_*` value to `true`.'
Set-Content -LiteralPath $reportPath -Value ($report -join [Environment]::NewLine) -Encoding UTF8
Write-Host "Wrote $reportPath"

if ($reportRows | Where-Object { $_.Status -eq "not runnable" }) {
    Write-Warning "Some placeholders remain unresolved in localized command files. Provide the corresponding flags if those experiments are being reproduced."
}

Write-Host ""
Write-Host "Localized command files are in: $OutputDirResolved"
