# Study Buddy installer for Windows. In PowerShell, run:
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/endahiro/study-buddy-alu/main/install.ps1 | iex"
#
# First checks this computer has what Study Buddy needs, then installs whatever is
# missing: Ollama (runs the AI models), uv (installs Python apps), Study Buddy, and the
# default AI model. Safe to run again: it skips what's there and updates Study Buddy.
$ErrorActionPreference = "Stop"

$Package = "https://github.com/endahiro/study-buddy-alu/releases/download/v1.0.0/studybuddy-1.0.0-py3-none-any.whl"
$Model = "qwen2.5-coder:1.5b"
$MinRamGB = 3.5  # a "4 GB" computer reports a little under 4
$MinDiskGB = 3   # Ollama + the model + Study Buddy, with some room to spare

function Ok($Text)   { Write-Host "  $([char]0x2713) " -ForegroundColor Green -NoNewline; Write-Host $Text }
function Todo($Text) { Write-Host "  * " -ForegroundColor Cyan -NoNewline; Write-Host $Text }
function Warn($Text) { Write-Host "  ! " -ForegroundColor Yellow -NoNewline; Write-Host $Text }
function Fail($Text) { Write-Host "`nError: $Text" -ForegroundColor Red; exit 1 }
function Step($Number, $Text) {
    Write-Host "`n[$Number/4] " -ForegroundColor Cyan -NoNewline; Write-Host $Text
}
function Has($Command) { [bool](Get-Command $Command -ErrorAction SilentlyContinue) }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                "$env:USERPROFILE\.local\bin;$env:LOCALAPPDATA\Programs\Ollama"
}
function Ollama-Running {
    try { ollama list *> $null; return $LASTEXITCODE -eq 0 } catch { return $false }
}

Write-Host "`nStudy Buddy installer"
Write-Host "A free AI coding tutor for your terminal" -ForegroundColor DarkGray

# --- Before installing: check this computer ---------------------------------------------

Write-Host "`nChecking your computer"
Ok "System: Windows"

try {
    Invoke-WebRequest "https://github.com" -Method Head -UseBasicParsing -TimeoutSec 10 | Out-Null
    Ok "Internet connection"
} catch {
    Fail "No internet connection. Connect and run this again."
}

$RamGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
if ($RamGB -ge $MinRamGB) { Ok "Memory: $RamGB GB" }
else { Warn "Memory: $RamGB GB. 4 GB or more is recommended; answers may be slow or fail." }

$Drive = (Get-Item $env:USERPROFILE).PSDrive
$DiskGB = [math]::Floor($Drive.Free / 1GB)
if ($DiskGB -ge $MinDiskGB) { Ok "Free disk space: $DiskGB GB" }
else { Fail "Only $DiskGB GB of free disk space; Study Buddy needs about $MinDiskGB GB. Free some up and run this again." }

if (Has "ollama") { Ok "Ollama: already installed" } else { Todo "Ollama: will be installed" }
if (Has "uv") { Ok "uv: already installed" } else { Todo "uv: will be installed" }
if (Has "studybuddy") { Todo "Study Buddy: will be updated" } else { Todo "Study Buddy: will be installed" }

# --- 1. Ollama ----------------------------------------------------------------------------

Step 1 "Ollama (runs the AI model on your computer)"
if (Has "ollama") {
    Ok "Already installed"
} else {
    if (Has "winget") {
        winget install --id Ollama.Ollama --exact --silent --accept-source-agreements --accept-package-agreements | Out-Null
    } else {
        $Setup = Join-Path $env:TEMP "OllamaSetup.exe"
        Invoke-WebRequest "https://ollama.com/download/OllamaSetup.exe" -OutFile $Setup -UseBasicParsing
        Start-Process $Setup -ArgumentList "/SILENT" -Wait
    }
    Refresh-Path
    if (-not (Has "ollama")) { Fail "Ollama didn't install. Install it from https://ollama.com/download, then run this again." }
    Ok "Installed"
}

# Ollama must be running to download the model. Start it if it isn't.
if (-not (Ollama-Running)) {
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    $Tries = 0
    while (-not (Ollama-Running)) {
        $Tries++
        if ($Tries -gt 30) { Fail "Ollama didn't start. Open the Ollama app, then run this again." }
        Start-Sleep -Seconds 1
    }
    Ok "Started"
}

# --- 2. uv --------------------------------------------------------------------------------

Step 2 "uv (installs Study Buddy and the Python it needs)"
if (Has "uv") {
    Ok "Already installed"
} else {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" | Out-Null
    Refresh-Path
    if (-not (Has "uv")) { Fail "uv didn't install. See https://docs.astral.sh/uv/ and run this again." }
    Ok "Installed"
}

# --- 3. Study Buddy -------------------------------------------------------------------------

Step 3 "Study Buddy"
# --force so running the installer again updates it.
uv tool install --quiet --force $Package
if ($LASTEXITCODE -ne 0) { Fail "Installing Study Buddy failed (see the message above)." }
# Redirected native output can throw under ErrorActionPreference=Stop in Windows PowerShell.
try { uv tool update-shell *> $null } catch { }
Ok "Installed"

# --- 4. The AI model ------------------------------------------------------------------------

Step 4 "AI model $Model (about 1 GB, only the first time)"
ollama pull $Model
if ($LASTEXITCODE -ne 0) { Fail "Downloading the model failed. Check your internet connection and run this again." }
Ok "Ready"

Write-Host "`n$([char]0x2713) Study Buddy is installed." -ForegroundColor Green
Write-Host "Open a NEW terminal, go to a folder with your code or notes, and run:`n"
Write-Host "    studybuddy`n" -ForegroundColor Cyan
