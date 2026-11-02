@echo off
REM ---------------------------------------------------------------------------------------------
REM Full reproduction of the P2DU study on Windows: creates the Python environment (first run only), installs the
REM pinned libraries and runs every stage of the pipeline. Double-click this file, or run it from
REM a command prompt with optional arguments, for example:
REM     run_pipeline.bat                    full run from a clean state: the results of the previous run
REM                                         (outputs\ and outputs_extra\, fitted models included) are moved to
REM                                         _archive\run_<date>_<time>; only the 2 most recent archives are kept
REM     run_pipeline.bat --resume           continue an interrupted full run without archiving;
REM                                         origins already fitted (outputs\data\parts) are not refitted
REM     run_pipeline.bat --quick            smoke test (a few minutes)
REM     run_pipeline.bat --stage figures    one stage only (outputs are kept)
REM     run_pipeline.bat --workers 4        fix the number of parallel fitting processes (default: detected)
REM Only a run without arguments archives the previous outputs.
REM Log: outputs\logs\pipeline.log
REM ---------------------------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM Arguments: --resume is handled here and removed; everything else goes to run_pipeline.py unchanged
REM (string replacement keeps comma-separated stage lists intact).
set "ARGS= %*"
set "RESUME=0"
if not "%ARGS:--resume=%"=="%ARGS%" set "RESUME=1"
set "ARGS=%ARGS:--resume=%"
set "NOSPACE=%ARGS: =%"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating the virtual environment .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Python was not found. Install Python 3.12 or later from python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

echo [2/4] Installing the pinned libraries ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt --quiet || ".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet

if not defined NOSPACE if "%RESUME%"=="0" (
    echo [3/4] Full run: clearing the results of the previous run ^(kept in _archive, last 2 runs^) ...
    ".venv\Scripts\python.exe" tools\clear_previous_run.py
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
set "TEEOPT="
REM A run with arguments (one stage, --quick, --workers) appends to the log, so that the log of the last full
REM run, from which tools\gen_results_tex.py reads the run time, is kept.
if defined NOSPACE set "TEEOPT=-Append"
if "%RESUME%"=="1" (
    echo [3/4] Resuming: previous outputs are kept; the log is appended.
    set "TEEOPT=-Append"
)

REM ---- Parallel model fitting (stages rq1_fit and sens_fit): one process per block of 4 origins, at most 8.
REM tools\detect_workers.ps1 sets the number of processes from this machine when the run starts: logical
REM processors (2 threads per process, n_jobs in config.yaml), free RAM (about 1.5 GB per process, 3 GB kept free)
REM and the 8 blocks of origins. Close other memory-hungry programs before starting. The results do not depend on
REM the number of processes. To force a value: run_pipeline.bat --workers 4, or "set P2DU_WORKERS=4" before the run.
set "WORKERS="
if not "%ARGS:--workers=%"=="%ARGS%" goto :workers_done
if defined P2DU_WORKERS set "WORKERS=%P2DU_WORKERS%"
if defined WORKERS goto :workers_set
for /f "tokens=1-4" %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -File "tools\detect_workers.ps1"') do (
    set "WORKERS=%%A"
    echo       This machine: %%B logical processors, %%C GB of %%D GB RAM free
)
if not defined WORKERS set "WORKERS=1"
:workers_set
echo       Parallel model fitting: %WORKERS% processes x 2 threads
set "ARGS=%ARGS% --workers %WORKERS%"
:workers_done

if not exist "outputs\logs" mkdir "outputs\logs"
echo [4/4] Running the pipeline. Progress is shown here and written to outputs\logs\pipeline.log
REM stderr is merged into stdout by cmd before PowerShell sees it: Windows PowerShell 5.1 turns every stderr line of a
REM native program into an error record ("python.exe : ... NativeCommandError") when it is redirected inside PowerShell.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:PYTHONUTF8='1'; cmd /d /c '.venv\Scripts\python.exe -u run_pipeline.py %ARGS% 2>&1' | Tee-Object %TEEOPT% -FilePath 'outputs\logs\pipeline.log'"

echo.
echo Finished. Tables: outputs\tables   Figures: outputs\figures   Logs: outputs\logs
pause
