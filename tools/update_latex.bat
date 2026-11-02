@echo off
REM ---------------------------------------------------------------------------------------------
REM Refresh and rebuild the LaTeX manuscript from the outputs of this folder. Since 01/11/2026 every step lives in
REM ..\Source-Latex\build_all.bat (refresh of Macros and Figs, check_claims.py, every PDF, DSS and anonymity checks),
REM so running that file once is enough; this script only points it at this code folder, for a manuscript folder
REM elsewhere. Usage: tools\update_latex.bat [path\to\Source-Latex]
REM ---------------------------------------------------------------------------------------------
setlocal
cd /d "%~dp0\.."
set "TEX=%~1"
if "%TEX%"=="" set "TEX=..\Source-Latex"
if not exist "%TEX%\main.tex" (
    echo %TEX%\main.tex was not found. Give the folder of the LaTeX manuscript as the first argument.
    pause
    exit /b 1
)
set "P2DU_CODE=%CD%"
call "%TEX%\build_all.bat"
exit /b %ERRORLEVEL%
