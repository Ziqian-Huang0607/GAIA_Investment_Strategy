@echo off
setlocal
cd /d %~dp0
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -e .
python -m gaia_ai full %*
echo.
echo Outputs are in the outputs folder.
pause
