@echo off
set "ROOT=C:\Users\zakho\Desktop\Profolio Project\Project 2"
set "SUB=sales-forecasting-gold-gpr"

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" ^
  -NoProfile ^
  -ExecutionPolicy Bypass ^
  -NoExit ^
  -Command "Set-Location -LiteralPath '%ROOT%\%SUB%'; & '%ROOT%\.venv\Scripts\Activate.ps1'; $Host.UI.RawUI.WindowTitle='Project 2 - Gold GPR (venv)'"
