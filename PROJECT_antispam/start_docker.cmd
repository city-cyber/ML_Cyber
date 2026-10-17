@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_docker.ps1" %*
exit /b %ERRORLEVEL%
