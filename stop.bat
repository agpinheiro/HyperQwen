@echo off
rem Para o vLLM e a UI (os que start.bat sobe).
cd /d "%~dp0"
docker compose --profile single stop
