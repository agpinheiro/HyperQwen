@echo off
rem Sobe o vLLM (KV q2_1, 262K, porta 11434) + UI llama.cpp em :8088, com os logs neste terminal.
rem Ctrl+C para tudo. Pronto quando aparecer "Application startup complete" (~4 min).
cd /d "%~dp0"
docker compose --profile single up --attach single
pause
