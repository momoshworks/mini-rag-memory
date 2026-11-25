@echo off
title Mini RAG + Dynamic Memory Web Dashboard
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   🧠 نظام الذاكرة الذكية Mini RAG + Dynamic Memory
echo   🌐 الرابط: http://127.0.0.1:8040
echo ========================================================
start "" "http://127.0.0.1:8040"
python api_server.py
pause