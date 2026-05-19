@echo off
chcp 65001 >nul
title Brawl Stars Bot - Dashboard + Backend + Bot
echo ================================================
echo  BRAWL STARS BOT - Soberana Omega
echo  Backend API + Dashboard + Bot
echo ================================================
echo.

REM Verificar Python
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale Python 3.12+ e adicione ao PATH.
    pause
    exit /b 1
)

REM Verificar dependencias criticas
echo [INFO] A verificar dependencias...
py -3.12 -c "import fastapi, uvicorn, ultralytics, toml" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Dependencias em falta. A instalar...
    py -3.12 -m pip install toml fastapi uvicorn ultralytics
    if errorlevel 1 (
        echo [ERRO] Falha ao instalar dependencias.
        pause
        exit /b 1
    )
)

REM Criar pastas necessarias se nao existirem
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs

echo.
echo [INFO] A iniciar Backend API (porta 8003)...
start "Backend API" cmd /k "py -3.12 -m api_server"
timeout /t 3 /nobreak >nul

echo [INFO] A iniciar Bot com Dashboard (porta 8765)...
start "Bot + Dashboard" cmd /k "py -3.12 -m brawl_bot.wrapper"
timeout /t 5 /nobreak >nul

echo.
echo ================================================
echo  SISTEMAS INICIADOS
echo ================================================
echo.
echo  Dashboard:     http://localhost:8765
echo  API Docs:      http://localhost:8003/docs
echo  API Base:      http://localhost:8003
echo.
echo  Comandos uteis:
echo    - Treino:     python train.py --schema core --epochs 50
echo    - Testes:     python -m pytest tests/
echo.
echo  Feche esta janela para manter os servicos a correr.
echo  (Use as janelas individuais para parar cada servico)
echo.
pause
