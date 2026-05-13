@echo off
REM Script para iniciar o servidor API do Soberana Omega Bot
REM Porta: 8003

echo ========================================
echo Iniciando Soberana Omega Bot API
echo ========================================
echo.

REM Navegar para o diretório do bot
cd /d "%~dp0"

echo Diretório atual: %CD%
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python não encontrado no PATH!
    echo Por favor, instale Python e adicione ao PATH.
    pause
    exit /b 1
)

echo [OK] Python encontrado
python --version
echo.

REM Verificar se api.py existe
if not exist "api.py" (
    echo [ERRO] api.py não encontrado no diretório atual!
    echo Por favor, execute este script no diretório backend/brawl_bot/
    pause
    exit /b 1
)

echo [OK] api.py encontrado
echo.

REM Verificar dependências
echo Verificando dependências...
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] uvicorn não encontrado. Instalando...
    pip install uvicorn fastapi
)

python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] fastapi não encontrado. Instalando...
    pip install fastapi
)

echo [OK] Dependências verificadas
echo.

REM Iniciar servidor
echo Iniciando servidor na porta 8003...
echo Pressione Ctrl+C para parar o servidor
echo ========================================
echo.

python api.py

pause
