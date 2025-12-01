@echo off
REM =============================================================================
REM Setup do Ambiente Principal - Pipeline de Detecção de Fraudes
REM =============================================================================
REM Autor: Eduardo Barbante Rodrigues
REM =============================================================================

echo.
echo ============================================================================
echo SETUP DO AMBIENTE PRINCIPAL (venv_main)
echo ============================================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale Python 3.11+ e adicione ao PATH.
    pause
    exit /b 1
)

echo [1/4] Criando ambiente virtual...
python -m venv ..\venv_main
if errorlevel 1 (
    echo ERRO ao criar ambiente virtual!
    pause
    exit /b 1
)

echo [2/4] Ativando ambiente...
call ..\venv_main\Scripts\activate.bat

echo [3/4] Atualizando pip...
python -m pip install --upgrade pip --quiet

echo [4/4] Instalando dependencias...
pip install -r ..\requirements\requirements_main.txt --quiet
if errorlevel 1 (
    echo ERRO ao instalar dependencias!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo AMBIENTE PRINCIPAL CONFIGURADO COM SUCESSO!
echo ============================================================================
echo.
echo Para ativar o ambiente:
echo   venv_main\Scripts\activate
echo.
echo Para executar o pipeline:
echo   cd src
echo   python fraud_detection.py
echo.

pause
