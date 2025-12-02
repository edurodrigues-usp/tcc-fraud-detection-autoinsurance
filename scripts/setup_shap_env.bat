@echo off
REM =============================================================================
REM Setup do Ambiente SHAP - Analise de Interpretabilidade
REM =============================================================================
REM Autor: Eduardo Barbante Rodrigues
REM 
REM IMPORTANTE: Use Python 3.11 (ou 3.10) - NAO use Python 3.12+
REM
REM Se voce tem multiplas versoes do Python, use:
REM   py -3.11 -m venv venv_shap
REM =============================================================================

echo.
echo ============================================================================
echo SETUP DO AMBIENTE SHAP (venv_shap)
echo ============================================================================
echo.
echo Este ambiente e separado do principal porque:
echo   - Ambiente Main: NumPy 1.26.4 (para PyCaret)
echo   - Ambiente SHAP: NumPy 2.0+ (para SHAP)
echo.
echo Se voce tem multiplas versoes do Python instaladas, use:
echo   py -3.11 -m venv venv_shap
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale Python 3.11 e adicione ao PATH.
    pause
    exit /b 1
)

echo [1/5] Criando ambiente virtual...
python -m venv ..\venv_shap
if errorlevel 1 (
    echo ERRO ao criar ambiente virtual!
    pause
    exit /b 1
)

echo [2/5] Ativando ambiente...
call ..\venv_shap\Scripts\activate.bat

echo [3/5] Atualizando pip...
python -m pip install --upgrade pip --quiet

echo [4/5] Instalando dependencias (pode demorar alguns minutos)...
pip install -r ..\requirements\requirements_shap.txt
if errorlevel 1 (
    echo ERRO ao instalar dependencias!
    pause
    exit /b 1
)

echo [5/5] Verificando instalacao...
python -c "import numpy, shap; print(f'NumPy: {numpy.__version__}'); print(f'SHAP: {shap.__version__}')"
if errorlevel 1 (
    echo ERRO: Falha ao importar bibliotecas!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo AMBIENTE SHAP CONFIGURADO COM SUCESSO!
echo ============================================================================
echo.
echo Para ativar o ambiente:
echo   venv_shap\Scripts\activate
echo.
echo Para executar a analise SHAP:
echo   cd src
echo   python fraud_detection_shap_analysis.py
echo.

pause
