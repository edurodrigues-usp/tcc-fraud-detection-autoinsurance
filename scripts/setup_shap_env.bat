@echo off
REM =============================================================================
REM Setup do Ambiente SHAP - Analise de Interpretabilidade
REM =============================================================================
REM Autor: Eduardo Barbante Rodrigues
REM 
REM IMPORTANTE: Este ambiente usa versoes ESPECIFICAS para compatibilidade!
REM =============================================================================

echo.
echo ============================================================================
echo SETUP DO AMBIENTE SHAP (venv_shap)
echo ============================================================================
echo.
echo ATENCAO: Este ambiente usa NumPy 1.26.4 (NAO 2.x) para compatibilidade!
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale Python 3.11+ e adicione ao PATH.
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
pip install -r ..\requirements\requirements_shap.txt --quiet
if errorlevel 1 (
    echo ERRO ao instalar dependencias!
    pause
    exit /b 1
)

echo [5/5] Verificando instalacao...
python -c "import numpy, pandas, shap; print(f'NumPy: {numpy.__version__}'); print(f'Pandas: {pandas.__version__}'); print(f'SHAP: {shap.__version__}')"
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
echo Versoes instaladas:
echo   NumPy:  1.26.4 (OBRIGATORIO - nao atualizar!)
echo   Pandas: 2.1.4
echo   SHAP:   0.50.0
echo.
echo Para ativar o ambiente:
echo   venv_shap\Scripts\activate
echo.
echo Para executar a analise SHAP:
echo   cd src
echo   copy ..\outputs\best_model_final_full.pkl .
echo   copy ..\data\fraud_oracle.csv .
echo   python fraud_detection_shap_analysis.py
echo.

pause
