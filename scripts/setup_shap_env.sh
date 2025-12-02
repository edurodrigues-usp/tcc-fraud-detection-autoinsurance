#!/bin/bash
# =============================================================================
# Setup do Ambiente SHAP - Análise de Interpretabilidade
# =============================================================================
# Autor: Eduardo Barbante Rodrigues
#
# IMPORTANTE: Use Python 3.11 (ou 3.10) - NÃO use Python 3.12+
# =============================================================================

set -e

echo ""
echo "============================================================================"
echo "SETUP DO AMBIENTE SHAP (venv_shap)"
echo "============================================================================"
echo ""
echo "Este ambiente é separado do principal porque:"
echo "  - Ambiente Main: NumPy 1.26.4 (para PyCaret)"
echo "  - Ambiente SHAP: NumPy 2.0+ (para SHAP)"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python3 não encontrado!"
    echo "Por favor, instale Python 3.11"
    exit 1
fi

echo "[1/5] Criando ambiente virtual..."
python3 -m venv ../venv_shap

echo "[2/5] Ativando ambiente..."
source ../venv_shap/bin/activate

echo "[3/5] Atualizando pip..."
pip install --upgrade pip --quiet

echo "[4/5] Instalando dependências (pode demorar alguns minutos)..."
pip install -r ../requirements/requirements_shap.txt

echo "[5/5] Verificando instalação..."
python -c "import numpy, shap; print(f'NumPy: {numpy.__version__}'); print(f'SHAP: {shap.__version__}')"

echo ""
echo "============================================================================"
echo "AMBIENTE SHAP CONFIGURADO COM SUCESSO!"
echo "============================================================================"
echo ""
echo "Para ativar o ambiente:"
echo "  source venv_shap/bin/activate"
echo ""
echo "Para executar a análise SHAP:"
echo "  cd src"
echo "  python fraud_detection_shap_analysis.py"
echo ""
