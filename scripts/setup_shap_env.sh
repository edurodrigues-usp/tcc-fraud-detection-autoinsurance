#!/bin/bash
# =============================================================================
# Setup do Ambiente SHAP - Análise de Interpretabilidade
# =============================================================================
# Autor: Eduardo Barbante Rodrigues
#
# IMPORTANTE: Este ambiente usa versões ESPECÍFICAS para compatibilidade!
# =============================================================================

set -e

echo ""
echo "============================================================================"
echo "SETUP DO AMBIENTE SHAP (venv_shap)"
echo "============================================================================"
echo ""
echo "ATENÇÃO: Este ambiente usa NumPy 1.26.4 (NÃO 2.x) para compatibilidade!"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python3 não encontrado!"
    echo "Por favor, instale Python 3.11+"
    exit 1
fi

echo "[1/5] Criando ambiente virtual..."
python3 -m venv ../venv_shap

echo "[2/5] Ativando ambiente..."
source ../venv_shap/bin/activate

echo "[3/5] Atualizando pip..."
pip install --upgrade pip --quiet

echo "[4/5] Instalando dependências (pode demorar alguns minutos)..."
pip install -r ../requirements/requirements_shap.txt --quiet

echo "[5/5] Verificando instalação..."
python -c "import numpy, pandas, shap; print(f'NumPy: {numpy.__version__}'); print(f'Pandas: {pandas.__version__}'); print(f'SHAP: {shap.__version__}')"

echo ""
echo "============================================================================"
echo "AMBIENTE SHAP CONFIGURADO COM SUCESSO!"
echo "============================================================================"
echo ""
echo "Versões instaladas:"
echo "  NumPy:  1.26.4 (OBRIGATÓRIO - não atualizar!)"
echo "  Pandas: 2.1.4"
echo "  SHAP:   0.50.0"
echo ""
echo "Para ativar o ambiente:"
echo "  source venv_shap/bin/activate"
echo ""
echo "Para executar a análise SHAP:"
echo "  cd src"
echo "  cp ../outputs/best_model_final_full.pkl ."
echo "  cp ../data/fraud_oracle.csv ."
echo "  python fraud_detection_shap_analysis.py"
echo ""
