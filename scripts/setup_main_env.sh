#!/bin/bash
# =============================================================================
# Setup do Ambiente Principal - Pipeline de Detecção de Fraudes
# =============================================================================
# Autor: Eduardo Barbante Rodrigues
# =============================================================================

set -e

echo ""
echo "============================================================================"
echo "SETUP DO AMBIENTE PRINCIPAL (venv_main)"
echo "============================================================================"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python3 não encontrado!"
    echo "Por favor, instale Python 3.11+"
    exit 1
fi

echo "[1/4] Criando ambiente virtual..."
python3 -m venv ../venv_main

echo "[2/4] Ativando ambiente..."
source ../venv_main/bin/activate

echo "[3/4] Atualizando pip..."
pip install --upgrade pip --quiet

echo "[4/4] Instalando dependências..."
pip install -r ../requirements/requirements_main.txt --quiet

echo ""
echo "============================================================================"
echo "AMBIENTE PRINCIPAL CONFIGURADO COM SUCESSO!"
echo "============================================================================"
echo ""
echo "Para ativar o ambiente:"
echo "  source venv_main/bin/activate"
echo ""
echo "Para executar o pipeline:"
echo "  cd src"
echo "  python fraud_detection.py"
echo ""
