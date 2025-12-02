# Detecção de Fraudes em Seguros Automotivos com Machine Learning e XAI

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ICMC-USP](https://img.shields.io/badge/ICMC-USP-green.svg)](https://www.icmc.usp.br/)

> **MBA em Inteligência Artificial e Big Data - ICMC/USP**  
> **Autor:** Eduardo Barbante Rodrigues  
> **Orientadora:** Profa. Dra. Cibele Maria Russo Novelli  
> **Ano:** 2025

---

## 📋 Sobre o Projeto

Este repositório contém o código-fonte do Trabalho de Conclusão de Curso (TCC) que propõe um sistema de detecção de fraudes em seguros automotivos baseado em aprendizado de máquina, integrando três perspectivas complementares:

1. **Desempenho Preditivo** - Métricas robustas para dados desbalanceados (MCC, G-Mean, Kappa)
2. **Viabilidade Econômica** - Análise de ROI e Benefício Líquido
3. **Interpretabilidade** - Explicabilidade das decisões via SHAP (XAI)

### 🏆 Resultados Principais

| Métrica | Valor |
|---------|-------|
| **Modelo Campeão** | CatBoost + SMOTEENN |
| **MCC** | 0.3144 |
| **Recall (Taxa de Captura)** | 52.7% |
| **ROI** | 943% |
| **Benefício Líquido** | R$ 3.508.000 |

---

## 📁 Estrutura do Repositório

```
tcc-fraud-detection-autoinsurance/
│
├── README.md                          # Este arquivo
├── LICENSE                            # Licença MIT
│
├── src/                               # Código-fonte principal
│   ├── fraud_detection.py             # Pipeline completo (treino, validação, teste)
│   └── fraud_detection_shap_analysis.py  # Análise SHAP (interpretabilidade)
│
├── requirements/                      # Dependências separadas por ambiente
│   ├── requirements_main.txt          # Pipeline principal
│   └── requirements_shap.txt          # Análise SHAP (ambiente separado)
│
├── scripts/                           # Scripts auxiliares
│   ├── setup_main_env.sh              # Setup ambiente principal (Linux/Mac)
│   ├── setup_shap_env.sh              # Setup ambiente SHAP (Linux/Mac)
│   ├── setup_main_env.bat             # Setup ambiente principal (Windows)
│   └── setup_shap_env.bat             # Setup ambiente SHAP (Windows)
│
├── data/                              # Dados (não versionados)
│   └── .gitkeep
│
├── outputs/                           # Resultados gerados (não versionados)
│   └── .gitkeep
│
└── docs/                              # Documentação adicional
    └── COMPATIBILITY_NOTES.md         # Notas sobre compatibilidade de versões
```

---

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11+
- Git

### 1. Clone o Repositório

```bash
git clone https://github.com/edurodrigues-usp/tcc-fraud-detection-autoinsurance.git
cd tcc-fraud-detection-autoinsurance
```

### 2. O Dataset já está incluído! ✅

O arquivo `data/fraud_oracle.csv` já está no repositório (3.5MB).

Fonte original: [Kaggle - Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)

### 3. Configure os Ambientes

⚠️ **IMPORTANTE:** Este projeto requer **dois ambientes virtuais separados** devido a incompatibilidades entre versões de bibliotecas (ver [Notas de Compatibilidade](docs/COMPATIBILITY_NOTES.md)).

#### Ambiente Principal (Pipeline de ML)

```bash
# Linux/Mac
python -m venv venv_main
source venv_main/bin/activate
pip install -r requirements/requirements_main.txt

# Windows
python -m venv venv_main
venv_main\Scripts\activate
pip install -r requirements/requirements_main.txt
```

#### Ambiente SHAP (Interpretabilidade)

⚠️ **IMPORTANTE:** Este ambiente usa NumPy 2.0+ (diferente do Main que usa 1.26.4).

```bash
# Linux/Mac
python3 -m venv venv_shap
source venv_shap/bin/activate
pip install -r requirements/requirements_shap.txt

# Windows
python -m venv venv_shap
venv_shap\Scripts\activate
pip install -r requirements/requirements_shap.txt

# Windows (se tiver múltiplas versões do Python)
py -3.11 -m venv venv_shap
venv_shap\Scripts\activate
pip install -r requirements/requirements_shap.txt
```

> **Por quê dois ambientes?** PyCaret requer NumPy 1.26.4, mas SHAP 0.50.0 requer NumPy >= 2.0. Ver [docs/COMPATIBILITY_NOTES.md](docs/COMPATIBILITY_NOTES.md) para detalhes.

### 4. Execute o Pipeline

#### Etapa 1: Treinar e Avaliar Modelos

```bash
# Ativar ambiente principal
source venv_main/bin/activate  # Linux/Mac
# ou
venv_main\Scripts\activate     # Windows

# Executar pipeline (modo FAST para teste rápido)
cd src
python fraud_detection.py

# Para execução completa (TCC), edite FAST_MODE = False no script
```

**Saídas geradas em `outputs/`:**
- `best_model_final_full.pkl` - Modelo completo para SHAP
- `best_model_final_light.pkl` - Modelo leve para deploy
- `model_comparison_FINAL_V3.csv` - Comparação de todos os modelos
- `champion_cv_results.csv` - Resultados da validação cruzada

#### Etapa 2: Análise SHAP (Interpretabilidade)

```bash
# ⚠️ TROCAR para ambiente SHAP
deactivate
source venv_shap/bin/activate  # Linux/Mac
# ou
venv_shap\Scripts\activate     # Windows

# Executar análise SHAP (da pasta src/)
cd src
python fraud_detection_shap_analysis.py
```

**Saídas geradas em `outputs/shap_results/`:**
- 23 visualizações PNG (summary plots, waterfalls, dependence plots)
- 1 HTML interativo (force plot)
- CSVs com valores SHAP e importâncias

---

## 📊 Metodologia

### Pipeline de Dados

```
Dataset Bruto (15.420 registros)
        │
        ▼
┌─────────────────────────────────────┐
│  LIMPEZA E DIVISÃO ESTRATIFICADA    │
│  Train (60%) / Val (20%) / Test (20%)│
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  FEATURE ENGINEERING (fit no train) │
│  • Isolation Forest (anomaly score) │
│  • Target Encoding (fraud rates)    │
│  • Variáveis de interação           │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  MODELAGEM COM BALANCEAMENTO        │
│  • SMOTE / ADASYN / SMOTEENN        │
│  • Otimização Bayesiana (Optuna)    │
│  • Threshold Tuning (Kappa)         │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  AVALIAÇÃO FINAL                    │
│  • Métricas técnicas (MCC, G-Mean)  │
│  • Métricas de negócio (ROI)        │
│  • Interpretabilidade (SHAP)        │
└─────────────────────────────────────┘
```

### Algoritmos Avaliados

| Categoria | Algoritmos |
|-----------|------------|
| **Baselines** | DummyClassifier, Logistic Regression |
| **Ensemble/Boosting** | Random Forest, XGBoost, LightGBM, CatBoost |

### Técnicas de Balanceamento

- SMOTE (Synthetic Minority Over-sampling)
- ADASYN (Adaptive Synthetic Sampling)
- SMOTEENN (SMOTE + Edited Nearest Neighbors)
- SMOTETomek (SMOTE + Tomek Links)

---

## 📈 Resultados Detalhados

### Top 5 Modelos (Validação)

| Rank | Modelo | Sampler | Score Composto | MCC | Recall |
|------|--------|---------|----------------|-----|--------|
| 1 | CatBoost | SMOTEENN | 0.4325 | 0.3144 | 52.7% |
| 2 | LightGBM | SMOTETomek | 0.4256 | 0.3087 | 50.5% |
| 3 | CatBoost | Nenhum | 0.4218 | 0.3151 | 44.0% |
| 4 | CatBoost | SMOTETomek | 0.4208 | 0.3106 | 45.7% |
| 5 | XGBoost | SMOTETomek | 0.4198 | 0.3073 | 46.7% |

### Análise SHAP - Top 5 Variáveis

1. **Fault_Policy_Holder** - Culpa do segurado (preditor dominante)
2. **Is_Third_Party_Fault** - Culpa de terceiro
3. **BasePolicy_fraud_rate** - Taxa histórica de fraude da apólice
4. **Make_fraud_rate** - Taxa histórica de fraude por fabricante
5. **Year** - Ano do sinistro

---

## ⚠️ Notas de Compatibilidade

Este projeto requer **dois ambientes virtuais separados** devido a conflitos entre:
- **PyCaret 3.3.2** → requer NumPy 1.26.4
- **SHAP 0.50.0** → requer NumPy >= 2.0

**Solução:** Ambiente Main (treino) separado do Ambiente SHAP (interpretabilidade).

Detalhes completos em [docs/COMPATIBILITY_NOTES.md](docs/COMPATIBILITY_NOTES.md).

---

## 📚 Referências

- **Dataset:** [Fraud Oracle - Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- **SHAP:** Lundberg & Lee (2017) - [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- **Métricas Robustas:** Huayanay, Bazán & Russo (2024) - Performance of evaluation metrics for classification in imbalanced data

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Eduardo Barbante Rodrigues**
- LinkedIn: [/in/eduardorodrigues01](https://linkedin.com/in/eduardorodrigues01)
- GitHub: [@edurodrigues-usp](https://github.com/edurodrigues-usp)

---

## 🙏 Agradecimentos

- Profa. Dra. Cibele Maria Russo Novelli (Orientadora)
- Profa. Dra. Solange Oliveira Rezende
- ICMC-USP
- Porto Seguro (contexto profissional)
