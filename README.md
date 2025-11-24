# 🎓 TCC: Detecção de fraudes em seguros automotivos com aprendizado de máquina e inteligência artificial explicável (XAI)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/Institution-ICMC--USP-red.svg)](https://www.icmc.usp.br/)

**Trabalho de Conclusão de Curso (TCC)**  
**Autor:** Eduardo Barbante Rodrigues  
**Orientadora:** Profa. Dra. Cibele M. Russo  
**Instituição:** Instituto de Ciências Matemáticas e de Computação (ICMC-USP)  
**Ano:** 2025

---

## 📋 Sobre o Projeto

Este repositório contém o código-fonte completo, dados e documentação do Trabalho de Conclusão de Curso que propõe um sistema de detecção de fraudes em seguros automotivos utilizando técnicas de aprendizado de máquina. O trabalho integra três perspectivas de avaliação:

- **🎯 Técnica:** Métricas especializadas para dados desbalanceados (MCC, G-Mean, Kappa)
- **💰 Econômica:** Análise de viabilidade financeira (ROI, Benefício Líquido)
- **🔍 Interpretabilidade:** Técnicas de XAI (SHAP) para transparência das decisões

### 📊 Principais Resultados

| Métrica | Valor |
|---------|-------|
| **MCC** | 0,3144 |
| **G-Mean** | 0,69 |
| **Kappa** | 0,2924 |
| **Recall** | 52,72% |
| **Precision** | 26,08% |
| **ROI** | **943%** |
| **Benefício Líquido** | **R$ 3.508.000** |

**Modelo Campeão:** CatBoost + SMOTEENN  
**Ganho vs. Baseline:** +23,9% (R$ 676.000)

---

## 🗂️ Estrutura do Repositório

```
tcc-fraud-detection-autoinsurance/
│
├── data/                          # Dados
│   ├── fraud_oracle.csv          # Dataset principal (Kaggle)
│   └── README.md                 # Descrição dos dados
│
├── src/                          # Código-fonte
│   ├── preprocessing/            # Pré-processamento
│   │   ├── feature_engineering.py
│   │   └── data_cleaning.py
│   │
│   ├── models/                   # Modelagem
│   │   ├── train_pipeline.py
│   │   ├── optimization.py
│   │   └── evaluation.py
│   │
│   ├── interpretability/         # Análise SHAP
│   │   ├── shap_analysis.py
│   │   └── shap_visualizations.py
│   │
│   └── utils/                    # Utilitários
│       ├── metrics.py
│       └── plots.py
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
│
├── models/                       # Modelos treinados
│   ├── best_model_FINAL_V3.pkl
│   └── README.md
│
├── results/                      # Resultados
│   ├── figures/                  # Figuras para o TCC
│   ├── tables/                   # Tabelas (CSVs)
│   └── shap_results/            # Análises SHAP
│
├── config/                       # Configurações
│   ├── model_config.yaml
│   └── optuna_config.yaml
│
├── scripts/                      # Scripts utilitários
│   ├── setup_environment.bat     # Windows
│   ├── setup_environment.sh      # Linux/Mac
│   └── run_full_pipeline.py
│
├── docs/                         # Documentação
│   ├── INSTALL.md               # Instruções de instalação
│   ├── USAGE.md                 # Guia de uso
│   └── METHODOLOGY.md           # Metodologia detalhada
│
├── requirements.txt              # Dependências principais
├── requirements_shap.txt         # Ambiente SHAP (separado)
├── .gitignore                    # Arquivos ignorados
├── LICENSE                       # Licença MIT
└── README.md                     # Este arquivo
```

---

## 🚀 Início Rápido

### 1️⃣ **Pré-requisitos**

- Python 3.11+
- Git
- 4 GB RAM mínimo (recomendado: 8 GB)
- 2 GB espaço em disco

### 2️⃣ **Clonar Repositório**

```bash
git clone https://github.com/seu-usuario/tcc-fraud-detection-autoinsurance.git
cd tcc-fraud-detection-autoinsurance
```

### 3️⃣ **Criar Ambiente Virtual**

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ **Instalar Dependências**

#### Ambiente Principal (treinamento):
```bash
pip install -r requirements.txt
```

#### Ambiente SHAP (interpretabilidade - separado):
```bash
python -m venv shap_env
shap_env\Scripts\activate  # Windows
# source shap_env/bin/activate  # Linux/Mac
pip install -r requirements_shap.txt
```

**⚠️ Importante:** Ambientes separados para evitar conflitos de dependências!

### 5️⃣ **Executar Pipeline Completo**

```bash
python scripts/run_full_pipeline.py
```

**Tempo estimado:** ~45-60 minutos

**Saída:**
- Modelo treinado: `models/best_model_FINAL_V3.pkl`
- Métricas: `results/tables/model_comparison.csv`
- Figuras: `results/figures/`

---

## 📖 Guias Detalhados

### 🔧 [Instalação Completa](docs/INSTALL.md)
Instruções detalhadas de instalação em diferentes sistemas operacionais.

### 📘 [Guia de Uso](docs/USAGE.md)
Como executar cada componente do sistema separadamente.

### 🧪 [Metodologia](docs/METHODOLOGY.md)
Explicação detalhada das técnicas utilizadas.

---

## 🎯 Reproduzindo os Resultados do TCC

### Passo 1: Feature Engineering

```bash
python src/preprocessing/feature_engineering.py
```

**Saída:** `data/processed/fraud_oracle_engineered.csv`

### Passo 2: Treinamento e Otimização

```bash
python src/models/train_pipeline.py --optimize
```

**Tempo:** ~30-40 minutos  
**Saída:** Modelo otimizado com Optuna

### Passo 3: Avaliação Econômica

```bash
python src/models/evaluation.py --economic
```

**Saída:** Tabelas e figuras de análise econômica

### Passo 4: Análise SHAP

```bash
# Ativar ambiente SHAP
shap_env\Scripts\activate

# Executar análise
python src/interpretability/shap_analysis.py
```

**Tempo:** ~8-10 minutos  
**Saída:** 25 visualizações SHAP

---

## 📊 Dataset

### Fraud Oracle Dataset

**Fonte:** [Kaggle - Fraud Oracle Dataset](https://www.kaggle.com/datasets/mastmustu/fraud-oracle-dataset)

**Características:**
- **Instâncias:** 15.420
- **Features:** 33 (originais)
- **Target:** FraudFound_P (binário)
- **Desbalanceamento:** ~6% fraudes
- **Tamanho:** 3.6 MB

**Divisão:**
- Treino: 9.252 (60%)
- Validação: 3.084 (20%)
- Teste: 3.084 (20%)

### Feature Engineering

O pipeline aplica 154 features derivadas:
- Target Encoding
- Taxas de fraude por categoria
- Detecção de anomalias (Isolation Forest)
- Variáveis temporais
- Interações entre features

**Detalhes:** Ver `src/preprocessing/feature_engineering.py`

---

## 🏆 Modelo Campeão

### Arquitetura

**Algoritmo:** CatBoost  
**Balanceamento:** SMOTEENN  
**Otimização:** Optuna (100 trials)  
**Métrica de Otimização:** Kappa de Cohen

### Hiperparâmetros

```yaml
learning_rate: 0.05
depth: 6
iterations: 500
l2_leaf_reg: 3
border_count: 128
random_strength: 1
```

### Pipeline Completo

```
Raw Data → Feature Engineering → SMOTEENN → CatBoost → Threshold Tuning → Predições
```

---

## 📈 Análise de Interpretabilidade

### SHAP (SHapley Additive exPlanations)

**Variável mais importante:** `Fault_Policy_Holder` (culpa do segurado)

**TOP 5 Features:**
1. Fault_Policy_Holder (importância SHAP ~1.0)
2. Is_Third_Party_Fault (~0.55)
3. BasePolicy_fraud_rate (~0.55)
4. Year (~0.45)
5. RepNumber (~0.45)

**Visualizações:**
- Summary Beeswarm (TOP 20)
- Dependence Plots
- Waterfall (casos específicos)
- Force Plots interativos

**Detalhes:** Ver `results/shap_results/`

---

## 📝 Citação

Se você utilizar este trabalho, por favor cite:

```bibtex
@mastersthesis{rodrigues2025fraud,
  author       = {Eduardo Barbante Rodrigues},
  title        = {Detecção de Fraudes em Seguros Automotivos com Machine Learning: 
                  Uma Abordagem Integrando Avaliação Técnica, Econômica e Interpretabilidade},
  school       = {Instituto de Ciências Matemáticas e de Computação, Universidade de São Paulo},
  year         = {2025},
  address      = {São Carlos, SP, Brasil},
  note         = {Trabalho de Conclusão de Curso},
}
```

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para mais detalhes.

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Eduardo Barbante Rodrigues**

- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- LinkedIn: [Eduardo Barbante](https://www.linkedin.com/in/seu-perfil/)
- Email: eduardo.barbante@usp.br

---

## 🙏 Agradecimentos

- **Profa. Dra. Cibele M. Russo** - Orientação acadêmica
- **ICMC-USP** - Infraestrutura e suporte
- **Porto Seguro** - Contexto profissional e motivação
- **Comunidade Kaggle** - Dataset Fraud Oracle

---

## 📚 Referências Principais

1. **Chawla et al. (2002)** - SMOTE: Synthetic Minority Over-sampling Technique
2. **Lundberg & Lee (2017)** - A Unified Approach to Interpreting Model Predictions (SHAP)
3. **Prokhorenkova et al. (2018)** - CatBoost: unbiased boosting with categorical features
4. **Huayanay et al. (2024)** - Performance Evaluation of Machine Learning Models with Kappa

---

## 🔗 Links Úteis

- [Documentação do CatBoost](https://catboost.ai/)
- [Documentação do SHAP](https://shap.readthedocs.io/)
- [PyCaret Documentation](https://pycaret.org/)
- [Optuna Documentation](https://optuna.org/)

---

## ⭐ Se este projeto foi útil, considere dar uma estrela!

---

**Última atualização:** Novembro 2025  
**Versão:** 1.0.0
