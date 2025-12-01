# Notas de Compatibilidade de Versões

## Problema

Durante o desenvolvimento deste projeto, foram identificadas incompatibilidades críticas entre versões de bibliotecas do ecossistema Python de Machine Learning. Essas incompatibilidades impediam a execução conjunta do treinamento de modelos e da análise SHAP em um único ambiente virtual.

## Conflitos Identificados

### 1. NumPy 2.x vs Serialização de Modelos

**Sintoma:** Erro ao carregar modelos `.pkl` salvos com NumPy 1.x em ambiente com NumPy 2.x.

```python
ModuleNotFoundError: No module named 'numpy._core'
# ou
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**Causa:** NumPy 2.0 (lançado em 2024) mudou a estrutura interna de arrays, quebrando compatibilidade com objetos serializados em versões anteriores.

**Solução:** Usar NumPy 1.26.4 no ambiente SHAP.

### 2. SHAP vs XGBoost/LightGBM

**Sintoma:** Warnings ou erros ao calcular SHAP values com TreeExplainer.

```python
XGBoostError: feature_names mismatch
```

**Causa:** SHAP espera nomes de features em formato específico que pode divergir entre versões do XGBoost.

**Solução:** Garantir que as features transformadas pelo `ColumnTransformer` tenham nomes consistentes.

### 3. PyCaret vs NumPy 2.x

**Sintoma:** PyCaret não funciona com NumPy 2.x.

```python
ImportError: cannot import name 'np' from 'numpy'
```

**Causa:** PyCaret 3.3.x foi desenvolvido para NumPy 1.x e não é compatível com as mudanças do NumPy 2.0.

**Solução:** Este projeto não usa PyCaret no pipeline final, mas caso seja necessário, manter NumPy 1.26.4.

## Arquitetura de Solução

A solução adotada foi criar **dois ambientes virtuais separados**:

```
┌─────────────────────────────────────────────────────────────┐
│                    AMBIENTE PRINCIPAL                        │
│                      (venv_main)                            │
│                                                             │
│  NumPy >= 1.24.0, < 2.0.0                                  │
│  scikit-learn, imbalanced-learn                            │
│  XGBoost, LightGBM, CatBoost                               │
│  Optuna                                                     │
│                                                             │
│  📝 Usado para: Treino, Validação, Teste                    │
│  📦 Saída: best_model_final_full.pkl                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ (arquivo .pkl)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    AMBIENTE SHAP                            │
│                     (venv_shap)                             │
│                                                             │
│  NumPy == 1.26.4 (FIXO!)                                   │
│  Pandas == 2.1.4 (FIXO!)                                   │
│  SHAP == 0.50.0                                            │
│  + mesmas libs de ML (para deserialização)                 │
│                                                             │
│  📝 Usado para: Análise de Interpretabilidade              │
│  📊 Saída: Gráficos SHAP, CSVs                             │
└─────────────────────────────────────────────────────────────┘
```

## Procedimento de Migração entre Ambientes

### Windows

```batch
REM Desativar ambiente atual
deactivate

REM Ativar ambiente SHAP
venv_shap\Scripts\activate

REM Verificar versões
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import shap; print(f'SHAP: {shap.__version__}')"
```

### Linux/Mac

```bash
# Desativar ambiente atual
deactivate

# Ativar ambiente SHAP
source venv_shap/bin/activate

# Verificar versões
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import shap; print(f'SHAP: {shap.__version__}')"
```

## Versões Testadas e Funcionais

| Biblioteca | Ambiente Main | Ambiente SHAP |
|------------|---------------|---------------|
| Python | 3.11.x | 3.11.x |
| NumPy | 1.26.4 | 1.26.4 |
| Pandas | 2.1.4 | 2.1.4 |
| scikit-learn | 1.4.2 | 1.4.2 |
| XGBoost | 2.0.3 | 2.0.3 |
| LightGBM | 4.3.0 | 4.3.0 |
| CatBoost | 1.2.7 | 1.2.7 |
| SHAP | - | 0.50.0 |
| imbalanced-learn | 0.12.0 | 0.12.0 |
| Optuna | 3.6.1 | - |

## Alternativas Consideradas

### 1. Google Colab
- **Prós:** Ambiente pré-configurado, fácil de compartilhar
- **Contras:** Limitações de tempo de execução, dependência de internet

### 2. Docker
- **Prós:** Reprodutibilidade total
- **Contras:** Complexidade adicional para usuários não técnicos

### 3. Ambiente Único com Downgrades
- **Prós:** Simplicidade
- **Contras:** Conflitos inevitáveis entre dependências

A solução de dois ambientes virtuais foi escolhida por oferecer o melhor equilíbrio entre reprodutibilidade e praticidade.

## Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'numpy._core'"

```bash
# Reinstalar NumPy com versão específica
pip uninstall numpy -y
pip install numpy==1.26.4 --no-cache-dir
```

### Erro: "feature_names mismatch"

Verificar se o `ColumnTransformer` está gerando nomes de features consistentes:

```python
# No script de treino
feature_names = preprocessor.get_feature_names_out()
print(feature_names[:10])
```

### Erro: "SHAP TreeExplainer not supported"

Verificar se o modelo é baseado em árvores:

```python
# Modelos suportados: XGBoost, LightGBM, CatBoost, RandomForest, DecisionTree
from shap import TreeExplainer
explainer = TreeExplainer(model)
```

## Referências

- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [SHAP GitHub Issues](https://github.com/slundberg/shap/issues)
- [scikit-learn Persistence](https://scikit-learn.org/stable/modules/model_persistence.html)
