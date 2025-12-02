"""
================================================================================
ANÁLISE SHAP STANDALONE - Versão PREMIUM (compatível com model_package_full)
================================================================================

- Lê o modelo salvo em .pkl pelo pipeline final:
  model_package_full = {
      "pipeline": final_pipeline,
      "threshold": best_threshold,
      "feature_engineering": fe,
      "numeric_cols": numeric_cols,
      "categorical_cols": categorical_cols,
      "model_name": best_model_name,
      "sampler_name": best_sampler_name,
      "metrics_test": metrics_test,
      "business_test": business_test,
      "fast_mode": FAST_MODE,
  }

- Aplica o mesmo Feature Engineering do treinamento (objeto fe salvo no pacote).
- Usa o preprocessor + model do pipeline para gerar a matriz de features.
- Calcula SHAP usando o modelo final.
- Gera 23 figuras + 1 HTML interativo.

Autor: Eduardo Barbante Rodrigues
Orientadora: Profa. Dra. Cibele M. Russo
Instituição: ICMC-USP
Data: Novembro 2025
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.compose import ColumnTransformer

print("=" * 80)
print("ANÁLISE SHAP STANDALONE - Versão PREMIUM (model_package_full)")
print("=" * 80)
print(f"SHAP version: {shap.__version__}")
print(f"NumPy version: {np.__version__}")
print("=" * 80)

# ============================================================================
# 1. CONFIGURAÇÕES
# ============================================================================

# Detecta o diretório raiz do projeto (um nível acima de /src)
SCRIPT_DIR = Path(__file__).parent  # /src
PROJECT_ROOT = SCRIPT_DIR.parent     # raiz do projeto

# Diretórios
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR_BASE = PROJECT_ROOT / "outputs"
OUTPUT_DIR = OUTPUT_DIR_BASE / "shap_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Arquivos de entrada
MODEL_FILE = OUTPUT_DIR_BASE / "best_model_final_full.pkl"
DATA_FILE = DATA_DIR / "fraud_oracle.csv"

RANDOM_STATE = 42
SAMPLE_SIZE = 1000  # amostra máxima para SHAP (para não ficar insano)

print(f"\n📁 Diretório do projeto: {PROJECT_ROOT}")
print("\n📂 Configurações:")
print(f"   Modelo (.pkl): {MODEL_FILE}")
print(f"   Dataset:       {DATA_FILE}")
print(f"   Saída:         {OUTPUT_DIR}/")
print(f"   Amostra:       {SAMPLE_SIZE} registros")

# ============================================================================
# FEATURE ENGINEERING AVANÇADO 
# ============================================================================
class AdvancedFeatureEngineering:
    """
    Feature engineering avançado:
        - Price_Range
        - Fraud rates por categoria (Make, AccidentArea, BasePolicy)
        - Anomaly score (Isolation Forest)
        - Features binárias de risco
        - Interações e variáveis temporais
        - Total de sinistros anteriores

    Importante: o ajuste (fit) é feito APENAS no conjunto de treino,
    evitando vazamento (data leakage).
    """

    def __init__(self, contamination=0.1, random_state=RANDOM_STATE):
        self.fraud_rates = {}
        self.iso_forest = None
        self.global_fraud_rate = None
        self.contamination = contamination
        self.random_state = random_state

    def _common_feature_engineering(self, df):
        """Transformações que não dependem de alvo nem de ajuste."""
        df = df.copy()

        # 1. Price Range
        price_map = {
            "less than 20000": 1,
            "20000 to 29000": 2,
            "30000 to 39000": 3,
            "40000 to 59000": 4,
            "60000 to 69000": 5,
            "more than 69000": 6,
        }
        if "VehiclePrice" in df.columns:
            df["Price_Range"] = df["VehiclePrice"].map(price_map).fillna(3)

        # 4. Features binárias
        if "Fault" in df.columns:
            df["Is_Third_Party_Fault"] = (df["Fault"] == "Third Party").astype(int)
        if "Age" in df.columns:
            age_numeric = (
                df["Age"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )
            df["Is_Young_Driver"] = (age_numeric < 26).astype(int)
            df["Is_Senior_Driver"] = (age_numeric > 60).astype(int)
        if "AccidentArea" in df.columns:
            df["Is_Urban_Accident"] = (df["AccidentArea"] == "Urban").astype(int)
        if "BasePolicy" in df.columns:
            df["Is_Comprehensive_Coverage"] = df["BasePolicy"].str.contains(
                "Collision|All Perils", na=False
            ).astype(int)
        if "VehicleCategory" in df.columns:
            df["Is_Sport_Vehicle"] = (df["VehicleCategory"] == "Sport").astype(int)
            df["Is_Utility_Vehicle"] = (df["VehicleCategory"] == "Utility").astype(int)

        # 5. Risk Score simples
        df["Risk_Score"] = 0
        if "Is_Third_Party_Fault" in df.columns:
            df["Risk_Score"] += df["Is_Third_Party_Fault"] * 2
        if "Is_Young_Driver" in df.columns:
            df["Risk_Score"] += df["Is_Young_Driver"] * 1.5
        if "Is_Urban_Accident" in df.columns:
            df["Risk_Score"] += df["Is_Urban_Accident"] * 1

        # 7. Diferenças temporais
        month_map = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }
        if "Month" in df.columns:
            df["Month_Numeric"] = df["Month"].map(month_map).fillna(6)
        if "MonthClaimed" in df.columns:
            df["MonthClaimed_Numeric"] = df["MonthClaimed"].map(month_map).fillna(6)

        if {"WeekOfMonthClaimed", "WeekOfMonth"}.issubset(df.columns):
            df["Week_Diff"] = df["WeekOfMonthClaimed"] - df["WeekOfMonth"]
        if {"MonthClaimed_Numeric", "Month_Numeric"}.issubset(df.columns):
            df["Month_Diff"] = df["MonthClaimed_Numeric"] - df["Month_Numeric"]
            df["Claim_Delay"] = (
                df["MonthClaimed_Numeric"] != df["Month_Numeric"]
            ).astype(int)

        # 8. Sinistros anteriores
        if "NumberOfSuppliments" in df.columns:
            suppliments_map = {
                "none": 0,
                "1 to 2": 1,
                "3 to 4": 3,
                "more than 5": 6,
            }
            df["Suppliments_Numeric"] = df["NumberOfSuppliments"].map(
                suppliments_map
            ).fillna(0)

        if "NumberOfCars" in df.columns:
            cars_map = {
                "1 vehicle": 1,
                "2 vehicles": 2,
                "3 to 4": 3,
                "more than 4": 5,
            }
            df["Cars_Numeric"] = df["NumberOfCars"].map(cars_map).fillna(1)

        if {"Suppliments_Numeric", "Cars_Numeric"}.issubset(df.columns):
            df["Total_Previous_Claims"] = (
                df["Suppliments_Numeric"] + df["Cars_Numeric"]
            )

        # Interações adicionais (usam colunas numéricas criadas)
        if "Age" in df.columns and "Price_Range" in df.columns:
            age_numeric = (
                df["Age"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )
            df["Age_Price_Interaction"] = age_numeric * df["Price_Range"]

        if {"Is_Urban_Accident", "Is_Third_Party_Fault"}.issubset(df.columns):
            df["Urban_ThirdParty"] = (
                df["Is_Urban_Accident"] * df["Is_Third_Party_Fault"]
            )

        if {"Is_Young_Driver", "Is_Sport_Vehicle"}.issubset(df.columns):
            df["Young_Sport"] = df["Is_Young_Driver"] * df["Is_Sport_Vehicle"]

        return df

    def fit(self, df):
        """Ajusta fraud rates e IsolationForest no conjunto de treino."""
        df = df.copy()
        if "FraudFound_P" not in df.columns:
            raise ValueError("Coluna 'FraudFound_P' é obrigatória para o fit.")

        self.global_fraud_rate = df["FraudFound_P"].mean()

        # Fraud rate encoding (no treino)
        for col in ["Make", "AccidentArea", "BasePolicy"]:
            if col in df.columns:
                fr = df.groupby(col)["FraudFound_P"].mean()
                self.fraud_rates[col] = fr.to_dict()

        # Feature engineering comum
        df_fe = self._common_feature_engineering(df)

        # Fraud rate encoding ANTES do Isolation Forest (para evitar mismatch)
        for col in ["Make", "AccidentArea", "BasePolicy"]:
            if col in df_fe.columns and col in self.fraud_rates:
                df_fe[f"{col}_fraud_rate"] = df_fe[col].map(
                    self.fraud_rates[col]
                ).fillna(self.global_fraud_rate)

        # Isolation Forest em colunas numéricas
        numeric_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
        if "FraudFound_P" in numeric_cols:
            numeric_cols.remove("FraudFound_P")

        X_numeric = df_fe[numeric_cols].fillna(0)

        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        iso_labels = self.iso_forest.fit_predict(X_numeric)
        df_fe["anomaly_score"] = (iso_labels == -1).astype(int)

        # Atualizar Risk_Score com anomaly_score
        if "Risk_Score" in df_fe.columns:
            df_fe["Risk_Score"] = df_fe["Risk_Score"] + df_fe["anomaly_score"] * 3

        return df_fe

    def transform(self, df):
        """Aplica o feature engineering com parâmetros ajustados no treino."""
        if self.global_fraud_rate is None or self.iso_forest is None:
            raise RuntimeError("Chame fit() antes de transform().")

        df = df.copy()
        df_fe = self._common_feature_engineering(df)

        # Fraud rate encoding com taxas aprendidas no treino
        for col in ["Make", "AccidentArea", "BasePolicy"]:
            if col in df_fe.columns and col in self.fraud_rates:
                df_fe[f"{col}_fraud_rate"] = df_fe[col].map(
                    self.fraud_rates[col]
                ).fillna(self.global_fraud_rate)

        # Isolation Forest em colunas numéricas (mesmas features usadas no fit)
        numeric_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
        if "FraudFound_P" in numeric_cols:
            numeric_cols.remove("FraudFound_P")

        X_numeric = df_fe[numeric_cols].fillna(0)
        iso_labels = self.iso_forest.predict(X_numeric)
        df_fe["anomaly_score"] = (iso_labels == -1).astype(int)

        if "Risk_Score" in df_fe.columns:
            df_fe["Risk_Score"] = df_fe["Risk_Score"] + df_fe["anomaly_score"] * 3

        return df_fe

    def fit_transform(self, df):
        """Atalho: fit + transform no treino."""
        df_fe = self.fit(df)
        return df_fe

# ============================================================================
# 2. CARREGAR MODELO DO PKL (model_package_full)
# ============================================================================

print("\n" + "=" * 80)
print("2. CARREGANDO MODELO DO PKL")
print("=" * 80)

with open(MODEL_FILE, "rb") as f:
    model_package = pickle.load(f)

# Tenta o formato novo (model_package_full); cai no antigo se necessário
pipeline = model_package.get("pipeline", None)
final_model = None
preprocessor = None
fe = model_package.get("feature_engineering", None)

best_threshold = model_package.get("threshold", None)
model_name = model_package.get("model_name", "desconhecido")
sampler_name = model_package.get("sampler_name", "desconhecido")
fast_mode = model_package.get("fast_mode", None)
metrics_test = model_package.get("metrics_test", {})
business_test = model_package.get("business_test", {})

# Compatibilidade com versão antiga (que salvava apenas 'model')
if pipeline is None and "model" in model_package:
    print("⚠️  Pacote não possui 'pipeline', usando chave 'model' (versão antiga).")
    final_model = model_package["model"]
else:
    print("✅ Pipeline carregado do pacote.")
    # pipeline é um ImbPipeline com steps: preprocess -> sampler (opcional) -> model
    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocess", None)
        final_model = pipeline.named_steps.get("model", pipeline)
    else:
        final_model = pipeline

print(f"   Modelo interno: {type(final_model).__name__}")
print(f"   Nome lógico:    {model_name}")
print(f"   Sampler:        {sampler_name}")
if fast_mode is not None:
    print(f"   FAST_MODE:      {fast_mode}")

if best_threshold is not None:
    print(f"   Threshold salvo: {best_threshold:.3f}")

# ============================================================================
# 3. CARREGAR DADOS E APLICAR FEATURE ENGINEERING DO PACOTE
# ============================================================================

print("\n" + "=" * 80)
print("3. CARREGANDO DADOS E FEATURE ENGINEERING (do pacote)")
print("=" * 80)

df = pd.read_csv(DATA_FILE)
print(f"✅ Dataset carregado: {df.shape[0]} registros, {df.shape[1]} variáveis")

# Remover PolicyNumber se o treino removeu (id único / leakage)
if "PolicyNumber" in df.columns:
    print("⚠️  Removendo PolicyNumber (identificador único - potencial leakage).")
    df = df.drop(columns=["PolicyNumber"])

# Aplicar o MESMO Feature Engineering usado no treino
if fe is not None:
    print("⚙️  Aplicando AdvancedFeatureEngineering.transform() (sem refazer fit)...")
    df_fe = fe.transform(df.copy())
else:
    print("⚠️  Nenhum feature_engineering encontrado no pacote. Usando dataset bruto.")
    df_fe = df.copy()

print(f"✅ Após FE: {df_fe.shape[0]} registros, {df_fe.shape[1]} variáveis")

# Separar X e y
if "FraudFound_P" in df_fe.columns:
    X = df_fe.drop("FraudFound_P", axis=1)
    y = df_fe["FraudFound_P"]
else:
    X = df_fe
    y = None
    print("⚠️  Coluna 'FraudFound_P' não encontrada. Algumas análises por classe serão puladas.")

# Amostra aleatória para SHAP
if len(X) > SAMPLE_SIZE:
    sample_idx = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    y_sample = y.iloc[sample_idx].reset_index(drop=True) if y is not None else None
else:
    X_sample = X.reset_index(drop=True)
    y_sample = y.reset_index(drop=True) if y is not None else None

print(f"📊 Amostra SHAP: {X_sample.shape[0]} registros, {X_sample.shape[1]} features")

# ============================================================================
# 4. TRANSFORMAR DADOS USANDO O PREPROCESSOR DO PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("4. TRANSFORMANDO DADOS COM PREPROCESSOR DO PIPELINE")
print("=" * 80)

# Se não houver preprocessor, usamos X_sample direto
if preprocessor is not None:
    print("   Usando ColumnTransformer 'preprocess' do pipeline.")
    X_enc = preprocessor.transform(X_sample)
    # Descobrir nomes das features pós-encoding
    if isinstance(preprocessor, ColumnTransformer):
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_enc.shape[1])]
    else:
        feature_names = [f"feature_{i}" for i in range(X_enc.shape[1])]
else:
    print("⚠️  Nenhum preprocessor encontrado no pipeline. Usando features originais.")
    X_enc = X_sample.values
    feature_names = X_sample.columns.astype(str).tolist()

# Converter para denso se vier esparso
if hasattr(X_enc, "toarray"):
    X_enc = X_enc.toarray()

X_shap = pd.DataFrame(X_enc, columns=feature_names)
print(f"✅ Dados transformados para SHAP: {X_shap.shape}")
print(f"   Nº de features após encoding: {X_shap.shape[1]}")

# ============================================================================
# 5. CRIAR EXPLAINER SHAP
# ============================================================================

print("\n" + "=" * 80)
print("5. CRIANDO EXPLAINER SHAP")
print("=" * 80)

model_type = type(final_model).__name__
print(f"🔍 Tipo de modelo: {model_type}")

# Usar a API genérica do SHAP (ela escolhe o melhor explainer possível)
# Para modelos de árvore (XGB, LGBM, RF, CatBoost) ela vai cair em TreeExplainer.
background = shap.sample(X_shap, min(100, len(X_shap)))  # background menor pra acelerar
explainer = shap.Explainer(final_model, background)
shap_values = explainer(X_shap)

print("✅ SHAP values calculados!")
print(f"   Formato: {shap_values.values.shape}")

# ============================================================================
# 6. GERAR VISUALIZAÇÕES (23 + 1 HTML)
# ============================================================================

print("\n" + "=" * 80)
print("6. GERANDO VISUALIZAÇÕES (23 + 1 HTML)")
print("=" * 80)

plt.style.use("seaborn-v0_8-darkgrid")
viz_count = 0

# Importância média absoluta por feature
feature_importance = np.abs(shap_values.values).mean(axis=0)
top_20_idx = np.argsort(feature_importance)[-20:]
feature_names_top20 = [feature_names[i] for i in top_20_idx]

# ============================================================================
# CATEGORIA 1: VISÃO GLOBAL (4)
# ============================================================================

print("\n" + "-" * 80)
print("CATEGORIA 1: VISÃO GLOBAL")
print("-" * 80)

# 1. Summary Plot (Beeswarm)
viz_count += 1
print(f"\n📊 {viz_count}/23: Summary Beeswarm (TOP 20)...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values[:, top_20_idx], X_shap.iloc[:, top_20_idx],
                  max_display=20, show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_summary_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   ✅ Salvo: {OUTPUT_DIR}/01_summary_beeswarm.png")

# 2. Summary Plot (Bar)
viz_count += 1
print(f"\n📊 {viz_count}/23: Summary Bar (TOP 20)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[:, top_20_idx], X_shap.iloc[:, top_20_idx],
                  plot_type="bar", max_display=20, show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_summary_bar.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   ✅ Salvo: {OUTPUT_DIR}/02_summary_bar.png")

# 3. Heatmap de SHAP Values (TOP 20)
viz_count += 1
print(f"\n📊 {viz_count}/23: Heatmap de SHAP Values (TOP 20)...")
plt.figure(figsize=(14, 10))
shap_values_top20 = shap_values.values[:, top_20_idx]
sns.heatmap(
    shap_values_top20.T,
    cmap="RdBu_r",
    center=0,
    yticklabels=feature_names_top20,
    cbar_kws={"label": "SHAP value"},
)
plt.xlabel("Amostras")
plt.ylabel("Features")
plt.title("Heatmap de SHAP Values (TOP 20 Features)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   ✅ Salvo: {OUTPUT_DIR}/03_heatmap.png")

# 4. Force Plot Interativo (HTML)
viz_count += 1
print(f"\n📊 {viz_count}/23: Force Plot Interativo (HTML)...")
shap.save_html(
    str(OUTPUT_DIR / "04_force_plot_interactive.html"),
    shap.force_plot(explainer.expected_value, shap_values.values, X_shap),
)
print(f"   ✅ Salvo: {OUTPUT_DIR}/04_force_plot_interactive.html")

# ============================================================================
# CATEGORIA 2: ANÁLISE POR CLASSE (2)
# ============================================================================

print("\n" + "-" * 80)
print("CATEGORIA 2: ANÁLISE POR CLASSE")
print("-" * 80)

if y_sample is not None:
    y_arr = y_sample.values
    fraud_mask = (y_arr == 1)
    legit_mask = (y_arr == 0)

    # 5. SHAP Summary - Fraude
    viz_count += 1
    print(f"\n📊 {viz_count}/23: SHAP Summary - Fraude...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values[fraud_mask][:, top_20_idx],
        X_shap.iloc[fraud_mask, :].iloc[:, top_20_idx],
        max_display=20,
        show=False,
    )
    plt.title("SHAP Summary - Casos de FRAUDE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_summary_fraud.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {OUTPUT_DIR}/05_summary_fraud.png")

    # 6. SHAP Summary - Legítimo
    viz_count += 1
    print(f"\n📊 {viz_count}/23: SHAP Summary - Legítimo...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values[legit_mask][:, top_20_idx],
        X_shap.iloc[legit_mask, :].iloc[:, top_20_idx],
        max_display=20,
        show=False,
    )
    plt.title("SHAP Summary - Casos LEGÍTIMOS")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_summary_legit.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {OUTPUT_DIR}/06_summary_legit.png")
else:
    print("   ⚠️  Target não disponível, pulando análise por classe.")
    viz_count += 2

# ============================================================================
# CATEGORIA 3: FEATURES INDIVIDUAIS (10)
# ============================================================================

print("\n" + "-" * 80)
print("CATEGORIA 3: FEATURES INDIVIDUAIS (TOP 10)")
print("-" * 80)

top_10_idx = np.argsort(feature_importance)[-10:][::-1]

for i, feat_idx in enumerate(top_10_idx, 1):
    viz_count += 1
    feat_name = feature_names[feat_idx]
    print(f"\n📊 {viz_count}/23: Dependence Plot - {feat_name}...")
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feat_idx,
        shap_values.values,
        X_shap,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    safe_name = str(feat_name).replace("/", "_").replace(" ", "_")[:50]
    plt.savefig(
        OUTPUT_DIR / f"{viz_count:02d}_dependence_{i}_{safe_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"   ✅ Salvo: {OUTPUT_DIR}/{viz_count:02d}_dependence_{i}_{safe_name}.png")

# ============================================================================
# CATEGORIA 4: CASOS ESPECÍFICOS (6)
# ============================================================================

print("\n" + "-" * 80)
print("CATEGORIA 4: CASOS ESPECÍFICOS")
print("-" * 80)

if y_sample is not None:
    # Probabilidades
    if hasattr(final_model, "predict_proba"):
        proba = final_model.predict_proba(X_shap.values)[:, 1]
    else:
        proba = final_model.predict(X_shap.values)

    y_arr = y_sample.values
    fraud_positions = np.where(y_arr == 1)[0]
    legit_positions = np.where(y_arr == 0)[0]

    if len(fraud_positions) > 0 and len(legit_positions) > 0:
        # Fraude: top 2 mais confiantes
        fraud_proba = proba[fraud_positions]
        top_fraud_pos = fraud_positions[np.argsort(fraud_proba)[-2:][::-1]]

        # Legítimo: top 2 mais confiantes (menor probabilidade de fraude)
        legit_proba = proba[legit_positions]
        top_legit_pos = legit_positions[np.argsort(legit_proba)[:2]]

        # Casos medianos
        fraud_median_pos = fraud_positions[len(fraud_positions) // 2]
        legit_median_pos = legit_positions[len(legit_positions) // 2]

        # 17. Waterfall - Fraude (médio)
        viz_count += 1
        print(f"\n📊 {viz_count}/23: Waterfall - Fraude (caso médio)...")
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[fraud_median_pos], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{viz_count:02d}_waterfall_fraud_median.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(
            f"   ✅ Salvo: {OUTPUT_DIR}/{viz_count:02d}_waterfall_fraud_median.png"
        )

        # 18. Waterfall - Legítimo (médio)
        viz_count += 1
        print(f"\n📊 {viz_count}/23: Waterfall - Legítimo (caso médio)...")
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap_values[legit_median_pos], max_display=15, show=False
        )
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{viz_count:02d}_waterfall_legit_median.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(
            f"   ✅ Salvo: {OUTPUT_DIR}/{viz_count:02d}_waterfall_legit_median.png"
        )

        # 19–20. Waterfall - Fraude TOP 2
        for i, pos in enumerate(top_fraud_pos, 1):
            viz_count += 1
            print(
                f"\n📊 {viz_count}/23: Waterfall - Fraude TOP {i} (mais confiante)..."
            )
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap_values[pos], max_display=15, show=False)
            plt.title(f"Fraude TOP {i} - Probabilidade: {proba[pos]:.3f}")
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"{viz_count:02d}_waterfall_fraud_top{i}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(
                f"   ✅ Salvo: {OUTPUT_DIR}/{viz_count:02d}_waterfall_fraud_top{i}.png"
            )

        # 21–22. Waterfall - Legítimo TOP 2
        for i, pos in enumerate(top_legit_pos, 1):
            viz_count += 1
            print(
                f"\n📊 {viz_count}/23: Waterfall - Legítimo TOP {i} (mais confiante)..."
            )
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap_values[pos], max_display=15, show=False)
            plt.title(f"Legítimo TOP {i} - Probabilidade: {proba[pos]:.3f}")
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"{viz_count:02d}_waterfall_legit_top{i}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(
                f"   ✅ Salvo: {OUTPUT_DIR}/{viz_count:02d}_waterfall_legit_top{i}.png"
            )
    else:
        print("   ⚠️  Poucos casos de fraude/legítimo para análise de casos específicos.")
        viz_count += 6
else:
    print("   ⚠️  Target não disponível, pulando casos específicos.")
    viz_count += 6

# ============================================================================
# CATEGORIA 5: ANÁLISE ESTATÍSTICA (1)
# ============================================================================

print("\n" + "-" * 80)
print("CATEGORIA 5: ANÁLISE ESTATÍSTICA")
print("-" * 80)

# 23. Violin Plot
viz_count += 1
print(f"\n📊 {viz_count}/23: Violin Plot de SHAP Values (TOP 20)...")
plt.figure(figsize=(12, 10))
shap_df_violin = pd.DataFrame(
    shap_values.values[:, top_20_idx], columns=feature_names_top20
)
shap_df_melted = shap_df_violin.melt(
    var_name="Feature", value_name="SHAP Value"
)
sns.violinplot(data=shap_df_melted, y="Feature", x="SHAP Value", orient="h")
plt.title("Distribuição de SHAP Values (TOP 20 Features)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / f"{viz_count:02d}_violin_plot.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print(f"   ✅ Salvo: {OUTPUT_DIR}/{viz_count:02d}_violin_plot.png")

# ============================================================================
# 7. EXPORTAR DADOS
# ============================================================================

print("\n" + "=" * 80)
print("7. EXPORTANDO DADOS")
print("=" * 80)

importance_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
importance_df.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)
print(f"✅ Feature importance: {OUTPUT_DIR}/shap_feature_importance.csv")

shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_df.to_csv(OUTPUT_DIR / "shap_values_sample.csv", index=False)
print(f"✅ SHAP values (amostra): {OUTPUT_DIR}/shap_values_sample.csv")

# ============================================================================
# CONCLUSÃO
# ============================================================================

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA!")
print("=" * 80)

print(f"\n📊 Resultados salvos em: {OUTPUT_DIR}/")
print(f"\n📈 Visualizações geradas: {viz_count}/23 + 1 HTML")

print(f"\n🏆 CATEGORIA 1: VISÃO GLOBAL (4)")
print("   01. Summary Beeswarm (TOP 20)")
print("   02. Summary Bar (TOP 20)")
print("   03. Heatmap de SHAP Values")
print("   04. Force Plot Interativo (HTML)")

print(f"\n🎯 CATEGORIA 2: ANÁLISE POR CLASSE (2)")
print("   05. SHAP Summary - Fraude")
print("   06. SHAP Summary - Legítimo")

print(f"\n🔍 CATEGORIA 3: FEATURES INDIVIDUAIS (10)")
print("   07–16. Dependence Plots (TOP 10)")

print(f"\n💡 CATEGORIA 4: CASOS ESPECÍFICOS (6)")
print("   17. Waterfall - Fraude (médio)")
print("   18. Waterfall - Legítimo (médio)")
print("   19–20. Waterfall - Fraude TOP 2")
print("   21–22. Waterfall - Legítimo TOP 2")

print(f"\n📊 CATEGORIA 5: ANÁLISE ESTATÍSTICA (1)")
print("   23. Violin Plot (TOP 20)")

print("\n📁 Dados exportados:")
print("   - shap_feature_importance.csv - Ranking completo")
print("   - shap_values_sample.csv      - Valores SHAP brutos (amostra)")

print("\n" + "=" * 80)
