"""
================================================================================
DETECÇÃO DE FRAUDES EM SEGUROS AUTOMOTIVOS
PIPELINE FINAL COM FEATURE ENGINEERING + SMOTE + OPTUNA + THRESHOLD TUNING
================================================================================

Objetivo:
    - Pipeline transparente, reprodutível.
    - Feature engineering avançado SEM data leakage.
    - Tratamento de desbalanceamento com SMOTE e variações.
    - Otimização de hiperparâmetros com Optuna (RF, XGB, LGBM, CatBoost).
    - Otimização de limiar (threshold) para maximizar MCC.
    - Métricas técnicas (MCC, G-Mean, Kappa, Precision, Recall, F1, Accuracy).
    - Métricas de negócio (Net Benefit, ROI, Taxa de Captura de Fraudes).

Autor: Eduardo Barbante Rodrigues
Orientadora: Profa. Dra. Cibele M. Russo
Instituição: ICMC-USP
Data: Novembro 2025
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.dummy import DummyClassifier
from sklearn.base import clone

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score

# Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    print("✅ Optuna disponível!")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna NÃO instalado. O script rodará sem otimização de hiperparâmetros.")

# XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  xgboost não está instalado. XGBClassifier será ignorado.")

# LightGBM
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️  lightgbm não está instalado. LGBMClassifier será ignorado.")

# CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  catboost não está instalado. CatBoostClassifier será ignorado.")


# ============================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS (paths relativos à raiz do projeto)
# ============================================================================
from pathlib import Path
import argparse

# Detecta o diretório raiz do projeto (um nível acima de /src)
SCRIPT_DIR = Path(__file__).parent  # /src
PROJECT_ROOT = SCRIPT_DIR.parent     # raiz do projeto

# Diretórios de entrada e saída
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Arquivos
DATA_FILE = DATA_DIR / "fraud_oracle.csv"

print(f"📁 Diretório do projeto: {PROJECT_ROOT}")
print(f"📂 Dados: {DATA_DIR}")
print(f"📂 Saídas: {OUTPUT_DIR}")

# ============================================================================
# ARGUMENTOS DE LINHA DE COMANDO
# ============================================================================
parser = argparse.ArgumentParser(
    description="Pipeline de Detecção de Fraudes em Seguros Automotivos",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Exemplos de uso:
  python fraud_detection.py              # Modo FULL (padrão)
  python fraud_detection.py --fast       # Modo FAST (teste rápido)
  python fraud_detection.py --no-cv      # Modo FULL sem validação cruzada
  python fraud_detection.py --fast --no-cv
    """
)
parser.add_argument(
    "--fast", 
    action="store_true",
    help="Modo rápido: menos trials, menos samplers (para teste)"
)
parser.add_argument(
    "--no-cv", 
    action="store_true",
    help="Desabilita validação cruzada 5-fold do modelo campeão"
)
args = parser.parse_args()

# ============================================================================
# CONFIGURAÇÕES GERAIS + FAST/FULL MODE
# ============================================================================

RANDOM_STATE = 42
FRAUD_COST = 40_000       # custo médio de uma fraude não detectada
INVESTIGATION_COST = 1000 # custo médio por investigação / sindicancia

# Configurações via argumentos de linha de comando
FAST_MODE = args.fast
RUN_CHAMPION_CV = not args.no_cv

print("\n🚀 Modo selecionado:", "FAST" if FAST_MODE else "FULL")
print(f"📊 Validação cruzada: {'SIM' if RUN_CHAMPION_CV else 'NÃO'}")

if FAST_MODE:
    # 🔥 MODO FAST: foco em rapidez para debug / testes
    N_TRIALS_OPTUNA = 20
    SAMPLERS = {
        "none": None,
        "smote": SMOTE(random_state=RANDOM_STATE),
    }
    ADVANCED_MODELS = ["XGBClassifier"]  # modelo campeão para tabular
else:
    # 🧠 MODO FULL: execução completa para resultados finais do TCC
    N_TRIALS_OPTUNA = 50
    SAMPLERS = {
        "none": None,
        "smote": SMOTE(random_state=RANDOM_STATE),
        "adasyn": ADASYN(random_state=RANDOM_STATE),
        "smoteenn": SMOTEENN(random_state=RANDOM_STATE),
        "smotetomek": SMOTETomek(random_state=RANDOM_STATE),
    }
    ADVANCED_MODELS = ["RandomForest", "XGBClassifier", "LGBMClassifier", "CatBoost"]

BASELINE_MODELS = ["DummyClassifier", "LogisticRegression"]


# ============================================================================
# FUNÇÕES AUXILIARES DE MÉTRICAS
# ============================================================================

def optimize_threshold(y_true, y_proba, metric="kappa"):
    """
    Otimiza o threshold para maximizar a métrica escolhida.
    Trabalha com probabilidades (saída de predict_proba).

    Args:
        y_true: array-like, rótulos verdadeiros (0/1)
        y_proba: array-like, probabilidades preditas para a classe 1
        metric: 'mcc', 'gmean' ou 'kappa'

    Returns:
        best_threshold, best_score, y_pred, metrics_dict
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    scores = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        if metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "gmean":
            score = geometric_mean_score(y_true, y_pred)
        elif metric == "kappa":
            score = cohen_kappa_score(y_true, y_pred)
        else:
            raise ValueError(f"Métrica desconhecida: {metric}")

        scores.append(score)

    best_idx = int(np.argmax(scores))
    best_threshold = float(thresholds[best_idx])
    best_score = float(scores[best_idx])

    # Predições com melhor threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "gmean": geometric_mean_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    return best_threshold, best_score, y_pred, metrics


def calculate_business_metrics(
    y_true,
    y_pred,
    fraud_cost=FRAUD_COST,
    investigation_cost=INVESTIGATION_COST,
):
    """
    Calcula métricas de negócio (custos, economia, ROI, etc.)

    Args:
        y_true: rótulos verdadeiros
        y_pred: predições (0/1)
        fraud_cost: custo de uma fraude não detectada
        investigation_cost: custo por investigação aberta

    Returns:
        dict com total_cost, savings, net_benefit, roi, fraud_catch_rate, cost_per_fraud
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    cost_missed_frauds = fn * fraud_cost
    cost_investigations = (tp + fp) * investigation_cost
    total_cost = cost_missed_frauds + cost_investigations

    savings = tp * fraud_cost
    net_benefit = savings - cost_investigations

    roi = (net_benefit / cost_investigations * 100) if cost_investigations > 0 else 0.0

    total_frauds = tp + fn
    fraud_catch_rate = (tp / total_frauds * 100) if total_frauds > 0 else 0.0

    cost_per_fraud = (cost_investigations / tp) if tp > 0 else float("inf")

    return {
        "total_cost": total_cost,
        "savings": savings,
        "net_benefit": net_benefit,
        "roi": roi,
        "fraud_catch_rate": fraud_catch_rate,
        "cost_per_fraud": cost_per_fraud,
    }


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
# OPTUNA – FUNÇÕES DE OTIMIZAÇÃO
# ============================================================================

def build_model(model_name, params=None):
    """
    Constrói um modelo sklearn/catboost/xgboost/lightgbm a partir do nome e dos params.
    """
    if model_name == "RandomForest":
        if params is None:
            params = {
                "n_estimators": 300,
                "max_depth": 20,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": "balanced",
            }
        params = {
            **params,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        return RandomForestClassifier(**params)

    elif model_name == "XGBClassifier":
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost não está disponível.")
        if params is None:
            params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "scale_pos_weight": 1.0,
            }
        params = {
            **params,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
        return XGBClassifier(**params)

    elif model_name == "LGBMClassifier":
        if not LGBM_AVAILABLE:
            raise RuntimeError("lightgbm não está disponível.")
        if params is None:
            params = {
                "n_estimators": 300,
                "max_depth": -1,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "scale_pos_weight": 1.0,
            }
        params = {
            **params,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        return LGBMClassifier(**params)

    elif model_name == "CatBoost":
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("catboost não está disponível.")
        if params is None:
            params = {
                "depth": 6,
                "learning_rate": 0.1,
                "iterations": 300,
                "l2_leaf_reg": 3.0,
            }
        params = {
            **params,
            "random_state": RANDOM_STATE,
            "verbose": False,
            "thread_count": -1,
        }
        return CatBoostClassifier(**params)

    else:
        raise ValueError(f"Modelo não suportado em build_model: {model_name}")


def create_optuna_objective(
    model_name,
    preprocessor,
    sampler,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    Cria função objetivo do Optuna para um modelo específico.
    Otimiza MCC no conjunto de validação (com threshold tuning).
    """

    def objective(trial):
        # Define espaço de busca por modelo
        if model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", "balanced_subsample"]
                ),
            }

        elif model_name == "XGBClassifier":
            if not XGB_AVAILABLE:
                raise RuntimeError("xgboost não está disponível.")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight", 1.0, 20.0
                ),
            }

        elif model_name == "LGBMClassifier":
            if not LGBM_AVAILABLE:
                raise RuntimeError("lightgbm não está disponível.")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight", 1.0, 20.0
                ),
            }

        elif model_name == "CatBoost":
            if not CATBOOST_AVAILABLE:
                raise RuntimeError("catboost não está disponível.")
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "iterations": trial.suggest_int("iterations", 200, 800),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }

        else:
            raise ValueError(f"Modelo não suportado em Optuna: {model_name}")

        model = build_model(model_name, params)

        steps = [("preprocess", clone(preprocessor))]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.append(("model", model))

        pipe = ImbPipeline(steps=steps)
        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_val)[:, 1]
        best_thr, best_kappa, _, _ = optimize_threshold(y_val, y_proba, metric="kappa")
        return best_kappa

    return objective


def optimize_with_optuna(
    model_name,
    preprocessor,
    sampler,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials,
):
    """
    Otimiza hiperparâmetros com Optuna e retorna:
        - best_params
        - best_threshold
        - metrics (dict)
        - business (dict)
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna não está disponível no ambiente.")

    print(f"   ⏳ Otimizando {model_name} com Optuna ({n_trials} trials)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    objective = create_optuna_objective(
        model_name, preprocessor, sampler, X_train, y_train, X_val, y_val
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    print(f"   ✅ Melhor Kappa (val): {best_score:.4f}")
    print(f"      Melhores parâmetros: {best_params}")

    # Reconstroi modelo com melhores parâmetros e avalia na validação
    model = build_model(model_name, best_params)

    steps = [("preprocess", clone(preprocessor))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    pipe = ImbPipeline(steps=steps)

    pipe.fit(X_train, y_train)
    y_proba_val = pipe.predict_proba(X_val)[:, 1]
    best_thr, _, y_pred_val, metrics = optimize_threshold(y_val, y_proba_val, metric="kappa")
    business = calculate_business_metrics(y_val, y_pred_val)

    return best_params, best_thr, metrics, business


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("DETECÇÃO DE FRAUDES EM SEGUROS - PIPELINE FINAL (BASELINES + MODELOS AVANÇADOS)")
    print("=" * 80)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # 1. CARREGAMENTO E SPLIT (60/20/20)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. CARREGAMENTO DO DATASET")
    print("=" * 80)

    df = pd.read_csv(DATA_FILE)
    print(f"📊 Dataset original: {df.shape[0]} registros, {df.shape[1]} variáveis")
    print(f"   Taxa de fraude: {df['FraudFound_P'].mean() * 100:.2f}%")

    if "PolicyNumber" in df.columns:
        print("\n⚠️  Removendo PolicyNumber (identificador único - data leakage!)")
        df = df.drop("PolicyNumber", axis=1)

    # Split 60/20/20 estratificado
    df_train_full, df_test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["FraudFound_P"],
        random_state=RANDOM_STATE,
    )

    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.25,  # 0.25 de 80% = 20%
        stratify=df_train_full["FraudFound_P"],
        random_state=RANDOM_STATE,
    )

    print("\n✅ Divisão estratificada (sem leakage para feature engineering):")
    for name, part in [
        ("Treino", df_train),
        ("Validação", df_val),
        ("Teste", df_test),
    ]:
        print(
            f"   {name}: {len(part)} registros "
            f"({len(part) / len(df) * 100:.1f}%) - "
            f"{part['FraudFound_P'].mean() * 100:.2f}% fraudes"
        )

    # ----------------------------------------------------------------------
    # 2. FEATURE ENGINEERING (FIT APENAS NO TREINO)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. FEATURE ENGINEERING AVANÇADO")
    print("=" * 80)

    fe = AdvancedFeatureEngineering()
    df_train_fe = fe.fit_transform(df_train)
    df_val_fe = fe.transform(df_val)
    df_test_fe = fe.transform(df_test)

    print(
        f"✅ Feature Engineering completo. "
        f"Total de variáveis (após FE): {df_train_fe.shape[1]}"
    )

    # Separa X e y
    X_train = df_train_fe.drop("FraudFound_P", axis=1)
    y_train = df_train_fe["FraudFound_P"]

    X_val = df_val_fe.drop("FraudFound_P", axis=1)
    y_val = df_val_fe["FraudFound_P"]

    X_test = df_test_fe.drop("FraudFound_P", axis=1)
    y_test = df_test_fe["FraudFound_P"]

    # ----------------------------------------------------------------------
    # 3. PREPROCESSOR (ONE-HOT PARA CATEGÓRICAS + PASS-THROUGH PARA NUMÉRICAS)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. PRÉ-PROCESSAMENTO (NUMÉRICAS + CATEGÓRICAS)")
    print("=" * 80)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"📊 Colunas numéricas: {len(numeric_cols)}")
    print(f"📊 Colunas categóricas: {len(categorical_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )

    # ----------------------------------------------------------------------
    # 4. AVALIAÇÃO DE MODELOS
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. AVALIAÇÃO DE MODELOS (BASELINES + AVANÇADOS)")
    print("=" * 80)

    results = []
    best_models_config = {}

    print(f"🧪 Samplers em uso ({'FAST' if FAST_MODE else 'FULL'}): {list(SAMPLERS.keys())}")

    for sampler_name, sampler in SAMPLERS.items():
        print(f"\n🔁 Estratégia de reamostragem: {sampler_name}")

        # --------------------------------------------------------------
        # 4.1 Baseline 1: DummyClassifier
        # --------------------------------------------------------------
        if "DummyClassifier" in BASELINE_MODELS:
            dummy = DummyClassifier(
                strategy="stratified",
                random_state=RANDOM_STATE,
            )
            steps_dummy = [("preprocess", clone(preprocessor))]
            if sampler is not None:
                steps_dummy.append(("sampler", sampler))
            steps_dummy.append(("model", dummy))
            pipe_dummy = ImbPipeline(steps=steps_dummy)
            pipe_dummy.fit(X_train, y_train)

            y_proba_val = pipe_dummy.predict_proba(X_val)[:, 1]
            thr_dummy, _, y_pred_val, metrics_dummy = optimize_threshold(y_val, y_proba_val, metric="kappa")
            business_dummy = calculate_business_metrics(y_val, y_pred_val)

            composite_dummy = (
                metrics_dummy["mcc"]
                + metrics_dummy["gmean"]
                + metrics_dummy["kappa"]
            ) / 3.0

            results.append(
                {
                    "Model": "DummyClassifier",
                    "Sampler": sampler_name,
                    "Threshold": thr_dummy,
                    "Composite_Score_Tuned": composite_dummy,
                    "MCC_Tuned": metrics_dummy["mcc"],
                    "G-Mean_Tuned": metrics_dummy["gmean"],
                    "Kappa_Tuned": metrics_dummy["kappa"],
                    "Precision": metrics_dummy["precision"],
                    "Recall": metrics_dummy["recall"],
                    "F1": metrics_dummy["f1"],
                    "Accuracy": metrics_dummy["accuracy"],
                    "Net_Benefit": business_dummy["net_benefit"],
                    "ROI": business_dummy["roi"],
                    "Fraud_Catch_Rate": business_dummy["fraud_catch_rate"],
                }
            )

            best_models_config[("DummyClassifier", sampler_name)] = {
                "model_name": "DummyClassifier",
                "sampler_name": sampler_name,
                "params": None,
                "threshold": thr_dummy,
                "type": "baseline",
            }

            print(
                f"   -> DummyClassifier ({sampler_name}): "
                f"MCC={metrics_dummy['mcc']:.4f}, "
                f"G-Mean={metrics_dummy['gmean']:.4f}, "
                f"Kappa={metrics_dummy['kappa']:.4f}, "
                f"ROI={business_dummy['roi']:.1f}%"
            )

        # --------------------------------------------------------------
        # 4.2 Baseline 2: Logistic Regression
        # --------------------------------------------------------------
        if "LogisticRegression" in BASELINE_MODELS:
            log_reg = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1,
            )

            steps_lr = [("preprocess", clone(preprocessor))]
            if sampler is not None:
                steps_lr.append(("sampler", sampler))
            steps_lr.append(("model", log_reg))

            pipe_lr = ImbPipeline(steps=steps_lr)
            pipe_lr.fit(X_train, y_train)

            y_proba_val = pipe_lr.predict_proba(X_val)[:, 1]
            thr_lr, _, y_pred_val, metrics_lr = optimize_threshold(y_val, y_proba_val, metric="kappa")
            business_lr = calculate_business_metrics(y_val, y_pred_val)

            composite_lr = (
                metrics_lr["mcc"] + metrics_lr["gmean"] + metrics_lr["kappa"]
            ) / 3.0

            results.append(
                {
                    "Model": "LogisticRegression",
                    "Sampler": sampler_name,
                    "Threshold": thr_lr,
                    "Composite_Score_Tuned": composite_lr,
                    "MCC_Tuned": metrics_lr["mcc"],
                    "G-Mean_Tuned": metrics_lr["gmean"],
                    "Kappa_Tuned": metrics_lr["kappa"],
                    "Precision": metrics_lr["precision"],
                    "Recall": metrics_lr["recall"],
                    "F1": metrics_lr["f1"],
                    "Accuracy": metrics_lr["accuracy"],
                    "Net_Benefit": business_lr["net_benefit"],
                    "ROI": business_lr["roi"],
                    "Fraud_Catch_Rate": business_lr["fraud_catch_rate"],
                }
            )

            best_models_config[("LogisticRegression", sampler_name)] = {
                "model_name": "LogisticRegression",
                "sampler_name": sampler_name,
                "params": None,
                "threshold": thr_lr,
                "type": "baseline",
            }

            print(
                f"   -> LogisticRegression ({sampler_name}): "
                f"MCC={metrics_lr['mcc']:.4f}, "
                f"G-Mean={metrics_lr['gmean']:.4f}, "
                f"Kappa={metrics_lr['kappa']:.4f}, "
                f"ROI={business_lr['roi']:.1f}%"
            )

        # --------------------------------------------------------------
        # 4.3 Modelos avançados com Optuna (ou defaults se Optuna indisponível)
        # --------------------------------------------------------------
        for model_name in ADVANCED_MODELS:
            # Pula modelos não disponíveis
            if model_name == "XGBClassifier" and not XGB_AVAILABLE:
                continue
            if model_name == "LGBMClassifier" and not LGBM_AVAILABLE:
                continue
            if model_name == "CatBoost" and not CATBOOST_AVAILABLE:
                continue

            try:
                if OPTUNA_AVAILABLE:
                    best_params, best_thr, metrics, business = optimize_with_optuna(
                        model_name,
                        preprocessor,
                        sampler,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        n_trials=N_TRIALS_OPTUNA,
                    )
                else:
                    print(
                        f"   ⚠ Optuna indisponível – "
                        f"treinando {model_name} ({sampler_name}) com parâmetros default."
                    )
                    model = build_model(model_name, params=None)
                    steps = [("preprocess", clone(preprocessor))]
                    if sampler is not None:
                        steps.append(("sampler", sampler))
                    steps.append(("model", model))
                    pipe = ImbPipeline(steps=steps)
                    pipe.fit(X_train, y_train)

                    y_proba_val = pipe.predict_proba(X_val)[:, 1]
                    best_thr, _, y_pred_val, metrics = optimize_threshold(y_val, y_proba_val, metric="kappa")
                    business = calculate_business_metrics(y_val, y_pred_val)
                    best_params = None  # usaremos defaults depois

                composite = (
                    metrics["mcc"] + metrics["gmean"] + metrics["kappa"]
                ) / 3.0

                results.append(
                    {
                        "Model": model_name,
                        "Sampler": sampler_name,
                        "Threshold": best_thr,
                        "Composite_Score_Tuned": composite,
                        "MCC_Tuned": metrics["mcc"],
                        "G-Mean_Tuned": metrics["gmean"],
                        "Kappa_Tuned": metrics["kappa"],
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1": metrics["f1"],
                        "Accuracy": metrics["accuracy"],
                        "Net_Benefit": business["net_benefit"],
                        "ROI": business["roi"],
                        "Fraud_Catch_Rate": business["fraud_catch_rate"],
                    }
                )

                best_models_config[(model_name, sampler_name)] = {
                    "model_name": model_name,
                    "sampler_name": sampler_name,
                    "params": best_params,
                    "threshold": best_thr,
                    "type": "advanced",
                }

                print(
                    f"   -> {model_name} ({sampler_name}): "
                    f"MCC={metrics['mcc']:.4f}, "
                    f"G-Mean={metrics['gmean']:.4f}, "
                    f"Kappa={metrics['kappa']:.4f}, "
                    f"ROI={business['roi']:.1f}%"
                )

            except Exception as e:
                print(f"   ⚠ Erro ao treinar {model_name} ({sampler_name}): {e}")

    # ----------------------------------------------------------------------
    # 5. RESULTADOS FINAIS (VALIDAÇÃO) E SELEÇÃO DO MELHOR MODELO
    # ----------------------------------------------------------------------
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values(
        "Composite_Score_Tuned", ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("5. RESULTADOS NA VALIDAÇÃO - TOP 10 (Score Composto)")
    print("=" * 80)
    if not metrics_df.empty:
        print(
            f"{'Rank':<5} {'Model':<20} {'Sampler':<12} "
            f"{'Score':<8} {'MCC':<8} {'G-Mean':<8} {'Kappa':<8} {'ROI%':<8}"
        )
        print("-" * 80)

        for i, row in metrics_df.head(10).iterrows():
            print(
                f"{i+1:<5} {row['Model']:<20} {row['Sampler']:<12} "
                f"{row['Composite_Score_Tuned']:.4f} "
                f"{row['MCC_Tuned']:.4f} "
                f"{row['G-Mean_Tuned']:.4f} "
                f"{row['Kappa_Tuned']:.4f} "
                f"{row['ROI']:.1f}"
            )

        best_row = metrics_df.iloc[0]
        best_model_name = best_row["Model"]
        best_sampler_name = best_row["Sampler"]
        best_threshold = best_row["Threshold"]

        print("\n🏆 MELHOR COMBINAÇÃO (Validação):")
        print(
            f"   Modelo: {best_model_name} | Amostrador: {best_sampler_name} "
            f"| Threshold*: {best_threshold:.3f}"
        )
        print(
            f"   MCC={best_row['MCC_Tuned']:.4f}, "
            f"G-Mean={best_row['G-Mean_Tuned']:.4f}, "
            f"Kappa={best_row['Kappa_Tuned']:.4f}, "
            f"ROI={best_row['ROI']:.1f}%"
        )
        print("   *Threshold otimizado para maximizar MCC na validação.")
    else:
        raise RuntimeError("Nenhum resultado foi gerado. Verifique o pipeline.")

    # ----------------------------------------------------------------------
    # 6. RE-TRAIN NO TREINO+VALIDAÇÃO E AVALIAÇÃO NO TESTE
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (HOLD-OUT)")
    print("=" * 80)

    # Reaproveita DF já com FE
    X_train_full_fe = pd.concat([X_train, X_val], axis=0)
    y_train_full_fe = pd.concat([y_train, y_val], axis=0)

    config = best_models_config[(best_model_name, best_sampler_name)]
    sampler_final = SAMPLERS.get(best_sampler_name, None)

    # Reconstrói modelo "do zero" com os melhores parâmetros (evita bug de #features)
    if config["type"] == "baseline":
        if best_model_name == "DummyClassifier":
            final_model = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
        elif best_model_name == "LogisticRegression":
            final_model = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Baseline desconhecido: {best_model_name}")
    else:
        final_model = build_model(best_model_name, params=config["params"])

    steps_final = [("preprocess", clone(preprocessor))]
    if sampler_final is not None:
        steps_final.append(("sampler", sampler_final))
    steps_final.append(("model", final_model))

    final_pipeline = ImbPipeline(steps=steps_final)

    # Ajusta no treino+val
    final_pipeline.fit(X_train_full_fe, y_train_full_fe)

    # Avaliação no teste (usando threshold ótimo aprendido na validação)
    if hasattr(final_pipeline.named_steps["model"], "predict_proba"):
        y_proba_test = final_pipeline.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= best_threshold).astype(int)
    else:
        # fallback (não deve acontecer aqui, mas fica por segurança)
        y_pred_test = final_pipeline.predict(X_test)

    metrics_test = {
        "mcc": matthews_corrcoef(y_test, y_pred_test),
        "gmean": geometric_mean_score(y_test, y_pred_test),
        "kappa": cohen_kappa_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "f1": f1_score(y_test, y_pred_test, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred_test),
    }
    business_test = calculate_business_metrics(y_test, y_pred_test)

    print("\n📊 MÉTRICAS TÉCNICAS (TESTE):")
    print(f"   MCC:      {metrics_test['mcc']:.4f}")
    print(f"   G-Mean:   {metrics_test['gmean']:.4f}")
    print(f"   Kappa:    {metrics_test['kappa']:.4f}")
    print(f"   Precision:{metrics_test['precision']:.4f}")
    print(f"   Recall:   {metrics_test['recall']:.4f}")
    print(f"   F1-Score: {metrics_test['f1']:.4f}")
    print(f"   Accuracy: {metrics_test['accuracy']:.4f}")

    print("\n💰 MÉTRICAS DE NEGÓCIO (TESTE):")
    print(f"   Net Benefit:       R$ {business_test['net_benefit']:,.2f}")
    print(f"   ROI:               {business_test['roi']:.1f}%")
    print(f"   Taxa de Captura:   {business_test['fraud_catch_rate']:.1f}%")
    print(f"   Custo por fraude:  R$ {business_test['cost_per_fraud']:,.2f}")

    # ----------------------------------------------------------------------
    # 7. VALIDAÇÃO CRUZADA 5-FOLD NOS 80% (TREINO+VAL) – MODELO CAMPEÃO
    # ----------------------------------------------------------------------
    if RUN_CHAMPION_CV:
        print("\n" + "=" * 80)
        print("7. VALIDAÇÃO CRUZADA 5-FOLD NOS 80% (TREINO+VAL) – MODELO CAMPEÃO")
        print("=" * 80)

        # Dados brutos dos 80% (treino+val) SEM FE
        df_trainval_raw = df_train_full.copy()
        y_tv = df_trainval_raw["FraudFound_P"]

        skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )

        cv_results = []

        for fold, (idx_tr, idx_va) in enumerate(skf.split(df_trainval_raw, y_tv), start=1):
            print(f"\n----- FOLD {fold}/5 -----")

            df_tr_raw = df_trainval_raw.iloc[idx_tr].copy()
            df_va_raw = df_trainval_raw.iloc[idx_va].copy()

            # FE por fold (fit no treino, transform no val)
            fe_cv = AdvancedFeatureEngineering()
            df_tr_fe = fe_cv.fit_transform(df_tr_raw)
            df_va_fe = fe_cv.transform(df_va_raw)

            X_tr_cv = df_tr_fe.drop("FraudFound_P", axis=1)
            y_tr_cv = df_tr_fe["FraudFound_P"]
            X_va_cv = df_va_fe.drop("FraudFound_P", axis=1)
            y_va_cv = df_va_fe["FraudFound_P"]

            # Preprocessador por fold
            numeric_cols_cv = X_tr_cv.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols_cv = X_tr_cv.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            preprocessor_cv = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_cols_cv),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols_cv),
                ]
            )

            # Modelo campeão + sampler
            sampler_cv = SAMPLERS.get(best_sampler_name, None)

            if config["type"] == "baseline":
                if best_model_name == "DummyClassifier":
                    model_cv = DummyClassifier(
                        strategy="stratified", random_state=RANDOM_STATE
                    )
                elif best_model_name == "LogisticRegression":
                    model_cv = LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        n_jobs=-1,
                    )
                else:
                    raise ValueError(f"Baseline desconhecido: {best_model_name}")
            else:
                model_cv = build_model(best_model_name, params=config["params"])

            steps_cv = [("preprocess", preprocessor_cv)]
            if sampler_cv is not None:
                steps_cv.append(("sampler", sampler_cv))
            steps_cv.append(("model", model_cv))

            pipe_cv = ImbPipeline(steps=steps_cv)
            pipe_cv.fit(X_tr_cv, y_tr_cv)

            # Avaliação no fold (com threshold tuning se tiver predict_proba)
            if hasattr(pipe_cv.named_steps["model"], "predict_proba"):
                y_proba_va_cv = pipe_cv.predict_proba(X_va_cv)[:, 1]
                thr_cv, _, y_pred_va_cv, metrics_cv = optimize_threshold(y_va_cv, y_proba_va_cv, metric="mcc")
            else:
                y_pred_va_cv = pipe_cv.predict(X_va_cv)
                thr_cv = 0.5
                metrics_cv = {
                    "mcc": matthews_corrcoef(y_va_cv, y_pred_va_cv),
                    "gmean": geometric_mean_score(y_va_cv, y_pred_va_cv),
                    "kappa": cohen_kappa_score(y_va_cv, y_pred_va_cv),
                    "precision": precision_score(y_va_cv, y_pred_va_cv, zero_division=0),
                    "recall": recall_score(y_va_cv, y_pred_va_cv, zero_division=0),
                    "f1": f1_score(y_va_cv, y_pred_va_cv, zero_division=0),
                    "accuracy": accuracy_score(y_va_cv, y_pred_va_cv),
                }

            business_cv = calculate_business_metrics(y_va_cv, y_pred_va_cv)

            cv_results.append(
                {
                    "fold": fold,
                    "threshold": thr_cv,
                    "mcc": metrics_cv["mcc"],
                    "gmean": metrics_cv["gmean"],
                    "kappa": metrics_cv["kappa"],
                    "precision": metrics_cv["precision"],
                    "recall": metrics_cv["recall"],
                    "f1": metrics_cv["f1"],
                    "accuracy": metrics_cv["accuracy"],
                    "net_benefit": business_cv["net_benefit"],
                    "roi": business_cv["roi"],
                    "fraud_catch_rate": business_cv["fraud_catch_rate"],
                    "cost_per_fraud": business_cv["cost_per_fraud"],
                }
            )

            print(
                f"Fold {fold}: MCC={metrics_cv['mcc']:.4f} | "
                f"G-Mean={metrics_cv['gmean']:.4f} | "
                f"Kappa={metrics_cv['kappa']:.4f} | "
                f"Recall={metrics_cv['recall']:.4f} | "
                f"ROI={business_cv['roi']:.1f}%"
            )

        cv_df = pd.DataFrame(cv_results)
        print("\n===== MÉDIAS CV (5-FOLD) =====")
        print(cv_df.mean(numeric_only=True))
        print("\n===== DESVIO-PADRÃO CV (5-FOLD) =====")
        print(cv_df.std(numeric_only=True))

        cv_df.to_csv(OUTPUT_DIR / "champion_cv_results.csv", index=False)
        print(f"\n💾 Resultados da CV do campeão salvos em: {OUTPUT_DIR / 'champion_cv_results.csv'}")

    # ----------------------------------------------------------------------
    # 8. SALVANDO RESULTADOS E MODELOS
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("8. SALVANDO RESULTADOS E MODELOS")
    print("=" * 80)

    # 8.1 Salvar CSV de comparação de modelos
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison_FINAL_V3.csv", index=False)
    print(f"💾 Resultados salvos em: {OUTPUT_DIR / 'model_comparison_FINAL_V3.csv'}")

    # 8.2 Pacote COMPLETO (para SHAP / reprodução / produção)
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

    with open(OUTPUT_DIR / "best_model_final_full.pkl", "wb") as f:
        pickle.dump(model_package_full, f)

    print(f"💾 Modelo COMPLETO salvo em: {OUTPUT_DIR / 'best_model_final_full.pkl'}")
    print("   → Contém pipeline + FE + colunas + métricas")
    print("   → Use este arquivo para SHAP e explicabilidade.")

    # 8.3 Pacote LEVE (para Google Colab)
    model_package_light = {
        "pipeline": final_pipeline,
        "threshold": best_threshold,
        "model_name": best_model_name,
        "sampler_name": best_sampler_name,
    }

    with open(OUTPUT_DIR / "best_model_final_light.pkl", "wb") as f:
        pickle.dump(model_package_light, f)

    print(f"💾 Modelo LEVE salvo em: {OUTPUT_DIR / 'best_model_final_light.pkl'}")
    print("   → Sem feature engineering nem objetos customizados")
    print("   → Use este arquivo no Google Colab.")

    # ----------------------------------------------------------------------
    # FINALIZAÇÃO
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXECUÇÃO CONCLUÍDA!")
    print("=" * 80)
    print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
