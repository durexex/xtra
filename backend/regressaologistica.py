# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Importando a base de dados
dados = pd.read_csv(r'C:\Projetos\xtra\dataset\diabetes.csv')

# gerando arquivo com as correlações
with open('corr.txt', 'w', encoding='utf-8') as f:
    f.write(dados.corr().to_string())


# 2) Separar target das outras características
X = dados.drop(columns=["Outcome"])
y = dados["Outcome"]

# 3) Definir colunas numéricas
num_cols = list(X.columns)
robust_cols = ["DiabetesPedigreeFunction"]
standard_cols = ["Pregnancies", "SkinThickness", "BMI", "BloodPressure", "Glucose", "Insulin", "Age"]

# definição dos zeros que serão tratados
ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}
LOG1P_COLS = {"Insulin"}  

def zero_to_nan(X_array, columns):
    """
    Recebe numpy array (após seleção de colunas) e devolve numpy array.
    Assume que as colunas estão na ordem passada em 'columns'.
    """
    X_array = np.asarray(X_array, dtype=float).copy()
    col_idx = {c: i for i, c in enumerate(columns)}
    for c in ZERO_AS_MISSING:
        if c in col_idx:
            i = col_idx[c]
            X_array[X_array[:, i] == 0, i] = np.nan
    return X_array

def log1p_selected(X_array, columns):
    X_array = np.asarray(X_array, dtype=float).copy()
    col_idx = {c: i for i, c in enumerate(columns)}
    for c in LOG1P_COLS:
        if c in col_idx:
            i = col_idx[c]
            # log1p requer valores >= -1 (aqui tudo é >=0 após imputação)
            X_array[:, i] = np.log1p(X_array[:, i])
    return X_array

# 5) Pré-processamento numérico
robust_preprocess = Pipeline(steps=[
    ("zero_to_nan", FunctionTransformer(zero_to_nan, kw_args={"columns": robust_cols}, feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(log1p_selected, kw_args={"columns": robust_cols}, feature_names_out="one-to-one")),
    ("scaler", RobustScaler()),
])

standard_preprocess = Pipeline(steps=[
    ("zero_to_nan", FunctionTransformer(zero_to_nan, kw_args={"columns": standard_cols}, feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num_robust", robust_preprocess, robust_cols),
        ("num_standard", standard_preprocess, standard_cols),
    ],
    remainder="drop"
)

# 6) Pipeline final: preprocess + regressão logística
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000, n_jobs=-1, solver="lbfgs"))
])

# 7) Split (ex.: 80/20 treino/teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y  # mantém proporção 0/1
)

# 8) Treinar e avaliar
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.15
y_pred = (y_pred_proba >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# print(f"Threshold: {threshold}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
