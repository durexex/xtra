# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Importando a base de dados
dados = pd.read_csv(r'C:\Projetos\xtra\dataset\diabetes.csv')

# gerando arquivo com as correlações
with open('corr.txt', 'w', encoding='utf-8') as f:
    f.write(dados.corr().to_string())


# 2) Separar target das outras características
X = dados.drop(columns=["Outcome"])
y = dados["Outcome"]

# 3) Definir colunas numéricas (exceto Age, que será categorizada)
num_cols = [c for c in X.columns if c != "Age"]

# definição dos zeros que serão tratados
ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}
LOG1P_COLS = {"Insulin", "DiabetesPedigreeFunction", "SkinThickness"}  
#Criando as faixas das idades
AGE_BINS = [14, 21, 28, 35, 41, 48, np.inf]
AGE_LABELS = ["14-20", "21-27", "28-34", "35-40", "41-47", "48+"]

def zero_to_nan(X_array):
    """
    Recebe numpy array (após seleção de colunas) e devolve numpy array.
    Assume que as colunas estão na ordem 'num_cols'.
    """
    X_array = np.asarray(X_array, dtype=float).copy()
    col_idx = {c: i for i, c in enumerate(num_cols)}
    for c in ZERO_AS_MISSING:
        if c in col_idx:
            i = col_idx[c]
            X_array[X_array[:, i] == 0, i] = np.nan
    return X_array

def log1p_selected(X_array):
    X_array = np.asarray(X_array, dtype=float).copy()
    col_idx = {c: i for i, c in enumerate(num_cols)}
    for c in LOG1P_COLS:
        if c in col_idx:
            i = col_idx[c]            
            X_array[:, i] = np.log1p(X_array[:, i])
    return X_array

def age_to_category(X_array):
    """
    Recebe apenas a coluna Age e devolve uma coluna categórica codificando os intervalos.
    """
    arr = np.asarray(X_array, dtype=float).ravel()
    cats = pd.cut(arr, bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return np.asarray(cats, dtype=object).reshape(-1, 1)


# 5) Pré-processamento numérico
numeric_preprocess = Pipeline(steps=[
    ("zero_to_nan", FunctionTransformer(zero_to_nan, feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(log1p_selected, feature_names_out="one-to-one")),    
])

#Aqui pega as faixas e transforma num "índice"
age_preprocess = Pipeline(steps=[
    ("bin_age", FunctionTransformer(age_to_category, feature_names_out="one-to-one")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, num_cols),
        ("age", age_preprocess, ["Age"]),
    ],
    remainder="drop"
)

# 6) Pipeline final: preprocess + árvore de decisão
base_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    "classifier__max_depth": [None, 5, 10, 15],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__max_features": [None, "sqrt", "log2"],
    "classifier__ccp_alpha": [0.0, 0.001, 0.01],
    "classifier__class_weight": [None, "balanced"],
}

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
)

# 7) Split (ex.: 80/20 treino/teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y  # mantém proporção 0/1
)

# 8) Treinar e avaliar
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = (y_pred_proba >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Best params (roc_auc cv): {grid.best_params_}")
print(f"Best CV roc_auc:          {grid.best_score_:.4f}")
print(f"Threshold:                {threshold}")
print(f"Accuracy:                 {acc:.4f}")
print(f"Precision:                {prec:.4f}")
print(f"Recall:                   {rec:.4f}")
print(f"F1-score:                 {f1:.4f}")
print(f"ROC AUC:                  {roc_auc:.4f}")
