# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

dados = pd.read_csv(r'C:\Projetos\xtra\dataset\diabetes.csv')

with open('corr.txt', 'w', encoding='utf-8') as f:
    f.write(dados.corr().to_string())

X = dados.drop(columns=["Outcome"])
y = dados["Outcome"]

robust_cols = ["DiabetesPedigreeFunction"]
standard_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]

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
            X_array[:, i] = np.log1p(X_array[:, i])
    return X_array

robust_preprocess = Pipeline(steps=[
    ("zero_to_nan", FunctionTransformer(zero_to_nan, kw_args={"columns": robust_cols}, feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])

standard_preprocess = Pipeline(steps=[
    ("zero_to_nan", FunctionTransformer(zero_to_nan, kw_args={"columns": standard_cols}, feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(log1p_selected, kw_args={"columns": standard_cols}, feature_names_out="one-to-one")),
    ("scaler", StandardScaler()),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num_robust", robust_preprocess, robust_cols),
        ("num_standard", standard_preprocess, standard_cols),
    ],
    remainder="drop"
)

base_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", MLPClassifier(random_state=42, max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20))
])

param_grid = {
    "classifier__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "classifier__activation": ["relu", "tanh"],
    "classifier__solver": ["adam", "lbfgs"],
    "classifier__alpha": [1e-4, 1e-3, 1e-2],
    "classifier__learning_rate_init": [0.001, 0.01],
}

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y  
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Best params (roc_auc cv): {grid.best_params_}")
print(f"Accuracy:                 {acc:.4f}")
print(f"Precision:                {prec:.4f}")
print(f"Recall:                   {rec:.4f}")
print(f"F1-score:                 {f1:.4f}")
print(f"ROC AUC:                  {roc_auc:.4f}")
