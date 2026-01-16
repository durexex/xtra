import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

dados = pd.read_csv(r'C:\Projetos\xtra\dataset\diabetes.csv')

with open('corr.txt', 'w', encoding='utf-8') as f:
    f.write(dados.corr().to_string())

X = dados.drop(columns=["Outcome"])
y = dados["Outcome"]

num_cols = [c for c in X.columns if c != "Age"]

ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}

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

numeric_preprocess = Pipeline(steps=[
    ("zero_to_nan", FunctionTransformer(zero_to_nan, feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="median")),
])


preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, num_cols),        
    ],
    remainder="drop"
)

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
