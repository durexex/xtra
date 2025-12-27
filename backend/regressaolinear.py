# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
            # log1p requer valores >= -1 (aqui tudo é >=0 após imputação)
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
    ("scaler", StandardScaler()),
])

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

# 6) Pipeline final: preprocess + regressão linear
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
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

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# MAPE: cuidado com y=0 (divide por zero). Aqui uso uma versão "safe".
y_test_np = y_test.to_numpy()
eps = 1e-9
mape = np.mean(np.abs((y_test_np - y_pred) / np.maximum(np.abs(y_test_np), eps))) * 100

print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# 9) (Opcional) transformar predição em classe para inspecionar acurácia
# (isso já vira um "classificador improvisado"; o correto é LogisticRegression)
y_class = (y_pred >= 0.5).astype(int)
acc = (y_class == y_test_np).mean()
print(f"Accuracy com threshold 0.5 (apenas referência): {acc:.4f}")
