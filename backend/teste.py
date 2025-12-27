import numpy as np
import pandas as pd


# Importando a base de dados
dados = pd.read_csv(r'C:\Projetos\xtra\dataset\diabetes.csv')

# gerando arquivo com as correlações
with open('corr.txt', 'w', encoding='utf-8') as f:
    f.write(dados.corr().to_string())


# 2) Separar target das outras características
X = dados.drop(columns=["Outcome"])
y = dados["Outcome"]

# 3) Definir colunas numéricas
num_cols = X.columns.tolist()

# definição dos zeros que serão tratados
ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}
LOG1P_COLS = {"Insulin", "DiabetesPedigreeFunction", "SkinThickness"}  

# def zero_to_nan(X_array):
    # """
    # Recebe numpy array (após seleção de colunas) e devolve numpy array.
    # Assume que as colunas estão na ordem 'num_cols'.
    # """
X_array = ZERO_AS_MISSING.astype(float).copy()
print(X_array)
col_idx = {c: i for i, c in enumerate(num_cols)}
print(col_idx)
for c in ZERO_AS_MISSING:
    if c in col_idx:
        i = col_idx[c]
        X_array[X_array[:, i] == 0, i] = np.nan

print(X_array)


