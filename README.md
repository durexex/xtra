# Xtra Data Dashboard (frontend)

Interface React para explorar datasets e acionar as rotinas de preparo/modelagem do backend Flask.

## Rodando

1) Backend: `cd ../backend && python api.py` (http://localhost:5000).  
2) Frontend: `npm install` (primeira vez) e `npm start` (http://localhost:3000).

## Funcionalidades principais

- Upload de CSV e navegações: head, describe, info, null-values, correlação, scatter/boxplot/histograma.  
- Modelos: KNN, regressão linear múltipla e rede neural (MLP) com pré-processamento automatizado.  
- Categorizar colunas, baixar versões reduzidas/corrigidas do dataset.

## Novo: botão log1p

- No menu lateral clique em `log1p`, escolha a coluna e confirme.  
- Backend executa: zeros conhecidos -> NaN -> mediana -> `np.log1p` -> histograma da coluna transformada (dataset em memória é atualizado).

## Observações do pré-processamento (MLP)

- Colunas biométricas (Glucose, BloodPressure, SkinThickness, Insulin, BMI) têm zeros tratados como faltantes e imputação por mediana.  
- `Insulin` recebe log1p e depois escalonamento; DPF usa RobustScaler; demais numéricas usam StandardScaler.  
- Split estratificado, grid com early stopping (`early_stopping=True`, `validation_fraction=0.15`).

## Scripts úteis

- `npm start` – ambiente de desenvolvimento.  
- `npm test` – runner interativo.  
- `npm run build` – bundle de produção.
