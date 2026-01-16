# Xtra Data Dashboard (Frontend)

Interface web para exploracao de datasets CSV e acionamento de rotinas de preparo e modelagem em uma API Flask.

## Visao geral
Este projeto fornece uma UI React para carregar arquivos CSV, inspecionar estatisticas, gerar visualizacoes e executar fluxos de preparo/modelagem com apoio de um backend Flask.

## Funcionalidades
- Upload de CSV e visualizacao de amostras (head), estatisticas (describe) e info do DataFrame
- Analise de qualidade de dados com valores nulos customizados e correcoes
- Visualizacoes: scatter plot, boxplot, histograma e matriz de correlacao
- Transformacoes como log1p em colunas selecionadas
- Categorizacao de colunas em bins com comparativo antes/depois
- Modelagem com KNN e regressao linear multipla com metricas
- Download de datasets tratados ou reduzidos

## Stack
- React 19 + Create React App (react-scripts)
- Fetch API para integracao com a API Flask

## Requisitos
- Node.js e npm
- Backend Flask rodando em http://localhost:5000

## Como rodar (desenvolvimento)
1) Backend: `cd ../backend` e `python api.py`.
2) Frontend:

```bash
cd frontend
npm install
npm start
```

A aplicacao abre em http://localhost:3000.

## Scripts
- `npm start` - modo desenvolvimento
- `npm test` - runner interativo
- `npm run build` - bundle de producao

## Configuracao da API
A URL base da API esta definida diretamente no frontend em `frontend/src/App.js`. Para trocar host/porta, ajuste os endpoints no codigo.

## Estrutura do repositorio
- `frontend/` - app React
- `frontend/src/components/` - componentes reutilizaveis
- `frontend/src/App.js` - fluxo principal da UI

## Licenca
Nao definida.
