# Autonomous Data Science Engineer - Documentacao Tecnica

## Visao Geral

Este sistema implementa um **Engenheiro de ML Autonomo** que:
- Escreve codigo Python sob demanda
- Executa celula a celula (como Jupyter)
- Analisa outputs (texto + graficos) com Vision AI
- Decide se aceita, corrige ou reescreve codigo
- Atualiza documentacao automaticamente

## Arquitetura de Componentes

### 1. Brain (brain.py) - Cerebro Cognitivo

Implementa o loop OODA (Observe, Orient, Decide, Act):

```text
while not goals_achieved:
    1. OBSERVE: Ler metadata.json, ultimo output, graficos
    2. ORIENT: Comparar com GOALS.md
    3. DECIDE: Qual acao tomar?
    4. ACT: Executar acao
    5. REFLECT: Atualizar CHANGELOG e README
```

**Acoes Disponiveis:**
- `PLAN` - Planejar proximos passos
- `WRITE_CODE` - Escrever novo script
- `EDIT_CODE` - Editar script existente
- `RUN_STEP` - Executar um script
- `ANALYZE` - Analisar output/plot com Vision AI
- `ROLLBACK` - Reverter para estado anterior
- `UPDATE_CONFIG` - Atualizar config.yaml
- `STOP` - Parar execucao

### 2. Executor (executor.py) - Ambiente Persistente

Executa codigo Python mantendo estado entre chamadas:
- Carrega variaveis de pickles anteriores
- Captura stdout, stderr, exceptions
- Salva plots como imagens
- Persiste estado em pickle
- Gera metadata.json com schema dos dados

### 3. Vision Critic (vision_critic.py) - Analista Visual

Analisa graficos com **Intent Injection**:

```python
# Problema do prompt generico:
"Analise este grafico" -> "Vejo um grafico azul"

# Solucao - Intent Injection:
"""
[INTENT] Verificar distribuicao de nulos
[ESPERADO] Maioria das features com <5% nulos
[ALERTA] Features com >50% devem ser removidas
"""
```

**Templates de Intent Disponiveis:**
- `histogram` - Distribuicoes e outliers
- `roc_curve` - Performance e overfitting
- `null_barplot` - Dados faltantes
- `correlation_matrix` - Multicolinearidade
- `profit_curve` - Otimizacao financeira
- `learning_curve` - Bias vs variance
- `feature_importance` - Relevancia e leakage
- `shap_beeswarm` - Explainability
- `psi_chart` - Drift de populacao
- `confusion_matrix` - Erros do modelo

## Configuracao

### Variaveis de Ambiente (.env)

```env
GEMINI_KEY="sua-api-key-aqui"
MODEL_NAME="gemini-2.5-flash"
```

O sistema usa automaticamente `MODEL_NAME` para chamadas de LLM e Vision.

### Config.yaml

```yaml
pipeline:
  mode: DEV  # DEV (2% da base) ou PROD (100%)
  run_shap: false

xgboost_params:
  max_depth: 4
  learning_rate: 0.05
  min_child_weight: 10
  gamma: 0.5

business_params:
  ticket_medio: 10000
  ganho_tp: 1500
  perda_fp: -10000

feature_selection:
  force_drop_features: []
  correlation_threshold: 0.95
```

### Modo DEV vs PROD

| Modo | Dados | Uso |
|------|-------|-----|
| `DEV` | 2% da base | Desenvolvimento rapido, testes |
| `PROD` | 100% da base | Execucao final, producao |

Para mudar o modo, edite `config.yaml`:
```yaml
pipeline:
  mode: PROD  # Mude de DEV para PROD
```

## Fluxo de Execucao

### Fase 1: EDA (Exploratory Data Analysis)

```text
01_load_data    -> Carregar dados, gerar metadata
02_eda_nulls    -> Analisar nulos, decidir remocoes
03_eda_target   -> Analisar desbalanceamento
04_eda_distrib  -> Plotar distribuicoes
05_eda_corr     -> Analisar correlacoes
```

### Fase 2: Feature Engineering

```text
06_feature_cleanup   -> Remover features criticas
07_feature_transform -> Aplicar log, binning
08_feature_impute    -> Imputar nulos
09_feature_select    -> Selecao final
```

### Fase 3: Modelagem

```text
10_train_baseline    -> Treinar modelo inicial
11_evaluate_baseline -> Avaliar gap, AUC
12_tune_regular      -> Ajustar regularizacao
13_cross_validate    -> Validacao cruzada
```

### Fase 4: Avaliacao de Negocio

```text
14_find_threshold    -> Threshold otimo de lucro
15_evaluate_profit   -> Eficiencia financeira
16_stability_psi     -> Calcular PSI
17_final_report      -> Relatorio final
```

## Persistencia de Estado

### metadata.json

```json
{
  "current_step": "03_eda_distributions",
  "data": {
    "rows": 50000,
    "columns": 95,
    "target_rate": 0.087
  },
  "decisions": {
    "features_to_drop": ["feature_39", "feature_11"],
    "features_to_transform": {"renda": "log"},
    "imputation_strategy": "median"
  },
  "metrics": {
    "auc_train": 0.88,
    "auc_val": 0.82,
    "gap": 0.06
  },
  "warnings": ["feature_39 tem 93% nulos"]
}
```

### Pickles (state/)

Cada passo salva seu estado em pickle:
- `step_01_raw.pkl` - Dados brutos
- `step_05_features.pkl` - Features processadas
- `step_10_model.pkl` - Modelo treinado

## Comandos

```bash
# Modo autonomo (loop completo)
python brain.py --mode auto

# Modo passo a passo
python brain.py --mode step

# Apenas planejar proxima acao
python brain.py --mode plan

# Executar script especifico
python executor.py --step 02_eda_nulls

# Analisar grafico manualmente
python vision_critic.py --image reports/nulls.png --intent "Verificar nulos"

# Rollback para estado anterior
python executor.py --rollback step_01_raw
```

## Seguranca

1. **Backups Automaticos:** Antes de editar `src/`, salva em `backups/`
2. **Estado Imutavel:** Cada passo gera novo pickle
3. **Validacao Pos-Edicao:** Se codigo falhar, reverte automaticamente
4. **Limite de Tentativas:** Max 3 tentativas de corrigir mesmo erro

## Criterios de Parada

O agente para quando:
1. Todas as metas de GOALS.md atingidas
2. Estagnou por 5 iteracoes
3. Erro critico nao recuperavel
4. Usuario interrompe (Ctrl+C)

## Troubleshooting

### Pipeline Falha
- Verificar `CHANGELOG.md` para detalhes do erro
- Confirmar que `train.parquet` e `test.parquet` existem
- Verificar dependencias: `pip install -r requirements.txt`

### LLM Nao Responde
- Verificar `GEMINI_KEY` no `.env`
- Confirmar conexao com internet
- O agente faz fallback para decisao conservadora

### Vision Nao Analisa
- Verificar se imagem existe no caminho
- Confirmar que `MODEL_NAME` suporta vision
- Modelo deve ter capacidade multimodal

---

## Analises Disponiveis

### 1. Analise de Nulos
- Barplot de % nulos por feature
- Classificacao: Critico (>50%), Moderado (5-50%), OK (<5%)
- Decisao automatica de remocao/imputacao

### 2. Analise de Distribuicoes
- Histogramas com KDE
- Calculo de skewness e kurtosis
- Recomendacao de transformacoes (log, sqrt)

### 3. Analise de Correlacoes
- Matriz de correlacao (heatmap)
- Identificacao de multicolinearidade
- Lista de features redundantes

### 4. Curva ROC e Gap
- Comparacao treino vs validacao
- Deteccao automatica de overfitting
- Status: OK (<8%), OVERFITTING (>8%)

### 5. Analise Financeira
- Curva de lucro vs threshold
- Eficiencia financeira (% do potencial maximo)
- Matriz de confusao com custos

### 6. Elasticidade AUC vs Lucro
- Simulacao de degradacao do modelo
- Coeficiente de elasticidade
- Valor marginal por 1% de AUC

### 7. PSI (Population Stability Index)
- Comparacao de distribuicoes treino vs producao
- Status: OK (<0.1), Atencao (0.1-0.2), Critico (>0.2)

### 8. Analise de Erros
- Exemplos de Falsos Positivos
- Exemplos de Falsos Negativos
- Identificacao de patterns de erro
