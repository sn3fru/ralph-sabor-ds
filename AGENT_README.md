# Ralph DS v2.0 - Documentação Técnica

## Visão Geral

O Ralph DS v2.0 é um **Agente Autônomo AGNÓSTICO** que resolve qualquer problema de Data Science:

- **Detecta automaticamente** o tipo de problema (classificação, regressão, etc.)
- **EDA obrigatória** antes de qualquer modelagem
- **STATE.md** como memória resumida (não relê todos os arquivos)
- **Planejamento dinâmico** baseado nos insights descobertos
- Escreve código Python sob demanda
- Executa célula a célula (como Jupyter)
- Analisa outputs (texto + gráficos) com Vision AI
- Decide se aceita, corrige ou reescreve código
- Atualiza documentação automaticamente (STATE.md, TASK_LIST.md, report.md)

## Arquitetura de Componentes

### 1. Brain (brain.py) - Cérebro Cognitivo

Implementa o loop OODA (Observe, Orient, Decide, Act) com detecção de tipo de problema:

```text
while not goals_achieved:
    1. OBSERVE: Ler STATE.md (memória resumida), report.md, metadata.json
    2. ORIENT: Comparar com GOALS.md, detectar tipo de problema
    3. DECIDE: Qual ação tomar? (EDA primeiro!)
    4. ACT: Executar ação
    5. UPDATE: Atualizar STATE.md (memória resumida)
```

**Tipos de Problema Suportados:**
- `binary_classification` - Target com 2 valores (AUC, F1)
- `multiclass_classification` - Target com 3-10 valores (F1 Macro)
- `regression` - Target contínuo (RMSE, MAE, R²)
- `unknown` - Não detectado (apenas EDA)

**Ações Disponíveis:**

- `PLAN` - Planejar próximos passos
- `WRITE_CODE` - Escrever novo script
- `EDIT_CODE` - Editar script existente
- `RUN_STEP` - Executar um script
- `ANALYZE` - Analisar output/plot com Vision AI
- `ROLLBACK` - Reverter para estado anterior
- `UPDATE_CONFIG` - Atualizar config.yaml
- `UPDATE_TASK_LIST` - Replanejar: add/remove/edit etapas em TASK_LIST
- `RUN_FROM_STEP` - Re-executar a partir de um step (alterar passado)
- `STOP` - Parar execução

### 2. Executor (executor.py) - Ambiente Persistente

Executor AGNÓSTICO que roda código Python para qualquer problema:

- Carrega variáveis de pickles anteriores
- Captura stdout, stderr, exceptions
- Salva plots como imagens em `runs/YYYYMMDD_HHMMSS/`
- **Paths injetados no namespace:**
  - `DATA_DIR` - Diretório de dados (context/data/ ou raiz)
  - `STATE_DIR` - Diretório de estado (state/)
  - `CONTEXT_DIR` - Diretório de contexto (context/)
  - `REPORTS_DIR` - Diretório da run atual
- **A cada step:** appenda em `runs/.../report.md` o output e refs às imagens
- Persiste estado em pickle
- Gera metadata.json com schema e tipo de problema

**Fluxo do report markdown:** A cada passo o agente (1) executa o script, (2) o executor escreve/appenda em `report.md` (stdout + imagens), (3) scripts que usam MarkdownLogger appendum conteúdo estruturado ao mesmo `report.md`, (4) analisamos imagens com `vision_critic.py`, (5) o brain usa o conteúdo de `report.md` da run atual como contexto para a LLM — assim o próprio markdown gerado serve para continuar a TASK_LIST. Ao editar ou refazer scripts, manter MarkdownLogger e REPORTS_DIR para que o report tenha sempre a versão atualizada do que cada step produz.

### 3. MarkdownLogger (markdown_logger.py) - Logging Estruturado

Logger que gera relatórios markdown para análise por LLMs. **Substitui `print()` para saída analítica.****Papel:** só **escrever** no report (métricas, seções, refs a imagens). **Não** analisa imagens com Vision — isso é feito pelo brain via Vision Critic (evitar redundância).

- **Uso nos scripts (notebooks/):** Use `REPORTS_DIR` (injetado pelo executor) e, para escrever no report único da run, `append_to_existing=True` e `report_filename="report.md"`.
- **API principal:** `log()`, `log_metric()`, `log_insight()`, `log_plot()`, `log_table()`, `log_parameters()`, `section()`. Não usar `print()` para métricas ou conclusões.
- **Exemplo:** `logger = MarkdownLogger(output_dir=REPORTS_DIR, run_name="11_evaluate_baseline", append_to_existing=True, report_filename="report.md")`; depois `logger.section("Métricas"); logger.log_metric("AUC", 0.85); logger.log_plot(fig, "ROC", context_description="...")`.
- **Não usar `use_vision_llm=True`** nos scripts: a análise das imagens é feita pelo **brain** após cada step via Vision Critic; usar vision no script duplicaria análise.

### 4. Vision Critic (vision_critic.py) - Analista Visual

**Papel:** analisar gráficos com Vision AI (Intent Injection). Usado **pelo brain** após cada step que gera imagens: o brain chama `vision_critic.analyze_plot()` e appenda a seção "Análises das imagens (contexto + Vision)" no `report.md`. Os scripts **não** chamam o Vision Critic diretamente.

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

(Logging estruturado nos scripts: **MarkdownLogger** — ver seção 3 acima.)

## Contexto do Agente (pasta context/)

O agente le **tudo** que estiver na pasta `context/` (arquivos `.md`, `.py`, `.txt`, `.yaml`, `.json`) e usa como contexto para planejamento, fluxo e geracao de codigo. Assim o agente nao parte do zero.

- **O que colocar:** documentacao, exemplos de codigo, convencoes, pipeline de referencia (ex.: `credit_scoring_pipeline.py`).
- **Ordem:** arquivos lidos em ordem alfabetica; pode prefixar com numeros (ex.: `01_objetivos.md`, `02_pipeline_exemplo.py`).
- **Limite:** o brain concatena ate ~60k caracteres para caber no contexto da LLM; arquivos muito grandes sao truncados.

O pipeline legado foi movido para `context/credit_scoring_pipeline.py`. Para executar: `python context/credit_scoring_pipeline.py` (cwd = raiz do projeto).

## Configuracao

### Variaveis de Ambiente (.env)

```env
GEMINI_KEY="sua-api-key-aqui"
MODEL_NAME="gemini-2.5-flash"
```

O sistema usa automaticamente `MODEL_NAME` para chamadas de LLM e Vision.

### Config.yaml

O `config.yaml` é a **Single Source of Truth** para parâmetros do pipeline. O agente deve **ler** e **respeitar** as seções `feature_selection` e `imputation_strategy` ao gerar ou editar código (steps 02, 05, 08, 09). Ver também **Diretriz de Modelagem (XGBoost Nativo)** abaixo e `context/02_diretriz_modelagem.md`.

```yaml
pipeline:
  mode: DEV  # DEV (2% da base) ou PROD (100%)
  run_shap: false

feature_selection:
  remove_high_correlation: false   # Em XGBoost: não remover por correlação (árvores lidam com colinearidade)
  correlation_threshold: 0.999     # Se remover: só duplicatas exatas (>0.999)
  remove_high_nulls: true
  null_threshold: 0.99             # Só remover colunas com >= 99% nulos
  imputation_strategy: "none"     # "none" = NÃO imputar (XGBoost usa NaN como informação)
  force_drop_features: []

xgboost_params:
  max_depth: 4
  learning_rate: 0.05
  min_child_weight: 10
  gamma: 0.5

business_params:
  ticket_medio: 10000
  ganho_tp: 1500
  perda_fp: -10000
```

### Diretriz de Modelagem (XGBoost Nativo) — "Let the Model Learn"

O agente deve tomar decisões alinhadas a esta diretriz ao comparar cenários (ex.: overfitting crítico + AUC baixo após imputação agressiva):

1. **NÃO IMPUTE DADOS** (quando `imputation_strategy: "none"` no config): O XGBoost lida nativamente com NaNs (Sparsity-aware Split Finding). Imputação por média/mediana destrói o sinal de ausência ("Missingness is information").
2. **NÃO REMOVA COLINEARIDADE** (exceto duplicatas >0.999): Árvores usam feature subsampling por nó; remover uma de um par correlacionado pode jogar fora nuance. Só remover "gêmeos" idênticos.
3. **FEATURE SELECTION**: Deve ser feita pelo modelo (Feature Importance ou SHAP) e não por heurísticas estatísticas prévias (correlação, % nulos baixo). Só remover colunas com **≥ null_threshold** nulos (ex.: 0.99 = coluna quase vazia).

Contexto completo em `context/02_diretriz_modelagem.md`. Ao detectar overfitting crítico + AUC baixo após steps de imputação/correlação agressivos, o agente deve **orientar** para desativar imputação e remoção por correlação (atualizar config e scripts conforme esta diretriz).

### Modo DEV vs PROD

| Modo     | Dados        | Uso                            |
| -------- | ------------ | ------------------------------ |
| `DEV`  | 2% da base   | Desenvolvimento rapido, testes |
| `PROD` | 100% da base | Execucao final, producao       |

Para mudar o modo, edite `config.yaml`:

```yaml
pipeline:
  mode: PROD  # Mude de DEV para PROD
```

## Fluxo de Execução (Genérico)

### Fase 0: Inicialização

```text
Ler GOALS.md → Ler context/ → Carregar STATE.md → Criar runs/
```

### Fase 1: EDA Obrigatória (sempre executa)

```text
01_load_data     -> Carregar dados, gerar metadata
02_eda_overview  -> Visão geral (shape, tipos, memória)
03_eda_nulls     -> Analisar nulos, decidir remoções
04_eda_target    -> Analisar target (DETECTA TIPO DE PROBLEMA!)
05_eda_distrib   -> Plotar distribuições
06_eda_corr      -> Analisar correlações
07_eda_drift     -> Comparar treino vs teste (se aplicável)
```

### Fase 2: Feature Engineering (dinâmico)

```text
08_feature_cleanup   -> Remover features críticas
09_feature_transform -> Aplicar transformações
10_feature_select    -> Seleção final
```

### Fase 3: Modelagem (adapta ao tipo)

```text
11_train_baseline    -> Treinar modelo inicial
12_evaluate_model    -> Avaliar métricas (AUC/RMSE/F1 conforme tipo)
13_tune_hyperparams  -> Otimização de hiperparâmetros
14_cross_validate    -> Validação cruzada
```

### Fase 4: Business Layer (opcional)

```text
15_find_threshold    -> Threshold ótimo (classificação)
16_business_metrics  -> Métricas de negócio (se aplicável)
17_stability_analysis-> Calcular PSI/drift
18_final_report      -> Relatório final
19_export            -> Exportar modelo e artefatos
```

## Persistência de Estado

### metadata.json

```json
{
  "current_step": "04_eda_target",
  "problem_type": "binary_classification",
  "target_column": "target",
  "target_info": {
    "n_unique": 2,
    "dtype": "int64",
    "class_balance": 0.087
  },
  "data": {
    "df_train": {
      "rows": 50000,
      "columns": 95,
      "column_names": ["feature_1", "feature_2", "..."]
    }
  },
  "decisions": {
    "features_to_drop": ["feature_39", "feature_11"],
    "features_to_transform": {"renda": "log"},
    "safe_features": ["feature_1", "feature_2"],
    "imputation_strategy": "none"
  },
  "metrics": {
    "auc_train": 0.88,
    "auc_val": 0.82,
    "gap": 0.06
  },
  "warnings": ["feature_39 tem 93% nulos"]
}
```

### STATE.md (memória resumida)

O agente lê STATE.md como contexto principal em vez de reler todos os arquivos:

```markdown
# Estado Atual do Projeto

**Step Atual:** 11_evaluate_model
**Tipo de Problema:** Classificação Binária

## Dados
- Treino: 50.000 linhas × 95 features
- Desbalanceamento: 8.7% positivos

## Decisões Tomadas
- Features removidas: 15
- Safe features: 80

## Métricas Atuais
| Métrica | Valor | Status |
|---------|-------|--------|
| AUC Val | 0.82 | ✅ |
| Gap | 6% | ✅ |

## Próximos Passos
1. tune_hyperparams
2. find_threshold
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
