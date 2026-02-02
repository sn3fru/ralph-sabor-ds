# Objetivos do Projeto: [NOME DO PROJETO]

> **Instruções**: Este arquivo define os critérios de sucesso para o agente.
> Edite as seções abaixo conforme seu problema específico.

## 1. Tipo de Problema

- [ ] Classificação Binária
- [ ] Classificação Multiclasse
- [ ] Regressão
- [ ] Série Temporal
- [ ] Outro: _______________

## 2. Metas de Performance Técnica

### Classificação Binária

| Métrica              | Aceitável | Excelente | Crítico |
| --------------------- | ---------- | --------- | -------- |
| AUC ROC (Validação) | > 0.81     | > 0.85    | < 0.75   |
| Gap Treino-Teste      | < 8%       | < 5%      | > 12%    |
| PSI (Estabilidade)    | < 0.2      | < 0.1     | > 0.25   |
| KS (Discriminação)  | > 0.30     | > 0.40    | < 0.20   |

### Classificação Multiclasse

| Métrica               | Aceitável | Excelente | Crítico |
| ---------------------- | ---------- | --------- | -------- |
| F1 Macro (Validação) | > 0.70     | > 0.80    | < 0.60   |
| Accuracy               | > 0.80     | > 0.85    | < 0.70   |
| Gap Treino-Teste       | < 8%       | < 5%      | > 12%    |

### Regressão

| Métrica          | Aceitável          | Excelente | Crítico |
| ----------------- | ------------------- | --------- | -------- |
| R² (Validação) | > 0.70              | > 0.85    | < 0.50   |
| RMSE              | Depende do domínio | -         | -        |
| MAE               | Depende do domínio | -         | -        |

## 3. Metas de Performance de Negócio (se aplicável)

| Métrica               | Meta   | Ideal |
| ---------------------- | ------ | ----- |
| Eficiência Financeira | > 75%  | > 85% |
| Taxa de Aprovação    | 70-80% | 75%   |
| Lucro/Economia         | > X    | > Y   |

## 4. Diretrizes de Modelagem

### Diretriz Padrão (XGBoost) — "Let the Model Learn"

- **Não impute** quando `imputation_strategy: "none"`. XGBoost usa NaN como informação.
- **Não remova colinearidade** (exceto duplicatas >0.999). Árvores lidam com correlação.
- **Remoção por nulos**: só remover colunas com >= 99% nulos.

### Se Overfitting (Gap > 8%):

1. Aumentar `min_child_weight` (ex: 10 -> 20)
2. Aumentar `gamma` (ex: 0.5 -> 1.0)
3. Reduzir `max_depth` (ex: 5 -> 4)
4. Reduzir `learning_rate` (ex: 0.05 -> 0.03)

### Se Underfitting (métricas baixas):

1. Aumentar `max_depth` (com cuidado)
2. Aumentar `n_estimators`
3. Revisar features (adicionar interações)

### Se Drift Alto (PSI > 0.2):

1. Identificar features com KS > 0.5
2. Adicionar em `force_drop_features` no config
3. Considerar usar apenas "safe features"

## 5. Restrições de Segurança

- NUNCA deletar dados originais em `context/data/`
- NUNCA commitar API keys
- SEMPRE fazer backup antes de editar código em `src/`
- MÁXIMO 3 tentativas de corrigir mesmo erro

## 6. Critérios de Parada

O agente PARA quando:

1. Todas as metas de performance atingidas, OU
2. Estagnou por 5 iterações sem melhoria, OU
3. Erro crítico não recuperável, OU
4. Usuário interrompe manualmente

---

## Exemplo: Credit Scoring

```yaml
# Metas específicas para Credit Scoring
tipo: Classificação Binária
target: target (0=adimplente, 1=inadimplente)

metricas_tecnicas:
  auc_min: 0.81
  gap_max: 0.08
  psi_max: 0.20

metricas_negocio:
  eficiencia_min: 75%
  taxa_aprovacao: 70-80%
  ticket_medio: R$ 10.000
  spread: 15%
```
