# Objetivos Estrategicos: Credit Scoring Agent

## 1. Metas de Performance Tecnica

| Metrica | Aceitavel | Excelente | Critico |
|---------|-----------|-----------|---------|
| AUC ROC (Validacao) | > 0.81 | > 0.83 | < 0.75 |
| Gap Treino-Teste | < 8% | < 5% | > 12% |
| PSI (Estabilidade) | < 0.2 | < 0.1 | > 0.25 |
| KS (Discriminacao) | > 0.30 | > 0.40 | < 0.20 |

## 2. Metas de Performance de Negocio

| Metrica | Meta | Ideal |
|---------|------|-------|
| Eficiencia Financeira | > 75% | > 85% |
| Taxa de Aprovacao | 70-80% | 75% |
| Lucro por Operacao | > R$ 500 | > R$ 800 |

## 3. Diretrizes de Otimizacao

### Se Overfitting (Gap > 8%):
1. Aumentar `min_child_weight` (ex: 10 -> 20)
2. Aumentar `gamma` (ex: 0.5 -> 1.0)
3. Reduzir `max_depth` (ex: 5 -> 4)
4. Reduzir `learning_rate` (ex: 0.05 -> 0.03)
5. Aumentar `subsample` (ex: 0.8 -> 0.6)

### Se Underfitting (AUC baixo):
1. Aumentar `max_depth` (com cuidado)
2. Aumentar `n_estimators`
3. Aumentar `learning_rate`
4. Revisar features (adicionar interacoes)

### Se Drift Alto (PSI > 0.2):
1. Identificar features com KS > 0.5
2. Adicionar em `force_drop_features` no config
3. Considerar retreino com dados mais recentes

### Se Eficiencia < 75%:
1. Verificar calibracao do threshold
2. Revisar features de negocio
3. Investigar se modelo esta rejeitando bons clientes

## 4. Criterio de Desempate

**Quando dois modelos atendem as metas minimas:**
- Priorizar MAIOR LUCRO NO TESTE (desde que PSI < 0.2)
- Um modelo com AUC 0.82 e lucro R$ 5M e melhor que AUC 0.80 e lucro R$ 4M

## 5. Restricoes de Seguranca

- NUNCA deletar `train.parquet` ou `test.parquet`
- NUNCA commitar API keys
- SEMPRE fazer backup antes de editar codigo em `src/`
- MAXIMO 3 tentativas de corrigir mesmo erro

## 6. Criterios de Parada

O agente PARA quando:
1. Todas as metas de performance atingidas, OU
2. Estagnou por 5 iteracoes sem melhoria, OU
3. Erro critico nao recuperavel, OU
4. Usuario interrompe manualmente
