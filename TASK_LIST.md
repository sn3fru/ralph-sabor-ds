## Tarefas do Pipeline

- [x] 01_load_data: Carregar dados brutos e gerar metadata inicial
- [x] 02_eda_nulls: Analisar valores faltantes e decidir estrategia
- [x] 03_eda_target: Analisar distribuicao do target e desbalanceamento
- [x] 04_eda_distributions: Plotar distribuicoes das features numericas
- [x] 05_eda_correlations: Analisar correlacoes e multicolinearidade
- [x] 06_feature_cleanup: Remover features criticas (>50% nulos, redundantes) **<-- ATUAL**
- [x] 07_feature_transform: Aplicar transformacoes (log, binning)
- [x] 08_feature_impute: Imputar valores faltantes
- [x] 09_feature_select: Selecionar features finais
- [x] 10_train_baseline: Treinar modelo baseline com parametros default
- [ ] 11_evaluate_baseline: Avaliar metricas e gap treino-teste **<-- ATUAL**
- [ ] 12_tune_regularization: Ajustar regularizacao se overfitting
- [ ] 13_cross_validate: Validacao cruzada para estimar variancia
- [ ] 14_find_threshold: Encontrar threshold otimo de lucro
- [ ] 15_evaluate_profit: Calcular metricas financeiras
- [ ] 16_stability_psi: Calcular PSI e verificar estabilidade
- [ ] 17_final_report: Gerar relatorio final