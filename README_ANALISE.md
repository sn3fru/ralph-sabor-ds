# Relatório de Análise: Credit Scoring

> **Última sincronização:** 2026-01-31 08:32:35
> **Fonte:** `credit_scoring_20260129_134511.md`

---

# 📊 Relatório de Execução: Credit Scoring Model

**Data/Hora:** 2026-01-29 13:45:11
**Execução:** `credit_scoring_20260129_134511`
**Modo Vision:** `Ativado` (gemini)

---


# Credit Scoring: Maquina de Decisao de Credito

ℹ️ Pipeline de analise executiva de risco de credito com Machine Learning

ℹ️ Modo de execucao: DEV


## 0. Configuracao do Pipeline

ℹ️ Modo: DEV

ℹ️ SHAP: Desativado

ℹ️ Correlation Threshold: 0.98

ℹ️ Max Depth: 3

ℹ️ Learning Rate: 0.07

ℹ️ N Estimators (DEV): 500


## 1. Setup & Infraestrutura

✅ GPU NVIDIA detectada - XGBoost pode usar aceleração GPU

**Pandas Version:** 2.2.3

**NumPy Version:** 2.1.3

**XGBoost Version:** 3.0.5

**GPU Available:** True


## 2. Engenharia de Dados

ℹ️ Carregando arquivos parquet...

**Train Raw Shape (Original):** 496,758 linhas × 1112 colunas

**Test Raw Shape (Original):** 27,377 linhas × 1112 colunas

ℹ️ [DEV MODE] Aplicando amostragem estratificada de 10,000 linhas para acelerar desenvolvimento...

ℹ️ [DEV MODE] Treino reduzido para 10,000 linhas (amostragem estratificada)

ℹ️ [DEV MODE] Teste reduzido para 551 linhas

**Train Raw Shape (Apos Sampling):** 10,000 linhas

**Test Raw Shape (Apos Sampling):** 551 linhas

**💡 Insight (dev_mode):** Modo DEV ativo: usando amostra de 10,000 linhas de treino e 551 linhas de teste para desenvolvimento rapido. Execute em modo PROD para usar dataset completo (496,758 linhas).

**DataFrame Unificado:** 10,551 linhas × 1113 colunas

**Com Label:** 10,000

**Sem Label:** 551

**Treino:** 8,000 amostras

**Validação:** 2,000 amostras

**Teste Cego:** 551 amostras

**Features:** 1110


## 3. EDA Executiva

ℹ️ Dimensões: 10,551 linhas × 1113 colunas

**Memória utilizada:** 90.58 MB

### Distribuição da Target

| Métrica | Valor |
|---------|-------|
| Classe 0 | 802 |
| Classe 1 | 9198 |
| Taxa de Balanceamento | 0.087 |

⚠️ PROBLEMA DESBALANCEADO - Necessário ajuste de estratégia de modelagem

**💡 Insight (overfitting):** O dataset está severamente desbalanceado (razão 0.087). Será necessário usar scale_pos_weight no XGBoost para compensar.

### Top 10 Features com Mais Nulos

| Métrica | Valor |
|---------|-------|
| feature_39 | 93.65% |
| feature_983 | 86.73% |
| feature_555 | 83.29% |
| feature_1016 | 83.29% |
| feature_104 | 82.82% |
| feature_903 | 82.41% |
| feature_417 | 81.72% |
| feature_402 | 81.30% |
| feature_779 | 81.01% |
| feature_547 | 80.31% |

**Features com 0% nulos:** 29

**Features com >0% e ≤10% nulos:** 808

**Features com >10% e ≤50% nulos:** 190

**Features com >50% nulos:** 83

**💡 Insight (data_quality):** A feature feature_39 tem 93.6% de valores nulos. Devido à alta cardinalidade de nulos, o XGBoost provavelmente está usando essa ausência como uma categoria informativa (ex: cliente sem histórico específico). Esta feature pode estar capturando padrões de 'novos clientes' ou 'dados não coletados'.

### Distribuição de Valores Nulos nas Features

![Distribuição de Valores Nulos nas Features](credit_scoring_20260129_134511_images/img_001_credit_scoring_20260129_134511.png)

**Contexto Técnico:** Distribuição de valores nulos nas 1110 features do dataset de credit scoring. Gráfico à esquerda: histograma da distribuição de percentuais de nulos (mediana: 1.61%). Gráfico à direita: top 15 features com maior percentual de nulos (máximo: 93.65%). Total de features com >50% nulos: 83. O XGBoost usa sparse-aware split finding para lidar com nulos, tratando ausência como informação.

> 🤖 **Análise Visual Automática:**
>
> Como Head de Risco, minha análise visual confirma integralmente os dados numéricos apresentados. A geometria do histograma de nulos é fortemente assimétrica à esquerda, com um pico proeminente próximo a zero, validando visualmente a mediana de 1.61%. O gráfico à direita, com as top 15 features, corrobora a presença de valores nulos extremamente altos, com a `feature_39` atingindo o máximo de 93.65%. Não há contradições entre texto e imagem.
> 
> O principal sinal de alerta é a existência de 83 features (aproximadamente 7.5% das 1110 totais) com mais de 50% de nulos. Este cenário, combinado com o **severo desbalanceamento de classes (0.0872)** no contexto global, sugere um ambiente de modelagem complexo. Embora o XGBoost seja robusto a nulos, tratando a ausência como informação, uma proporção tão alta de features extremamente esparsas pode introduzir ruído, aumentar a complexidade do modelo e o risco de *overfitting*, ou capturar relações espúrias que não generalizam bem em dados de produção (especialmente em contextos de *data drift*).
> 
> Para o negócio, é imperativo investigar a fundo a origem e a relevância dessas 83 features com alta incompletude. Recomendo uma avaliação rigorosa para determinar se a "informação da ausência" é verdadeiramente preditiva ou se a remoção de features excessivamente esparsas simplificaria o modelo e aumentaria sua robustez e interpretabilidade para o score de risco. Priorizar features com maior completude pode levar a um modelo mais estável e confiável.

**💡 Insight (geral):** XGBoost lida nativamente com nulos através de 'sparse-aware split finding'. Isso evita imputação arbitrária e preserva informação de padrões de missingness.


### 3.1. Análise de Drift Temporal Completa

ℹ️ Iniciando análise de drift temporal completa (KS Test + PCA + t-SNE)...

**Treino (Referência):** 10,000 amostras

**Teste (Atual/Cego):** 551 amostras

ℹ️ Calculando drift nas top 20 features...

### Top 10 Features com Maior Instabilidade (KS Statistic)

| Métrica | Valor |
|---------|-------|
| feature_757 | KS=0.6887, p=0.0000 |
| feature_1024 | KS=0.3730, p=0.0000 |
| feature_708 | KS=0.3477, p=0.0000 |
| feature_383 | KS=0.2449, p=0.0000 |
| feature_372 | KS=0.2360, p=0.0000 |
| feature_868 | KS=0.2264, p=0.0000 |
| feature_735 | KS=0.2198, p=0.0000 |
| feature_402 | KS=0.2114, p=0.0003 |
| feature_425 | KS=0.2006, p=0.0000 |
| feature_613 | KS=0.1929, p=0.0000 |

ℹ️ Processando mapa visual (PCA & t-SNE) com 10k pontos...

ℹ️ Executando t-SNE (pode demorar)...

### Análise de Drift Temporal Completa

![Análise de Drift Temporal Completa](credit_scoring_20260129_134511_images/img_002_credit_scoring_20260129_134511.png)

**Contexto Técnico:** Análise completa de drift temporal usando KS Test, PCA e t-SNE. Top esquerdo: Distribuição de KS Statistics das 20 features analisadas. Top direito: Distribuição da feature com maior drift (feature_757, KS=0.6887). Bottom esquerdo: PCA 2D mostrando estrutura global. Bottom direito: t-SNE 2D mostrando agrupamentos locais. Contornos azuis: densidade do treino. Contornos vermelhos: densidade do teste. Se os contornos vermelhos formarem 'ilhas' onde não há contornos azuis, indica regiões não exploradas pelo treino (risco de falha do modelo).

> 🤖 **Análise Visual Automática:**
>
> Como Head de Risco, minha análise visual valida os números fornecidos e revela preocupações críticas.
> 
> 1.  **Observações sobre a forma/geometria da curva:**
>     O histograma de KS estatísticas (top esquerdo) mostra claramente que **muitas features sofrem drift**, com 5 features acima do limite crítico (0.2) e uma com KS próximo a 0.7. A `feature_757` (top direito, KS=0.6887) corrobora esse drift extremo, com as distribuições de Treino (azul) e Teste (vermelho) completamente deslocadas e com picos em valores distintos. Nos gráficos de redução de dimensionalidade, o PCA (bottom esquerdo) indica que a estrutura global do Teste se estende a regiões de baixa densidade do Treino. Mais alarmante, o t-SNE (bottom direito) exibe múltiplas **"ilhas" de contornos vermelhos (Teste) sem correspondência em azul (Treino)**, indicando agrupamentos locais no Teste que o modelo nunca viu.
> 
> 2.  **Sinais de problemas (drift, etc):**
>     Há um **drift temporal severo e multidimensional** confirmado visualmente e pelos valores de KS. O contexto global de `n_features_raw = 1110` amplifica o problema, pois o drift em múltiplas features cria um shift significativo na distribuição de dados. A presença de "ilhas" de dados no Teste completamente fora da distribuição de Treino representa um **risco extremo de falha do modelo**, pois ele fará previsões em território completamente inexplorado.
> 
> 3.  **Conclusões práticas para o negócio:**
>     A performance do modelo de Credit Scoring está em **alto risco de degradação severa em produção**. As previsões para os clientes cujos perfis se enquadram nas "ilhas" de dados do Teste serão intrinsecamente não confiáveis. Para o negócio, isso se traduz em um **aumento inaceitável nas perdas por inadimplência** (aprovação de maus clientes) ou na **perda de market share** (rejeição de bons clientes). É imperativo **pausar a implantação ou exigir revalidação rigorosa**. Recomendo investigação imediata das causas do drift e o retreinamento do modelo com dados mais representativos, além de implementação de monitoramento contínuo de drift.

⚠️ CONCLUSÃO: 11 features com drift alto (KS > 0.15). Observe as linhas de contorno no t-SNE: se as linhas vermelhas formam 'ilhas' onde não há linhas azuis, o modelo falhará nessas regiões.

**💡 Insight (drift):** 11 features apresentam drift significativo (KS > 0.15). Isso indica que a distribuição dos dados de teste difere substancialmente do treino. Recomenda-se: (1) Retreinar modelo com dados mais recentes, (2) Aplicar técnicas de domain adaptation, ou (3) Usar sample weights baseados em validação adversária.

**Features com Drift Alto (KS > 0.15):** 11

**Features com Drift Moderado (0.1 < KS <= 0.15):** 1

**Features Estáveis (KS <= 0.1):** 8

**KS Médio:** 0.1906

**KS Máximo:** 0.6887


#### 3.1.1. Sugestão de Ação: Remover Features com Drift Crítico

🚨 🚨 ATENÇÃO AGENTE: 1 features têm Drift Crítico (KS > 0.5). Elas estão distorcendo a realidade entre Treino e Teste e devem ser removidas imediatamente.

```json
{
  "force_drop_features": [
    "feature_757"
  ]
}
```

**💡 Insight (drift_action):** **AÇÃO RECOMENDADA:** Adicione as seguintes features na lista 'force_drop_features' do config.yaml: 'feature_757'. Essas features têm KS > 0.5, indicando que a distribuição mudou drasticamente entre treino e teste. Manter essas features pode causar falhas graves do modelo em produção.


## 4. Feature Selection (Conservadora)

⚠️ [FEATURE DROP] Removendo 1 features banidas pelo Agente...

ℹ️ [FEATURE DROP] Features removidas: feature_757

```json
['feature_757']
```

ℹ️ Modo DEV: Mantendo todas as features (exceto constantes) para visibilidade completa do Agente.

**💡 Insight (feature_selection):** Em modo DEV, não aplicamos filtros de correlação ou importância para evitar corte prematuro. O XGBoost lida bem com multicolinearidade através de divisão de importância entre variáveis. Features removidas apenas por variância zero (constantes).

**Features Iniciais:** 1110

**Features Removidas (Variância Zero):** 0

**Features Mantidas:** 1109

ℹ️ Processando Holdout (X_val) com a mesma seleção de features...

**Holdout Processado:** 2,000 amostras × 1109 features

**💡 Insight (data_split):** Holdout separado e processado: 2,000 amostras serão usadas apenas para avaliação final (calibração e métricas financeiras). Este conjunto não foi usado durante o treino.

**Features Mantidas:** 1109

**Features Removidas (Variância Zero):** 0

**Features Removidas (Forçadas pelo Agente):** 1

**Shape Final Treino:** (8000, 1109)

### Amostra dos Dados Após Feature Selection

| feature_1 | feature_2 | feature_3 | feature_4 | feature_5 | feature_6 | feature_7 | feature_8 | feature_9 | feature_10 | feature_11 | feature_12 | feature_13 | feature_14 | feature_15 | feature_16 | feature_17 | feature_18 | feature_19 | feature_20 | feature_21 | feature_22 | feature_23 | feature_24 | feature_25 | feature_26 | feature_27 | feature_28 | feature_29 | feature_30 | feature_31 | feature_32 | feature_33 | feature_34 | feature_35 | feature_36 | feature_37 | feature_38 | feature_39 | feature_40 | feature_41 | feature_42 | feature_43 | feature_44 | feature_45 | feature_46 | feature_47 | feature_48 | feature_49 | feature_50 | feature_51 | feature_52 | feature_53 | feature_54 | feature_55 | feature_56 | feature_57 | feature_58 | feature_59 | feature_60 | feature_61 | feature_62 | feature_63 | feature_64 | feature_65 | feature_66 | feature_67 | feature_68 | feature_69 | feature_70 | feature_71 | feature_72 | feature_73 | feature_74 | feature_75 | feature_76 | feature_77 | feature_78 | feature_79 | feature_80 | feature_81 | feature_82 | feature_83 | feature_84 | feature_85 | feature_86 | feature_87 | feature_88 | feature_89 | feature_90 | feature_91 | feature_92 | feature_93 | feature_94 | feature_95 | feature_96 | feature_97 | feature_98 | feature_99 | feature_100 | feature_101 | feature_102 | feature_103 | feature_104 | feature_105 | feature_106 | feature_107 | feature_108 | feature_109 | feature_110 | feature_111 | feature_112 | feature_113 | feature_114 | feature_115 | feature_116 | feature_117 | feature_118 | feature_119 | feature_120 | feature_121 | feature_122 | feature_123 | feature_124 | feature_125 | feature_126 | feature_127 | feature_128 | feature_129 | feature_130 | feature_131 | feature_132 | feature_133 | feature_134 | feature_135 | feature_136 | feature_137 | feature_138 | feature_139 | feature_140 | feature_141 | feature_142 | feature_143 | feature_144 | feature_145 | feature_146 | feature_147 | feature_148 | feature_149 | feature_150 | feature_151 | feature_152 | feature_153 | feature_154 | feature_155 | feature_156 | feature_157 | feature_158 | feature_159 | feature_160 | feature_161 | feature_162 | feature_163 | feature_164 | feature_165 | feature_166 | feature_167 | feature_168 | feature_169 | feature_170 | feature_171 | feature_172 | feature_173 | feature_174 | feature_175 | feature_176 | feature_177 | feature_178 | feature_179 | feature_180 | feature_181 | feature_182 | feature_183 | feature_184 | feature_185 | feature_186 | feature_187 | feature_188 | feature_189 | feature_190 | feature_191 | feature_192 | feature_193 | feature_194 | feature_195 | feature_196 | feature_197 | feature_198 | feature_199 | feature_200 | feature_201 | feature_202 | feature_203 | feature_204 | feature_205 | feature_206 | feature_207 | feature_208 | feature_209 | feature_210 | feature_211 | feature_212 | feature_213 | feature_214 | feature_215 | feature_216 | feature_217 | feature_218 | feature_219 | feature_220 | feature_221 | feature_222 | feature_223 | feature_224 | feature_225 | feature_226 | feature_227 | feature_228 | feature_229 | feature_230 | feature_231 | feature_232 | feature_233 | feature_234 | feature_235 | feature_236 | feature_237 | feature_238 | feature_239 | feature_240 | feature_241 | feature_242 | feature_243 | feature_244 | feature_245 | feature_246 | feature_247 | feature_248 | feature_249 | feature_250 | feature_251 | feature_252 | feature_253 | feature_254 | feature_255 | feature_256 | feature_257 | feature_258 | feature_259 | feature_260 | feature_261 | feature_262 | feature_263 | feature_264 | feature_265 | feature_266 | feature_267 | feature_268 | feature_269 | feature_270 | feature_271 | feature_272 | feature_273 | feature_274 | feature_275 | feature_276 | feature_277 | feature_278 | feature_279 | feature_280 | feature_281 | feature_282 | feature_283 | feature_284 | feature_285 | feature_286 | feature_287 | feature_288 | feature_289 | feature_290 | feature_291 | feature_292 | feature_293 | feature_294 | feature_295 | feature_296 | feature_297 | feature_298 | feature_299 | feature_300 | feature_301 | feature_302 | feature_303 | feature_304 | feature_305 | feature_306 | feature_307 | feature_308 | feature_309 | feature_310 | feature_311 | feature_312 | feature_313 | feature_314 | feature_315 | feature_316 | feature_317 | feature_318 | feature_319 | feature_320 | feature_321 | feature_322 | feature_323 | feature_324 | feature_325 | feature_326 | feature_327 | feature_328 | feature_329 | feature_330 | feature_331 | feature_332 | feature_333 | feature_334 | feature_335 | feature_336 | feature_337 | feature_338 | feature_339 | feature_340 | feature_341 | feature_342 | feature_343 | feature_344 | feature_345 | feature_346 | feature_347 | feature_348 | feature_349 | feature_350 | feature_351 | feature_352 | feature_353 | feature_354 | feature_355 | feature_356 | feature_357 | feature_358 | feature_359 | feature_360 | feature_361 | feature_362 | feature_363 | feature_364 | feature_365 | feature_366 | feature_367 | feature_368 | feature_369 | feature_370 | feature_371 | feature_372 | feature_373 | feature_374 | feature_375 | feature_376 | feature_377 | feature_378 | feature_379 | feature_380 | feature_381 | feature_382 | feature_383 | feature_384 | feature_385 | feature_386 | feature_387 | feature_388 | feature_389 | feature_390 | feature_391 | feature_392 | feature_393 | feature_394 | feature_395 | feature_396 | feature_397 | feature_398 | feature_399 | feature_400 | feature_401 | feature_402 | feature_403 | feature_404 | feature_405 | feature_406 | feature_407 | feature_408 | feature_409 | feature_410 | feature_411 | feature_412 | feature_413 | feature_414 | feature_415 | feature_416 | feature_417 | feature_418 | feature_419 | feature_420 | feature_421 | feature_422 | feature_423 | feature_424 | feature_425 | feature_426 | feature_427 | feature_428 | feature_429 | feature_430 | feature_431 | feature_432 | feature_433 | feature_434 | feature_435 | feature_436 | feature_437 | feature_438 | feature_439 | feature_440 | feature_441 | feature_442 | feature_443 | feature_444 | feature_445 | feature_446 | feature_447 | feature_448 | feature_449 | feature_450 | feature_451 | feature_452 | feature_453 | feature_454 | feature_455 | feature_456 | feature_457 | feature_458 | feature_459 | feature_460 | feature_461 | feature_462 | feature_463 | feature_464 | feature_465 | feature_466 | feature_467 | feature_468 | feature_469 | feature_470 | feature_471 | feature_472 | feature_473 | feature_474 | feature_475 | feature_476 | feature_477 | feature_478 | feature_479 | feature_480 | feature_481 | feature_482 | feature_483 | feature_484 | feature_485 | feature_486 | feature_487 | feature_488 | feature_489 | feature_490 | feature_491 | feature_492 | feature_493 | feature_494 | feature_495 | feature_496 | feature_497 | feature_498 | feature_499 | feature_500 | feature_501 | feature_502 | feature_503 | feature_504 | feature_505 | feature_506 | feature_507 | feature_508 | feature_509 | feature_510 | feature_511 | feature_512 | feature_513 | feature_514 | feature_515 | feature_516 | feature_517 | feature_518 | feature_519 | feature_520 | feature_521 | feature_522 | feature_523 | feature_524 | feature_525 | feature_526 | feature_527 | feature_528 | feature_529 | feature_530 | feature_531 | feature_532 | feature_533 | feature_534 | feature_535 | feature_536 | feature_537 | feature_538 | feature_539 | feature_540 | feature_541 | feature_542 | feature_543 | feature_544 | feature_545 | feature_546 | feature_547 | feature_548 | feature_549 | feature_550 | feature_551 | feature_552 | feature_553 | feature_554 | feature_555 | feature_556 | feature_557 | feature_558 | feature_559 | feature_560 | feature_561 | feature_562 | feature_563 | feature_564 | feature_565 | feature_566 | feature_567 | feature_568 | feature_569 | feature_570 | feature_571 | feature_572 | feature_573 | feature_574 | feature_575 | feature_576 | feature_577 | feature_578 | feature_579 | feature_580 | feature_581 | feature_582 | feature_583 | feature_584 | feature_585 | feature_586 | feature_587 | feature_588 | feature_589 | feature_590 | feature_591 | feature_592 | feature_593 | feature_594 | feature_595 | feature_596 | feature_597 | feature_598 | feature_599 | feature_600 | feature_601 | feature_602 | feature_603 | feature_604 | feature_605 | feature_606 | feature_607 | feature_608 | feature_609 | feature_610 | feature_611 | feature_612 | feature_613 | feature_614 | feature_615 | feature_616 | feature_617 | feature_618 | feature_619 | feature_620 | feature_621 | feature_622 | feature_623 | feature_624 | feature_625 | feature_626 | feature_627 | feature_628 | feature_629 | feature_630 | feature_631 | feature_632 | feature_633 | feature_634 | feature_635 | feature_636 | feature_637 | feature_638 | feature_639 | feature_640 | feature_641 | feature_642 | feature_643 | feature_644 | feature_645 | feature_646 | feature_647 | feature_648 | feature_649 | feature_650 | feature_651 | feature_652 | feature_653 | feature_654 | feature_655 | feature_656 | feature_657 | feature_658 | feature_659 | feature_660 | feature_661 | feature_662 | feature_663 | feature_664 | feature_665 | feature_666 | feature_667 | feature_668 | feature_669 | feature_670 | feature_671 | feature_672 | feature_673 | feature_674 | feature_675 | feature_676 | feature_677 | feature_678 | feature_679 | feature_680 | feature_681 | feature_682 | feature_683 | feature_684 | feature_685 | feature_686 | feature_687 | feature_688 | feature_689 | feature_690 | feature_691 | feature_692 | feature_693 | feature_694 | feature_695 | feature_696 | feature_697 | feature_698 | feature_699 | feature_700 | feature_701 | feature_702 | feature_703 | feature_704 | feature_705 | feature_706 | feature_707 | feature_708 | feature_709 | feature_710 | feature_711 | feature_712 | feature_713 | feature_714 | feature_715 | feature_716 | feature_717 | feature_718 | feature_719 | feature_720 | feature_721 | feature_722 | feature_723 | feature_724 | feature_725 | feature_726 | feature_727 | feature_728 | feature_729 | feature_730 | feature_731 | feature_732 | feature_733 | feature_734 | feature_735 | feature_736 | feature_737 | feature_738 | feature_739 | feature_740 | feature_741 | feature_742 | feature_743 | feature_744 | feature_745 | feature_746 | feature_747 | feature_748 | feature_749 | feature_750 | feature_751 | feature_752 | feature_753 | feature_754 | feature_755 | feature_756 | feature_758 | feature_759 | feature_760 | feature_761 | feature_762 | feature_763 | feature_764 | feature_765 | feature_766 | feature_767 | feature_768 | feature_769 | feature_770 | feature_771 | feature_772 | feature_773 | feature_774 | feature_775 | feature_776 | feature_777 | feature_778 | feature_779 | feature_780 | feature_781 | feature_782 | feature_783 | feature_784 | feature_785 | feature_786 | feature_787 | feature_788 | feature_789 | feature_790 | feature_791 | feature_792 | feature_793 | feature_794 | feature_795 | feature_796 | feature_797 | feature_798 | feature_799 | feature_800 | feature_801 | feature_802 | feature_803 | feature_804 | feature_805 | feature_806 | feature_807 | feature_808 | feature_809 | feature_810 | feature_811 | feature_812 | feature_813 | feature_814 | feature_815 | feature_816 | feature_817 | feature_818 | feature_819 | feature_820 | feature_821 | feature_822 | feature_823 | feature_824 | feature_825 | feature_826 | feature_827 | feature_828 | feature_829 | feature_830 | feature_831 | feature_832 | feature_833 | feature_834 | feature_835 | feature_836 | feature_837 | feature_838 | feature_839 | feature_840 | feature_841 | feature_842 | feature_843 | feature_844 | feature_845 | feature_846 | feature_847 | feature_848 | feature_849 | feature_850 | feature_851 | feature_852 | feature_853 | feature_854 | feature_855 | feature_856 | feature_857 | feature_858 | feature_859 | feature_860 | feature_861 | feature_862 | feature_863 | feature_864 | feature_865 | feature_866 | feature_867 | feature_868 | feature_869 | feature_870 | feature_871 | feature_872 | feature_873 | feature_874 | feature_875 | feature_876 | feature_877 | feature_878 | feature_879 | feature_880 | feature_881 | feature_882 | feature_883 | feature_884 | feature_885 | feature_886 | feature_887 | feature_888 | feature_889 | feature_890 | feature_891 | feature_892 | feature_893 | feature_894 | feature_895 | feature_896 | feature_897 | feature_898 | feature_899 | feature_900 | feature_901 | feature_902 | feature_903 | feature_904 | feature_905 | feature_906 | feature_907 | feature_908 | feature_909 | feature_910 | feature_911 | feature_912 | feature_913 | feature_914 | feature_915 | feature_916 | feature_917 | feature_918 | feature_919 | feature_920 | feature_921 | feature_922 | feature_923 | feature_924 | feature_925 | feature_926 | feature_927 | feature_928 | feature_929 | feature_930 | feature_931 | feature_932 | feature_933 | feature_934 | feature_935 | feature_936 | feature_937 | feature_938 | feature_939 | feature_940 | feature_941 | feature_942 | feature_943 | feature_944 | feature_945 | feature_946 | feature_947 | feature_948 | feature_949 | feature_950 | feature_951 | feature_952 | feature_953 | feature_954 | feature_955 | feature_956 | feature_957 | feature_958 | feature_959 | feature_960 | feature_961 | feature_962 | feature_963 | feature_964 | feature_965 | feature_966 | feature_967 | feature_968 | feature_969 | feature_970 | feature_971 | feature_972 | feature_973 | feature_974 | feature_975 | feature_976 | feature_977 | feature_978 | feature_979 | feature_980 | feature_981 | feature_982 | feature_983 | feature_984 | feature_985 | feature_986 | feature_987 | feature_988 | feature_989 | feature_990 | feature_991 | feature_992 | feature_993 | feature_994 | feature_995 | feature_996 | feature_997 | feature_998 | feature_999 | feature_1000 | feature_1001 | feature_1002 | feature_1003 | feature_1004 | feature_1005 | feature_1006 | feature_1007 | feature_1008 | feature_1009 | feature_1010 | feature_1011 | feature_1012 | feature_1013 | feature_1014 | feature_1015 | feature_1016 | feature_1017 | feature_1018 | feature_1019 | feature_1020 | feature_1021 | feature_1022 | feature_1023 | feature_1024 | feature_1025 | feature_1026 | feature_1027 | feature_1028 | feature_1029 | feature_1030 | feature_1031 | feature_1032 | feature_1033 | feature_1034 | feature_1035 | feature_1036 | feature_1037 | feature_1038 | feature_1039 | feature_1040 | feature_1041 | feature_1042 | feature_1043 | feature_1044 | feature_1045 | feature_1046 | feature_1047 | feature_1048 | feature_1049 | feature_1050 | feature_1051 | feature_1052 | feature_1053 | feature_1054 | feature_1055 | feature_1056 | feature_1057 | feature_1058 | feature_1059 | feature_1060 | feature_1061 | feature_1062 | feature_1063 | feature_1064 | feature_1065 | feature_1066 | feature_1067 | feature_1068 | feature_1069 | feature_1070 | feature_1071 | feature_1072 | feature_1073 | feature_1074 | feature_1075 | feature_1076 | feature_1077 | feature_1078 | feature_1079 | feature_1080 | feature_1081 | feature_1082 | feature_1083 | feature_1084 | feature_1085 | feature_1086 | feature_1087 | feature_1088 | feature_1089 | feature_1090 | feature_1091 | feature_1092 | feature_1093 | feature_1094 | feature_1095 | feature_1096 | feature_1097 | feature_1098 | feature_1099 | feature_1100 | feature_1101 | feature_1102 | feature_1103 | feature_1104 | feature_1105 | feature_1106 | feature_1107 | feature_1108 | feature_1109 | feature_1110 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.165121136112006 | 3.0000005316514966 | 5.000519986133703 | -4.0 | -4.82056783152296 | -2.4309165526675787 | 1.0000042455385871 | -3.533333333333333 | 3.00039048200122 | -3.994199933328247 | 3.4509650417101603 | 0.08246001392645572 | 3.021330822848022 | -1.9984542084062302 | -1.9999996681999999 | 3.00000000210138 | -2.940309880619761 | -4.361089456149959 | 2.0773480662983426 | 0.0030120481927710845 | 0.028439229682836572 | -2.9988605330709923 | 2.034480548237093 | 0.009473363228699551 | 2.00000018317978 | -2.98894620486367 | 3.400046534624669 | 1.0000000039999963 | -3.743464388195438 | -3.9117421335379894 | 2.00000013825 | 5.165888105182063 | 3.0000001100464146 | -4.934426229508197 | -2.8318101241668856 | -1.67802101109215 | -0.9350180505415162 | nan | nan | 2.2017543859649122 | -1.2 | 2.011552525875431 | nan | -2.973520557932233 | -1.9429126546472757 | -1.935914363859285 | -2.294320798158097 | 0.10655919950831577 | 3.0007567351021933 | 5.024242424242424 | 4.239024390243903 | 0.20295400005481926 | -0.9999999972913043 | 4.031728665207877 | nan | 5.054513115639142 | 5.255563032896535 | 1.8345458060491735 | 0.00012040828843770852 | -3.999998181117018 | 4.000462993401133 | -0.9849939975990396 | nan | 2.0000004631847133 | -4.720947724708537 | 4.334012219959266 | -4.999997533797335 | -2.830215976222509 | 6.773116307207492 | 2.0066009612459696 | 4.6 | 3.002776125322229 | -3.9986606777514284 | 5.0406361701348565 | -1.9876862374870699 | 5.027813885544991 | 5.002565774331543 | -2.9994142769981695 | -0.999999760000004 | 1.5998476527408365 | -3.6 | 2.669761037209985 | 3.1499246100794536 | 1.8181818181818183 | 4.043829296424452 | 3.0526315789473686 | 4.0428021447266795 | -2.625387561553894 | -1.511111111111111 | -3.0 | 5.077656504471556 | 0.01805467221177116 | -1.9836728529344512 | 0.027217806041335453 | 2.021019852082522 | 1.0037064560954512 | 4.005305039787799 | -0.8865619774957368 | 1.1131020210479012 | 3.1156939494319107 | -2.998464220778766 | 5.069190486653257 | 9.6033319597035e-06 | -1.8938682702889575 | 3.427350427350427 | 5.00007499910001 | 2.1230819427274605 | -4.950299050958705 | -4.992016528548131 | nan | 5.103896103896104 | -1.4910918446850379 | -1.990867579908676 | nan | -1.2912223456068346 | 4.239372822299652 | -2.380952380952381 | 1.380952380952381 | -0.9698464912280702 | -2.996042216358839 | 2.9849246231155777 | -0.3947548012769859 | -4.575193798449613 | 2.0009120453631883 | -2.1998509459197604 | 6.1912527908250095 | nan | 3.0000003003486264 | 4.821727611787347 | 3.010470051039731 | 0.001775422965416216 | 5.000003013723542 | -2.876412429378531 | 0.0001156737998843262 | -4.719137783061101 | 5.636363636363637 | -4.805177246413266 | 5.562757942427356e-06 | 3.084443950431193 | -4.0 | -4.99999798207144 | -0.9963469257047498 | -2.557716837522797 | -1.8860008786059452 | 4.1109779071758865 | -1.0 | nan | 3.0520454058684945 | 4.000001612382507 | nan | 4.03619963164907 | -0.9999993507999999 | -4.9862215156332805 | -1.9419988102320047 | 2.0078825947068926 | nan | 2.1657445252456005 | -3.951703500112183 | 2.2070592737052652 | 0.0 | 1.0063380651200156 | 8.908340473265687 | -0.9996927808568847 | 2.075478721434127 | nan | -3.8619430256961493 | -1.9193209406514444 | -1.8321241399861932 | 2.0 | 4.2 | -0.9653924307848616 | 0.49150762322094665 | 4.00727485458091 | 1.0 | 5.330082132758853 | -2.9644162544909483 | 6.923119816272759 | -1.9102419012590182 | 0.006349206349206349 | nan | -2.2068636796949477 | -2.4986833647805264 | 1.0207009447148552 | 1.4303089262200968 | -0.612755109258769 | -4.957359397202902 | 0.11538461538461539 | 7.0 | 4.033287558934397 | -3.8113303415522686 | 4.405429864253394 | 3.188888888888888 | -3.8046504794075817 | nan | -3.9891842301543874 | -1.2871909705646702 | 4.0 | 3.401785313901057 | 4.000000065436634 | -3.471033958163196 | -2.25 | -3.999960888942775 | -0.9831273604143004 | 5.0013809743011794 | 1.0054211539496407 | -2.989329268292683 | 1.0 | -2.9896078180785786 | nan | 2.0085025015771305 | 5.022337400116194 | 5.072256141531657 | 0.011482996594657555 | -1.9999990132501468 | 0.00254862686415035 | -0.2506349374609944 | 5.000494559867487 | 4.887600506570145 | -0.9999985153124996 | 3.0676040061633283 | 4.6595235739291745 | 4.347151820770567 | 5.047379330766953 | 0.7538461538461538 | -2.0 | -0.906057185735463 | -0.9417879417879418 | 4.097625968992248 | -4.866596759697472 | -3.9880239520958085 | 5.024973584261117 | 5.13627455076746 | -4.99999835138198 | 4.000000152999991 | 4.838574423480084 | -0.11089140574823242 | 2.000832223933992 | 2.0000006293309176 | 4.621858266887186 | -1.953464322647363 | 1.041209995615958 | 5.955555555555556 | -3.750202593192869 | -0.9974256675047617 | -1.0 | -0.9315068493150684 | 3.6911873611453965 | -1.3819958245146267 | -1.2291666666666667 | -0.9572818366453842 | 1.726341865802667e-06 | 3.0 | 1.0000004328 | -2.0546301616043348 | 4.0000524176169785 | -2.991360551304601 | 3.015921152388173 | -0.8746048049344336 | -5.0 | 2.0092174322018885 | -1.9411657559198543 | -1.2467185667890175 | -4.5268170728268124 | -4.9116364706419064 | 5.007619531398819 | -1.8918512955313556 | 1.025500189945549 | -0.933748272654995 | nan | 2.4540152534768955 | 3.0 | -1.871043243378312 | 6.429563042126878 | -3.942528735632184 | 4.405172413793103 | 3.016107226000718 | 0.47899778924097275 | -0.8658835403999459 | 4.002781499316291 | -3.624041205041363 | -1.9910332417710608 | 4.161223563749451 | -1.9999428545152318 | -2.0 | 1.017814465932164 | -5.0 | -3.619047619047619 | 4.857142857142857 | 4.0 | -1.7249478537160594 | -1.9999999005194002 | 5.569230769230769 | 2.004818765432099 | 1.0590033709245468 | nan | -0.9998558758314856 | -2.5420236725504486 | 0.1411285761136698 | -0.9687685901249257 | nan | 2.8 | 4.0371137923210325 | -4.058519281467407 | 2.4559710494571774 | 2.0017843280437893 | 5.005144423547884 | -0.991082413383242 | 5.003036651712607 | -2.8079575596816975 | 2.0 | 4.014022687111669 | -0.21138316880310004 | 2.724017969897335 | -4.438319970845481 | 5.4823216292641135 | -0.9850841184663799 | 0.16463414634146342 | -3.9954660452480093 | nan | nan | 1.9708178282649556 | -2.999996118378835 | 0.14634146341463417 | -0.9731310727741423 | nan | 5.013497144944033 | 3.0 | 3.0255507214730173 | -0.9952488465419929 | -4.0 | -2.971208043715909 | 2.3363914373088686 | -0.947275687434033 | 1.519207747085966e-05 | -0.569564519636069 | 2.112167557150657 | -3.958724202626642 | 3.015341841431836 | 2.0689610964741854 | 5.003673909633294 | -1.999989760928588 | 4.4296875 | 5.23042114873179 | 2.0000050379040877 | -1.7164750957854407 | -0.9522809123649459 | 1.0324716686437276 | 1.0124593845982905 | 2.021978021978022 | -0.7796610169491525 | -2.4290962403354546 | 2.6772717108290216 | 3.2340624540122342 | -2.8355368907690024 | -2.9999997577070063 | -0.06331542594013806 | 5.190281066363749 | 0.32375754657579914 | 3.00205369512681 | 6.071186779729882 | 6.829222905670406 | 5.857142857142857 | nan | -0.9623247663551402 | 2.0273323456489956 | 2.3365551809696017 | -3.9242894998764695 | -0.8532780968521483 | 1.0075541196545503 | 4.045287232860476 | 4.021987315010571 | 5.966666666666667 | -0.9986618595185079 | 0.001650813303332976 | 6.999965755784471 | 0.06142979452054795 | 4.011095516273424 | 5.001156159847523 | -4.93055150919182 | nan | 4.4296875 | 1.0068593461534976 | -3.0 | 3.0879289215686274 | 2.0 | -1.54872346442801 | -3.986117221488977 | 1.001236690793987 | -3.8542682389084555 | -2.7407276904278697 | -1.1051781348236114 | -0.9910672595156755 | 5.026368886153083 | nan | -4.998945750585823 | -1.9764705882352942 | 5.8 | 0.001319286121275386 | nan | -0.7413226922678686 | 4.011512752570692 | nan | -4.576446228735898 | -4.999999798188532 | nan | -4.980585873882111 | 0.13392857142857142 | 2.8 | 2.071845784706369 | 2.023738228054331 | 0.01754991094909613 | 3.062200956937799 | 0.41469816272965887 | -3.0 | 2.0465890183028286 | -3.0 | 3.0 | nan | -0.7949195025182636 | -1.9866151100535396 | -1.967247200417478 | nan | -4.999998477312866 | 1.3608231815731533 | -5.0 | 3.0 | 2.0300409649522075 | -0.5512675509325623 | -2.99999763920455 | 5.091097592036187 | -3.4651162790697674 | -2.968682998840111 | 1.1346153846153846 | 2.6153846153846154 | 1.0000008831814173 | 5.0047462896961274 | 0.0 | -4.9552631477597755 | 0.055360281195079075 | -1.9719268148452709 | 4.045564491951307 | -4.999804460783786 | 3.1534800713860798 | 1.1676514032496308 | 1.035487959442332 | -3.9933454650586855 | 1.001338612285185 | -3.981040803348792 | 5.024193548387097 | 2.0627156194930065 | -3.996785751207383 | -4.8999283924095955 | 4.018685950413223 | -4.9999993196 | -0.6801625991870041 | 3.8806451612903228 | 5.218508106327821 | 6.340705890054692 | 2.015020750171423 | -4.745460565506294 | 2.0 | nan | -1.8563467492260062 | -5.0 | -2.8141411216875927 | nan | 4.182233834095363 | -4.990769230769231 | 4.1421319796954315 | -0.3616438356164383 | 2.001978929181913 | -1.9999997968180168 | 0.15377314200631367 | -4.9967902906291535 | -4.9924568177741655 | 2.1124330755502676 | -3.7948774106912513 | -0.9546370967741935 | -4.989080665837148 | -3.962647189928206 | 5.006594403000453 | 2.0252789663200863 | -3.857142857142857 | -4.930287526583213 | 3.521415270018622 | -0.9887540669064877 | 0.18583730253780398 | 5.017100244289204 | 1.8466990167283872 | 4.4 | -1.9783342286719308 | 5.018101842520743 | -4.9999965726869755 | 3.1724137931034484 | -2.999997572899043 | 3.6166666666666667 | 1.1772727272727272 | 4.560026828055337 | 1.0302013827540122 | -4.900972970393305 | -2.282560706401766 | -1.8660714285714286 | 1.1416248770304964 | 3.0011458171360994 | 5.071963813461692 | 5.070697571499696 | 4.006762453711612 | 0.007773938795656465 | 2.038852764401954 | 2.957754489620731 | 1.0252673796791445 | 3.4572361706470707 | -0.12 | -3.0 | -3.9328185050592572 | -3.989355444164091 | -1.6550566769300947 | 5.0 | -2.583007965890933 | -2.9615216066866896 | 2.0489651706079277 | -3.181827799324001 | 3.0475907198096372 | 1.2418300653594772 | 0.0999789221624586 | -4.444444444444445 | 5.000000008473429 | -3.835734870317003 | 4.135055517263349 | -3.4755760467267427 | 2.699978122344774 | -1.0 | -2.9878596161244815 | -1.9661016949152543 | nan | -2.984558488006124 | -2.9999999884 | 1.083172535216976 | 5.9 | 7.864864864864865 | -0.7452513966480447 | -4.99648782663718 | 1.0642873726532316 | 0.09330143540669857 | 4.318014705882353 | 4.189655172413794 | nan | 5.114599824098505 | 3.0194219616671214 | 4.006992043749354 | -4.999039577491316 | -4.90592657076049 | -4.864271304495551 | 5.51298074441175 | nan | -0.9999968499996424 | 5.100151702990926 | 3.038961038961039 | -4.955414250002265 | nan | 3.0604274134119382 | 0.015040349490685919 | 4.0137591258315295 | 0.8333333333333339 | 7.9 | -1.6432973465523544 | 1.0132581229104323 | 1.229236575067767 | -4.900744416873449 | -4.999998294908335 | -4.826418975403773 | 3.6143527833668676 | 0.2830745341614907 | 3.5606130887485494 | 0.001685916488574392 | 3.230766038877901 | 5.007351982624125 | -1.4775258334246177 | 1.4309859154929576 | -0.9661448614097279 | 6.008445945945946 | 1.0336489972818486 | -4.366197183098592 | -0.8 | 5.04 | -3.3843797856049003 | 2.841950770971849 | 5.234686763967779 | 9.249034164963449 | 3.0705327675624705 | 4.034682080924855 | -2.967839356787136 | -1.9963917817523207 | 6.34375 | -0.9857348544329665 | -2.949003711507189 | -1.3394696592217294 | 5.000000542418659 | -0.9999991114860193 | 2.0587033743783216 | -1.9922480620155039 | 4.921348314606742 | -3.7049679133469624 | 2.352340775003883 | -2.889128896176144 | 1.2639833448993754 | nan | 1.106032906764168 | 5.463405962914946 | -3.8819055371556974 | 6.8939522631595125 | 3.0067402728094756 | -1.8447456671154283 | 4.168987795225447 | 7.912162162162162 | -3.0 | 4.09095225849051 | 3.0 | -0.9103342666626845 | nan | nan | 4.06346118776974 | -0.9992101523581689 | nan | -3.947417282483568 | 4.052682551618193 | -4.6597393636953015 | 0.7291276138283365 | 3.0230393297649525 | 1.0000000839999952 | 3.000006139110888 | 5.873786407766991 | 0.25900959294788695 | -4.572080363893886 | 2.6356118831083544 | -2.986281307693005 | 3.141304347826087 | -2.927827360393468 | 7.310344827586206 | -1.996890614133191 | 5.000000005675999 | 3.126079932389268 | 4.054672600127145 | 5.000296558350859 | 3.0 | 3.001961380663021 | 3.021694131420603 | 4.00084061869536 | -3.9801015624803266 | -1.9988117085244244 | 0.0025833201115245015 | -3.8934730488214155 | 3.03125 | -2.0 | 1.0627712977914623 | 4.00246170678337 | 4.359375 | 5.0031347703493285 | 2.0 | -2.979420731707317 | -4.0 | -1.9994752459511296 | -3.2792496751561853 | 5.057840616966581 | 5.000000045218455 | 3.0314739444967405 | -2.990024154143469 | -1.9869332355903087 | 2.8920687267780654 | 5.495726495726496 | -4.996247203889598 | 0.0097853128167327 | -0.9267461669505963 | -2.999992202733172 | -0.9999996266925827 | 5.000029640406394 | 5.000004358915434 | 1.032394840110365 | -0.9490760179336322 | 4.8 | 6.627358490566038 | 1.109069385898912 | 8.478252027439684 | 5.508916041181733 | 3.007303811125342 | -3.992355216398978 | 0.10657691504512352 | -2.820841934581943 | 0.3889479277364506 | 1.0576147213280351 | 5.333333333333333 | 6.650355878322875 | 4.018442716472051 | 4.045858088088095 | 2.865958378268715 | 4.124965065972493 | 1.021440428808576 | 0.14124293785310735 | 3.1626413079099787 | 4.085631018029087 | -3.8112769030671814 | -2.754523359438293 | 4.000004974261862 | 1.0050632911392405 | 1.0000001374107121 | -4.981967787114846 | 4.0080178654946 | 4.555555555555555 | -4.988533988533988 | -3.8666666666666667 | -4.996966018951338 | 5.0864864864864865 | 0.4612253912165573 | 3.042195540308748 | 3.0066722577144125 | 7.0 | -2.885100042571307 | nan | 3.0519068597580508 | 3.0023182674374556 | 2.12633127342143 | -4.98337835657981 | 0.25770308123249297 | -2.6558978211870774 | -0.6343248865187725 | 0.0068672196090518636 | -0.4565952552585212 | 2.087397610910184 | -0.534076404494382 | 4.045964865935808 | 3.072938421308039 | 0.3688962445455704 | -2.866056572379368 | 2.3059335465648205 | 6.666666666666666 | 0.05506255726135723 | -4.9999990087798025 | 5.000001038396827 | 5.745952365352978 | 0.10093357201486423 | -2.992917624228577 | 0.3445394230975974 | -1.1978125851070183 | -2.0 | -2.1554054054054053 | 4.0332876618809665 | -0.7999999999999998 | -1.3963906118292604 | -3.9999725130608574 | nan | 0.43918346747213377 | -2.701957895066857 | nan | 3.000022837502592 | 0.0036772232793499267 | -1.0666666666666664 | 5.021798365122616 | -1.4935382103926527 | 4.002631578947368 | -4.996637525218561 | -2.8735807264633544 | 2.0000010195043423 | -0.9898438936694987 | -1.0 | 5.177464019146546 | 1.9726221535799158 | 0.08433571495923314 | 0.7281644445084126 | -0.9757643549589858 | -2.981599810790891 | -1.9953979354283662 | 1.020594227180269 | -2.0 | -0.9475424180390458 | -2.9999996080891176 | -3.9895294976571334 | 3.0853063652587744 | nan | -4.958005249343832 | -3.9010431511144046 | 1.9811407356547713e-06 | 3.1914772201486783 | 0.2925024042900112 | -3.9945085156047377 | 0.12890625 | 1.1273158977179738 | 0.20324977833080787 | -4.078575331238902 | 5.000000000000701 | 3.7803030303030303 | -3.912318836680853 | 1.0214107640416483 | -3.9999149457353793 | 3.4820867601568706 | 6.2 | 0.007889529330214393 | 2.6395153342222737 | 1.0000026804887931 | 5.036763183182173 | 1.0000073006299628 | -0.5893703884040601 | 3.0000632360078474 | nan | -3.8659339461046014 | nan | 0.3093822136723724 | 1.1846123339282173 | -4.913093858632677 | 0.08382226165570436 | nan | -0.9323652773283049 | 3.4597961219982905 | nan | -3.9975031210986267 | 3.024160583941606 | -1.471994391868209 | -4.818140602229215 | nan | -4.708333333333333 | -2.9930727382151248 | 4.000032266576012 | 5.346534653465347 | nan | -1.9999958365581387 | -2.9938555067897363 | -1.976905311778291 | 3.447947017194808 | 1.9777777777777779 | 1.4615384615384617 | 5.2851784481004955 | -1.9674025729161797 | -3.9999993508 | 3.6 | -0.6298633017875921 | 5.082800126327369e-05 | 4.12 | 2.031605562579014 | -4.990844034436284 | 2.0242157735445967 | -0.9410019695193519 | nan | -3.8236434108527133 | nan | 8.621642346609043 | 1.9307316972361999 | 1.0646153846153845 | 1.073168222741841 | -4.997309972531318 | 2.037024235535533 | 1.012311965565242 | 3.0 | -2.9902532628848713 | 0.47498702804993437 | 0.6028119507908611 | 5.001364032080476 | 0.017529171681230703 | -3.9865449427287802 | -4.210876803551609 | 8.477704798488148 | -2.9957567187041168 | 4.056234670970714 | -0.8788244553582673 | 1.6446821152703506 | -3.986839067391406 | -2.6851520572450807 | 2.0332001229634185 | 9.0 | 2.222642066420664 | -4.999682585534596 | 4.306536402905481 | 5.342304103119092 | 0.016664903327161487 | 0.002010712115585615 | 1.0000014309259908 | 4.000000008625403 | -4.9543478260869565 | 2.9938271604938267 | 8.333333333333334 | -2.998969632748721 | 3.000000946979636 | 3.0232898307143685 | -1.911293010852254 | 2.5620689655172413 | 2.0 | -1.9398189251359528 | 5.002445158066089 | -0.8783463849936259 | 1.7362035753920457 | 4.000000082832667 | 4.015241682488482 | -4.860918816865953 | 4.007897085470625 | 0.0041235067928246764 | -3.94 | 2.725486859801301 | 1.4749653686018596 | 1.1875074360499702 | -0.6347177003096959 | nan | -2.1760124212238203 | nan | nan | nan | 3.008780487804878 | 1.038679509160507 | 2.0000040645881834 | -2.767296121677115 | -0.9784346532812791 | 5.00881537663432 | 1.2651566620979304 | 4.087536661166117 | 1.4285714285714284 | 7.193713450292398 | -3.99546485260771 | 2.002446769870726 | -4.963069492483992 | -0.09044453185372947 | 1.018195182608795 | nan | -4.987705730715843 | 2.0000234209753835 | nan | 5.021187214611873 | -3.967439399342663 | 2.3716814159292037 | 5.0000006718182455 | 5.362179825023716 | 0.48620876568065663 | -2.8783783783783785 | -2.6745250874129463 | 5.017379528269947 | -2.2442080404833695 | 1.0000001911745828 | -4.969083698682619 | nan | nan | 2.541538461538461 | -0.5512594837232399 | -4.941201585119513 | -3.973423056155201 | 0.11742223444622406 | -0.9423631123919308 | nan | 2.1689997240462493e-07 | 2.000000004585987 | 3.7914680355876285 | 4.018867924528302 | 0.16144667646130006 | 1.1288659793814433 | -0.17882202705796357 | 1.0523076923076924 | 4.1822694154946465 | nan | -4.999898122460164 | -0.9896472392638037 | 5.6034482758620685 | 3.0000000035643564 | 4.005443102568464 | 1.0387036605269293 | -3.75 | 6.416579223504722 | -0.8141283648513851 | -0.4556783740945449 | 1.0395647217693353 | -2.8846153846153846 | 2.03083031647787 | 7.833333333333333 | -2.9890719777554207 | 0.37735849056603776 | -3.7050369152376117 | 1.0055439617581474 | 4.020859545725449 | 3.1850068521056336 | -2.989316450352231 | 2.108097636617189 | -3.926981432856303 | 2.000000157308671 | 5.44028387856142 | nan | 3.571428571428571 | -4.999999469654548 | nan | 1.0000011992000002 | 4.014283496351498 | 2.159398852138927 | -4.981120320718714 | -4.960341066825302 | -2.9556178839206098 | -3.5957823443392596 | 9.064638546979157 | -2.999996496651811 | 3.0384615384615383 | 0.03352603349667285 | 0.2636871508379888 | 1.0004094072692535 | 5.005722813857801 | 0.05747803408684685 | -4.952349216927691 | 0.10084033613445378 | -4.999999512164489 | -3.7578125 | 2.488888888888889 | 0.10809451985922575 | 4.4052524056476425 | 0.2728284671676473 | 3.084207215685899 | 3.0 | 8.076923076923077 | -4.744749428155542 | 4.1644542772861355 | -4.992625648630254 | 3.0153846153846153 | nan | -4.838408628547972 | nan | -4.75211592716081 | 1.5534090909090907 | -4.599529780564263 | -3.7663397475260245 | 0.031777720429785924 | -0.98729003914977 | 3.43859649122807 | 1.7666666666666666 | 6.454545454545455 | -0.11449364240182636 | -2.0177468162229393 | 6.274580443449458 | -4.793846153846154 | 5.272850107639881 | 2.0651796534098796 | nan | 2.245780005464116 | nan | 4.0634696755994355 | 3.3684210526315788 | -3.9864708156165443 | -1.8125 | 3.8207547169811322 | 5.0501105379513636 | -4.069767441860465 | -1.9999992868890972 | 0.3492723492723493 | 0.0039316742137687315 | -0.974574567507295 | 4.6467065868263475 | nan | 1.0096000956428692 | 4.5 | 3.5428359919399033 | -3.6366813211588322 | 4.868577223386429 | 2.1333333333333337 | -1.9253294289897511 | 6.600985221674877 | 0.003522424028124931 | -4.911578011933477 | 3.124930594114381 | -4.999999337795909 | -0.9967416939829875 | 5.0584217089732775 | 5.115987756027538 | -4.9002008633796095 | -0.9576024476624405 | 3.9492003762935086 | -0.9789475341031391 | -0.9999984545666892 | -1.9677860827455522 | -0.8181818181818181 | 0.06875056554393146 | 7.44367816091954 | 5.02015873015873 | 4.014959846370297 | 0.00014838855328017084 | 1.004163033996627 | 1.6400000000000001 | 5.000006100675996 | 5.000000360319318 | 5.436572052401747 | -2.979316050744622 | -2.9999917298097056 | 2.00000630775552 | nan | 6.0 | 4.057087345352724 | -1.996035987917626 | 3.3052948627775063 | -3.999999744000018 | 1.769908293529202e-06 | -4.998041102097312 | 3.042784086512656 | 1.3853418964865885 | nan | -0.9150365609134139 | -0.9986043715377678 | 1.0000004368966988 | 1.4963680387409202 | 4.057762629377156 | 3.000000653972635 | -4.94773423869477 | 1.0066555359233982 | -3.9776164254081383 | -0.9999999073850616 | nan | 2.296875 | -3.9445487331249565 | 1.0000551393369594 | -0.7431528534094702 | -2.937968324770058 | 0.0030373082157892283 |
| 4.014332180059522 | 3.0000001701745393 | 5.000199994666809 | -4.0 | -4.914008377881295 | -2.9110807113543093 | 1.0000008488057885 | -4.0 | 3.0002440512507627 | -3.9996755797701677 | 3.5127998678663515 | 0.0745625238376833 | 3.0145815531674556 | -2.0 | nan | 3.000000000210138 | -2.973945104140208 | -4.643728013958459 | 2.0 | 0.0030120481927710845 | 0.0 | -2.999782778255571 | 2.009177802601572 | 0.011713004484304932 | 2.0000000184722064 | -2.993999368354564 | 3.406070614138555 | 1.0000000799999273 | -3.823227689981542 | -3.976976208749041 | nan | 5.226858152282151 | 3.00000007249067 | -4.967213114754099 | -2.9622605268816744 | -2.024796195652174 | -0.9783393501805054 | nan | nan | 2.1666666666666665 | nan | nan | nan | -2.9947028471476753 | nan | nan | -1.982736635605689 | 0.06264408774230201 | 3.0004908552014227 | 5.0 | 4.024390243902439 | 0.2386572751514333 | -0.9999998826231847 | 4.00036469730124 | nan | 5.062835745533806 | 5.307033221284716 | 1.92987007595447 | 0.0005391150293547257 | -3.9999999633814016 | 4.001591756087602 | -1.0 | nan | 2.000000060191083 | -4.731108755854901 | 4.191446028513238 | -5.0 | -2.965920298727399 | 6.689639571290555 | 2.003084504471619 | 3.0 | 3.0236686390532546 | nan | 5.029992084960392 | -1.986486321818553 | 5.045894974230131 | 5.000801117469186 | -2.999633923123856 | -0.9999998009222255 | 1.6278370191203964 | -3.7777777777777777 | 2.6436921845810106 | 3.1757325161266876 | 1.9 | 4.018454440599769 | 3.0 | 4.000001041200549 | -3.769542221411636 | -1.488888888888889 | -3.0 | nan | 0.00596472463874397 | -1.9640136062895355 | 0.021621621621621623 | 2.0 | 1.0025142436272931 | nan | -0.9345163113628736 | 1.0640085045076346 | 3.0212714917446553 | -2.998724182330219 | 4.997373901834585 | 1.2791237147416633e-06 | nan | 3.5 | 5.0 | 2.1038618460138445 | -4.955735344169907 | -4.995527395567564 | nan | 5.0 | -1.66494946559183 | -1.9726027397260273 | nan | 1.165312471887919 | 4.0130662020905925 | -1.8421052631578947 | 0.6393442622950818 | nan | -2.984168865435356 | 2.607594936708861 | -0.5803617162023037 | -4.7844961240310075 | 2.0003354220229808 | -2.0548914030562972 | 6.701418864545888 | nan | 3.000000292025585 | 4.852227251421109 | 3.000998628906143 | 0.002621358648730432 | 5.000000845908567 | -2.7581382835620123 | 0.009109311740890687 | -4.99999761869573 | 5.199999999999999 | -4.801112639986891 | 1.3894921309753695e-06 | 3.0533174822543705 | -4.0 | -4.999999586000002 | -0.9977164760988311 | -2.502184614149775 | -1.8928666316510339 | 4.05343380715876 | -1.0 | nan | 3.0437041219649914 | 4.000000033720287 | nan | 4.004050670645006 | -0.999999646126 | -4.991372549019608 | -1.9644970414201184 | 2.0052476521769784 | nan | 2.0115090488256895 | -3.994440393388677 | 2.131087260013232 | 0.2639305972639306 | 1.005956650282429 | 8.806082923765118 | -0.9999725078118101 | nan | 1.0164609053497942 | -3.935561297047304 | -1.949358596735534 | -1.9635147340873393 | 2.0 | 4.315337423312883 | -0.973012446024892 | 0.049541882446673666 | nan | 1.0 | 5.170968039136484 | -2.999982449208429 | 7.011218203887857 | -2.0896010145351673 | 0.03492063492063492 | nan | -2.913250714966635 | -2.881606502311887 | 1.0273158352827307 | 1.4479134873783153 | -0.5200420601138229 | -4.994440393388677 | 0.13559322033898305 | 7.0 | 4.000012519465585 | -3.898043768584724 | 4.1484162895927605 | 3.55 | -3.8960009746486404 | 2.0014078558355624 | -3.9765934040184114 | -3.0 | 4.0 | 3.2945323653584677 | 4.000000070988538 | -3.904891859862641 | -1.5 | nan | -0.9794823239812153 | 5.00090477626629 | 1.003089817448513 | -2.989329268292683 | 1.0103546652944484 | -2.9998446996475003 | nan | 2.0060909601449004 | 5.008333907284569 | 5.010480265371857 | 0.0 | -1.9999990701397663 | 0.0007140529161563777 | -0.2604771876717744 | 5.0005986777343265 | 4.953771001716463 | -0.9999996431749999 | 3.0612480739599386 | 4.489425948241241 | 4.29201295238684 | 5.035534498075215 | 1.6610169491525424 | nan | -0.9123215497223616 | -0.9833679833679834 | nan | -4.978446031499436 | -4.0 | 5.001636466330095 | 5.00879076745164 | -4.9999998337501435 | 4.000004480199739 | 4.012578616352202 | 0.022600864902903695 | 2.000071907278156 | 2.0000007170368366 | 4.513972780435494 | -1.9831954498448812 | 1.8636562911003947 | 6.5777777777777775 | -3.9002618517017824 | -0.9999957446629574 | -1.0 | -0.863013698630137 | 3.0 | -1.597062068122936 | -1.45 | -0.9791851344831068 | 0.0 | 3.0 | 1.000000235916 | -2.5015724768893546 | 4.002271430069075 | -2.995727447515387 | 3.0636846095526913 | -0.9142623496089824 | -5.0 | 2.000140798584527 | -1.9598907103825136 | -1.5543333132758388 | -4.914851053760501 | -4.96238934301224 | 5.002857324274557 | -1.9188884716485166 | 1.0 | -0.5711346199434066 | nan | 2.053835800807537 | 3.0 | -1.9743059806933714 | 6.5106149274994145 | -3.945887445887446 | 4.517857142857142 | 3.0 | 0.02210759027266028 | -1.0 | 4.000235094622357 | -3.8527431576542854 | -2.0 | 4.080182886430314 | -1.9999933342682865 | -2.0 | 1.031419372345965 | -5.0 | -4.0 | 4.5 | 0.05441589002486058 | -1.8403378808386028 | -1.999999123340618 | 5.3559322033898304 | 2.0252633456790123 | 1.1167478240814752 | nan | -0.9999279379157427 | -3.249838736288117 | -0.8004638885957375 | -0.9556213017751479 | nan | 1.65 | 4.075915958423517 | -4.341987156843424 | 2.0313630880579012 | 2.0013562307334443 | nan | -0.9921806413299517 | 5.002241242613597 | -2.045623342175066 | 2.0 | 4.011058541868551 | -0.07603824210417742 | 3.1634804879997924 | -4.972120991253644 | 5.61133194666672 | -0.9937789165277882 | 0.04878048780487805 | -3.995295704971516 | -3.9992960720822186 | nan | 2.3713031710021673 | -2.9999996319505833 | 0.36585365853658536 | -0.9940828402366864 | 0.0007040270346381301 | 5.037950833197305 | 3.0 | 3.010174165184982 | -0.9968996685360137 | -4.0 | -2.9822093489599144 | 2.1773700305810397 | -0.9968928286900456 | 9.495048419287287e-06 | -0.03641056262854869 | 1.0097412707294195 | -4.0 | 3.012809574654253 | 2.018355605203144 | 3.596145063984615 | -1.9999970810784542 | 5.428571428571429 | 5.015877426269202 | 2.000000759039778 | -2.0 | -0.9347739095638256 | 1.0039698216669921 | 1.002427330188895 | 2.032967032967033 | -0.9830508474576272 | -2.9025809308840613 | 1.0000950763210819 | 3.0 | -2.9828489471450705 | -2.9999995436942672 | -0.3703776360961255 | 5.039594806918061 | nan | 3.001909833136019 | 6.110635536348082 | 7.364022827646342 | 5.5 | nan | -0.9772196261682243 | 2.0240143546037195 | 2.545357876647434 | -3.921477662733442 | -0.4919188998136367 | nan | nan | 4.014376321353065 | 2.5 | -0.9986630178433701 | 0.0021260157859022133 | 6.9999799011438375 | 0.03595890410958904 | 4.019981412639405 | 5.000197538116379 | nan | nan | 3.489795918367347 | 1.0031184970209348 | -3.0 | 3.05218545751634 | 6.295454545454546 | -1.638267785995109 | -3.9923742484235225 | 1.0007264356935546 | -3.9446614344216564 | -2.690230821410165 | -1.0420450576318547 | -0.9995285260904433 | 5.00019481096006 | nan | -5.0 | -1.9593582887700536 | 1.4 | 0.0 | nan | -0.6823798359752542 | 4.0 | nan | -4.805758780057227 | -4.9999982509672805 | nan | nan | 0.0 | 1.0 | 2.0845900238487993 | 2.0094540013104534 | 0.0 | 3.0354330708661417 | -0.3648293963254593 | -3.0 | 2.016638935108153 | -3.0 | nan | nan | -0.8221462369706506 | -2.0 | -1.997166381300512 | nan | -4.99999817443031 | 1.4863098270880637 | nan | 3.0 | 2.019230769230769 | -0.48241496424545904 | -2.999999526101266 | 5.023236661227851 | -3.5 | -2.9757685494358723 | 1.0386825817860301 | 2.2542372881355934 | 1.0 | 5.016925359033446 | 0.0 | -4.977046500896334 | 1.2917398945518448 | -1.9994500534393922 | 3.957612257783845 | -4.94717095245843 | 4.06508875739645 | 1.5214180206794683 | 1.0215462610899873 | -3.9813846285942724 | 1.0032170671080405 | -3.986668603833773 | 5.012096774193548 | 2.009233283707674 | -4.0 | -4.830233478432924 | 4.0 | -4.9999999986 | -0.9643451782741086 | 3.47741935483871 | 5.000013258194785 | 6.247274908825547 | 2.0000130812825874 | -4.955016132188809 | 2.0 | nan | -1.9944272445820432 | -4.999984249998213 | -2.774306057929289 | nan | 4.0235140431090795 | -4.995897435897436 | 4.025380710659898 | -0.6102564102564103 | 2.0 | -1.9999999499604144 | 0.07366516502421983 | -4.964726631393298 | -4.994666127401415 | 1.4378698224852071 | -4.020349718323518 | -0.9818548387096774 | -4.993764703347986 | -3.984491715939266 | 5.000856945934465 | 2.0558094737440387 | -3.9523809523809526 | -4.945085192902562 | 3.260707635009311 | -0.9911508201920406 | 0.008913530896968221 | 5.0040714867355245 | 1.6207689157592062 | nan | -1.9839015667198965 | 5.002025562105994 | -4.999996618695112 | 3.1724137931034484 | -2.9999992986879107 | 3.4166666666666665 | 1.0 | 5.9903067634528995 | 1.0209712670753093 | -4.931965646767802 | -2.8852097130242824 | -1.9851190476190477 | 1.1824471724316024 | 3.0 | 5.097913031915132 | 5.015599933617303 | 4.0 | 0.017851266864100032 | 2.0223542230501566 | 2.5112486006999353 | 1.0457219251336898 | 3.0 | -1.0 | -3.0 | -4.0 | -3.9976384422927 | -2.110454640687199 | 5.0 | -2.061873340780506 | -3.0 | 2.000009751236181 | -3.5172835246268983 | 3.0710059171597632 | 0.6833036244800951 | 0.09117428857149752 | -4.444444444444445 | 5.000000009706385 | -3.9481268011527377 | 4.12068769682704 | -3.6377227003020614 | 2.8905540986870895 | nan | -2.99358782541786 | -1.9322033898305084 | nan | -2.9943210903756343 | -2.999999980844 | 1.049343103303179 | 4.3 | 8.0 | -0.7541899441340782 | -4.997847377616337 | 1.013312463071673 | 0.028708133971291863 | nan | 4.64367816091954 | nan | 5.035180299032541 | 3.0131987956970816 | 4.00840038001893 | -4.999691120399217 | -4.986150074438489 | -4.880430960458991 | 5.55163358446827 | nan | -0.9999946224993896 | 5.114266912656652 | nan | -4.979729769363923 | nan | 3.000842193915149 | 0.0 | 4.0 | -3.5999999999999996 | 7.0 | -1.9171258850166912 | 0.8483355442259717 | 1.0102144280212864 | -4.925558312655087 | -4.999999137262044 | -4.827661460087946 | 3.3870967741935485 | 0.20543478260869563 | 3.234918301055039 | 0.00045245482180650753 | 3.6838766165082295 | 5.000000370367871 | -1.293131777850805 | 1.1690140845070423 | -0.9830263163689587 | nan | 1.0581042201450757 | -4.661971830985916 | -1.0 | 5.0 | -3.609271523178808 | 2.9989106753812638 | 5.0900787450916125 | 9.264972734782196 | 3.0173503064592175 | 4.011560693641618 | nan | -1.9380492724391298 | 6.468070652173913 | -0.982362365934652 | -2.911509396178546 | -1.5097817867276533 | 5.000000446326503 | -0.9999995569162236 | 2.0601327692664553 | -2.0 | 5.01123595505618 | -3.88757156590701 | 2.7316028823605634 | -2.9179690610116356 | nan | nan | 1.0146252285191957 | 5.05758058027655 | -3.9484342411366757 | 6.453521165157053 | 3.0031173670915905 | -4.662616633440747 | 4.002598324037696 | nan | -3.0 | 4.067985805132532 | 3.0 | -0.9595030692846908 | 5.0 | nan | 4.000729438939882 | -0.9989391597382989 | nan | -3.9692524787403913 | 4.001609466333713 | -4.931301932467718 | 0.8105264263701915 | 3.0090760996043753 | nan | 3.0000032642528094 | 5.0 | nan | -4.689739219050826 | 4.50586268652216 | -2.9937630059581304 | 3.0 | -2.9842778591362493 | 5.620689655172415 | -1.999718938932063 | 5.0 | 1.2829656992510543 | 4.030514939605848 | 5.000245427600711 | 3.0 | 3.000965793769513 | 3.0182969897951866 | 4.0 | -3.9859256252276625 | -1.8875647541913716 | 0.008165136781068512 | -3.944960900374922 | 3.4042119565217392 | -1.1666666666666667 | 1.525358850524693 | 4.00510576221736 | 4.272108843537415 | 5.0040154600798274 | 2.0 | -2.9939024390243905 | -4.0 | -1.999161851171943 | -3.8885859810512065 | 5.013496143958869 | 5.000000398481537 | 3.015476710665384 | -2.995906138728688 | -1.9888316155743482 | 2.8942450799348114 | 5.944444444444445 | -4.983726594129554 | 0.01807795079701465 | -1.2106757524134015 | -2.999998861440433 | -0.9999993275937131 | 5.000118330461481 | 5.00000107030931 | 1.0100652554814693 | nan | 4.666666666666667 | 6.363636363636363 | 1.0755125304502688 | 9.38078351240305 | -0.9942959610304045 | 3.001352544918504 | nan | 0.012066801996459351 | -2.9227788163411117 | 0.07651434643995748 | 1.0278215574473781 | 4.0 | 6.879867417833061 | 4.0053753873539275 | 4.055890988117194 | 1.9234092464079753 | 5.004754572387347 | nan | 0.0 | 3.642941674536509 | nan | -3.988385948455967 | -2.90359168241966 | 4.000005165822326 | 1.0202531645569621 | 1.0000001128214269 | -4.9866946778711485 | 4.005806476170556 | 2.2222222222222223 | -4.923177723177723 | -3.2 | -4.997192356477651 | 5.043243243243243 | 0.2587708228167592 | 2.648616125150421 | 3.004924551688414 | nan | -3.0 | nan | 3.0112162283924753 | 3.001745009671146 | 2.021866117911604 | -4.984054307498736 | 0.03361344537815126 | -4.012772351615327 | -0.7432364427802363 | 0.007316051191556205 | -0.6648506600856996 | 2.134862808479526 | nan | 4.012679963016774 | 3.381672479754764 | 0.0711084137183117 | -2.9417637271214643 | 2.0000196462444166 | 2.0 | 0.036750276985416046 | -4.9999993260559075 | 5.0000002295596255 | 0.5204769982068829 | 0.06074067562202574 | nan | 0.1899643998165037 | -1.1850533427107741 | -2.0 | -2.8986486486486487 | 4.009197161927707 | -3.0 | -1.9449546092976933 | -3.9999924534496594 | nan | 0.42113861908244754 | -3.328985938896117 | nan | 3.000018900002145 | 0.0018631547368009147 | -1.1666666666666665 | 5.0 | -0.21947624115096787 | 4.005263157894737 | -5.0 | -2.946615143925931 | 2.000000193199999 | -0.9993001090914043 | -1.0 | 5.259334815409075 | 1.9967252527038721 | 0.031936327998458705 | 0.5861311130588428 | -0.9947800149142431 | -2.991049301707863 | -2.0 | 1.023308465270404 | -2.0 | -0.9724814251158596 | -2.9999994903553002 | -3.9949148629679003 | 3.276923076923077 | nan | -4.916010498687664 | -3.9225902069201393 | 2.1719911012750012e-07 | 3.0273604984731177 | 0.16018375451762934 | -3.9653220368289097 | -1.7278911564625852 | 1.0083075696735582 | 0.025263531780203578 | -4.937269498702363 | 5.00000000000059 | 4.011363636363637 | -3.5481233982032974 | 1.0114386273647162 | -3.9982287449392713 | 4.355289713267225 | 5.0 | 0.0 | 2.509518477043673 | 1.0000001109986385 | 5.003237184142261 | 1.0000019406814553 | -0.9770328476721974 | 3.0002213260274666 | nan | -4.574725229789226 | -4.98559670781893 | 0.0 | 1.4142011834319526 | -4.930475086906141 | 0.10086923539331555 | nan | -1.0 | 3.5614248959942816 | nan | -3.986267166042447 | 3.0179619982721784 | -1.6682790045566072 | -4.721417808025699 | nan | -4.666666666666667 | -2.9983472661613337 | 4.000021301409079 | 5.333333333333333 | nan | -1.9999996883325053 | -2.9960903393413174 | -1.9953810623556583 | 4.238760858961854 | 2.022222222222222 | 1.7636363636363637 | 5.282325188552432 | -1.9985344004436472 | -3.999999520849 | 2.6 | -0.7504353312302838 | 2.949256963677371e-06 | 4.0 | 2.0126422250316054 | -4.9973746902219895 | 2.0102617055367635 | -0.9371751378511674 | nan | -3.484126984126984 | nan | 9.046269773704875 | 2.768374654216246 | 1.0287179487179487 | 1.0107721643256198 | -5.0 | 2.0415372921800112 | 1.000466357472304 | 3.875 | -2.992757795268278 | 0.07743491133290602 | 2.4573813708260097 | 5.006897258464256 | nan | -3.9997130403525207 | -4.4306326304106545 | 4.501426348547718 | -2.9976617637554948 | 4.038167306803417 | -0.9677799796517217 | 1.3416518122400476 | -3.9837276309436667 | -2.3989266547406083 | 2.030740854595758 | 9.0 | 2.013549815498155 | -4.999869678144955 | 4.226869935376597 | 5.176810489835278 | 0.020389372970652193 | 0.019855595666261817 | 1.000000861212839 | 4.000000094095298 | -4.928260869565217 | 3.1944444444444446 | 6.5 | -2.999878975970022 | 3.0000000066611467 | 3.023517174214407 | nan | 2.275862068965517 | 2.0 | -1.9579972448993497 | 5.000307728888563 | -0.992958173981667 | 3.382197292623763 | 4.000000007268733 | 4.01230018577874 | -4.958612519897827 | 4.012765151856627 | 8.288131360498936e-05 | -3.9607365439093485 | 2.18818094062787 | 1.398682220461724 | 1.012849494348602 | -0.6664813785436354 | nan | -1.9735277326996643 | nan | nan | nan | 3.0 | 1.0299754303996576 | 2.00000039368501 | -2.834177687573163 | -0.9871723368655884 | 5.005762464139166 | 1.0290261495540554 | 4.05500671362147 | 1.5789473684210527 | nan | -3.997732426303855 | nan | -4.973594687126054 | 2.615915145040476 | 1.0935541827973165 | nan | -4.976331360946745 | 2.0105537160720384 | nan | 5.004566210045662 | -4.0 | 2.588495575221239 | 5.0000001109986405 | 5.017555123278745 | 0.06135902636916857 | -2.8783783783783785 | -3.999939110489426 | 5.0027084979122 | -2.8480582153443845 | 1.0000001000417107 | -4.968778941229315 | nan | nan | 4.436483516483516 | -0.5797938689130697 | -4.964455265213671 | -3.9828840313461655 | 0.04303647451291313 | -0.9769452449567724 | nan | 1.6044527049623713e-06 | 2.0000002281528664 | 3.9008442827808647 | nan | -0.8668799516676051 | 1.0 | -0.9418395876182325 | 1.0215384615384615 | 4.184541450388335 | nan | -4.99999898627934 | -1.0 | 5.875 | 3.0000000347524742 | 4.02041163463174 | 1.0281339861661616 | -3.75 | 5.041972717733473 | -0.9678983237651457 | -0.7163358329826246 | 1.1123190841244623 | -3.0 | 2.0100432091556697 | 7.5 | -2.9987947046999035 | 0.48484848484848486 | -3.664315853473129 | 1.0139123416678004 | 4.011919740414542 | nan | -2.984317424335987 | 2.009319836057283 | -3.9456134467362975 | 2.000000148546929 | 5.040681389608111 | nan | 0.7142857142857135 | -4.99999832572919 | 5.000003893773144 | 1.000000943664 | 4.019251668995498 | 2.070646707295426 | -4.972637360836083 | -4.964497041420119 | -2.915064007921718 | -2.9595628415300546 | 9.000470671294934 | -2.9999993621954197 | 3.01105216622458 | 0.01526714505067165 | 0.2547486033519553 | 1.0041659332277182 | 5.008560475123713 | 0.10727851497861268 | -4.954181939353549 | 0.012605042016806723 | -4.99999679109459 | -3.687074829931973 | 2.511111111111111 | nan | 4.027870382837911 | 1.523922232715345 | 3.0647123951201483 | 3.0 | 7.915254237288136 | -4.824079850280723 | 4.134969325153374 | -4.9998667035569335 | 3.061224489795918 | nan | -5.0 | nan | -4.759612021730514 | -0.7242424242424242 | -4.325068870523416 | -3.860187680559203 | 0.03265405139149213 | nan | 2.9230769230769234 | -0.11111111111111116 | 5.833333333333333 | -0.07311719111441484 | -1.7562370043206794 | 6.318140526140179 | -4.849230769230769 | 4.873974901059141 | 2.023631285600805 | nan | 2.0400921209606584 | nan | 4.6016057285450795 | 3.013520038628682 | nan | -1.8125 | 3.9696969696969697 | 5.016422781345405 | -4.333333333333333 | -1.999999360861396 | 0.0997920997920998 | 0.35281607429633555 | -0.9719785714927562 | 4.886227544910179 | nan | 1.0106533386188545 | 4.5 | 3.1587523121661776 | -3.9888400405816706 | 6.120053778118768 | -0.2666666666666666 | -1.9648609077598829 | 6.672413793103448 | 0.0 | -4.964897803732385 | 3.0688506385341476 | -4.99999994448164 | -0.9917972533432452 | 5.105875930803185 | 5.087867829759759 | -4.875759545683131 | -0.9831147247198939 | 3.4148635936030103 | -0.9822854724609722 | nan | -1.981998105063691 | -2.0 | 0.060493189110930136 | nan | 5.007301587301587 | 4.000332636138665 | 0.02534155026306075 | 1.0018341904417312 | 0.72 | 5.000001360405489 | nan | 6.0 | -2.962274774774775 | -2.9999891503246827 | 2.0000015809222136 | nan | 3.2 | nan | -1.9972414976883923 | 4.011218203887857 | -3.999976453361665 | 0.0 | nan | 3.0333350373906054 | 1.2206271250472231 | nan | -0.9879518771953015 | -0.9909430544548851 | 1.0000002546309594 | 1.3453947368421053 | nan | 3.0000011959075583 | -4.919734660033168 | 1.0001775833846345 | -3.992534699833346 | -0.9999992034737345 | nan | 1.945578231292517 | -3.9517197715489085 | 1.0000042950281358 | -0.8399511863782365 | -2.9401985807400983 | 0.01680346541349943 |
| 4.169006024348916 | 3.000002285408843 | 5.0003199914668945 | -3.9114735002912058 | -4.987120822992571 | -2.982216142270862 | 1.0000000787689365 | -4.0 | 3.0018791946308725 | -3.9993645121177077 | 4.682103611021454 | 0.06385874794864707 | 3.0036942285554034 | -2.0 | -1.999999896 | 3.000000001260828 | -2.9959121793243586 | -4.916208128465058 | 2.0 | nan | 0.0 | -2.9998379739447287 | 2.0251545218458538 | nan | 2.0000000031986502 | -2.9962101273818296 | 3.377044132185864 | 1.0000000451999589 | -3.8043883124942384 | -4.0 | 2.0000000433333334 | 4.982943069338628 | 3.0000000348451152 | -4.950819672131147 | -2.951532219652252 | -1.4951612903225806 | -0.9422382671480144 | nan | nan | 2.357142857142857 | nan | 2.0007014949367323 | nan | -2.97938533222431 | nan | -1.8808198937407652 | 4.113961407491487 | 0.012054548544095605 | 3.0004704029013634 | 5.0 | 4.336585365853659 | 0.04424867479804628 | -0.9999999954855071 | 4.025164113785558 | nan | 5.076299502615085 | 5.245667631455189 | 1.401507935990236 | 9.104941622458885e-05 | -3.9999994326620447 | 4.00000830481437 | -0.9849939975990396 | nan | 2.000000144458599 | -4.704619531152062 | 4.171079429735234 | -5.0 | -2.8394559414695144 | 6.720698144822789 | 2.0276327797043256 | 6.199999999999999 | 3.001812415043045 | -4.0 | 5.046964446167872 | -1.9483963318933801 | 5.028504563108417 | 5.001902897307874 | -2.9976388041488713 | -0.9999999150000014 | 1.5339588807688025 | -3.4222222222222225 | 2.8042040894016793 | 3.202472002779588 | 1.6923076923076918 | 4.03921568627451 | 3.0 | 4.110684555258322 | -2.761480941090644 | -1.1111111111111112 | -3.0 | nan | 0.052309891242923796 | -1.9204133787568327 | 0.06308426073131955 | 2.0747478742337355 | 1.0010570950551 | nan | -0.9990755505965786 | 1.018691412316619 | 3.0940480393826832 | -2.9987910166388896 | 4.831406438903602 | 1.6713017919222971e-06 | nan | 3.5555555555555554 | 5.000704991540101 | 2.750928819783568 | -4.868915369539905 | -4.989217875516623 | 3.0000000073491657 | 5.0 | -1.868650389572049 | -1.7534246575342465 | nan | 1.2653448505435092 | 4.049128919860627 | -0.7142857142857144 | 2.84 | nan | -3.0 | 2.9327731092436977 | -0.6149868752707088 | -4.95968992248062 | 2.0000999394367938 | -2.2317909369087854 | 6.509405231030574 | 2.007712521455794 | 3.000000120964894 | 4.883977595316459 | 3.0009189658152047 | 0.0008506022322014576 | 5.000000737438206 | -2.8260715236489484 | 0.002891844997108155 | -4.998021175939638 | 6.769230769230769 | -4.8050710131610375 | 0.0 | 3.0178324878736165 | -4.0 | -4.999999622243245 | -0.9989621353533628 | -2.3332633289990596 | -1.9625743608575474 | 4.009864702860079 | -0.9994011976047904 | -1.9525644128576305 | 3.044254658385093 | 4.000000166721943 | 3.0 | 4.008031757672112 | -0.9999999475 | -4.996820349761526 | -1.9826996745891172 | 2.0016981800580256 | nan | 2.1777863676384706 | -3.9391032832248896 | 2.0086970208305277 | 0.8385051718385053 | 1.002789400837733 | 8.240457379663237 | -0.99982090303517 | nan | nan | -3.793179297695235 | -1.915600443158209 | -1.8448472760568724 | 2.0 | 3.98159509202454 | -0.9212598425196851 | 0.1874117263607721 | 4.000675345477506 | 1.0173076923076922 | 5.140794248014003 | -2.988132955055553 | nan | -2.5013974988354177 | 0.0031746031746031746 | nan | -2.7397521448999047 | -2.651744800845964 | 1.027894198126051 | 1.4712226856321506 | -0.5723770255547318 | -4.93910328322489 | 0.27941176470588236 | 7.0 | 4.00005185276079 | -3.9472740053245343 | 4.273303167420814 | 2.705555555555555 | -3.9861377220508323 | nan | -3.9933316044308307 | 0.26742033712408375 | 4.000000171374671 | 3.2418621210804095 | 4.000000092284584 | -3.7633795938997867 | -2.25 | nan | -0.9730283792021928 | 5.000380958427912 | 1.0013968704059308 | -2.9786585365853657 | 1.008148671383892 | -2.991413184676385 | 0.0 | 2.00188201710076 | 5.00353679196029 | 5.118047209268785 | 0.0 | -1.9999991838432405 | 0.000167850503815205 | -0.2688053319442799 | 5.000312353600518 | 4.626603825481938 | -0.9999998385260416 | 3.102272727272727 | 5.078760131130234 | 4.449422921151843 | 5.068700029612081 | 0.4117647058823529 | nan | -0.9095285710744703 | -0.8752598752598753 | 4.10640746124031 | -4.936441352414247 | -3.9880239520958085 | 5.008700787769641 | 5.4265065142831 | -4.999999971212146 | 4.00000260653831 | 4.779874213836478 | -0.007666213665098454 | 2.0012995172865846 | nan | 4.615882073300776 | -1.971302998965874 | 1.0657606313020604 | 8.911111111111111 | -3.9843496758508916 | -0.9996309392715544 | -1.0 | -0.9315068493150684 | 3.0 | 1.134053921973761 | -1.4130434782608696 | -0.9622863811308475 | 0.0 | 3.00000116603036 | 1.0000005002 | -2.8159317095750906 | 4.0000873626949645 | -2.9972446495388985 | 3.0 | -0.99413587162897 | -5.0 | 2.005878381214609 | -1.9272859744990893 | -1.8734231150156366 | -4.793865885358869 | -4.841733905398979 | 5.00103181154359 | -1.8141194141945174 | 1.0653887552235026 | -1.6383593132208447 | nan | 2.265753089440842 | 3.0 | -1.9451184712019562 | nan | -3.9689021092482424 | 4.24076354679803 | 3.0 | 0.051584377302873984 | -1.0 | 4.003485479452998 | -3.663922487869839 | -2.0 | 4.02625462242263 | -1.9999874745222066 | -2.0 | 1.0236719525542801 | -4.991376515593594 | -4.0 | 4.928571428571429 | 4.0 | -1.9533730094484416 | -1.999999941324317 | 4.882352941176471 | 2.000527160493827 | 1.1153557023542024 | nan | -0.9999279379157427 | -2.4980126735188866 | -0.08742108037987772 | -0.9947481155002678 | nan | 2.75 | 4.17771304882939 | -1.121478264426241 | 2.118214716525935 | 2.000824039593484 | 5.0261950727299665 | -0.9938476950336751 | 5.003540421566963 | -2.763395225464191 | 2.0090556274256146 | 4.082426038875905 | -0.23825927635012567 | 2.7373918163230497 | -4.567055393586005 | 5.5559908548242145 | -0.9995427655434116 | 0.10365853658536585 | -3.9971454188351427 | nan | nan | 1.3523573229318446 | -2.999991791172196 | 0.29268292682926833 | -0.9856654446595543 | nan | 5.0349241662634885 | 3.0 | 3.0601992648524505 | -0.9944125150570551 | -4.0 | -2.9864088466053005 | 2.229357798165138 | -0.9952991260010443 | 7.31118728285121e-05 | -0.28946473249650806 | 0.35698155924308983 | -3.9380863039399623 | 3.003474591101496 | 2.0503090436917075 | 0.37999908631939494 | -1.9999974194818009 | 4.209424083769633 | 5.0865319731671494 | 2.0000009952349256 | -1.9693486590038314 | -0.968187274909964 | 1.0170959153507488 | 1.006216265551657 | 2.010989010989011 | -0.30508474576271194 | -2.7478929854078182 | 1.000108027703998 | 3.0 | -2.7881409225312255 | -2.99999995910828 | -1.4330684827847144 | 5.2821733272171905 | nan | 3.0004841691391078 | 5.88817458813873 | 6.114163234832305 | 5.928571428571429 | nan | -0.9299065420560748 | 2.025145143326634 | 2.3712154194471387 | -3.7769100081995806 | -0.5744047619047619 | 1.0124814255549561 | nan | 4.0625792811839325 | 6.4 | -0.9996766390426649 | 0.0009070918129359302 | 6.999998869919538 | 0.206763698630137 | 4.010825881643726 | 5.0000665014239125 | -4.994689703475008 | nan | 3.43979057591623 | 1.0033796491316207 | -3.0 | 3.198733660130719 | 2.0 | -1.6449907903046892 | -3.9647064574473285 | 1.0016755280716916 | -3.879968711626414 | -1.9341558142979223 | -1.1205029689137271 | -0.9815681540065021 | 5.037556359444576 | nan | -4.999810657332998 | -1.8631016042780748 | 4.6 | 0.0 | nan | -0.6464808459615315 | 4.011865680543319 | nan | -4.96357977126073 | -4.999997533415396 | -4.999999970603338 | nan | 0.0 | 2.2 | 2.109205815553592 | 2.0178381058633943 | 0.005402874936205952 | 3.011985018726592 | -0.3359580052493438 | -2.9999996992116142 | 2.0615640599001663 | -3.0 | nan | nan | -0.8130648426328831 | -1.981093215800964 | -1.9918033796640868 | nan | -4.999999676243391 | nan | nan | 10.0 | 2.0153618570778336 | -0.4945190153709119 | -2.9999999046297705 | 5.00713900969609 | -4.0 | -2.9914589996836667 | 1.1408045977011494 | 3.911764705882353 | 1.0 | 5.022572993886559 | 0.0 | -4.931180007387445 | 0.09226713532513164 | -1.9980722746767636 | 4.04551774460176 | -4.995521620459186 | 3.0192775054578407 | 1.171344165435746 | 1.0937896070975919 | -3.966613682359324 | 1.0017287164008268 | -3.9946543819894114 | 5.07258064516129 | 2.0160501101467125 | -3.9998680412860557 | -4.788996300274496 | 4.001942148760331 | -4.9999989527999995 | -0.7462882685586572 | 4.32258064516129 | 5.000015357853247 | 6.542928136835389 | 2.0045678269041365 | -4.805976414224051 | 2.0 | nan | -1.9430340557275543 | -5.0 | -2.1127643960469964 | nan | 4.037230568256041 | -4.988205128205128 | 4.101522842639594 | -0.4482758620689655 | 2.0 | -1.9999998793475795 | 0.030225691814559322 | -4.9717813051146384 | -4.998221670055593 | 2.6069118919141574 | -3.216872039348959 | -0.936491935483871 | -4.997739578799746 | -3.9988934593351253 | 5.001819989937006 | 2.0550771634291327 | -3.870748299319728 | -4.96048857782789 | 3.6256983240223466 | -0.9557055430985602 | 0.05647330456396342 | 5.00266406186755 | 1.0952502569154339 | nan | -1.911508657801588 | 5.004016328507284 | -4.999999145997151 | nan | -2.999999682635356 | 3.8166666666666664 | 1.0 | 6.0 | 1.0056105675268332 | -4.982501247780343 | -2.4834437086092715 | -1.8125 | 1.738525909673921 | 3.0000006549429474 | 5.068574682908589 | 5.041101952757648 | 4.0 | 0.037430075682790394 | 2.006150926153738 | 2.041602918354331 | 1.1540106951871658 | 3.0 | 0.13636363636363624 | -3.0 | -3.7372700981894518 | nan | -1.794653670063506 | 5.0 | 2.530264189214642 | -2.967895773316706 | 2.0779225613533967 | -3.741026189722072 | 3.0103801952465297 | 2.620320855614973 | 0.13495652250463824 | -4.444444444444445 | 5.000000000005296 | -3.8962536023054755 | 4.106207153377833 | -3.7583163649326887 | 2.952578308842318 | nan | -2.9589620826743044 | -1.9830508474576272 | nan | -2.9983079613424994 | -2.99999999485 | 1.0305885961241907 | 6.4 | 8.0 | -0.7541899441340782 | -4.993561015501498 | 1.0027293484328157 | 0.16507177033492823 | nan | 2.0 | -2.0 | 5.1273526824978015 | 3.0098644938625174 | 4.023258310977892 | -4.999728523788375 | -4.9759248347799305 | -4.1262261992465215 | 5.3068809500913545 | nan | -0.9999975249997191 | 5.154454343996854 | nan | -4.978032280644466 | -0.9819511713871117 | 3.037477629224129 | 0.26778515035880196 | 4.244973667791668 | -1.2666666666666666 | 9.4 | -1.6620054777054616 | 1.0426148926718164 | 1.4861333913241905 | -4.875930521091812 | -4.999999824398321 | -4.722125374985963 | 3.6212471131639723 | 0.3301296720061022 | 2.732136052008574 | 0.0033517768313049913 | 3.146110322656216 | 5.004402506356452 | -0.3934348413480846 | 1.1267605633802817 | -0.9582750297377436 | 5.181587837837838 | 1.0500061487383026 | -4.197183098591549 | -0.8 | 5.04 | -2.84688995215311 | 5.394092680848692 | 5.018675523104445 | nan | 3.03998114097124 | 4.023121387283237 | nan | -1.9647385103011092 | 6.162364130434783 | -1.0 | -2.7688921129048794 | -1.5086638129455445 | 5.000000262168144 | -0.9999991918964639 | 2.0583417301078266 | -2.0 | 5.685393258426966 | -3.770510528306838 | 2.782582559012754 | -2.9179690610116356 | 1.2820204645119377 | nan | 1.0 | 5.194886541060882 | -3.9832968850386354 | 7.293710628182723 | 3.000432492540464 | -2.422502146032598 | 4.000768850405151 | 7.611486486486486 | -3.0 | 4.057493669684927 | 3.0 | -0.9769145780292758 | nan | nan | 4.050331286851863 | -0.9999832657072247 | nan | -3.77852872368079 | 4.007384800346456 | -4.812980663427052 | 0.6539672656639645 | 3.0041889690481733 | 1.000000377999978 | 3.000003938770955 | 5.728155339805825 | nan | -2.4658143185156174 | 2.7452503583825347 | -2.9932407017367586 | 3.2242990654205608 | -2.9587742808985658 | 7.551724137931035 | -1.9995901449026705 | 5.000000021568798 | 0.4519499333018673 | 4.045772409408773 | 5.000235201450682 | 3.0 | 3.000868966116528 | 3.0287906154673556 | 4.157700067249496 | -3.9522855461019155 | -1.998263211945215 | 1.6774805918990267e-05 | -3.834979825407338 | 2.4870923913043477 | 2.72972972972973 | 1.256565709452289 | 4.019876002917578 | 5.696335078534031 | 5.003501973236112 | 2.0 | -2.987042682926829 | -4.0 | -1.9995736373352926 | -3.9437060402354325 | 5.075192802056555 | 5.000000077439988 | 3.0507982681393924 | -2.996486494541096 | -1.9531963591010564 | 2.8945348906753834 | 6.777777777777778 | -4.972154222766218 | 0.11548880493872662 | 0.3282226007950029 | -2.999998507147743 | -0.9999999401241062 | 5.000034815967395 | 5.000000600514208 | 1.0308569379662267 | -0.9952725816574555 | 4.766666666666667 | 5.0 | 1.160856250997155 | 7.119413536269532 | -0.9080901120023107 | 3.0034637967353714 | nan | 0.04692582510097998 | -2.7465585363189993 | 0.08926673751328373 | 1.248805778510707 | 4.0 | 6.696008642340365 | 4.025182448073322 | 4.070256968433575 | -1.2307040693317695 | 4.438085934080489 | nan | 0.0 | 3.2844342940484736 | nan | -3.972607210321589 | -2.9846070753443152 | 4.000007252514111 | 1.0025316455696203 | 1.0000001301785695 | -4.930147058823529 | 4.005806476170556 | 6.0 | -4.590663390663391 | nan | -4.998853406073456 | 5.0864864864864865 | 0.484698384654215 | 3.058995327102804 | 3.0077113292716016 | 7.0 | -2.984557258407833 | nan | 3.05245921817517 | 3.0070883406595437 | 2.0156233633418896 | -4.969273785071352 | 0.0 | -2.5837716003005258 | 0.8736729889960286 | 0.002389194679306057 | -0.8729154070514008 | 2.1585691281300874 | nan | 4.054946506406023 | 3.293224031185508 | 0.17140964648361698 | -2.784525790349418 | 2.000024384623199 | 5.333333333333333 | 0.04035613560237782 | -4.99999965959895 | 5.000000065947246 | 0.39683915524958335 | 0.034100566920242235 | nan | 0.25127100423743126 | -1.1891378193827375 | -2.0 | -2.5912162162162162 | 4.025207581887929 | -1.9 | -1.705228686669325 | -3.922078762063017 | -4.0 | 0.3341697760873208 | -3.7768631130598744 | nan | 3.000018112502056 | 0.002052200680878446 | -0.5999999999999996 | 5.009096302155428 | 0.03050618253022197 | 4.015789473684211 | -4.369199731002017 | -2.8718264079553166 | 2.0000001762864854 | -0.9993179192211581 | -1.0 | 5.884860536257974 | 1.9246132806821628 | 0.06568493685615961 | 0.2841761716144128 | -0.9858314690529456 | -2.9651702680761076 | -2.0 | 1.0183017836396315 | -1.811175337186898 | -0.910344404724572 | nan | -3.9976531653729417 | 3.0523458417432137 | nan | nan | -3.4230177278686655 | 5.536796987676438e-07 | 3.0927773836330545 | 0.05994235317491583 | -3.999903172819227 | -3.162303664921466 | 1.0027054396735646 | 0.09126169534571023 | -4.349687542685426 | 5.0000000000009095 | 3.443181818181818 | -3.705931198740264 | 1.0529403138290072 | -3.9983130904183537 | -0.7811677238046497 | 8.2 | 0.0 | 2.683453237410072 | 1.0000016210799254 | 5.0371864354936635 | 1.0000019734625363 | -0.7348290198134761 | 3.0 | 2.0000001311032745 | -4.843475258186312 | nan | 1.4854827452269697 | 1.088808337109198 | -4.913093858632677 | 0.12736296574696715 | nan | -0.9341154519723491 | 3.5609018627245232 | nan | -4.0 | 3.0140113912067332 | -1.9136347704171048 | -4.864523400972494 | nan | -4.611111111111111 | -2.9983942333935083 | 4.00001914118916 | 5.644628099173554 | nan | -1.9999993825902684 | -2.9980951385463546 | -1.6466512702078522 | 2.375387473758761 | 2.7777777777777777 | 1.545454545454545 | 5.282325188552432 | -1.9994173690090336 | -3.9999992497 | 3.2 | -0.9239091482649842 | 0.00012252656424363264 | 4.12 | 2.0189633375474085 | -4.995821444929534 | 2.010138270725773 | -0.9342168386668304 | -0.9999999066625269 | -3.263671875 | nan | 7.017230837414959 | 2.925187479502072 | 1.0825641025641026 | 1.018725128504498 | -5.0 | 2.058022973561048 | 1.0016851734246803 | 3.0 | -2.9955996695135343 | 0.2892897475811128 | 0.6581722319859402 | 5.002741919862464 | nan | -3.9989823897639227 | -4.44062153163152 | -0.03725038900414945 | -2.999152342049905 | 4.009937050292014 | -0.989187759727994 | 2.3101604278074865 | -4.0 | -2.8855098389982112 | 2.0688595142944974 | 9.0 | 2.1757047970479704 | -4.999941763455179 | 4.04944145772556 | 5.1214027306228385 | 0.026018486045885077 | 0.015529503545513646 | 1.000000052134497 | 4.000000043911139 | -4.934782608695652 | 2.7253086419753085 | 8.333333333333334 | -3.0 | 3.0000002032853543 | 3.057846366460995 | nan | 1.0059760956175299 | 2.0756972111553784 | -1.9839497007158025 | 5.000414430765506 | -0.9617556000728464 | 3.012382574067786 | 4.000000006431285 | 4.014108636329611 | -4.758893865916411 | 4.080593543501582 | 0.00017308335456540894 | -3.6857223796033995 | 2.1623880185670807 | 1.0549483013232588 | 1.0693238151893714 | -0.6664813785436354 | -4.0 | -2.1867519761888192 | nan | nan | 3.0 | 3.034338275753749 | 1.0233825588643077 | 2.000000352187066 | -1.750689111697291 | -0.9076036437999628 | 5.002284119621812 | 1.1267016133377312 | 4.066990984460257 | 1.0476190476190474 | nan | -3.99546485260771 | nan | -4.958785553612135 | 6.474796473820145 | 1.0448881153370821 | nan | -4.998681879968695 | 2.0006182932097722 | nan | 5.012054794520548 | -3.97855822168717 | 2.851769911504425 | 5.000000344454221 | 5.112983955716116 | -0.7662424271564574 | nan | -3.9999964060388216 | 5.00067712447805 | -2.152222750292334 | 1.000000091450136 | -4.960114641637751 | nan | nan | 2.6523076923076925 | 1.903869564771159 | -4.9353966119112025 | -3.994232504358926 | 0.0933476216604997 | -0.9654178674351584 | nan | 1.8659656716910518e-07 | 2.0000000204458597 | 3.77380281987325 | 4.0 | -0.15738236142196094 | 1.0644329896907216 | -0.8809986015950902 | 1.0707692307692307 | 4.230658197695725 | nan | -4.81580983317139 | -1.0 | 5.228448275862069 | nan | 4.055791801326756 | 1.0556462267816895 | nan | 5.7030430220356765 | -0.9164603053914353 | -0.3651636536761447 | 1.1508657955477346 | -3.0 | 2.064696952002803 | 9.25 | -2.9999114137362035 | 7.746835443037975 | -1.8414812165914998 | 1.0278523547194698 | 4.082047546520098 | nan | -2.964048266392877 | 2.116805332751047 | -3.988184881477926 | 2.0000008898973385 | 5.09150422380387 | 2.000000092674922 | 5.0 | -4.999999950172728 | nan | 1.0000001076 | 4.071417481757491 | 2.1547837406070647 | -4.926891246461437 | -4.984676854636075 | -2.9486855330219615 | -3.5150423347144657 | 9.100105267419734 | -2.9999991947341034 | 3.0402298850574714 | 0.049001012076338155 | 0.25027932960893856 | 1.0044477390659747 | 5.022071904382002 | 0.11149899243070953 | -4.855214928357214 | 0.0 | -4.999999626806866 | -3.507853403141361 | 2.888888888888889 | 0.2410433856800639 | 4.252712714285633 | 0.5738538851983157 | 3.02075445642083 | 3.0000008745227698 | 6.764705882352941 | -4.781388158073178 | 4.219211822660099 | -4.999474198595993 | 3.2432432432432434 | nan | -5.0 | nan | -4.611491531021302 | 0.28333333333333344 | -5.0 | -3.9779358505559035 | 0.06193755156950814 | nan | 3.52112676056338 | 4.51111111111111 | 7.738095238095238 | -0.02284403923283662 | -1.5620475864858023 | 6.287231998625276 | -4.9 | 5.284043557360391 | 2.017262093360051 | nan | 2.1345249322153093 | 3.058739330238292 | 4.440490398177281 | 3.41405118300338 | nan | -1.8125 | 3.0 | 5.010106326981788 | -5.0 | -1.9999998623821065 | 0.7484407484407485 | 0.03809420984205388 | -0.9603624562269382 | 3.5009980039920157 | nan | 1.0101816145213447 | 4.5 | 2.688817300857791 | -3.8549205275617178 | 5.297756078460436 | 2.533333333333333 | -1.9033674963396778 | 6.49384236453202 | 0.0 | -4.788498159197664 | 3.0799555802332037 | -4.999999812182572 | -0.9939402969003652 | 5.061429757107714 | 5.6103201252061154 | -4.836952203393592 | -0.9681403335912344 | 3.8908748824082786 | -0.8052753779713364 | -1.0 | -1.988630382145489 | -2.0 | 0.4372213608553112 | nan | 5.0 | 4.001117756265166 | 0.0028529008442198566 | 1.0065926472775295 | 0.8 | 5.000001065191707 | 5.000000090459456 | 5.998244674521004 | -2.977781835889944 | -2.9999905566521514 | 2.0000009919999946 | nan | 5.2 | nan | -1.9988938968240597 | nan | -3.999997224000196 | 0.0 | nan | 3.0239846920180606 | 1.2871174914998111 | nan | -0.9641847321704748 | -0.989881693648817 | 1.0000002374478105 | 1.7437070938215102 | 4.003507474683661 | 3.0000001697727035 | -4.709590186452623 | 1.000352896160186 | -3.934312270924668 | -0.9999999583712041 | nan | 1.9267015706806285 | -3.9240302635804536 | 1.0000086033416418 | 0.20373930762943093 | -2.9444289059267197 | 0.005472209535328924 |


## 5. Configuração do Modelo XGBoost

✅ Usando aceleração GPU

### Parâmetros do Modelo

| Parâmetro | Valor |
|-----------|-------|
| `objective` | binary:logistic |
| `eval_metric` | auc |
| `tree_method` | gpu_hist |
| `max_depth` | 3 |
| `learning_rate` | 0.0700 |
| `subsample` | 0.6000 |
| `colsample_bytree` | 0.8000 |
| `min_child_weight` | 15 |
| `gamma` | 1.2000 |
| `scale_pos_weight` | 0.0850 |
| `random_state` | 42 |
| `n_jobs` | -1 |
| `verbosity` | 0 |
| `device` | cuda |

**scale_pos_weight (Balanceamento):** 0.085


## 6. Validação Cruzada Temporal

**AUC Médio (CV):** 0.7559 ± 0.0321

**AUC Fold 1:** 0.7343

**AUC Fold 2:** 0.7322

**AUC Fold 3:** 0.8013

**💡 Insight (overfitting):** A validação cruzada temporal mostra AUC médio de 0.7559 com desvio padrão de 0.0321. Esta é a métrica REALISTA de treino sem vazamento temporal. Se a AUC de teste for próxima desta, o modelo está generalizando bem.


## 7. Treinamento do Modelo Final

**Treino Final:** 6,800 amostras

**Validação Final:** 1,200 amostras

ℹ️ Modo DEV: n_estimators=500 (configuracao rapida)

ℹ️ Treinando modelo...

**AUC Treino:** 0.8476

**AUC Validação:** 0.8216

**Gap (Overfitting):** 2.59%

✅ Modelo generalizando bem. Gap aceitável.

### Curva ROC - Treino vs Validação

![Curva ROC - Treino vs Validação](credit_scoring_20260129_134511_images/img_003_credit_scoring_20260129_134511.png)

**Contexto Técnico:** Curva ROC do modelo XGBoost (Validação). AUC: 0.8216. Gap Treino-Val: 0.0259 (Aceitável). O contexto global indica desbalanceamento: Severe (razão: 0.0872), o que torna a curva ROC sensível. 
**Análise da Curva ROC:**

- **AUC-ROC:** 0.8216
  - Bom poder discriminativo
  - O modelo consegue distinguir bem entre bons e maus pagadores

- **Forma da Curva:**
  - A curva sobe gradualmente no início, indicando que o modelo identifica gradualmente os casos de maior risco
  - Curva acima da diagonal indica boa separação de classes



**Geometria da Curva ROC (Validação):**
- **Comportamento Inicial (Janela 6 pts):** Crescimento Agressivo (Excelente) (Slope=6.799).
- **Eficiência:** Atinge 80% da performance em x=0.265.
- **Estabilidade:** Zona estável (95% do pico) entre x=0.602 e x=1.000 (largura=0.398).
- **Pico:** x=1.000, y=1.000.

> 🤖 **Análise Visual Automática:**
>
> Análise da Curva ROC (Validação) do Modelo XGBoost:
> 
> 1.  **Observações sobre a forma/geometria da curva:**
>     A curva de validação (dourada) exibe um poder discriminativo consistente (AUC: 0.8216). A geometria visual confirma o "Crescimento Agressivo (Excelente)" inicial, com um slope elevado, capturando True Positives rapidamente com baixos False Positives. A afirmação "sobe gradualmente no início" no contexto técnico parece contradizer ligeiramente o "Crescimento Agressivo"; a imagem e o slope de 6.799 indicam um início abrupto e eficiente, seguido por uma ascensão mais gradual. Atinge 80% dos True Positives em aproximadamente 26.5% de False Positives, o que é um bom balanço. A zona estável após x=0.602, onde a curva se horizontaliza, indica retornos decrescentes de TPR para aumentos de FPR.
> 
> 2.  **Sinais de problemas:**
>     O `overfitting_gap` de 0.0259 é "Aceitável", e visualmente a curva de validação segue bem a de treino, ligeiramente abaixo, confirmando a ausência de overfitting severo. O `target_imbalance_status` "Severe" (razão 0.0872) é crucial: um AUC de 0.8216 nesse cenário é robusto e demonstra que o modelo é eficaz na identificação da classe minoritária (risco), superando o desafio do desbalanceamento. A menção de `toxic_features_drift` (`feature_757`) não é visível diretamente na ROC, mas é um ponto de atenção para monitoramento futuro.
> 
> 3.  **Conclusões práticas para o negócio:**
>     O modelo apresenta excelente capacidade de classificação para o risco de crédito, especialmente em um ambiente de dados severamente desbalanceado. A capacidade de identificar agressivamente os casos de maior risco no início da curva (low FPR, high TPR) é valiosa para a área de risco, permitindo ações proativas e mitigação. O gap aceitável e a robustez do AUC validam o modelo para implantação. A escolha do ponto de corte para decisões (e.g., conceder crédito) pode ser otimizada considerando a eficiência de 80% do TPR e a zona de estabilidade, alinhando a tolerância a False Positives com a estratégia de negócio.


## 9. SHAP Values - Explainability

ℹ️ SHAP pulado no modo DEV para acelerar desenvolvimento. Execute em modo PROD para análise completa.

**💡 Insight (explainability):** Análise SHAP não foi executada. Para ativar, configure 'run_shap: true' no config.yaml e execute em modo PROD.


## 10. Calibração de Probabilidades

ℹ️ [13:46:30] Início da calibração

ℹ️ Calibrando probabilidades (método isotônico, cv=3)...

ℹ️ Usando Holdout para avaliação final: 2,000 amostras

**Brier Score (Antes da Calibração):** 0.1771

**Brier Score (Depois da Calibração):** 0.0687

**Melhoria no Brier Score:** 0.1084 (61.2%)

### Curva de Calibração - Antes vs Depois

![Curva de Calibração - Antes vs Depois](credit_scoring_20260129_134511_images/img_004_credit_scoring_20260129_134511.png)

**Contexto Técnico:** Curva de calibração comparando probabilidades antes e depois da calibração isotônica. Quanto mais próxima da diagonal (linha preta tracejada), melhor a calibração. Brier Score melhorou de 0.1771 para 0.0687 (61.2% de melhoria).

> 🤖 **Análise Visual Automática:**
>
> **Análise da Curva de Calibração (Head de Risco)**
> 
> 1.  **Observações Geométricas e Validação:**
>     A curva `Antes` (vermelha) demonstra uma severa subestimação da fração de positivos observada. Consistentemente acima da linha de calibração perfeita (diagonal preta), o modelo atribuía probabilidades significativamente menores do que a realidade (ex: previsto 40% de probabilidade, observado ~80% de positivos). A curva `Depois` (verde) valida integralmente a melhoria de 61.2% no Brier Score, posicionando-se muito mais próxima à diagonal. No entanto, observamos uma oscilação notável em probabilidades baixas (e.g., ~0.25 previstos com 0% observados, seguido de um salto acentuado), indicando potencial ruído ou escassez de dados nesses bins.
> 
> 2.  **Sinais de Problemas (Contexto Global):**
>     A principal causa da descalibração inicial e da instabilidade em baixas probabilidades na curva pós-calibração reside no `desbalanceamento severo` da classe-alvo (apenas 8.72% de positivos). Modelos em datasets desbalanceados frequentemente tendem a "empurrar" as probabilidades em direção à classe majoritária, resultando em subestimação do risco real para a minoria. A calibração isotônica corrige esse viés, mas a escassez de eventos em bins de baixa probabilidade pode gerar as flutuações pontuais observadas na curva verde, pois há pouca informação para uma calibração precisa nessas faixas. O `overfitting_status` "Acceptable" (gap 0.0259) sugere que o problema não era a discriminação do modelo, mas a interpretação das suas probabilidades.
> 
> 3.  **Conclusões Práticas para o Negócio:**
>     A calibração realizada é um `sucesso operacional crítico`. Agora, as probabilidades de risco do modelo são muito mais `confiáveis`, permitindo:
>     *   **Melhor Tomada de Decisão:** Precificação mais acurada de produtos de crédito, definição de limites e cálculo de Expected Loss (EL) baseados em estimativas de risco fidedignas.
>     *   **Estratificação de Risco Robusta:** A gestão de portfólio e segmentação de clientes se beneficiam de uma compreensão mais real do risco, evitando a subestimação generalizada anterior.
>     *   **Conformidade Regulatória:** A acurácia das probabilidades é vital para modelos usados em contextos regulatórios.
>     Apesar da instabilidade marginal em probabilidades muito baixas, a melhoria global é substancial, endossando o uso das probabilidades calibradas para decisões de negócio.

✅ Calibração concluída

**💡 Insight (calibration):** As probabilidades foram calibradas usando método isotônico. O Brier Score melhorou de 0.1771 para 0.0687 (61.2% de melhoria). Isso garante que probabilidades de 0.7 realmente significam 70% de chance de default, essencial para decisões financeiras precisas.

ℹ️ [13:46:57] Fim da calibração


## 11. Análise de Impacto Financeiro

**Ticket Médio:** 10,000.00

**Ganho TP:** 1,500.00

**Perda FP:** -10,000.00

### Parâmetros Financeiros

| Parâmetro | Valor |
|-----------|-------|
| `TICKET_MEDIO` | 10000 |
| `GANHO_TP` | 1500 |
| `PERDA_FP` | -10000 |
| `PERDA_FN` | 0 |
| `GANHO_TN` | 0 |

ℹ️ Otimização financeira usando Holdout: 2,000 amostras

**Threshold Ótimo:** 0.8384

**Lucro Real (Amostra):** 1,504,500.00

**Lucro Potencial Máximo (Teórico):** 2,737,500.00

**Eficiência Financeira (% do Potencial):** 54.96%

ℹ️ **IMPORTANTE:** A Eficiência Financeira de 54.96% é uma métrica agnóstica ao tamanho da amostra. Ela funciona igual em modo DEV (10k linhas) e PROD (500k linhas). Meta: > 75% de eficiência.

ℹ️ **Geometria da Curva de Lucro:**
- **Comportamento Inicial (Janela 5 pts):** Estagnado/Plano (Crítico) (Slope=0.000).
- **Eficiência:** Atinge 80% da performance em x=0.545.
- **Estabilidade:** Zona estável (95% do pico) entre x=0.788 e x=0.889 (largura=0.101).
- **Pico:** x=0.838, y=1504500.000.

### Curvas de Lucro e Taxa de Aprovação vs Threshold

![Curvas de Lucro e Taxa de Aprovação vs Threshold](credit_scoring_20260129_134511_images/img_005_credit_scoring_20260129_134511.png)

**Contexto Técnico:** Análise completa de otimização financeira. Topo: Curva de lucro vs threshold. O threshold ótimo (0.8384) maximiza lucro em 1,504,500. Zonas verdes indicam lucro, zonas vermelhas indicam prejuízo. Base: Taxa de aprovação vs threshold. No threshold ótimo, aprovamos 89.2% dos casos. Esta visualização permite balancear lucro máximo com volume de negócios.

> 🤖 **Análise Visual Automática:**
>
> Como Head de Risco, minha análise da otimização financeira e das curvas de desempenho é a seguinte:
> 
> 1.  **Observações de Geometria e Validação:**
>     A geometria visual das curvas é altamente consistente com os dados fornecidos. A Curva de Lucro Esperado atinge seu pico próximo a 1.5e6, precisamente validando o `Lucro Máximo = 1,504,500` e o `Threshold Ótimo = 0.838`. A curva apresenta um platô inicial, um crescimento gradual até o ótimo e uma queda acentuada para thresholds mais altos. No gráfico inferior, o `Threshold Ótimo` alinha-se perfeitamente com a `Taxa Ótima = 89.2%` de aprovação. Notavelmente, a ausência de "Zonas de Prejuízo" visíveis (apenas zona verde) é um sinal positivo, indicando que o modelo sempre gera lucro no range de thresholds avaliado.
> 
> 2.  **Sinais de Problemas (Contexto Global):**
>     A `severa desbalanceamento de classes` (razão 0.0872) explica a alta taxa de aprovação (89.2%) no ponto ótimo, refletindo uma predominância de clientes "bons" na base. O `overfitting_gap` de 0.0259 é `aceitável`, e a suavidade das curvas corrobora que o modelo não está sobreajustado de forma prejudicial. O `drift` na 'feature_757' é um alerta que requer monitoramento contínuo para garantir que a performance se mantenha estável em produção.
> 
> 3.  **Conclusões Práticas para o Negócio:**
>     O modelo oferece um `threshold ótimo (0.838)` que maximiza o lucro esperado em R$ 1.504.500, aprovando um substancial `89.2%` dos casos. Isso representa um excelente equilíbrio entre rentabilidade e volume de negócios, fundamental para o crescimento da carteira de crédito. Contudo, a `eficiência financeira de 54.96%` indica que ainda há um `lucro potencial máximo de R$ 2.737.500` não capturado. Futuras iterações devem focar em aprimorar o modelo para fechar essa lacuna, explorando novos recursos ou técnicas para otimizar a captura de lucro remanescente.

**💡 Insight (financeiro):** A zona de lucro máximo é plana (flat-top), variando menos de 2.9% entre os thresholds 0.788 e 0.889. Isso indica que o modelo é robusto a pequenas variações na política de corte nesta faixa.

### Matriz de Confusão (Threshold Ótimo)

| Métrica | Valor |
|---------|-------|
| True Negatives (TN) | 73 |
| False Positives (FP) | 102 |
| False Negatives (FN) | 142 |
| True Positives (TP) | 1683 |

### Matriz de Confusão e Custos Financeiros

![Matriz de Confusão e Custos Financeiros](credit_scoring_20260129_134511_images/img_006_credit_scoring_20260129_134511.png)

**Contexto Técnico:** Matriz de confusão e custos financeiros no threshold ótimo (0.8384). Esquerda: Contagem de acertos/erros (TN=73, FP=102, FN=142, TP=1683). Direita: Impacto financeiro por célula. Erro Tipo I (FP): Aprovamos caloteiro = 102 × 10,000 = -1,020,000. Erro Tipo II (FN): Negamos bom pagador = 142 × 0 = 0. Lucro Total: 1,504,500.

> 🤖 **Análise Visual Automática:**
>
> Como Cientista de Dados Sênior e Head de Risco, minha análise é a seguinte:
> 
> 1.  **Geometria da Curva e Validação Numérica**:
>     As matrizes visuais de Contagem e Custos Financeiros confirmam com precisão os dados numéricos fornecidos no contexto técnico. Não há discrepâncias. A Matriz de Contagem evidencia um grande volume de True Positives (1683) e um número relativamente baixo de False Positives (102) e False Negatives (142), considerando o desbalanceamento inerente da classe (total de 'negativos' é 175, 'positivos' é 1825 no set analisado). A Matriz de Custos reforça que o maior lucro advém dos True Positives (R$ 2.52M) e a maior perda dos False Positives (R$ -1.02M).
> 
> 2.  **Sinais de Problemas e Contexto Global**:
>     O `overfitting_status` "Acceptable" (gap de 0.0259) é um bom sinal de generalização do modelo. Contudo, o `target_imbalance_status` "Severe" (0.0872) é evidente na Matriz de Confusão, com a classe minoritária (0) tendo apenas 73 True Negatives. A existência de `toxic_features_drift` (`feature_757`) é uma preocupação latente, exigindo monitoramento pós-deploy para evitar degradação de performance ao longo do tempo.
> 
> 3.  **Conclusões Práticas para o Negócio**:
>     O modelo é lucrativo, gerando R$ 1.50M de lucro real, o que representa 54.96% do potencial máximo. Este é um resultado sólido para um sistema de Credit Scoring. A principal alavanca financeira são os True Positives. A maior perda é claramente a aprovação indevida de maus pagadores (FPs). Dada a ausência de custo financeiro para False Negatives (negados bons pagadores), o foco estratégico deve ser a contínua mitigação de FPs para proteger e aumentar o capital, e o monitoramento proativo da `feature_757` para garantir a estabilidade e longevidade do modelo em produção.

**💡 Insight (financeiro):** O threshold ótimo para maximizar lucro é 0.8384, gerando lucro esperado de 1,504,500.00 (amostra atual). **MÉTRICA PRINCIPAL:** O modelo capturou **54.96%** de todo o dinheiro disponível na mesa (lucro potencial máximo teórico: 2,737,500.00). Esta métrica de eficiência é AGNÓSTICA ao tamanho da amostra (funciona igual em DEV e PROD). Uma eficiência > 75% é considerada excelente em crédito. Com este threshold, temos 1683 aprovações corretas (TP) e 102 aprovações incorretas (FP). Este threshold é mais conservador que o padrão de 0.5, refletindo o risco assimétrico do negócio onde perder 10.000 em um calote é muito pior do que perder 1.500 em juros de um bom pagador.


## 11.5. Análise de Elasticidade: Sensibilidade do Lucro à AUC

ℹ️ Simulando degradação controlada da AUC para medir elasticidade do lucro...

### Curva de Elasticidade: AUC vs Lucro

![Curva de Elasticidade: AUC vs Lucro](credit_scoring_20260129_134511_images/img_007_credit_scoring_20260129_134511.png)

**Contexto Técnico:** Análise de elasticidade entre AUC e lucro mantendo threshold fixo (0.838). O gráfico mostra como o lucro varia quando degradamos a qualidade do modelo (injetando ruído). Coeficiente de elasticidade: 7.30 (valores > 1 indicam que lucro cresce mais rápido que AUC). Na zona de alta performance (AUC > 0.71), aumentar 1% de AUC gera aproximadamente 20,585 de lucro adicional. Curva convexa/exponencial indica que ganhos de performance no topo valem mais do que na base.

> 🤖 **Análise Visual Automática:**
>
> Como Head de Risco, minha análise visual da curva de elasticidade, confrontada com os contextos fornecidos, revela o seguinte:
> 
> **1. Observações sobre a forma/geometria da curva:**
> A imagem valida inequivocamente o contexto local: a curva é nitidamente convexa e demonstra um comportamento quase exponencial, especialmente para AUCs acima de 0.65. Isso corrobora que a elasticidade é alta (7.30), indicando que pequenos ganhos de performance do modelo (AUC) no topo da faixa de desempenho se traduzem em aumentos desproporcionais e substanciais de lucro, conforme a anotação amarela destaca. O "Modelo Atual (AUC=0.751)" está posicionado na parte mais inclinada da curva, com lucro estimado alinhado ao lucro real informado (R$ 1.504.500).
> 
> **2. Sinais de problemas (overfitting, drift, etc) considerando o contexto global:**
> Visualmente, a curva de simulação é suave, e o modelo atual se alinha bem à tendência, não indicando overfitting *diretamente por este gráfico*. Contudo, o Contexto Global aponta para um "Severe" class imbalance (0.0872), o que por si só torna a construção de um modelo robusto para lucro e performance um desafio. A presença de "toxic_features_drift" em 'feature_757' é um alerta crítico: embora a curva mostre a sensibilidade ao ruído, o *drift real* em produção pode levar a uma degradação de performance e lucro muito mais acentuada do que o cenário simulado, exigindo monitoramento rigoroso.
> 
> **3. Conclusões práticas para o negócio:**
> Este gráfico é um poderoso argumento para investir continuamente em melhorias marginais de performance do modelo. A alta elasticidade demonstra que cada ponto percentual adicional de AUC, especialmente na faixa operacional atual (>0.70), gera um retorno de lucro considerável. O lucro real (R$ 1.5M) está numa região de alto valor. Para fechar a lacuna até o "profit_potential_max" (R$ 2.7M), além de otimizar a performance do AUC, precisamos explorar a dinâmica do threshold fixo (0.838) e investigar o impacto e mitigação do drift em 'feature_757'. É vital priorizar a robustez do modelo em produção.

**Coeficiente de Elasticidade:** 7.30

**Valor Marginal (1% AUC):** 20,585

**AUC Atual:** 0.7510

**Lucro no AUC Atual:** 1,504,500

**💡 Insight (elasticity):** **Diagnóstico de Elasticidade:** ALTO RISCO. Modelo muito sensível: qualquer degradação causará prejuízo massivo. Monitoramento crítico necessário. O coeficiente de 7.30 indica que a relação AUC-Lucro é super-linear (convexa). **ROI de Investimento:** Se melhorar o modelo em 1% de AUC custar menos que 20,585, o investimento é justificado. Caso contrário, focar em estabilidade e monitoramento.

### Valor Marginal por Faixa de AUC

| Zona | AUC Min | AUC Max | Valor Marginal (1% AUC) |
|---|---|---|---|
| Zona da Morte | 0.50 | 0.65 | 30,941 |
| Zona de Crescimento | 0.65 | 0.80 | 88,505 |


## 12. Monitoramento de Drift (PSI)

ℹ️ Cálculo de PSI usando Holdout: 2,000 amostras

**PSI (Population Stability Index):** 0.0134

### Análise de Drift: Distribuição de Scores (PSI)

![Análise de Drift: Distribuição de Scores (PSI)](credit_scoring_20260129_134511_images/img_008_credit_scoring_20260129_134511.png)

**Contexto Técnico:** Análise visual de drift usando PSI (Population Stability Index = 0.0134). Esquerda: Histogramas comparando distribuições de scores entre treino (baseline) e validação (atual). Direita: Estimativa de densidade suave (KDE) para visualização mais clara das diferenças. Quanto mais sobrepostas as distribuições, menor o drift. PSI < 0.1 = Estável, PSI 0.1-0.2 = Atenção, PSI > 0.2 = Crítico (retreino necessário).

> 🤖 **Análise Visual Automática:**
>
> Como Head de Risco, analiso os dados e a imagem com a seguinte perspectiva:
> 
> **1. Observações sobre a forma/geometria da curva:**
> A geometria visual dos histogramas e das curvas KDE demonstra uma excelente sobreposição entre as distribuições de scores de Treino (Baseline) e Validação (Atual). Ambas são fortemente bimodais, concentrando-se em scores mais altos (próximos a 1.0), o que é típico de modelos de risco em datasets desbalanceados, onde a maioria das observações é de "bons pagadores". As distribuições são quase idênticas, confirmando visualmente o baixíssimo PSI de 0.0134. Não há contradição entre o valor numérico e a representação gráfica.
> 
> **2. Sinais de problemas (overfitting, drift, etc) considerando o contexto global:**
> O PSI de 0.0134 indica que o modelo de Credit Scoring está excepcionalmente estável, sem drift significativo na distribuição de seus scores. Este valor, estando muito abaixo do limite de "Estável" (< 0.1), é um forte sinal de robustez. Essa estabilidade é consistente com o `overfitting_status` "Acceptable" (`overfitting_gap` de 0.0259), mostrando que o modelo generaliza bem. A concentração de scores em valores altos é esperada, dada a `target_imbalance_status` "Severe" (class_1 muito maior que class_0). O fato de `feature_757` apresentar drift (`toxic_features_drift`) não impactou a estabilidade global dos scores do modelo, o que é positivo.
> 
> **3. Conclusões práticas para o negócio conectando contexto global + visual:**
> A estabilidade da distribuição dos scores é um pilar fundamental para a confiança operacional de um modelo de risco. O baixíssimo PSI garante que a lógica de risco do modelo está sendo aplicada de forma consistente ao longo do tempo. As decisões de crédito baseadas nestes scores mantêm a mesma base estatística do treinamento, o que valida a `financial_efficiency_percent` de 54.96% e o `profit_real`. Não há necessidade imediata de retreinar o modelo devido a drift. Contudo, manter a vigilância sobre a `feature_757` é crucial para mitigar riscos futuros.

✅ OK: PSI < 0.1 - Distribuição estável.

**💡 Insight (drift):** O PSI de 0.0134 indica distribuição estável. Status: OK. Este valor deve ser monitorado em produção para detectar drift temporal.


## 13. Análise de Erros - Casos Reais

ℹ️ Encontrados 102 Falsos Positivos e 142 Falsos Negativos


### 13.1. Exemplos de Falsos Positivos (Aprovamos mas Calotearam)

ℹ️ **Exemplo 1:** Score = 0.9805 (threshold = 0.8384)

### Top 5 Features (Valores)

| Métrica | Valor |
|---------|-------|
| feature_426 | 10.0000 |
| feature_968 | 9.8333 |
| feature_565 | 9.8000 |
| feature_242 | 9.6111 |
| feature_991 | 9.1000 |

**💡 Insight (error_analysis):** Este cliente foi aprovado com score 0.9805 mas caloteou. As features mais altas são feature_426, feature_968, feature_565. Verifique se alguma delas está agindo como 'falso sinal' de bom pagador. Possíveis causas: data leakage, feature enganosa, ou padrão raro não capturado pelo modelo.

ℹ️ **Exemplo 2:** Score = 0.9673 (threshold = 0.8384)

### Top 5 Features (Valores)

| Métrica | Valor |
|---------|-------|
| feature_683 | 9.8658 |
| feature_868 | 9.0000 |
| feature_162 | 8.9277 |
| feature_991 | 8.6496 |
| feature_565 | 8.3000 |

**💡 Insight (error_analysis):** Este cliente foi aprovado com score 0.9673 mas caloteou. As features mais altas são feature_683, feature_868, feature_162. Verifique se alguma delas está agindo como 'falso sinal' de bom pagador. Possíveis causas: data leakage, feature enganosa, ou padrão raro não capturado pelo modelo.

ℹ️ **Exemplo 3:** Score = 0.9517 (threshold = 0.8384)

### Top 5 Features (Valores)

| Métrica | Valor |
|---------|-------|
| feature_589 | 9.2725 |
| feature_991 | 9.0264 |
| feature_868 | 9.0000 |
| feature_1071 | 8.7908 |
| feature_162 | 8.7679 |

**💡 Insight (error_analysis):** Este cliente foi aprovado com score 0.9517 mas caloteou. As features mais altas são feature_589, feature_991, feature_868. Verifique se alguma delas está agindo como 'falso sinal' de bom pagador. Possíveis causas: data leakage, feature enganosa, ou padrão raro não capturado pelo modelo.


### 13.2. Exemplos de Falsos Negativos (Negamos mas Pagaram)

ℹ️ **Exemplo 1:** Score = 0.7822 (threshold = 0.8384)

### Top 5 Features (Valores)

| Métrica | Valor |
|---------|-------|
| feature_67 | -5.0000 |
| feature_117 | -5.0000 |
| feature_286 | -5.0000 |
| feature_395 | -5.0000 |
| feature_709 | -5.0000 |

**💡 Insight (error_analysis):** Este cliente foi negado com score 0.7822 mas pagou. As features mais baixas são feature_67, feature_117, feature_286. Verifique se o modelo está sendo muito conservador ou se há features que estão incorretamente penalizando bons pagadores. Possível oportunidade de ajuste fino.

ℹ️ **Exemplo 2:** Score = 0.8350 (threshold = 0.8384)

### Top 5 Features (Valores)

| Métrica | Valor |
|---------|-------|
| feature_67 | -5.0000 |
| feature_117 | -5.0000 |
| feature_395 | -5.0000 |
| feature_474 | -5.0000 |
| feature_671 | -5.0000 |

**💡 Insight (error_analysis):** Este cliente foi negado com score 0.8350 mas pagou. As features mais baixas são feature_67, feature_117, feature_395. Verifique se o modelo está sendo muito conservador ou se há features que estão incorretamente penalizando bons pagadores. Possível oportunidade de ajuste fino.

ℹ️ **Exemplo 3:** Score = 0.8003 (threshold = 0.8384)

### Top 5 Features (Valores)

| Métrica | Valor |
|---------|-------|
| feature_67 | -5.0000 |
| feature_289 | -5.0000 |
| feature_464 | -5.0000 |
| feature_474 | -5.0000 |
| feature_982 | -5.0000 |

**💡 Insight (error_analysis):** Este cliente foi negado com score 0.8003 mas pagou. As features mais baixas são feature_67, feature_289, feature_464. Verifique se o modelo está sendo muito conservador ou se há features que estão incorretamente penalizando bons pagadores. Possível oportunidade de ajuste fino.

**Taxa de Falsos Positivos:** 5.10%

**Taxa de Falsos Negativos:** 7.10%


---

## 📋 Resumo Executivo

**Data de Execução:** 2026-01-29 13:48:10

### Seções do Relatório:

1. Credit Scoring: Maquina de Decisao de Credito
2. 0. Configuracao do Pipeline
3. 1. Setup & Infraestrutura
   - 4 métricas registradas
4. 2. Engenharia de Dados
   - 11 métricas registradas
   - 1 insights identificados
5. 3. EDA Executiva
   - 5 métricas registradas
   - 3 insights identificados
   - 1 visualizações geradas
6. 3.1. Análise de Drift Temporal Completa
   - 7 métricas registradas
   - 1 insights identificados
   - 1 visualizações geradas
7. 3.1.1. Sugestão de Ação: Remover Features com Drift Crítico
   - 1 insights identificados
8. 4. Feature Selection (Conservadora)
   - 6 métricas registradas
   - 2 insights identificados
9. 5. Configuração do Modelo XGBoost
   - 1 métricas registradas
10. 6. Validação Cruzada Temporal
   - 4 métricas registradas
   - 1 insights identificados
11. 7. Treinamento do Modelo Final
   - 5 métricas registradas
   - 1 visualizações geradas
12. 9. SHAP Values - Explainability
   - 1 insights identificados
13. 10. Calibração de Probabilidades
   - 3 métricas registradas
   - 1 insights identificados
   - 1 visualizações geradas
14. 11. Análise de Impacto Financeiro
   - 7 métricas registradas
   - 2 insights identificados
   - 2 visualizações geradas
15. 11.5. Análise de Elasticidade: Sensibilidade do Lucro à AUC
   - 4 métricas registradas
   - 1 insights identificados
   - 1 visualizações geradas
16. 12. Monitoramento de Drift (PSI)
   - 1 métricas registradas
   - 1 insights identificados
   - 1 visualizações geradas
17. 13. Análise de Erros - Casos Reais
18. 13.1. Exemplos de Falsos Positivos (Aprovamos mas Calotearam)
   - 3 insights identificados
19. 13.2. Exemplos de Falsos Negativos (Negamos mas Pagaram)
   - 2 métricas registradas
   - 3 insights identificados

### Estatísticas da Execução:

- **Total de Seções:** 19
- **Total de Métricas:** 60
- **Total de Insights:** 21
- **Total de Visualizações:** 8

---

**Relatório gerado automaticamente pelo sistema de Credit Scoring**
