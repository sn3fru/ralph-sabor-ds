# Contexto do Projeto

Esta pasta contém **todo o contexto do projeto** que o agente lê automaticamente.

## Estrutura

```
context/
├── data/                  # 📁 DADOS DO PROJETO (READ-ONLY)
│   ├── train.parquet      # Dados de treino
│   └── test.parquet       # Dados de teste
│
├── README.md              # Este arquivo
├── 01_objetivos.md        # Documentação de negócio (opcional)
└── exemplos/              # Código de referência (opcional)
```

## O que colocar aqui

### Subpasta `data/`
- **Dados do projeto**: `.parquet`, `.csv`, `.json`
- Variável `DATA_DIR` é injetada no namespace dos scripts

### Documentação (opcional)
- Arquivos `.md` com regras de negócio, convenções
- Prefixe com números para ordenação (ex: `01_objetivos.md`)

### Exemplos de código (opcional)
- Trechos `.py` que servem de referência
- Pipeline legado, padrões de EDA

## Regras

- **Dados são READ-ONLY**: o agente nunca sobrescreve os originais
- **Não** coloque dados sensíveis ou secrets
