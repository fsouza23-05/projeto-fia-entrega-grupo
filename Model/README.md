# Model — CreditGuard AI

## Descrição do Projeto

O CreditGuard AI é um sistema de predição de inadimplência desenvolvido pela ProScore Analytics para apoiar instituições financeiras na concessão de crédito com maior segurança. O modelo utiliza dados financeiros, cadastrais e comportamentais do cliente para estimar a probabilidade de não pagamento antes da aprovação do crédito.

---

## Objetivo de Negócio

Instituições financeiras enfrentam o desafio permanente de equilibrar crescimento de carteira e controle de risco. A aprovação de clientes inadimplentes gera perdas financeiras diretas; critérios excessivamente restritivos bloqueiam boas oportunidades.

O objetivo é classificar cada solicitante em três níveis de risco — **BAIXO RISCO**, **MÉDIO RISCO** ou **ALTO RISCO** — antes da concessão, reduzindo a taxa de inadimplência da carteira sem comprometer o volume de aprovações.

| Classificação | Probabilidade | Ação recomendada |
|---|---|---|
| 🟢 BAIXO RISCO | < 30% | Aprovação automática recomendada |
| 🟡 MÉDIO RISCO | 30% – 69% | Análise complementar recomendada |
| 🔴 ALTO RISCO | ≥ 70% | Revisão manual obrigatória |

A métrica prioritária é o **Recall**: o custo de aprovar um inadimplente (perda financeira) supera o custo de negar um bom pagador (perda de receita).

---

## Metodologia

O projeto segue a metodologia **CRISP-DM**:

1. **Business Understanding** — definição do problema de risco de crédito e da métrica prioritária (Recall)
2. **Data Understanding** — análise exploratória do dataset Home Credit Default Risk (`exp_analysis.ipynb` em DataPipeline/)
3. **Data Preparation** — sanitização, tratamento de nulos, feature engineering e encoding (`data_preparation.ipynb` em DataPipeline/)
4. **Modeling** — treinamento do LightGBM Balanced com `scale_pos_weight` para lidar com o desbalanceamento de classes (`train.py`)
5. **Evaluation** — avaliação de métricas ROC-AUC, Recall, Gini, KS e análise de importância de features (`evaluation.ipynb`)
6. **Deployment** — serving via Streamlit containerizado com Docker Compose

**Dataset:** Home Credit Default Risk (Kaggle) — 307.511 registros, 119 features após encoding, TARGET binário (0 = adimplente, 1 = inadimplente). Desbalanceamento: 91,93% adimplentes / 8,07% inadimplentes.

---

## Modelo em Produção

**LightGBM Balanced v2** — `Model/artifacts/lgbm_balanced_model.joblib`

| Métrica | LightGBM Balanced (v2) | XGBoost Balanced (v1) |
|---|---|---|
| ROC-AUC | **0,7524** | 0,7509 |
| Recall  | **0,6606** | 0,6568 |
| Gini    | **0,5049** | 0,5019 |
| KS      | **0,3736** | 0,3694 |
| Tempo de treino | **2,89 s** | 4,00 s |

O LightGBM supera o XGBoost em 6 de 7 métricas e é 28% mais rápido. Usa crescimento folha-a-folha (`num_leaves=63`) com GOSS e EFB, convergindo mais eficientemente que o crescimento nível-a-nível do XGBoost.

---

## Artefatos

```
Model/artifacts/
├── lgbm_balanced_model.joblib   ← modelo LightGBM serializado (produção)
├── features.joblib              ← lista das 119 features na ordem exata esperada pelo modelo
└── medianas.joblib              ← medianas para imputação de nulos em produção
```

---

## Como Treinar o Modelo

### Pré-requisitos

1. Dataset `Dados/abt.csv` gerado pelo pipeline de dados (ver `DataPipeline/`)
2. Dependências instaladas:

```bash
pip install -r Model/requirements.txt
```

### Executar treinamento

A partir da **raiz do projeto**:

```bash
python Model/train.py
```

Saída esperada:
```
Treino: (246008, 119) | Teste: (61503, 119)
Modelo salvo:    Model/artifacts/lgbm_balanced_model.joblib
Features salvas: Model/artifacts/features.joblib  (119 colunas)

Métricas — LightGBM Balanced:
  accuracy    : 0.7046
  precision   : 0.1660
  recall      : 0.6606
  f1          : 0.2653
  roc_auc     : 0.7524
```

### Opções avançadas

```bash
# Lê ABT do MinIO e salva artefatos no MinIO (requer stack MLOps ativa)
python Model/train.py --minio

# Registra o modelo no PostgreSQL após o treino
python Model/train.py --db

# Ambos
python Model/train.py --minio --db
```

### Avaliar o modelo

```bash
jupyter lab Model/evaluation.ipynb
```

O notebook recria o split de teste deterministicamente (`random_state=42`) e calcula métricas, curva ROC, matriz de confusão e importância de features por gain.

---

## Serviço de Inferência

`Model/predict.py` expõe a função `predict(data: dict) -> dict`:

```python
import sys
sys.path.append(".")  # executar da raiz do projeto
from Model.predict import predict

resultado = predict({
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_CREDIT": 300000.0,
    "AMT_ANNUITY": 25000.0,
    "CNT_CHILDREN": 0,
    "DAYS_BIRTH": -(35 * 365),
    "EXT_SOURCE_1": 0.50,
    "EXT_SOURCE_2": 0.50,
    "EXT_SOURCE_3": 0.50,
})
# {"prediction": 0|1, "probability": float}
```

O serviço inicializa todas as 119 features com zero e preenche apenas as colunas fornecidas. Tenta carregar o artefato do MinIO primeiro; na ausência, usa fallback local de `Model/artifacts/`.
