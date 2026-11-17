# RAG Evaluation Report

- Dataset: `data\evals\golden.jsonl`
- Evaluated at: `2026-11-15T18:42:55.235375+00:00`
- Cases: `31`

## Retrieval metrics

| Metric | Value |
|---|---:|
| MRR | 0.000 |
| Recall@1 | 0.000 |
| Recall@3 | 0.000 |
| Recall@5 | 0.000 |
| Recall@10 | 0.000 |
| Precision@1 | 0.000 |
| Precision@3 | 0.000 |
| Precision@5 | 0.000 |
| Precision@10 | 0.000 |
| nDCG@1 | 0.000 |
| nDCG@3 | 0.000 |
| nDCG@5 | 0.000 |
| nDCG@10 | 0.000 |

## Answer metrics

| Metric | Value |
|---|---:|
| Abstention accuracy | 0.935 |
| Citation support rate | 0.953 |
| Citation coverage | 0.952 |
| Lexical answer F1 | 0.671 |

## Per-case results

### fastapi-routing-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-annotations-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-validation-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-parameters-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-config-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-dependency-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-dependency-use-cases-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-dependency-graph-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-errors-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-health-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-readiness-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### fastapi-health-ambiguous-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### otel-signals-001

- Abstained: `True`
- Confidence: `0.593`
- Retrieval MRR: `0.000`

### otel-rag-spans-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### otel-metrics-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### otel-metrics-privacy-001

- Abstained: `True`
- Confidence: `0.593`
- Retrieval MRR: `0.000`

### otel-exporter-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### otel-privacy-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### pydantic-models-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### pydantic-constraints-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### pydantic-config-identifier-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### pydantic-serialization-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### pydantic-nested-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### pydantic-secrets-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### pydantic-abstention-001

- Abstained: `False`
- Confidence: `0.993`
- Retrieval MRR: `0.000`

### cross-source-observability-001

- Abstained: `False`
- Confidence: `0.987`
- Retrieval MRR: `0.000`

### cross-source-validation-001

- Abstained: `False`
- Confidence: `0.990`
- Retrieval MRR: `0.000`

### unknown-database-001

- Abstained: `True`
- Confidence: `0.385`
- Retrieval MRR: `0.000`

### unknown-model-version-001

- Abstained: `True`
- Confidence: `0.392`
- Retrieval MRR: `0.000`

### unknown-latency-001

- Abstained: `True`
- Confidence: `0.392`
- Retrieval MRR: `0.000`

### unknown-license-001

- Abstained: `True`
- Confidence: `0.383`
- Retrieval MRR: `0.000`


## Run metadata

- corpus_version: `corpus-92b6dbbf925fedf5`
- retrieval_mode: `hybrid`
- embedding_model: `text-embedding-3-small`
- generation_model: `gpt-4o`
- evaluation_model: `gpt-4o-mini`
- dense_top_k: `10`
- sparse_top_k: `10`
- final_top_k: `5`
- dense_weight: `0.7`
- sparse_weight: `0.3`
- dataset: `data\evals\golden.jsonl`
- case_count: `31`
- cache_enabled: `True`
- cache_hits: `31`
- cache_misses: `0`
- duration_seconds: `0.024`