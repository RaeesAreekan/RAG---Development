# RAG Hybrid Search Pipeline

An OpenAI-first retrieval-augmented generation (RAG) system for answering questions over a small, versioned technical-documentation corpus.

This is a learning and portfolio PoC focused on retrieval quality, grounded generation, citations, abstention, and reproducible evaluation.

## Features

- Markdown, text, HTML, and PDF ingestion boundaries.
- Structure-aware chunking with deterministic chunk IDs.
- Dense retrieval using OpenAI embeddings or an offline fake provider.
- Sparse retrieval using BM25.
- Hybrid retrieval using weighted reciprocal-rank fusion.
- Optional deterministic keyword-overlap reranking.
- Structured answers with claim-level citations.
- Citation validity checks and confidence diagnostics.
- Explicit abstention when evidence is missing or weak.
- Human-approved golden evaluation data.
- Recall, precision, MRR, nDCG, citation, abstention, and answer-overlap metrics.
- Cached evaluation runs and dense/sparse/hybrid comparison reports.
- FastAPI endpoints for health, questions, and document listing.

## Architecture

```text
Corpus files
    |
    v
Ingestion and normalization
    |
    v
Structure-aware chunks + corpus manifest
    |
    +--> Dense index
    |
    +--> BM25 sparse index
    |
    v
Dense / sparse / hybrid retrieval
    |
    v
Grounded structured generation
    |
    v
Citation verification + confidence + abstention
    |
    v
CLI, evaluation reports, and FastAPI API
```

## Repository structure

```text
src/rag_pipeline/
├── ingestion/       # loaders, normalization, corpus manifests
├── chunking/        # chunking strategies and deduplication
├── providers/       # embedding and generation provider interfaces
├── indexing/        # dense and sparse persisted indexes
├── retrieval/       # retrieval, fusion, reranking, traces
├── generation/      # grounded prompts and answer generation
├── verification/    # citation checks and confidence scoring
├── evaluation/      # golden data, metrics, caching, reports
└── api/             # FastAPI application and contracts

data/
├── corpus/          # curated documentation snapshot
├── indexes/         # generated local indexes
└── evals/           # golden data, caches, and reports
```

## Requirements

- Python 3.11+
- An OpenAI API key for live embeddings and generation

Install the project:

```powershell
python -m pip install -e ".[dev]"
```

Optional cross-encoder reranking:

```powershell
python -m pip install -e ".[reranker]"
```

## Environment

Create `.env` from `.env.example`:

```env
OPENAI_API_KEY=your-key-here
```

Never commit `.env`, API keys, generated caches, or private documents.

Default models:

```text
Embedding:  text-embedding-3-small
Generation: gpt-4o
Evaluation: gpt-4o-mini
```

The pipeline falls back to deterministic fake embeddings when no key is configured. Live answer generation still requires an OpenAI key.

## Build indexes

Run from the repository root:

```powershell
python -m rag_pipeline.cli build-index `
  --corpus data/corpus
```

This creates dense and sparse indexes plus manifests connecting the corpus version, chunk IDs, and embedding model.

## Query retrieval

```powershell
python -m rag_pipeline.cli query `
  "How can dependencies be replaced during testing?"
```

## Evaluation

The human-approved PoC dataset is:

```text
data/evals/golden.jsonl
```

It covers lookup, exact-identifier, multi-hop, ambiguous, and unanswerable questions. Answerable cases contain reference answers, acceptable source references, required evidence units, and human relevance grades from 0 to 3.

Only `human-approved` cases are accepted by benchmark runs.

Run evaluation:

```powershell
python -m rag_pipeline.cli evaluate `
  --dataset data/evals/golden.jsonl `
  --output-dir data/evals/reports
```

Run retrieval comparison:

```powershell
python -m rag_pipeline.cli compare-eval `
  --dataset data/evals/golden.jsonl `
  --output-dir data/evals/comparison `
  --cache-dir data/evals/cache
```

The comparison evaluates dense-only, sparse-only BM25, hybrid, and hybrid-plus-keyword-overlap reranking.

### Metrics

| Metric | Meaning |
|---|---|
| Recall@k | Fraction of judged relevant chunks retrieved in the top k |
| Precision@k | Fraction of top-k results judged relevant |
| MRR | Rank of the first relevant result |
| nDCG@k | Ranking quality using graded relevance |
| Citation support rate | Fraction of citation checks supported by evidence |
| Citation coverage | Fraction of claims with supported citations |
| Abstention accuracy | Whether expected and actual abstentions match |
| Lexical answer F1 | Token-overlap baseline against the reference answer |

Lexical answer F1 is only a baseline. It can penalize correct paraphrases and is not semantic correctness.

## Human-in-the-loop golden data

```text
Corpus chunks
    |
    v
LLM-proposed evaluation cases
    |
    v
Human review and correction
    |
    v
golden.jsonl with human-approved cases
```

LLM-generated cases are authoring aids only. Reviewers must verify factual correctness, chunk IDs, relevance grades, evidence requirements, and abstention labels.

## FastAPI service

Start the API:

```powershell
uvicorn rag_pipeline.api.app:app --reload
```

Open the interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Implemented endpoints:

```text
GET  /health
POST /v1/ask
GET  /v1/documents
```

Example request:

```powershell
$body = @{
    question = "How can dependencies be replaced during testing?"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/v1/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

The answer response contains claims, citations, citation checks, confidence, abstention state, retrieved evidence, and the index version.

## Testing

Run the complete suite:

```powershell
python -m pytest -q
```

Run focused evaluation, API, and smoke tests:

```powershell
python -m pytest tests/evaluation tests/integration tests/smoke -q
```

Run linting and type checks when configured:

```powershell
ruff check .
mypy src
```

Tests should use fake providers and should not require network access or paid model calls.

## Demo setup

Build indexes using the configured corpus:

```powershell
python scripts/seed_demo.py
```

The script prints the corpus version, chunk count, and persisted index paths.

## Reproducibility note

Chunk IDs depend on the corpus version, document ID, and chunk position. The current PoC document IDs are derived from document paths. Run indexing and evaluation from the repository root and rebuild indexes after moving the project or changing the corpus.

If the corpus changes, regenerate or relabel the golden dataset because previous judgments may refer to stale chunk IDs.

## Current limitations

- The corpus is intentionally small and curated for a PoC.
- The golden dataset should eventually be expanded to at least 50 reviewed cases.
- Keyword-overlap reranking is a deterministic baseline, not a semantic cross-encoder.
- Lexical answer F1 does not measure semantic correctness.
- Token usage and cost tracking are not yet exposed by the provider abstraction.
- Upload-based ingestion is not exposed through the API yet.
- The dashboard and Docker packaging are follow-up work.
- Confidence scores are heuristic diagnostics, not calibrated probabilities.

## Planned extensions

1. Add Docker packaging and a reproducible containerized demo.
2. Add a Streamlit dashboard for evidence and evaluation inspection.
3. Add optional cross-encoder reranking.
4. Add judge-backed semantic answer and citation evaluation.
5. Expand the golden dataset to at least 50 reviewed cases.
6. Add token, latency, and cost telemetry to evaluation reports.

