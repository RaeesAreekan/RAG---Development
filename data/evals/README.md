# Evaluation Dataset

Each JSONL row is one evaluation case.

The dataset must contain at least 50 human-approved cases covering:

- lookup questions;
- exact keywords;
- exact identifiers;
- multi-hop questions;
- ambiguous questions;
- unanswerable questions.

`graded_chunk_judgments` maps chunk IDs to relevance grades:

- 0: irrelevant
- 1: weakly relevant
- 2: useful evidence
- 3: essential evidence

LLM-generated cases may begin with `review_status: llm-proposed`,
but benchmark runs should only use cases with:

`review_status: human-approved`