# Pydantic Models and Validation

Pydantic models describe structured data using Python type annotations. A model
validates input data and exposes the validated result through model attributes.

Models are useful for API request bodies, configuration objects, evaluation
records, and serialized pipeline results.

## Field Constraints

Fields may specify constraints such as:

- minimum and maximum numeric values;
- minimum string length;
- regular expression patterns;
- default values;
- whether a field is required.

A field named `embedding_model` may be constrained to a non-empty string.
A field named `top_k` may require a value greater than or equal to one.

Validation errors should be treated as structured data rather than silently
coercing invalid application configuration.

## Model Serialization

Pydantic models can be converted into dictionaries or JSON-compatible
representations. Serialization is useful when storing:

- document metadata;
- chunk records;
- retrieval traces;
- evaluation results;
- API responses.

A serialized object should preserve stable identifiers such as `document_id`,
`chunk_id`, and `corpus_version`.

## Nested Models

A model may contain a list of other models. For example, a generated answer
can contain a list of claim records, where each claim contains citation IDs.

Nested validation ensures that malformed citation records are rejected before
they reach the verification stage.

## Configuration Models

Application settings should be separated from user-provided content. A settings
model may load values from environment variables while still applying type
validation and safe defaults.

Sensitive values such as `OPENAI_API_KEY` should not be included in logs,
serialized reports, or error messages.

## Validation Behavior

An unanswerable question should not be represented as a successful answer with
fabricated content. The application response can instead include:

- `abstained: true`;
- a human-readable abstention reason;
- retrieved evidence, if any;
- confidence diagnostics.