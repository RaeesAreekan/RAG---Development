# OpenTelemetry Application Instrumentation

OpenTelemetry provides APIs and SDK components for collecting telemetry from
applications. The main telemetry signals are traces, metrics, and logs.

Instrumentation should make application behavior observable without changing
the business result of a request.

## Traces and Spans

A trace represents the path of an operation through one or more services.
Spans represent individual units of work within that trace.

A RAG query may create spans for:

1. request parsing;
2. query embedding;
3. dense retrieval;
4. sparse retrieval;
5. result fusion;
6. generation;
7. citation verification.

Each span should record duration and relevant non-sensitive attributes.

## Metrics

Useful pipeline metrics include:

- retrieval latency;
- generation latency;
- number of retrieved chunks;
- citation support rate;
- abstention rate;
- token usage;
- estimated model cost.

Metrics should use stable names and bounded label values. Raw user questions
should not be used as metric labels because they create high-cardinality data.

## Context Propagation

When a request passes through multiple services, trace context allows spans
from those services to be associated with the same trace.

A service should propagate context through supported request headers when making
downstream calls.

## Exporters

Telemetry can be exported to a backend using an exporter. Local development
may use a console exporter, while production deployments may use an OTLP
exporter.

Exporter configuration should be controlled through environment variables and
should not prevent the main application from starting when telemetry is
temporarily unavailable.

## Privacy and Reliability

Telemetry must avoid recording API keys, authorization headers, full document
contents, or sensitive user questions.

Instrumentation should be sampled or filtered when volume is high. Observability
failures should not cause a successful application request to fail unless the
system explicitly requires telemetry for compliance.