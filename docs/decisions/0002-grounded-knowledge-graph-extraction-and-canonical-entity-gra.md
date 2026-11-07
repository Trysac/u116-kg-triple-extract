# Grounded Knowledge Graph Extraction and Canonical Entity Graph Representation

Status: accepted (generated, pending human review)

## Context

Knowledge graph extraction using LLMs frequently introduces hallucinations, non-standard predicate naming, and duplicate entity nodes differing only by capitalization or whitespace. Existing solutions either require heavy runtime dependencies or fail to verify text grounding.

## Decision

We implemented a zero-dependency Python library (`trysac-kg-triple-extract`) using an LLMPort protocol. The library performs sentence-aware chunking, prompt injection protection by tag escaping, verbatim evidence verification against source text, predicate normalization to snake_case, and canonical node ID deduplication in an aggregated Graph class.

## Consequences

Users obtain verified, grounded knowledge graphs without external dependencies. Callers must supply an LLMPort implementation. RDF or Wikidata entity linking is out of scope.
