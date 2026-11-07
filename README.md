# trysac-kg-triple-extract

Extracts grounded subject-predicate-object triples from text using an injected LLM. Features verbatim evidence verification, predicate allowlists, and a merged, normalized knowledge graph. Zero external dependencies.

## The Problem

Knowledge graphs built by asking an LLM for triples fill up with relations the text never states, entity names that differ only in case, and predicates in ten spellings. Without grounding and normalisation, the graph cannot be trusted for retrieval. This library extracts triples chunk by chunk, keeps only those whose evidence is quoted from the source, and merges them into a normalized graph.

## Installation

```bash
pip install .
```

## Usage

```python
from kg_triple_extract import FakeLLM, Graph, extract_triples

text = "Ada Lovelace worked with Charles Babbage on the Analytical Engine."
llm = FakeLLM(
    lambda msgs: (
        '[{"subject": "Ada Lovelace", "predicate": "worked_with", "object": "Charles Babbage", "evidence": "Ada Lovelace worked with Charles Babbage"}]'
    )
)
extraction = extract_triples(text, llm, predicates=["worked_with"])
graph = Graph.from_triples(extraction.triples)
print(graph.to_json())
```

## API Summary

- **`extract_triples(text, llm, *, predicates=None, max_chars_per_call=4000, max_tokens=1024, max_input_chars=1_000_000, on_event=None) -> Extraction`**: Extracts triples from text.
- **`parse_reply(reply, chunk, chunk_index, *, predicates=None) -> tuple[list[Triple], list[Rejection]]`**: Validates LLM output against source.
- **`Graph.from_triples(triples)`**: Aggregates triples into a normalized structure.
- **Errors**: `KgError` (raised for `input_too_large` or `invalid_option`).
- **Limits**: Max 200 items per LLM call; fields limited to 1-200 characters; max 1,000,000 input characters.

## Why not LangChain's LLMGraphTransformer?

Unlike LangChain's solution, this library is provider-agnostic, enforces strict grounding via verbatim evidence verification, and uses explicit rejection codes for debugging hallucinations.

## Security

See [docs/SECURITY.md](docs/SECURITY.md) for the threat model and limits.