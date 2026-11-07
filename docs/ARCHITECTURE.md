# Architecture

`kg-triple-extract` extracts verified knowledge graph triples from raw text through an injected `LLMPort`.

## Workflow

1. **Chunking**: Splits documents by paragraphs, then sentences, and finally hard-cuts, packing chunks to `max_chars_per_call`.
2. **Tag Escaping**: Wraps content in `<document>` tags, escaping any existing `</document>` markers.
3. **LLM Interaction**: The `LLMPort` completes the prompt with the `SYSTEM_PROMPT` and the wrapped chunk.
4. **Parsing**: Validates that responses are JSON arrays, handles field missing errors, validates string lengths, normalizes predicates via NFKC/casefold/snake_case, and enforces grounding rules.
5. **Normalization**: `Graph` resolves node IDs (NFKC, casefold, whitespace collapsed) and aggregates duplicate edges.