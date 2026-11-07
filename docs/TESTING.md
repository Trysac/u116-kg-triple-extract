# Testing

## Tests

The repository contains a comprehensive suite in `tests/test_kg_triple_extract.py`. These tests cover:
- Acceptance criteria, including `RecordedLLM` basic extraction and `FakeLLM` behavior.
- Grounding rejection logic (evidence mismatch, subject/object absence).
- Predicate allowlist enforcement and normalization (`Worked With` to `worked_with`).
- Exception propagation and `invalid_json` handling.
- Chunking logic and size limit validation.
- Graph normalization, ID collapsing, and JSON serialization correctness.
- Event callbacks.