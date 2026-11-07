# Security

## Threat Model

### Prompt Injection
Untrusted documents may attempt prompt injection or tag closing attacks. The library mitigates this by:
- Setting an explicit system prompt directing the LLM to ignore instructions inside `<document>` tags.
- Escaping all occurrences of `</document` tags inside document chunks before LLM submission.

### Resource Exhaustion
To protect against unbounded computation and memory usage:
- Input texts exceeding `max_input_chars` (default 1,000,000) are rejected with `KgError('input_too_large')`.
- Per-reply items are capped at 200.
- Field lengths are strictly bounded between 1 and 200 characters.

### Non-Grounded Hallucinations
Triples claiming relations unsupported by the text are rejected using strict substring verification on whitespace-collapsed text and evidence containment checks.