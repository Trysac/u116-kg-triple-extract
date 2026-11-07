# Project record

## Objective

Extract grounded subject-predicate-object triples from text with an injected LLM, verbatim evidence checks, predicate allowlists, and a merged, normalized knowledge graph.

## Scope

- In-scope: The public API as defined in `src/kg_triple_extract/__init__.py`, including `extract_triples`, `parse_reply`, `Triple`, `Rejection`, `Extraction`, `Graph`, and `KgError`.
- Out-of-scope: Coreference resolution, entity linking to external databases (e.g., Wikidata), RDF/graph database exports, and automated retries for invalid LLM replies.

## Factory run u116 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4415 | 10561 | 0 | 0 | 0.0215 |
| fix-1 | gemini-3.8-flash | no | 45202 | 380 | 0 | 205365 | 0.0507 |
| fix-2 | gemini-3.8-flash | no | 13140 | 217 | 0 | 8817 | 0.0113 |
| fix-3 | gemini-3.8-flash | no | 92534 | 3407 | 0 | 568662 | 0.1248 |
| fix-4 | gemini-3.8-flash | no | 50624 | 1480 | 0 | 195467 | 0.0582 |
| audit | gemini-3.1-flash-lite | no | 51463 | 157 | 0 | 65211 | 0.0147 |
| docs | gemini-3.1-flash-lite | no | 136310 | 1510 | 0 | 0 | 0.0363 |

Total: $0.3176. Prices: Anthropic list prices.
