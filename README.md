# kg-triple-extract

[![PyPI version](https://img.shields.io/pypi/v/trysac-kg-triple-extract)](https://pypi.org/project/trysac-kg-triple-extract/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-kg-triple-extract)](LICENSE)
[![CI](https://github.com/Trysac/u116-kg-triple-extract/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u116-kg-triple-extract/actions/workflows/ci.yml)

Extract grounded subject-predicate-object triples from text with an injected LLM, verbatim evidence checks, predicate allowlists and a merged, normalised knowledge graph

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-kg-triple-extract
# or
uv add trysac-kg-triple-extract
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import kg_triple_extract

# normalize() trims surrounding whitespace and lower-cases the text.
print(kg_triple_extract.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    kg_triple_extract.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
