"""Behavior tests for the public API of kg_triple_extract."""

from __future__ import annotations

import pytest

import kg_triple_extract


def test_normalize_trims_and_lowercases() -> None:
    assert kg_triple_extract.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        kg_triple_extract.normalize(value)


def test_version_is_semver() -> None:
    parts = kg_triple_extract.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
