"""Byte-level tokenizer adapter for dependency-free token accounting.

Author: Olshansky Grigory
"""
from __future__ import annotations

from collections.abc import Sequence

__author__ = "Olshansky Grigory"


class ByteTokenizer:
    """Maps text to UTF-8 byte ids and back.

    A deterministic placeholder tokenizer used for token accounting in
    benchmarks and for forcing-plan tests. Byte counts are NOT a lower
    bound for model token counts and must not be read as model tokens;
    swap in a real model tokenizer through the same port for model-level
    numbers.
    """

    def encode(self, text: str) -> list[int]:
        """Return the UTF-8 byte ids of text."""
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        """Return the text decoded from UTF-8 byte ids."""
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")
