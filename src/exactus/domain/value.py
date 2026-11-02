"""Computed values with explicit rendering formats."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

from dataclasses import dataclass

from .errors import FormatError

FORMATS = ("dec", "hex", "bin")


@dataclass(frozen=True, slots=True)
class ComputeValue:
    """Exact integer result of one expression evaluation."""

    integer: int
    format: str = "dec"

    def __post_init__(self) -> None:
        """Reject non-int payloads and unknown output formats."""
        if type(self.integer) is not int:
            raise FormatError("value must be a plain int")
        if self.format not in FORMATS:
            raise FormatError(f"unknown format: {self.format}")

    def render(self) -> str:
        """Return the value rendered in the selected format."""
        if self.format == "hex":
            return hex(self.integer)
        if self.format == "bin":
            return bin(self.integer)
        return str(self.integer)
