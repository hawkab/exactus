"""Incremental bounded marker parsing; offsets are reused by every renderer."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import StringIO
import re
from .errors import BudgetError, GrammarError

SPAN_RE = re.compile(
    r"<(?P<tag>calc(?:-final)?)>"
    r"(?P<expr>(?:(?!</?calc(?:-final)?>)[\s\S])*)</(?P=tag)>"
)
_OPEN = ("<calc>", "<calc-final>")
_CLOSE = ("</calc>", "</calc-final>")
_TAGS = _OPEN + _CLOSE
_FULL_TAG = re.compile(r"</?calc(?:-final)?>")
_MAX_TAG = max(map(len, _TAGS))
MAX_TRANSCRIPT, MAX_OPERATIONS = 262144, 128


@dataclass(frozen=True, slots=True)
class Span:
    expression: str
    final: bool
    start: int = field(default=0, compare=False)
    end: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class ParsedTranscript:
    """Immutable validated text and offsets, not a second parser."""

    content: str
    spans: tuple[Span, ...]

    def render(self, values: Sequence[str], *, marks: bool = False) -> str:
        if len(values) != len(self.spans):
            raise GrammarError("computed values do not match transcript spans")
        parts, last = [], 0
        for span, value in zip(self.spans, values):
            parts.append(self.content[last : span.start])
            parts.append(f"⊢ {value} ∎" if marks else value)
            last = span.end
        parts.append(self.content[last:])
        return "".join(parts)


def parse(content: str) -> ParsedTranscript:
    machine = SpanMachine()
    machine.feed(content)
    return machine.finish()


def _tag_at(text: str, pos: int) -> tuple[str | None, bool]:
    for tag in _TAGS:
        if text.startswith(tag, pos):
            return tag, False
    # Only a bounded suffix can be a partial tag. Never copy a growing tail.
    if len(text) - pos < _MAX_TAG:
        tail = text[pos:]
        return None, any(tag.startswith(tail) for tag in _TAGS)
    return None, False


class SpanMachine:
    """Each character is scanned once plus bounded lookahead at chunk borders."""

    __slots__ = (
        "_text",
        "_length",
        "_pending",
        "_open",
        "_start",
        "_expr",
        "_found",
        "_parsed",
        "_fault",
    )

    def __init__(self):
        self._text, self._length = StringIO(), 0
        self._pending, self._open, self._start = "", None, 0
        self._expr, self._found = [], []
        self._parsed, self._fault = None, None

    def feed(self, chunk: str) -> list[Span]:
        if self._fault is not None:
            raise self._fault
        try:
            return self._feed(chunk)
        except (GrammarError, BudgetError) as exc:
            self._fault = exc
            raise

    def _feed(self, chunk: str) -> list[Span]:
        if not isinstance(chunk, str):
            raise GrammarError("stream chunks must be text")
        if self._length + len(chunk) > MAX_TRANSCRIPT:
            raise BudgetError("transcript exceeds the character budget")
        self._text.write(chunk)
        base = self._length - len(self._pending)
        self._length += len(chunk)
        text, self._pending = self._pending + chunk, ""
        self._parsed = None
        old_count, run_start = len(self._found), 0
        # The C scanner skips ordinary '<' and shifts without per-character
        # Python calls. Only complete tags and one bounded tail need handling.
        for match in _FULL_TAG.finditer(text):
            cut, tag = match.start(), match.group()
            if self._open is None:
                if tag in _CLOSE:
                    raise GrammarError("closing tag without an open frame")
                self._open, self._start = tag, base + cut
                self._expr.clear()
            elif tag in _OPEN:
                raise GrammarError("nested operation frame")
            elif tag == ("</calc>" if self._open == "<calc>" else "</calc-final>"):
                if len(self._found) >= MAX_OPERATIONS:
                    raise BudgetError("transcript exceeds the operation budget")
                self._expr.append(text[run_start:cut])
                self._found.append(
                    Span(
                        "".join(self._expr),
                        self._open == "<calc-final>",
                        self._start,
                        base + cut + len(tag),
                    )
                )
                self._open = None
                self._expr.clear()
            else:
                raise GrammarError("mismatched closing tag")
            run_start = match.end()
        cut = text.rfind("<", run_start)
        if cut >= 0 and len(text) - cut < _MAX_TAG and _tag_at(text, cut)[1]:
            if self._open is not None:
                self._expr.append(text[run_start:cut])
            self._pending = text[cut:]
        elif self._open is not None:
            self._expr.append(text[run_start:])
        return self._found[old_count:]

    def transcript(self) -> str:
        return self._text.getvalue()

    def spans(self) -> list[Span]:
        return self._found.copy()

    def has_open_frame(self) -> bool:
        return self._open is not None

    def has_pending_tag(self) -> bool:
        return bool(self._pending)

    def pending_close(self) -> str | None:
        if self._open is None:
            return None
        return "</calc>" if self._open == "<calc>" else "</calc-final>"

    def finish(self) -> ParsedTranscript:
        if self._fault is not None:
            raise self._fault
        if self._open is not None:
            raise GrammarError("incomplete operation frame at end of stream")
        if self._pending:
            raise GrammarError("stream ends inside a frame marker")
        if self._parsed is None:
            self._parsed = ParsedTranscript(self.transcript(), tuple(self._found))
        return self._parsed

    def render(self, values: Sequence[str]) -> str:
        return self.finish().render(values)
