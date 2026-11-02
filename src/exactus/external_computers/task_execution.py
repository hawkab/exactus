"""Execute digest-bound tasks with request-local result reuse."""

__author__ = "Olshansky Grigory"

import hashlib

from .task_specs import execute, parse
from .values import packed


class TaskExecution:
    """Own computed outputs, cache entries, and result evidence for one request."""

    def __init__(self, external_computer, sql_session, encode, aggregate=None):
        """Bind evaluators and the encoding policy for a private request snapshot."""
        self._external_computer = external_computer
        self._sql_session = sql_session
        self._encode = encode
        self._aggregate = aggregate
        self._cache = {}
        self.outputs = {}
        self.rows = []

    def run(self, identifier, specification):
        """Execute one resolved task and retain its canonical input digest.

        Args:
            identifier: Unique task identifier in the current graph.
            specification: Resolved operation fields from a private snapshot.

        Returns:
            The computed JSON value, also available through ``outputs``.
        """
        canonical = self._encode(specification)
        task = parse(specification)
        digest = hashlib.sha256(
            ("exactus-external-computer-v1\0" + canonical).encode()
        ).hexdigest()
        cached = digest in self._cache
        value = (
            self._cache[digest]
            if cached
            else execute(task, self._external_computer, self._sql_session)
        )
        self.outputs[identifier] = self._cache[digest] = value
        self.rows.append(
            {
                "id": identifier,
                "kind": specification["kind"],
                "value": value,
                "input_sha256": digest,
                "cached": cached,
            }
        )
        if self._aggregate is None:
            packed(value)
        else:
            self._aggregate.append(self.rows[-1])
        return value
