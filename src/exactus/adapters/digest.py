"""Hashlib-backed digest adapter with common and modern algorithms.

Author: Olshansky Grigory
"""
from __future__ import annotations

import hashlib
import zlib

__author__ = "Olshansky Grigory"

_DIGEST_BYTES = 32
_BLOCK = {"shake_128", "shake_256"}
_SAFE = ("adler32", "blake2b", "blake2s", "crc32", "md5", "sha1", "sha256", "sha512",
         "sha3_256", "sha3_512", "shake_128", "shake_256")


class HashlibHasher:
    """Computes integer digests over byte payloads using hashlib.

    Covers the most common algorithms (crc32, md5, sha1, sha256, sha512)
    and the modern ones (sha3, shake, blake2). Digests render as unsigned
    big-endian integers so they stay inside the expression grammar.
    """

    @property
    def algorithms(self) -> tuple[str, ...]:
        """Return the supported algorithm names."""
        return _SAFE

    def digest(self, algorithm: str, payload: bytes) -> int:
        """Return the digest of payload as an unsigned integer."""
        if algorithm not in _SAFE:
            raise ValueError(f"unsupported algorithm: {algorithm}")
        if algorithm == "crc32":
            return zlib.crc32(payload) & 0xFFFFFFFF
        if algorithm == "adler32":
            return zlib.adler32(payload) & 0xFFFFFFFF
        digest = hashlib.new(algorithm)
        digest.update(payload)
        raw = digest.digest(_DIGEST_BYTES) if algorithm in _BLOCK else digest.digest()
        return int.from_bytes(raw, "big")
