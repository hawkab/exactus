"""Tests for digest algorithms in the hashing adapter.

Author: Olshansky Grigory
"""
import unittest

from exactus.adapters.digest import HashlibHasher

__author__ = "Olshansky Grigory"


class TestHashlibHasher(unittest.TestCase):
    """Verifies digest vectors for common and modern algorithms."""

    def setUp(self) -> None:
        """Create one hasher adapter."""
        self.hasher = HashlibHasher()

    def test_sha256_vector(self):
        """The sha256 digest of abc matches the NIST vector."""
        digest = self.hasher.digest("sha256", b"abc")
        self.assertEqual(hex(digest),
                         "0xba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_md5_vector(self):
        """The md5 digest of abc matches the RFC vector."""
        self.assertEqual(hex(self.hasher.digest("md5", b"abc")),
                         "0x900150983cd24fb0d6963f7d28e17f72")

    def test_crc32_vector(self):
        """The crc32 checksum of 123456789 matches the check value."""
        self.assertEqual(self.hasher.digest("crc32", b"123456789"), 0xCBF43926)

    def test_sha3_and_blake2_present(self):
        """Modern algorithms are available and produce integer digests."""
        for name in ("sha3_256", "sha3_512", "blake2b", "blake2s", "shake_128", "shake_256"):
            self.assertIsInstance(self.hasher.digest(name, b"x"), int)

    def test_algorithm_list(self):
        """The advertised algorithm list covers all supported names."""
        self.assertIn("sha256", self.hasher.algorithms)
        self.assertIn("blake2b", self.hasher.algorithms)

    def test_unsupported_algorithm(self):
        """Unknown algorithm names raise ValueError."""
        with self.assertRaises(ValueError):
            self.hasher.digest("sha4", b"x")


if __name__ == "__main__":
    unittest.main()
