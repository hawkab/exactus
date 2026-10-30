"""Tests for the hash-indexed n-gram draft predictor.

Author: Olshansky Grigory
"""
import unittest

from exactus.adapters.ngram import NgramDraftPredictor
from exactus.application.predict import DraftService

__author__ = "Olshansky Grigory"


class TestNgramDraftPredictor(unittest.TestCase):
    """Verifies contour two drafting and its verify-protection contract."""

    def setUp(self) -> None:
        """Fit one predictor on a small corpus."""
        self.predictor = NgramDraftPredictor(order=3)
        self.predictor.fit([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5]])
        self.service = DraftService(self.predictor)

    def test_top_candidate(self):
        """The most frequent continuation ranks first."""
        candidates = self.predictor.predict([1, 2, 3], k=2)
        self.assertEqual(candidates[0].tokens, (5,))
        self.assertGreater(candidates[0].score, candidates[1].score)

    def test_k_limit(self):
        """Predictions respect the k limit."""
        self.assertLessEqual(len(self.predictor.predict([1, 2, 3], k=1)), 1)

    def test_no_match(self):
        """Unknown prefixes yield no drafts."""
        self.assertEqual(self.predictor.predict([9, 9, 9]), [])

    def test_shorter_order_fallback(self):
        """Shorter n-grams fill in when the longest one misses."""
        candidates = self.predictor.predict([7, 3], k=1)
        self.assertEqual(candidates[0].tokens, (5,))

    def test_service_delegates(self):
        """The draft service delegates to the predictor port."""
        self.assertEqual(self.service.suggest([1, 2, 3], k=1)[0].tokens, (5,))


if __name__ == "__main__":
    unittest.main()
