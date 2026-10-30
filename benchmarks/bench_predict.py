"""Benchmark: contour two draft accuracy on a held-out number split.

Author: Olshansky Grigory
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exactus.adapters.ngram import NgramDraftPredictor

__author__ = "Olshansky Grigory"

TRAIN = [
    "total <calc>37*184500</calc> ok",
    "total <calc>37*184500</calc> done",
    "total <calc>12345*6789</calc> ok",
    "check <calc>2^100</calc> fine",
    "check <calc>2^100</calc> done",
    "sum <calc>999999999+1</calc> ok",
]
TEST_IN_SAMPLE = [
    "total <calc>37*184500</calc>",
    "check <calc>2^100</calc>",
    "sum <calc>999999999+1</calc>",
]
TEST_HELD_OUT = [
    "total <calc>31*70273</calc>",
    "check <calc>2^87</calc>",
    "sum <calc>555555555+2</calc>",
]


def accuracy(predictor: NgramDraftPredictor, sequences: list[list[int]],
             blind_token: int) -> tuple[int, int, int, int]:
    """Return (ngram hits, blind hits, total) top-1 accuracy counts."""
    hits = blind_hits = total = 0
    for sequence in sequences:
        for position in range(1, len(sequence)):
            prefix, actual = sequence[:position], sequence[position]
            drafts = predictor.predict(prefix, k=1)
            hits += bool(drafts) and drafts[0].tokens[0] == actual
            blind_hits += blind_token == actual
            total += 1
    return hits, blind_hits, total


def main() -> None:
    """Measure top-1 accuracy in-sample and held-out against a blind baseline."""
    train = [list(s.encode("utf-8")) for s in TRAIN]
    predictor = NgramDraftPredictor(order=6)
    predictor.fit(train)
    counts = Counter(token for seq in train for token in seq)
    blind_token = counts.most_common(1)[0][0]
    for name, texts in (("in-sample (train prefixes)", TEST_IN_SAMPLE),
                        ("held-out (new numbers)", TEST_HELD_OUT)):
        hits, blind_hits, total = accuracy(predictor, [list(s.encode("utf-8")) for s in texts],
                                           blind_token)
        print(f"{name}: probe positions={total}")
        print(f"  ngram top-1 {hits}/{total} = {hits / total:.3f}")
        print(f"  blind      {blind_hits}/{total} = {blind_hits / total:.3f}")
    print("in-sample numbers measure memorization; held-out numbers measure")
    print("template-level prediction; both are toy diagnostics on byte ids,")
    print("not MTP acceptance. Drafts require a verifier in the calling loop;")
    print("no lossless guarantee is claimed.")


if __name__ == "__main__":
    main()
