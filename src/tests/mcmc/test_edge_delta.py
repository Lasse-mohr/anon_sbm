# tests/test_edge_delta_equivalence.py
"""
Unit tests that compare behaviour of sbm.edge_delta.EdgeDelta
and its NumPy-accelerated subclass NumpyEdgeDelta.

The tests are written against the public API actually used by
block-change proposers and the likelihood calculator:

    • increment(counts, blocks_i, blocks_j)              :contentReference[oaicite:0]{index=0}
    • __getitem__, __len__, items                       :contentReference[oaicite:1]{index=1}
"""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

import numpy as np
import pytest

from sbm.edge_delta import EdgeDelta, NumpyEdgeDelta


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _random_updates(
    n_blocks: int,
    n_updates: int,
    *, rng: random.Random
) -> tuple[list[int], list[int], list[int]]:
    """
    Create a *single* batch of updates without duplicate (i, j) pairs
    – this mirrors how `_compute_delta_edge_counts` builds its argument
    lists before calling ``increment`` once per batch.                   :contentReference[oaicite:2]{index=2}
    """
    pairs = random.sample(list(combinations(range(n_blocks), 2)), k=n_updates)
    counts = [rng.randint(-5, 5) for _ in range(n_updates)]
    blocks_i, blocks_j = zip(*pairs)   # already i < j
    return counts, list(blocks_i), list(blocks_j)


def _build_two_deltas(
    n_blocks: int,
    counts: list[int],
    blocks_i: list[int],
    blocks_j: list[int],
) -> tuple[EdgeDelta, NumpyEdgeDelta]:
    """
    Convenience wrapper: build and *increment once* – just like the
    real code does.                                                      :contentReference[oaicite:3]{index=3}
    """
    d_py  = EdgeDelta(n_blocks)
    d_np  = NumpyEdgeDelta(n_blocks)
    d_py.increment(counts, blocks_i, blocks_j)
    d_np.increment(counts, blocks_i, blocks_j)
    return d_py, d_np


# --------------------------------------------------------------------------- #
# public API parity tests                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_blocks, n_updates, seed", [
    (5,  6,  1),
    (7, 10, 42),
    (3,  2, 99),
])
def test_increment_equivalence(n_blocks: int, n_updates: int, seed: int) -> None:
    """After an identical batch update, both classes hold exactly the same deltas."""
    rng = random.Random(seed)
    counts, bi, bj = _random_updates(n_blocks, n_updates, rng=rng)
    d_py, d_np = _build_two_deltas(n_blocks, counts, bi, bj)

    # compare through the *public* interface ─ not private storage
    pairs = set(d_py.items()) | set(d_np.items())
    for (i, j), _ in pairs:
        assert d_py[i, j] == d_np[i, j], \
            f"Mismatch on pair {(i, j)}: python={d_py[i, j]}, numpy={d_np[i, j]}"

    assert len(d_py) == len(d_np), \
        f"__len__ diverged: python={len(d_py)}, numpy={len(d_np)}"

    # full dict comparison (order-independent)
    assert dict(d_py.items()) == dict(d_np.items())


def test_getitem_default_zero() -> None:
    """Both classes must return 0 for unseen (i,j) pairs."""
    d_py  = EdgeDelta(4)
    d_np  = NumpyEdgeDelta(4)
    for pair in ((0, 0), (0, 1), (2, 3)):
        assert d_py[pair] == d_np[pair] == 0


def test_negative_and_positive_counts() -> None:
    """Signed counts stay intact and are *not* silently truncated."""
    counts  = [  5, -3,  2]
    blocks_i = [0, 0, 1]
    blocks_j = [1, 2, 2]

    d_py, d_np = _build_two_deltas(3, counts, blocks_i, blocks_j)
    assert dict(d_py.items()) == { (0, 1): 5, (0, 2): -3, (1, 2): 2 }
    assert dict(d_py.items()) == dict(d_np.items())


# --------------------------------------------------------------------------- #
# integration smoke test – reproduces the exact public call-sequence used
# by `_compute_delta_edge_counts`                                            #
# --------------------------------------------------------------------------- #
def test_two_step_update_matches() -> None:
    """
    `_compute_delta_edge_counts` issues *two* successive ``increment`` calls
    on the *same* EdgeDelta instance.  Here we reproduce that pattern and make
    sure the NumPy implementation yields identical final deltas after both
    steps.                                                                   :contentReference[oaicite:4]{index=4}
    """
    n_blocks = 4
    # step-1: neighbour blocks
    counts1  = [ 2, -1]
    blocks1a = [0, 1]
    blocks1b = [2, 2]

    # step-2: intra / inter old blocks
    counts2  = [ 7, -4, -3]
    blocks2a = [0, 0, 1]
    blocks2b = [1, 0, 1]

    py = EdgeDelta(n_blocks)
    npd = NumpyEdgeDelta(n_blocks)

    py.increment(counts1, blocks1a, blocks1b)
    py.increment(counts2, blocks2a, blocks2b)

    npd.increment(counts1, blocks1a, blocks1b)
    npd.increment(counts2, blocks2a, blocks2b)

    assert dict(py.items()) == dict(npd.items())
