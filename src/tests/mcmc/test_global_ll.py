"""Tests that the *slow* and *fast* global Bernoulli log‑likelihood
implementations are numerically identical on a variety of synthetic block
partitions.

We build *real* ``BlockData`` instances by constructing a synthetic graph
(adjacency matrix) whose edge counts per block pair match a prescribed
connectivity matrix.  This avoids touching the rest of the SBM pipeline
while exercising exactly the code paths used by the likelihood routines.
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Functions under test
# ---------------------------------------------------------------------------
from sbm.likelihood import (
    compute_global_bernoulli_ll,
    compute_global_bernoulli_ll_fast,
)
from sbm.graph_data import GraphData

# ---------------------------------------------------------------------------
# Utility to create a BlockData instance whose *block_connectivity* matches a
# given integer matrix ``conn`` (undirected only, because BlockData does not
# implement the directed path).
# ---------------------------------------------------------------------------
try:
    from sbm.block_data import BlockData  # type: ignore
except ModuleNotFoundError:  # local fallback
    from block_data import BlockData  # type: ignore


def _node_ranges(sizes: Sequence[int]) -> List[Tuple[int, int]]:
    """Return (start, stop) index for each block (Python half‑open)."""
    ranges = []
    start = 0
    for sz in sizes:
        ranges.append((start, start + sz))
        start += sz
    return ranges


def build_block_data(
    block_sizes: Sequence[int],
    conn: np.ndarray,
    rng: random.Random,
) -> BlockData:
    """Construct a *consistent* BlockData (undirected) for testing."""

    B = len(block_sizes)
    assert conn.shape == (B, B)
    assert (conn == conn.T).all(), "Connectivity must be symmetric for undirected graphs."  # noqa: E501

    # Total number of nodes & adjacency matrix
    N = int(sum(block_sizes))
    adj = sp.dok_array((N, N), dtype=np.int64)

    ranges = _node_ranges(block_sizes)

    for r in range(B):
        nodes_r = list(range(*ranges[r]))

        # Diagonal block r==r
        e_rr = int(conn[r, r])
        if e_rr:
            # all unordered pairs inside block
            possible = [(u, v) for i, u in enumerate(nodes_r) for v in nodes_r[i + 1 :]]
            assert e_rr <= len(possible)
            chosen = rng.sample(possible, e_rr)
            for u, v in chosen:
                adj[u, v] = 1
                adj[v, u] = 1

        for s in range(r + 1, B):
            e_rs = int(conn[r, s])
            if not e_rs:
                continue
            nodes_s = list(range(*ranges[s]))
            possible = [(u, v) for u in nodes_r for v in nodes_s]
            assert e_rs <= len(possible)
            chosen = rng.sample(possible, e_rs)
            for u, v in chosen:
                adj[u, v] = 1
                adj[v, u] = 1

    adj = adj.tocsr()

    # Blocks mapping: node -> block_id (block IDs are 0..B-1)
    blocks = {node: b for b, (start, stop) in enumerate(ranges) for node in range(start, stop)}

    dummy_graph = GraphData(adj, directed=False)
    return BlockData(initial_blocks=blocks, graph_data=dummy_graph)

# ---------------------------------------------------------------------------
# Sanity helper
# ---------------------------------------------------------------------------

def assert_ll_equal(bd: BlockData):
    ll_slow = compute_global_bernoulli_ll(bd)
    ll_fast = compute_global_bernoulli_ll_fast(bd)
    assert np.isclose(ll_slow, ll_fast, rtol=1e-4, atol=1e-6), f"{ll_slow} != {ll_fast}"

# ==========================================================================
# TEST CASES
# ==========================================================================

# 1) Tiny hand-crafted graph -------------------------------------------------

def test_tiny_example():
    sizes = [3, 4]
    conn = np.array([[2, 5],
                     [5, 1]], dtype=np.int64)
    bd = build_block_data(sizes, conn, rng=random.Random(0))
    assert_ll_equal(bd)

# 2) Singleton block present -------------------------------------------------

def test_singleton_block():
    sizes = [1, 5, 2]
    conn = np.array([[0, 0, 0],
                     [0, 4, 3],
                     [0, 3, 1]], dtype=np.int64)
    # ensure symmetry
    conn = conn + conn.T - np.diag(conn.diagonal())
    bd = build_block_data(sizes, conn, rng=random.Random(1))
    assert_ll_equal(bd)

# 3) Random dense undirected graphs -----------------------------------------

@pytest.mark.parametrize("seed,B", [(2, 4), (3, 6)])
def test_random_dense(seed: int, B: int):
    rng = random.Random(seed)
    sizes = [rng.randint(2, 6) for _ in range(B)]

    conn = np.zeros((B, B), dtype=np.int64)
    for r in range(B):
        n_rr = sizes[r] * (sizes[r] - 1) // 2
        conn[r, r] = rng.randrange(n_rr + 1)
        for s in range(r + 1, B):
            n_rs = sizes[r] * sizes[s]
            val = rng.randrange(n_rs + 1)
            conn[r, s] = conn[s, r] = val

    bd = build_block_data(sizes, conn, rng)

    assert_ll_equal(bd)
