# src/tests/test_regressions.py
"""
Targeted regression tests for issues uncovered in the planted‐partition script.

* size-1 blocks must not crash likelihood computation
* Δ-dicts must use *matrix indices*, not block-IDs
"""

import numpy as np
import pytest
from scipy.sparse import csr_array

from sbm.block_data import BlockData
from sbm.likelihood import compute_global_bernoulli_ll
from sbm.block_change_proposers import NodeSwapProposer
from sbm.block_data import BlockData
from sbm.graph_data import GraphData


# -------------------------------------------------------------------
# 1. size-1 diagonal must be ignored (or handled gracefully)
# -------------------------------------------------------------------
@pytest.mark.parametrize("singletons", [1, 3])
def test_singleton_blocks_allowed(singletons):
    """
    A partition containing blocks of size 1 must not raise or return NaN.
    """
    n = 6
    # make a path graph (any sparse graph works)
    rows = np.arange(n-1)
    cols = rows + 1
    data = np.ones(n-1, dtype=int)
    A = csr_array((data, (rows, cols)), shape=(n, n))
    A = A + A.T

    # put the first `singletons` nodes into their own blocks
    blocks = {v: v if v < singletons else singletons for v in range(n)}
    bd = BlockData(
        initial_blocks=blocks,
        graph_data=GraphData(adjacency_matrix=A, directed=False)
    )

    ll = compute_global_bernoulli_ll(bd)
    assert np.isfinite(ll), "likelihood should be finite even with size-1 blocks"


# -------------------------------------------------------------------
# 2. Δ-dicts must reference matrix indices, not block-IDs
# -------------------------------------------------------------------
def test_delta_keys_are_matrix_indices():
    """
    When block IDs are non-contiguous (e.g. {0,10}), the delta_e keys
    must still be *matrix indices* (0 or 1), otherwise the likelihood
    updater crashes with an IndexError.
    """
    # two blocks with ids 0 and 10, one edge across
    adj = csr_array([[0,1],[1,0]])
    blocks = {0: 0, 1: 10}
    bd = BlockData(
        initial_blocks=blocks,
        graph_data=GraphData(adjacency_matrix=adj, directed=False)
    )

    proposer = NodeSwapProposer(bd)
    swap = [(0, 10)]  # move node 0 to block 10 -> will create a singleton & trigger Δ
    _, delta_e, _ = proposer.propose_change(swap)

    # the only valid matrix indices are 0 and 1
    valid = {0,1}
    for (r, s) in delta_e:
        assert r in valid and s in valid, (
            "delta_e must use matrix indices (0..B-1), "
            "not raw block IDs"
        )