import numpy as np
import scipy.sparse as sp
import pytest

from sbm.sampling import (
    sample_adjacency_matrix,
    sample_sbm_graph_from_fit,
)
from sbm.io import SBMFit
from sbm.graph_data import GraphData

# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
def csr_edges_between(adj: sp.csr_array, idx_a, idx_b, directed:bool = False):
    """number of non-zero entries with row in A and col in B"""
    sub = adj[idx_a][:, idx_b] # type: ignore

    if idx_a == idx_b and not directed:
        # self-edges, count only upper triangle
        return sub.count_nonzero() // 2 # type: ignore

    elif not directed:
        # undirected, count both directions
        return sub.count_nonzero() // 2 + sub[idx_a][:, idx_b].T.count_nonzero() // 2 # type: ignore

    else:
        return sub.count_nonzero() # type: ignore

def complete_block_edges(n, directed):
    return n * (n - 1) if directed else n * (n - 1) // 2

# --------------------------------------------------------------------
# 1. full connectivity should yield a complete bipartite/clique
# --------------------------------------------------------------------
def test_full_connectivity(rng, directed):
    n1, n2 = 3, 4
    sizes = [n1, n2]

    # maximum possible edges
    B = 2
    conn = sp.csr_array((B, B), dtype=int)
    conn[0, 0] = complete_block_edges(n1, directed)
    conn[1, 1] = complete_block_edges(n2, directed)
    conn[0, 1] = conn[1, 0] = n1 * n2

    adj = sample_adjacency_matrix(sizes, conn, directed=directed, rng=rng)

    idx0 = slice(0, n1)
    idx1 = slice(n1, n1 + n2)

    # --- within-block ------------------------------------------------
    assert csr_edges_between(adj, idx0, idx0, directed) == conn[0, 0], \
        (
            f"expected {conn[0, 0]} edges within block 0. Got {csr_edges_between(adj, idx0, idx0)}. "
            f"sizes: {sizes}, conn: {conn.toarray().flatten()}, directed: {directed}"
        )

    assert csr_edges_between(adj, idx1, idx1) == conn[1, 1], \
        (
            f"expected {conn[1, 1]} edges within block 1"
            f"sizes: {sizes}, conn: {conn}, directed: {directed}"
        )

    # --- between blocks ---------------------------------------------
    expect = conn[0, 1]
    assert csr_edges_between(adj, idx0, idx1) == expect,\
        (
            f"expected {expect} edges between blocks 0 and 1. Got {csr_edges_between(adj, idx0, idx1)}. "
            f"sizes: {sizes}, conn: {conn.toarray().flatten()}, directed: {directed}"
        )
    assert csr_edges_between(adj, idx1, idx0) == expect, \
        (
            f"expected {expect} edges between blocks 1 and 0. Got {csr_edges_between(adj, idx1, idx0)}. "
            f"sizes: {sizes}, conn: {conn.toarray().flatten()}, directed: {directed}"
        )
    if directed:
        # both directions filled
        assert csr_edges_between(adj, idx1, idx0) == expect, \
                f"expected {expect} edges between blocks 1 and 0. Got {csr_edges_between(adj, idx1, idx0)}"

    # --- no self-loops ----------------------------------------------
    assert adj.diagonal().sum() == 0, \
                "expected no self-loops in the adjacency matrix"


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # For reproducibility

    for directed in [False, True]:
        test_full_connectivity(rng, directed)
        print(f"Test for {'directed' if directed else 'undirected'} graph passed.")
