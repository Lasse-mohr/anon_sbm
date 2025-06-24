import numpy as np
import scipy.sparse as sp
import pytest

from sbm.sampling import (
    sample_adjacency_matrix,
    sample_sbm_graph_from_fit,
)
from sbm.io import SBMFit
from sbm.graph_data import GraphData

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(12345)

# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
def csr_edges_between(adj: sp.csr_array, idx_a, idx_b, directed:bool = False):
    """number of non-zero entries with row in A and col in B"""
    sub = adj[idx_a][:, idx_b] # type: ignore

    if (idx_a == idx_b) and not directed:
        # self-edges, count only upper triangle
        return sub.count_nonzero() // 2 # type: ignore

    if directed:
        # directed, count all edges
        print('')
        print(sub.toarray().tolist())
        print(adj.toarray().tolist())
        print('')

    return sub.sum() # type: ignore


def complete_block_edges(n, directed):
    return n * (n - 1) if directed else n * (n - 1) // 2

# --------------------------------------------------------------------
# 1. full connectivity should yield a complete bipartite/clique
# --------------------------------------------------------------------
@pytest.mark.parametrize("directed", [False, True])
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

    assert csr_edges_between(adj, idx1, idx1, directed) == conn[1, 1], \
        (
            f"expected {conn[1, 1]} edges within block 1"
            f"sizes: {sizes}, conn: {conn}, directed: {directed}"
        )

    # --- between blocks ---------------------------------------------
    expect = conn[0, 1]
    assert csr_edges_between(adj, idx0, idx1, directed) == expect ,\
        (
            f"expected {expect} edges between blocks 0 and 1. Got {csr_edges_between(adj, idx0, idx1)}. "
            f"sizes: {sizes}, conn: {conn.toarray().flatten()}, directed: {directed}"
        )
    assert csr_edges_between(adj, idx1, idx0, directed) == expect, \
        (
            f"expected {expect} edges between blocks 1 and 0. Got {csr_edges_between(adj, idx1, idx0)}. "
            f"sizes: {sizes}, conn: {conn.toarray().flatten()}, directed: {directed}"
        )
    if directed:
        # both directions filled
        assert csr_edges_between(adj, idx1, idx0, directed) == expect, \
                f"expected {expect} edges between blocks 1 and 0. Got {csr_edges_between(adj, idx1, idx0)}. "

    # --- no self-loops ----------------------------------------------
    assert adj.diagonal().sum() == 0, \
                "expected no self-loops in the adjacency matrix"

# --------------------------------------------------------------------
# 2. zero connectivity must yield zero edges between blocks
# --------------------------------------------------------------------
def test_zero_connectivity(rng):
    n1, n2 = 5, 6
    sizes = [n1, n2]
    conn = sp.csr_array([[10, 0],
                         [0, 15]], dtype=int)

    adj = sample_adjacency_matrix(sizes, conn, directed=False, rng=rng)

    idx0 = slice(0, n1)
    idx1 = slice(n1, n1 + n2)
    assert csr_edges_between(adj, idx0, idx1) == 0
    assert csr_edges_between(adj, idx1, idx0) == 0


# --------------------------------------------------------------------
# 3. large probabilistic block matches expected count ±3σ
# --------------------------------------------------------------------
def test_statistical_match(rng):
    n1, n2 = 100, 200
    sizes = [n1, n2]
    p = 0.15
    m = int(p * n1 * n2)

    conn = sp.csr_array((2, 2), dtype=int)
    conn[0, 1] = conn[1, 0] = m

    trials = 10
    errs = []
    for _ in range(trials):
        adj = sample_adjacency_matrix(sizes, conn, directed=False, rng=rng)
        idx0 = slice(0, n1)
        idx1 = slice(n1, n1 + n2)
        observed = csr_edges_between(adj, idx0, idx1)
        errs.append(observed - m)

    std = np.sqrt(n1 * n2 * p * (1 - p))
    assert max(map(abs, errs)) < 3 * std, \
        (
            f"expected observed edge counts to match {m} ± 3σ. Got {errs}. "
        )


# --------------------------------------------------------------------
# 4. directed vs undirected symmetry
# --------------------------------------------------------------------
def test_directed_flag(rng):
    sizes = [10, 10]
    B = 2
    conn = sp.csr_array((B, B), dtype=int)
    conn[0, 1] = conn[1, 0] = 10 * 10 / 2   # p = 0.5

    adj_d = sample_adjacency_matrix(sizes, conn, directed=True, rng=rng)
    assert (adj_d != adj_d.T).nnz > 0, \
        "expected directed graph to be asymmetric"

    adj_u = sample_adjacency_matrix(sizes, conn, directed=False, rng=rng)
    assert (adj_u != adj_u.T).nnz == 0


# --------------------------------------------------------------------
# 5. sampling via SBMFit wrapper
# --------------------------------------------------------------------
def test_sample_from_fit(rng):
    sizes = [2, 2]
    conn = sp.csr_array([[1, 2],
                         [2, 1]], dtype=int)
    fit = SBMFit(
        block_sizes    = sizes,
        block_conn     = conn,
        directed_graph = False,
        neg_loglike    = -1.0,
        metadata       = {},
    )
    g = sample_sbm_graph_from_fit(fit, rng)
    assert isinstance(g, GraphData)
    assert g.adjacency.shape == (4, 4)