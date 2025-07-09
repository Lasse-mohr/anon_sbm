""" 
Functions for sampling graph from SBM model
"""
# sbm/sampling.py
from typing import List, Optional

import numpy as np
from scipy.sparse import csr_array, coo_matrix
from sbm.graph_data import GraphData
from sbm.io import SBMFit


def sample_adjacency_matrix(
    block_sizes: List[int],
    block_connectivity: csr_array,
    rng: np.random.Generator,
    directed: bool = False,
) -> csr_array:
    """
    Draw a random graph from the *profile* Bernoulli SBM specified by
    `block_edge_counts` (edge counts m_rs) and `block_sizes`.

    :param block_sizes: Sizes of the blocks.
    :param block_connectivity: Sparse matrix of edge counts m_rs between blocks.
    :param directed: Whether the graph is directed or undirected.
    :param rng: Random number generator for reproducibility.

    :return: Sparse adjacency matrix of the sampled graph.
    """


    block_sizes = list(map(int, block_sizes))
    B = len(block_sizes)
    N = sum(block_sizes)

    # cumulative offsets → map local idx → global idx
    offsets = np.cumsum([0] + block_sizes)

    rows: list[int] = []
    cols: list[int] = []

    # ------------------------------------------------------------------
    for r in range(B):
        n_r = block_sizes[r]
        off_r = offsets[r]

        # -- diagonal block -------------------------------------------
        m_rr = int(block_connectivity[r, r]) # type: ignore
        if m_rr:
            if directed:
                n_poss = n_r * (n_r - 1)
                p = m_rr / n_poss

                mask = (rng.random((n_r, n_r)) < p).astype(int)
                mask[np.diag_indices(n_r)] = 0

                rr, cc = np.nonzero(mask)

                rows.extend(off_r + rr)
                cols.extend(off_r + cc)

            else:
                n_poss = n_r * (n_r - 1) // 2
                p = m_rr / n_poss
                triu_mask = rng.random((n_r, n_r)) < p
                tri_r, tri_c = np.triu_indices(n_r, k=1)
                sel = triu_mask[tri_r, tri_c]
                rr = tri_r[sel]; cc = tri_c[sel]

                rows.extend(off_r + rr)
                cols.extend(off_r + cc)

                rows.extend(off_r + cc)
                cols.extend(off_r + rr)

        # -- off-diagonal blocks --------------------------------------
        s_iter = range(B) if directed else range(r + 1, B)
        for s in s_iter:
            if s == r:
                continue

            m_rs = int(block_connectivity[r, s]) # type: ignore

            if m_rs == 0:
                continue

            n_s = block_sizes[s]
            off_s = offsets[s]
            n_poss = n_r * n_s
            p = m_rs / n_poss

            mask = rng.random((n_r, n_s)) < p
            rr, cc = np.nonzero(mask)
            rows.extend(off_r + rr)
            cols.extend(off_s + cc)

            if not directed:
                # mirror block
                rows.extend(off_s + cc)
                cols.extend(off_r + rr)

    data = np.ones(len(rows), dtype=np.int8)
    adj = coo_matrix((data, (rows, cols)), shape=(N, N))

    # ensure no duplicate edge
    adj.sum_duplicates() 
    adj.data.fill(1)

    # convert to csr format
    adj = csr_array(adj)
    adj.sort_indices()

    return adj


def sample_sbm_graph(
            block_sizes: List[int],
            block_connectivity: csr_array,
            directed:bool,
            rng: np.random.Generator,
            metadata: Optional[dict] = None
    )->GraphData:
    """
    Sample a graph from a Stochastic Block Model (SBM) given block sizes and connectivity.
    :param block_sizes: List of sizes for each block.
    :param block_connectivity: Sparse matrix representing connectivity between blocks.
    :param directed: Whether the graph is directed or undirected.
    :param rng: Random number generator for reproducibility.
    :param metadata: Optional metadata to include in the graph data.

    :return: GraphData object containing the sampled graph.
    """

    if metadata is None:
        metadata = {}

    # Validate inputs
    if not isinstance(block_sizes, list) or not all(isinstance(size, int) for size in block_sizes):
        raise ValueError("block_sizes must be a list of integers.")
    if not isinstance(block_connectivity, csr_array):
        raise ValueError("block_connectivity must be a scipy.sparse.csr_array.")
    if len(block_sizes) != block_connectivity.shape[0] or len(block_sizes) != block_connectivity.shape[1]: #type: ignore
        raise ValueError("block_sizes length must match the dimensions of block_connectivity.")
    if not isinstance(directed, bool):
        raise ValueError("directed must be a boolean value.")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy random Generator instance.")    

    adj = sample_adjacency_matrix(
        block_sizes=block_sizes,
        block_connectivity=block_connectivity,
        directed=directed,
        rng=rng
    )
    return GraphData(adjacency_matrix=adj, directed=directed)


def sample_sbm_graph_from_fit(sbm_fit: SBMFit, rng: np.random.Generator) -> GraphData:
    """
    Sample a graph from a Stochastic Block Model (SBM) fit.
    
    :param sbm_fit: SBMFit object containing block sizes and connectivity.
    :param rng: Random number generator for reproducibility.
    
    :return: GraphData object containing the sampled graph.
    """
    return sample_sbm_graph(
        block_sizes=sbm_fit.block_sizes,
        block_connectivity=sbm_fit.block_conn,
        directed=sbm_fit.directed_graph,
        rng=rng,
        metadata=sbm_fit.metadata
    )