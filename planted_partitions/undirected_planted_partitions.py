#!/usr/bin/env python3
"""
undirected_planted_partition.py
--------------------------------
Simple smoke-test for the swap-only MCMC implementation.

For each of 100 independent repetitions we

1.  Draw an undirected 100-node SBM with
        – B = 10 blocks of size 10
        – p_in  = 0.30   (within-block connection probability)
        – p_out = 0.05   (between-block probability)

2.  Build an initial *random* equal-size partition with
        UniformSmallBlockAssigner(min_block_size=10)             (code in block_assigner.py)

3.  Run the adaptive swap-only MCMC for `n_iter` iterations.

4.  Compute the Jaccard index between
        – the set of node pairs co-clustered in the *final* state, and
        – the same set for the planted partition.

The script prints the mean, standard deviation and a histogram
of the 100 Jaccard scores so you can eyeball whether the sampler
typically finds the planted structure.

Dependencies
------------
Only `numpy`, `scipy` and the local `sbm` package (already required by
your project).

Author: Von Nøgenmand
"""

from typing import Sequence, Hashable
from collections.abc import Sequence
import numpy as np
from scipy.sparse import csr_array
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from tqdm import tqdm

from sbm.graph_data import GraphData
from sbm.block_assigner import (
    UniformSmallBlockAssigner,
    MetisBlockAssigner,
)
from sbm.model import SBMModel




###############################################################################
# utility helpers
###############################################################################
def planted_blocks(n_nodes: int, block_size: int, rng) -> dict[int, int]:
    """Random planted partition: block 0 → nodes 0–9, block 1 → 10–19, …"""

    random_nodes = rng.permutation(n_nodes)
    # generate a random list of blocks for the nodes ensuring the correct block size
    random_blocks = np.arange(n_nodes) // block_size
    random_blocks = rng.permutation(random_blocks)

    return {v: block for (v, block) in zip(random_nodes, random_blocks)}


def sample_sbm(rng: np.random.Generator,
               blocks: dict[int, int],
               p_in: float,
               p_out: float) -> csr_array:
    """Generate an undirected loop-free adjacency matrix for a binary SBM."""
    n = len(blocks)
    adj = np.zeros((n, n), dtype=np.int8)

    # probability matrix look-up
    for u in range(n):
        for v in range(u + 1, n):          # u < v → strict upper triangle
            p = p_in if blocks[u] == blocks[v] else p_out
            if rng.random() < p:
                adj[u, v] = adj[v, u] = 1  # symmetrise

    return csr_array(adj)                 # sparse CSR


def misclassification_rate(
    true_labels: Sequence[Hashable],
    est_labels: Sequence[Hashable],
) -> float:
    """
    Percentage of vertices whose community label is wrong *after*
    optimally permuting the estimated labels to match the true ones.

    Parameters
    ----------
    true_labels : sequence
        Ground-truth block labels – length N.
    est_labels  : sequence
        Estimated block labels  – length N.

    Returns
    -------
    float
        Mis-classification rate in the interval [0, 1].

    Notes
    -----
    * Label sets may use arbitrary hashables (str, int, …) and need not
      have the same cardinality.  Any surplus estimated or true blocks
      are matched to “dummy” columns/rows filled with zeros.
    * Uses the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`)
      to maximise the number of correctly matched vertices.
    """
    true = np.asarray(true_labels)
    est  = np.asarray(est_labels)

    if true.shape != est.shape:
        raise ValueError("true_labels and est_labels must have the same length")

    # Map arbitrary labels to contiguous integers 0..T-1 and 0..E-1
    true_ids,  true_inv  = np.unique(true, return_inverse=True)
    est_ids,   est_inv   = np.unique(est,  return_inverse=True)

    T, E = len(true_ids), len(est_ids)
    N    = len(true)

    # Build contingency matrix C[e, t] = |{ i : est_i=e and true_i=t }|
    C = np.zeros((E, T), dtype=int)
    np.add.at(C, (est_inv, true_inv), 1)

    # Pad to square (Hungarian implementation needs it or we need to
    # maximise on rectangles by padding zeros).
    if E != T:
        dim = max(E, T)
        C_padded = np.zeros((dim, dim), dtype=int)
        C_padded[:E, :T] = C
        C = C_padded

    # Maximise trace(C[perm])  →  minimise −C for Hungarian
    row_ind, col_ind = linear_sum_assignment(-C)
    matched = C[row_ind, col_ind].sum()

    return 1.0 - matched / N


###############################################################################
# main loop
###############################################################################

def main(
    n_nodes = 100,
    block_size = 10,
    p_in = 0.30,
    p_out = 0.05,
    n_experiments = 10,
    n_iter = 3_000,
    rng_master = np.random.default_rng(42),
    temperature: float = 1
):
    init_scores = []
    final_scores = []

    for rep in tqdm(range(n_experiments)):
        rng = np.random.default_rng(rng_master.integers(2**32))

        # --- 1. plant graph -----------------------------------------------------
        planted = planted_blocks(n_nodes, block_size, rng)
        adj     = sample_sbm(rng, planted, p_in, p_out)

        # --- 2. initial random partition ---------------------------------------
        gdata    = GraphData(adjacency_matrix=adj, directed=False)
        assigner = MetisBlockAssigner(graph_data=gdata,
                                            rng=rng,
                                            min_block_size=block_size
                                        )
        init_blocks = assigner.compute_assignment()

        init_scores.append(
            misclassification_rate(
                true_labels=list(planted.values()),
                est_labels=list(init_blocks.blocks.values())
            )
        )

        sbm = SBMModel(
                initial_blocks=init_blocks,
                rng=rng,
                log=True,  # no logging
            )
        print(f"Initial ll {sbm.likelihood_calculator.ll:.3f}")

        sbm.fit(num_iterations=n_iter,
                min_block_size=block_size,
                initial_temperature=temperature,
                cooling_rate=0.999)

        final_blocks = sbm.get_block_assignments()

        # --- 4. score -----------------------------------------------------------
        final_scores.append(
            misclassification_rate(
                true_labels=list(planted.values()),
                est_labels=list(final_blocks.values())
            )
        )

    # --- print results ----------------------------------------------------------
    print(f"Initial misclassification rate: {np.mean(init_scores):.3f} ± {np.std(init_scores):.3f}")
    print(f"Final misclassification rate:   {np.mean(final_scores):.3f} ± {np.std(final_scores):.3f}")


if __name__ == "__main__":
    main(
        n_nodes=300,
        block_size=3,
        p_in=0.5,
        p_out=0.01,
        n_experiments=1,
        n_iter=5_000,
        rng_master=np.random.default_rng(42),
        temperature=1e-2
    )