# tests/test_likelihood.py
"""
End-to-end tests for the Bernoulli likelihood calculators.

 * We create a tiny 4-node undirected graph with two blocks.
 * We compare every Δℓ returned by the incremental code against the
   ground-truth global likelihood difference.
"""

from __future__ import annotations

from collections import Counter
from typing import Tuple, Dict

import numpy as np
#np.float_ = np.float64  # ensure float64 is used

import pytest
from scipy.sparse import csr_array

from sbm.likelihood import (
    compute_global_bernoulli_ll,
    LikelihoodCalculator
)

from sbm.block_data import BlockData
from sbm.graph_data import GraphData
from sbm.block_change_proposers import NodeSwapProposer 

##############################################################################
# Fixtures
##############################################################################


@pytest.fixture(scope="module")
def four_node_example() -> Tuple[BlockData, GraphData, Dict[int, int]]:
    """
    Graph:

        0──1   Block 0: {0,1}      Edges: (0,1)
        │  │   Block 1: {2,3}              (2,3)
        2──3                              (0,2) (1,3)

    The matrix is symmetrical (undirected, no loops).
    """
    adj = np.zeros((4, 4), dtype=int)
    edges = [(0, 1), (2, 3), (0, 2), (1, 3)]
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1  # symmetric

    adjacency = csr_array(adj)
    blocks = {0: 0, 1: 0, 2: 1, 3: 1}  # Node to block mapping

    graph_data: GraphData = GraphData(adjacency_matrix=adjacency, directed=False)
    
    return BlockData(graph_data=graph_data, initial_blocks=blocks), graph_data, blocks


##############################################################################
# Tests
##############################################################################


def test_edge_counter(four_node_example):
    """`_compute_edge_counts_between_node_and_blocks` returns correct counts."""
    block_data, *_ = four_node_example
    change_proposer = NodeSwapProposer(block_data)

    counts = change_proposer._compute_edge_counts_between_node_and_blocks(node=0)
    # Node 0 is linked to node-1 (block-0) and node-2 (block-1) → {0:1, 1:1}
    assert counts == Counter({0: 1, 1: 1})


def test_swap_same_block_zero_delta(four_node_example):
    """
    Swapping two nodes that are *already* in the same block must leave the
    likelihood unchanged.
    """
    block_data, *_ = four_node_example
    swap_proposer = NodeSwapProposer(block_data)
    calc = LikelihoodCalculator(block_data)

    # attempt to swap nodes 0 and 2
    swap = [(0, block_data.blocks[1]), (1, block_data.blocks[0])]
    swap, delta_e, delta_n = swap_proposer.propose_change(swap)
    delta = calc.compute_delta_ll(delta_e=delta_e, delta_n=delta_n)

    assert delta == pytest.approx(0.0, abs=1e-6)


def test_delta_ll_matches_global_recompute(four_node_example):
    """
    delta ll from the incremental calculator must equal the brute-force recomputed
    likelihood difference after the swap (0 ↔ 2).
    """
    block_data_old, graph_data, blocks_old = four_node_example
    adjacency = graph_data.adjacency
    swap_proposer = NodeSwapProposer(block_data_old)
    calc = LikelihoodCalculator(block_data_old)

    # ---------- perform swap 0 ↔ 2 -----------------------------------------
    blocks_new = blocks_old.copy()
    blocks_new[0], blocks_new[2] = blocks_new[2], blocks_new[0]

    block_data_new = BlockData(
        graph_data=GraphData(adjacency_matrix=adjacency, directed=False),
        initial_blocks=blocks_new
    )

    ll_old = compute_global_bernoulli_ll(block_data_old)
    ll_new = compute_global_bernoulli_ll(block_data_new)
    expected_delta = ll_new - ll_old

    swap = [(0, block_data_old.blocks[2]), (2, block_data_old.blocks[0])]
    swap, delta_e, delta_n = swap_proposer.propose_change(swap)

    delta_calc = calc.compute_delta_ll(delta_e=delta_e, delta_n=delta_n)

    msg = (
        f"Failed on swap (0 ↔ 2) with blocks {blocks_old} → {blocks_new}\n"
        f"delta_e: {delta_e}\n"
    )

    assert delta_calc == pytest.approx(expected_delta, rel=1e-6, abs=1e-6), msg


def test_delta_edge_counts_consistency(four_node_example):
    """
    The raw `delta_e` returned by `_compute_delta_edge_counts_swap` should turn
    the old connectivity into the new one *exactly* on every affected (r,s).
    """
    block_data_old, graph_data, blocks_old = four_node_example
    adjacency = graph_data.adjacency
    swap_proposer = NodeSwapProposer(block_data_old)
    calc = LikelihoodCalculator(block_data_old)

    i, j = 0, 2  # the same swap as above
    proposed_changes = [(i, block_data_old.blocks[j]), (j, block_data_old.blocks[i])]
    delta_e = swap_proposer._compute_delta_edge_counts(
        proposed_changes= proposed_changes
    )

    conn_expected = block_data_old.block_connectivity.copy()
    for (r, s), de in delta_e.items():
        conn_expected[r, s] += de

    # Ground-truth connectivity after the swap
    blocks_new = blocks_old.copy()
    blocks_new[i], blocks_new[j] = blocks_new[j], blocks_new[i]
    block_data_new = BlockData(
        graph_data=GraphData(adjacency_matrix=adjacency, directed=False),
        initial_blocks=blocks_new
    )

    for (r, s), de in delta_e.items():
        assert (
            conn_expected[r, s] == block_data_new.block_connectivity[r, s]
        ), f"Mismatch on block pair ({r},{s})"

###################################################
### Randomized tests
###################################################

def _er_graph_csr(n: int, p: float, *, seed: int) -> csr_array:
    """Undirected G(n,p) without self-loops, returned as CSR matrix."""
    rng = np.random.default_rng(seed)
    upper = rng.random((n, n)) < p                           # boolean mask
    upper = np.triu(upper, k=1)                              # keep strict upper
    adj = upper | upper.T                                    # symmetrise
    return csr_array(adj.astype(np.int8))

def _random_equal_blocks(n: int, block_size: int, *, seed: int) -> dict[int, int]:
    """Random permutation of vertices into equal-size blocks."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    blocks: dict[int, int] = {}
    for b in range(n // block_size):
        for v in perm[b * block_size : (b + 1) * block_size]:
            blocks[v] = b
    return blocks

def _pick_two_different_blocks(rng: np.random.Generator, blocks: dict[int, int]):
    n = len(blocks)
    i = rng.choice(n, size=1)[0]
    j = rng.choice(n-1, size=1)[0]
    if j >= i:
        j += 1
    
    return i, j

def _single_swap_calc_vs_bruteforce(
    graph_data: GraphData,
    blocks: dict[int, int],
    rng: np.random.Generator,
    tol: float = 1e-6,
    experiment_index: int = 0
):
    """One random swap and check incremental delta ll against brute force."""
    # choose vertices in different blocks
    i, j = _pick_two_different_blocks(rng, blocks)

    # incremental path -------------------------------------------------
    block_data = BlockData(graph_data=graph_data, initial_blocks=blocks)
    print(block_data.blocks)
    swap_proposer = NodeSwapProposer(block_data)
    calc = LikelihoodCalculator(block_data)

    swap_instr = [(i, blocks[j]), (j, blocks[i])]
    _, delta_e, delta_n = swap_proposer.propose_change(swap_instr)

    delta_ll = calc.compute_delta_ll(delta_e=delta_e, delta_n=delta_n)

    # brute-force path -------------------------------------------------
    ll_before = compute_global_bernoulli_ll(block_data)

    new_blocks = blocks.copy()
    new_blocks[i], new_blocks[j] = new_blocks[j], new_blocks[i]

    block_data_after = BlockData(graph_data=graph_data, initial_blocks=new_blocks)
    ll_after = compute_global_bernoulli_ll(block_data_after)
    delta_brute = ll_after - ll_before

    # compute the delta_e in the brute-force way
    # only storing non-zero deltas in upper triangular matrix
    delta_e_brute = block_data_after.block_connectivity - block_data.block_connectivity
    delta_e_brute = {
        (r, s): de for (r, s), de in zip(
            np.argwhere(delta_e_brute != 0),
            delta_e_brute[delta_e_brute != 0]
        )
        if r <= s
    }

    # comparison and print informatino in case of failure
    msg = (
        f"Failed on experiment {experiment_index}"
        f"\nGraph: {graph_data.adjacency.toarray()}"
        f"\nBlocks: {blocks}"
        f"\nswap:         {i} <-> {j} (blocks {blocks[i]} <-> {blocks[j]})"
        f"\ndelta_e:   {delta_e}"
        f"\ndelta_e_brute:   {delta_e_brute}"
        f"\ndelta_inc:    {delta_ll:.12g}"
        f"\ndelta_brute:  {delta_brute:.12g}"
    )

    assert delta_ll == pytest.approx(delta_brute, rel=tol, abs=tol), msg

def test_delta_ll_random_swaps_er():
    """100 random swaps on independent ER-20 graph must all match brute force delta ll."""
    for index in range(100):
        n, p, b = 20, 0.1, 2

        rng = np.random.default_rng(1)

        adj = _er_graph_csr(n, p, seed=42)
        graph = GraphData(adjacency_matrix=adj, directed=False)

        blocks = _random_equal_blocks(n, block_size=b, seed=2)

        _single_swap_calc_vs_bruteforce(graph_data=graph,
                                        blocks=blocks,
                                        rng=rng,
                                        experiment_index=index,
                                        tol=1e-6
                                        )
