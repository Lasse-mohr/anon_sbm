# src/tests/test_change_proposers_and_mover.py
"""
Unit-tests for
  • sbm.block_change_proposers           (helper + NodeSwapProposer)
  • sbm.node_mover                      (NodeMover)

All graphs are 4 undirected vertices:

    0──1   block 0 = {0,1}
    │  │
    2──3   block 1 = {2,3}

Edges: (0,1) (2,3)  plus two cross edges (0,2) (1,3).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike
import pytest
from scipy.sparse import csr_array

from sbm.graph_data import GraphData
from sbm.block_data import BlockData
from sbm.block_change_proposers import (
    NodeSwapProposer,
    ProposedValidChanges,
)
from sbm.edge_delta import EdgeDelta
from sbm.node_mover import NodeMover


###############################################################################
# helpers
###############################################################################
def _toy_graph() -> Tuple[csr_array, Dict[int, int]]:
    """Return 4×4 adjacency and initial {node: block} mapping."""
    adj = np.zeros((4, 4), dtype=int)
    edges = [(0, 1), (2, 3), (0, 2), (1, 3)]
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    blocks = {0: 0, 1: 0, 2: 1, 3: 1}
    return csr_array(adj), blocks


def _brute_block_connectivity(adj: csr_array, blocks: Dict[int, int]) -> Dict[Tuple[int, int], int]:
    """Return upper-triangle edge counts between blocks."""
    rows, cols = adj.nonzero() # type: ignore
    bc = defaultdict(int)

    for u, v in zip(rows, cols, strict=False):
        if u >= v:  # count each undirected edge once
            continue
        a, b = blocks[u], blocks[v]
        r, s = (a, b) if a <= b else (b, a)
        bc[(r, s)] += 1
    return bc


###############################################################################
# tests for _increment_delta_e
###############################################################################
def test_increment_delta_e_uses_sorted_key() -> None:
    e_delta = EdgeDelta(n_blocks=4)
    e_delta.increment(
        counts=[1],
        blocks_i=[3],
        blocks_j=[1],
    )          # block_i > block_j → key (1,3)
    assert ((1, 3), 1) in e_delta.items() and ((3, 1), 1) not in e_delta.items(), \
        f"Key mismatch: delta keys = {list(e_delta.items())}"

def test_increment_delta_e_overwrites_existing() -> None:
    e_delta = EdgeDelta(n_blocks=4)
    e_delta.increment(
        counts = [1],
        blocks_i = [0],
        blocks_j = [2],
    )          # set to 1
    e_delta.increment(
        counts = [3],
        blocks_i = [2],
        blocks_j = [0]
    )          # overwrite same pair
    assert e_delta[(0, 2)] == 3, \
        f"Value not overwritten, got {e_delta[(0, 2)]}, expected 3"


###############################################################################
# tests for NodeSwapProposer
###############################################################################
@pytest.fixture(scope="module")
def proposer() -> NodeSwapProposer:
    adj, blocks = _toy_graph()
    bd = BlockData(initial_blocks=blocks, graph_data=GraphData(adj, directed=False))
    return NodeSwapProposer(block_data=bd, rng=np.random.default_rng(0))

def test_compute_delta_edge_counts_matches_brute(proposer: NodeSwapProposer) -> None:
    # swap vertices 0 (block 0) and 2 (block 1)
    changes: ProposedValidChanges = [(0, 1), (2, 0)]
    delta = proposer._compute_delta_edge_counts(changes)          # type: ignore

    before = _brute_block_connectivity(
        proposer.block_data.graph_data.adjacency,
        proposer.block_data.blocks
    )

    # build new blocks mapping
    new_blocks = proposer.block_data.blocks.copy()
    new_blocks[0], new_blocks[2] = new_blocks[2], new_blocks[0]
    after = _brute_block_connectivity(proposer.block_data.graph_data.adjacency,
                                      new_blocks)

    brute_delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)}
    delta_dict = dict(delta.items())
    assert delta_dict == brute_delta, \
        f"\nexpected {brute_delta}\ngot      {delta_dict}"


def test_propose_change_returns_expected_structure(proposer: NodeSwapProposer) -> None:
    changes = [(0, 1), (2, 0)]
    new_changes, delta_e, delta_n = proposer.propose_change(changes=changes)

    assert new_changes == changes, "proposer changed explicit instruction"
    assert all(isinstance(k, tuple) and len(k) == 2 for k in delta_e.items()), "delta_e keys malformed"

    # check that all kays of delta_e are accesible in delta_n (not necessarily non-zero)
    for (i, j), _ in delta_e.items():
        try:
            _ = delta_n[(i, j)]
        except KeyError:
            pytest.fail(f"delta_n missing key ({i}, {j}) from delta_e: {delta_e.items()}")



###############################################################################
# tests for NodeMover
###############################################################################
def test_node_mover_updates_blocks_and_sizes() -> None:
    adj, blocks = _toy_graph()
    bd = BlockData(initial_blocks=blocks, graph_data=GraphData(adj, directed=False))
    mover = NodeMover(bd)

    changes = [(0, 1), (2, 0)]
    # reuse proposer to get correct delta_e
    prop = NodeSwapProposer(block_data=bd, rng=np.random.default_rng(0))
    delta_e = prop._compute_delta_edge_counts(changes)            # type: ignore

    mover.perform_change(changes, delta_e)

    assert bd.blocks[0] == 1 and bd.blocks[2] == 0, \
        f"blocks not swapped: {bd.blocks}"
    assert bd.block_sizes[0] == 2 and bd.block_sizes[1] == 2, \
        f"block_sizes wrong: {bd.block_sizes}"


def test_node_mover_updates_connectivity() -> None:
    adj, blocks = _toy_graph()
    bd = BlockData(initial_blocks=blocks, graph_data=GraphData(adj, directed=False))
    mover = NodeMover(bd)

    before = bd.block_connectivity.copy()
    changes = [(0, 1), (2, 0)]
    prop = NodeSwapProposer(block_data=bd, rng=np.random.default_rng(0))
    delta_e = prop._compute_delta_edge_counts(changes)            # type: ignore
    mover.perform_change(changes, delta_e)

    # brute recompute
    after_brute = _brute_block_connectivity(bd.graph_data.adjacency, bd.blocks)
    # connectivity matrix stores both triangles → pick upper
    after_matrix = {(r, s): int(bd.block_connectivity[r, s]) #type: ignore
                    for r, s in after_brute}

    msg = (f"\nexpected connectivity {after_brute}"
           f"\nobserved  connectivity {after_matrix}"
           f"\ndelta_e applied        {delta_e}")
    assert after_matrix == after_brute, msg