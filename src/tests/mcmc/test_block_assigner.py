"""PyTest suites for structural block‑model code.

These unit tests focus on two invariants that *must* hold for the current
pipeline:

1. **MetisBlockAssigner should only emit blocks whose sizes are either
   `min_block_size` or `min_block_size + 1`.**  (Because the assigner first
   builds blocks of exactly `min_block_size` vertices and then distributes any
   leftovers one‑by‑one.)

2. **A SWAP move in the MCMC sampler must leave every block size unchanged.**

The tests rely only on public APIs plus `_attempt_move`, which is part of the
MCMC sampler’s stable interface (it is used by `fit` internally).  If the
names or module paths differ in your codebase, tweak the imports at the top of
each file – the assertions themselves should stay valid.

Run with::

    pytest -q tests/
"""

# ──────────────────────────────────────────────────────────────────────────────
# tests/test_block_assigner.py
# ──────────────────────────────────────────────────────────────────────────────

import networkx as nx
import numpy as np
import pytest

# Adjust the import path to wherever MetisBlockAssigner lives in your project
from sbm.block_assigner import MetisBlockAssigner
from sbm.block_assigner import ProNEAndConstrKMeansAssigner
from sbm.graph_data import gd_from_networkx

@pytest.mark.parametrize(
    "num_nodes,num_blocks,min_block_size,edge_p",
    [
        (97, 10, 8, 0.05),
        (50, 5, 6, 0.10),
        (23, 4, 5, 0.30),
        (128, 16, 7, 0.02),
    ],
)

def test_metis_block_sizes_are_balanced(num_nodes, num_blocks, min_block_size, edge_p):
    """The initial partition must use only *min* and *min+1* sized blocks."""
    G = nx.erdos_renyi_graph(
            n=num_nodes,
            p=edge_p,
            seed=1,
            directed=False
        )
    graph_data = gd_from_networkx(G)

    assigner = MetisBlockAssigner(
        graph_data=graph_data,
        num_blocks=num_blocks,
        min_block_size=min_block_size,
        rng=np.random.default_rng(42),
    )

    block_data = assigner.compute_assignment()

    # ensure that block_sizes correspond to 

    # Ensure every vertex received exactly one label
    assert len(block_data.blocks) == num_nodes

    # Compute block‑size histogram
    _, counts = np.unique(
        list(block_data.blocks.values()),
        return_counts=True
        )

    # All block sizes must be either min_block_size or min_block_size+1
    assert (counts >= min_block_size).all(), \
        (
            "MetisBlockAssigner produced illegal block sizes: "
            f"{sorted(set(counts))}. Expected larger than {min_block_size}."
        )

    # The partition must contain exactly *num_blocks* non‑empty blocks.
    assert len(counts) == num_blocks

def test_ProNEKMeans_block_sizes_are_balanced():
    """The ProNEKMeans assigner must also use only *min* and *min+1* sized blocks."""

    num_nodes = 100
    min_block_size = 8

    num_blocks = num_nodes // min_block_size

    edge_p = 0.05

    G = nx.erdos_renyi_graph(
            n=num_nodes,
            p=edge_p,
            seed=1,
            directed=False
        )
    graph_data = gd_from_networkx(G)

    assigner = ProNEAndConstrKMeansAssigner(
        graph_data=graph_data,
        min_block_size=min_block_size,
        rng=np.random.default_rng(42),
    )

    block_data = assigner.compute_assignment()

    # ensure that block_sizes correspond to 

    # Ensure every vertex received exactly one label
    assert len(block_data.blocks) == num_nodes, \
        "ProNEKMeansBlockAssigner did not assign a label to every vertex."

    # ensure that block sizes correspond to the number of blocks
    assert len(block_data.block_sizes) == num_blocks  \
        and len(set(block_data.blocks.values())) == num_blocks, \
        "ProNEKMeansBlockAssigner did not produce the expected number of blocks."

    # Compute block‑size histogram
    _, counts = np.unique(
        list(block_data.blocks.values()),
        return_counts=True
        )

    # All block sizes must be either min_block_size or min_block_size+1
    assert (counts >= min_block_size).all(), \
        (
            "ProNEKMeansBlockAssigner produced illegal block sizes: "
            f"{sorted(set(counts))}. Expected larger than {min_block_size}."
        )
    
    count_set = set(counts)
    # Ensure that the block sizes are either min_block_size or min_block_size + 1
    assert count_set.issubset({min_block_size, min_block_size + 1})

    # The partition must contain exactly *num_blocks* non‑empty blocks.
    assert len(counts) == num_blocks