# ──────────────────────────────────────────────────────────────────────────────
# tests/test_swap_move.py
# ──────────────────────────────────────────────────────────────────────────────
import networkx as nx
import numpy as np
import pytest

from sbm.block_assigner import MetisBlockAssigner
from sbm.block_change_proposers import NodeSwapProposer
from sbm.likelihood import LikelihoodCalculator
from sbm.mcmc import MCMC
from sbm.graph_data import gd_from_networkx
from sbm.block_data import BlockData

def _sizes_from_block_data(block_data: BlockData, num_blocks):
    """Helper used in both reference and post‑move checks."""
    sizes = np.zeros(num_blocks, dtype=int)
    for bid, nodes in block_data.block_members.items():
        if bid < num_blocks:
            sizes[bid] = len(nodes)
    return sizes


@pytest.mark.parametrize(
    "num_nodes,num_blocks,min_block_size,iterations,edge_p",
    [
        (80, 8, 8, 250, 0.05),
        (30, 6, 4, 100, 0.25),
        (10, 5, 2, 100, 0.25),
    ],
)
def test_swap_move_preserves_block_sizes(num_nodes, num_blocks, min_block_size, iterations, edge_p):
    """After *every* accepted SWAP move, block‑size vector must be unchanged."""

    rng = np.random.default_rng(42)

    # old version of networkx can't take rng so use seed instead
    G = nx.erdos_renyi_graph(num_nodes, edge_p, seed=42, directed=False)
    graph_data = gd_from_networkx(G)

    # ── Build initial state ────────────────────────────────────────────────
    block_data = MetisBlockAssigner(
        graph_data=graph_data,
        num_blocks=num_blocks,
        min_block_size=min_block_size,
        rng=rng,
    ).compute_assignment()

    likelihood_calculator = LikelihoodCalculator(block_data=block_data)
    swap_proposer = NodeSwapProposer(block_data=block_data, rng=rng)

    mcmc = MCMC(
        block_data=block_data,
        likelihood_calculator=likelihood_calculator,
        change_proposer={"uniform_swap": swap_proposer},
        rng=rng,
    )

    reference_sizes = _sizes_from_block_data(block_data, num_blocks)

    # ── Run many candidate swap moves ──────────────────────────────────────
    for iter in range(iterations):
        _delta_ll, accepted = mcmc._attempt_move(move_type="uniform_swap", temperature=1.0)
        if accepted:
            current_sizes = _sizes_from_block_data(mcmc.block_data, num_blocks)
            assert np.array_equal(reference_sizes, current_sizes), (
                f"SWAP move {iter} changed block sizes:", reference_sizes, "→", current_sizes
            )

    # Final safeguard: after *all* moves sizes are still identical.
    final_sizes = _sizes_from_block_data(mcmc.block_data, num_blocks)
    assert np.array_equal(reference_sizes, final_sizes)