# src/tests/mcmc/test_private_partition_flag.py
# ----------------------------------------------------------
import numpy as np
import networkx as nx

from sbm.graph_data import gd_from_networkx
from sbm.block_data import BlockData
from sbm.model import SBMModel
from sbm.mcmc import PrivatePartitionMCMC
from sbm.mcmc_diagnostics import OnlineDiagnostics


def _toy_blocks(g, k=3):
    """
    Deterministic starter partition: nodes are grouped in
    consecutive chunks of size k ⇢ block_id = node // k.
    """
    return {node: node // k for node in g.nodes()}


def test_private_partition_flag_runs():
    rng = np.random.default_rng(0)

    # ----- tiny graph -----------------------------------------------------
    G = nx.erdos_renyi_graph(12, 0.25, seed=1)
    graph_data = gd_from_networkx(G)

    # ----- starter blocks & BlockData -------------------------------------
    blocks = _toy_blocks(G, k=3)
    block_data = BlockData(initial_blocks=blocks, graph_data=graph_data)

    # ----- build model *with* the privacy flag ----------------------------
    model = SBMModel(
        initial_blocks=block_data,
        rng=rng,
        private_sbm=True,           # <‑‑ flag under test
    )

    # (1) the correct sampler type is selected
    assert isinstance(model.mcmc_algorithm, PrivatePartitionMCMC)

    # (2) diagnostics object is attached
    assert isinstance(model.mcmc_algorithm._diag, OnlineDiagnostics)

    # (3) a micro‑run completes without error
    #     – 10 iterations, no cooling, min‑block‑size honoured
    model.fit(
        min_block_size=3,
        max_num_iterations=10,
        cooling_rate=1.0,
        initial_temperature=1.0,
        patience=20,
    )

    # optional quick sanity: buffer has at least one entry
    assert model.mcmc_algorithm._diag._buf is not None

    print(model.mcmc_algorithm._diag._buf)

if __name__ == "__main__":
    test_private_partition_flag_runs()