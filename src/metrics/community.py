"""
Metric functions to compare the community structure of two graphs.
"""
import numpy as np
from scipy.sparse import csr_array, csr_matrix

# external heavy‑duty packages (all listed in requirements.txt)
from infomap import Infomap  # type: ignore
import igraph as ig  # type: ignore
import leidenalg  # type: ignore

### Utility -----------------------------------------------------------------
def _modularity(adj: csr_array) -> float:
    g = ig.Graph.Adjacency(csr_matrix(adj), mode="UNDIRECTED")

    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=None)
    return g.modularity(part)

###############################################################################
# Infomap codelength difference -------------------------------------------
###############################################################################
def infomap_codelength_difference(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    directed: bool = False,
    rng: np.random.Generator = np.random.default_rng(1),
) -> float:
    """Absolute difference in Infomap codelength (compression) between graphs."""
    def _codelength(adj: csr_array) -> float:
        im = Infomap("--directed" if directed else "")
        rows, cols = adj.nonzero()
        for u, v in zip(rows, cols, strict=False):
            if u < v or directed:
                im.add_link(int(u), int(v))
        im.run(silent=True)
        return im.codelength
    return abs(_codelength(emp_adj) - _codelength(sur_adj))

###############################################################################
# Modularity distance via Leiden ------------------------------------------
###############################################################################
def leiden_modularity_difference(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    rng: np.random.Generator = np.random.default_rng(1),
) -> float:
    """Absolute difference in maximum modularity found by Leiden."""

    return abs(_modularity(emp_adj) - _modularity(sur_adj))