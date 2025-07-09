""" 
Metrics for comparing assortativity of two graph.
"""
import numpy as np
from scipy.sparse import csr_array
import networkx as nx

###############################################################################
#Assortativity coefficient difference ------------------------------------
###############################################################################
def assortativity_difference(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    rng: np.random.Generator = np.random.default_rng(1),
) -> float:
    """Absolute difference in degree‑assortativity (Pearson) coefficient."""
    def _assort(adj: csr_array) -> float:
        G = nx.from_scipy_sparse_matrix(adj)
        # NetworkX warns for disconnected graphs → ignore
        try:
            return float(nx.degree_pearson_correlation_coefficient(G))
        except Exception:
            return 0.0  # fallback (e.g., for trivial graphs)

    emp_assort = _assort(emp_adj)
    sur_assort = _assort(sur_adj)
    return abs(emp_assort - sur_assort)