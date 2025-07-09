""" 
Implementation of the clustering coefficient distance metric.
This module computes the absolute difference in the clustering coefficients
between two graphs represented by their adjacency matrices.
"""

# metrics/clustering.py
from typing import Optional
import numpy as np
import networkx as nx
from scipy.sparse import csr_array

###############################################################################
### Clustering coefficient distributional distance ----------------------------
###############################################################################
def clustering_distance(
        emp_adj: csr_array,
        sur_adj: csr_array,
        directed: Optional[bool] = False,
        rng: np.random.Generator = np.random.default_rng(1)
    ) -> float:

    """ 
    Compute absolute 
    
    Parameters
    ----------
    emp_adj, sur_adj : scipy.sparse.csr_matrix
        Adjacency of empirical and surrogate graphs (directed or undirected).
    directed : bool, optional
        If True, compute directed degree distribution.
        If False, compute undirected degree distribution.
    rng : np.random.Generator, optional
        Random number generator for sampling (default: np.random.default_rng(1)).
    Returns
    -------
    float
        Distance (lower = more similar).
    """

    if directed:
        raise NotImplementedError(
            "Directed clustering coefficient is not implemented yet."
        )
    else:
        # Undirected clustering coefficient
        emp_graph = nx.from_scipy_sparse_matrix(emp_adj)
        sur_graph = nx.from_scipy_sparse_matrix(sur_adj)

        emp_clustering = nx.average_clustering(emp_graph)
        sur_clustering = nx.average_clustering(sur_graph)

        return abs(emp_clustering - sur_clustering)

###############################################################################
# Average clustering coefficient difference -------------------------------
###############################################################################
def avg_clustering_difference(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    rng: np.random.Generator = np.random.default_rng(1),
) -> float:
    """Absolute difference in *average* clustering coefficient.

    (The existing *clustering_distance* compares the *distribution*; this
    variant is the scalar average.)
    """
    emp_C = nx.average_clustering(nx.from_scipy_sparse_matrix(emp_adj))
    sur_C = nx.average_clustering(nx.from_scipy_sparse_matrix(sur_adj))
    return abs(emp_C - sur_C)
