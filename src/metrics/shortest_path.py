""" 
Implementation of the shortest path distribution distance metric.
This module computes the Earth-mover distance (1-D Wasserstein distance)
between the distributions of all-pairs shortest-path lengths
in two graphs represented by their adjacency matrices.
"""
# metrics/shortest_path.py
from typing import Optional
import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance #  1-D EMD
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path

def shortest_path_distance(
        emp_adj: csr_array,
        sur_adj: csr_array,
        n_samples: Optional[int]=None,
        rng:np.random.Generator = np.random.default_rng(1)
    ) -> float:
    """
    Earth-mover (1-D Wasserstein) distance between the distributions
    of all-pairs shortest-path lengths.

    Parameters
    ----------
    emp_adj, sur_adj : scipy.sparse.csr_matrix
        Adjacency of empirical and surrogate graphs (undirected).
    n_samples : int, optional
        Number of samples to use for the distributions.
        If None, all pairs are used.

    Returns
    -------
    float
        Distance (lower = more similar).
    """

    emp_graph_size = emp_adj.shape[0] # type: ignore
    sur_graph_size = emp_adj.shape[0] # type: ignore

    if (emp_graph_size > 1000) or (sur_graph_size > 100) and (n_samples is not None):
        raise Warning(
            "Graph sizes large (>1000 nodes), "
            "consider setting n_samples=None to use all pairs."
        )

    if n_samples is None:
        G_emp = nx.from_scipy_sparse_array(emp_adj)
        G_sur = nx.from_scipy_sparse_array(sur_adj)

        def sp_hist(G) -> np.ndarray:
            lengths = dict(nx.all_pairs_shortest_path_length(G))
            vals = [d for lengths_u in lengths.values() for d in lengths_u.values()]
            return np.array(vals)

        x = sp_hist(G_emp)
        y = sp_hist(G_sur)

    else: # sample pairs and compute distances
        if n_samples > (emp_graph_size * (emp_graph_size - 1) / 2) \
            or n_samples > (sur_graph_size * (sur_graph_size - 1) / 2):
            raise ValueError(
                "n_samples exceeds the number of unique pairs in the graph."
            )
        emp_pair_part_1 = rng.choice(emp_graph_size, size=n_samples, replace=True)
        emp_pair_part_2 = rng.choice(emp_graph_size-1, size=n_samples, replace=True)
        emp_pair_part_2[emp_pair_part_2 >= emp_pair_part_1] += 1
        emp_pairs = np.column_stack((emp_pair_part_1, emp_pair_part_2))

        sur_pair_part_1 = rng.choice(sur_graph_size, size=n_samples, replace=True)
        sur_pair_part_2 = rng.choice(sur_graph_size-1, size=n_samples, replace=True)
        sur_pair_part_2[sur_pair_part_2 >= sur_pair_part_1] += 1
        sur_pairs = np.column_stack((sur_pair_part_1, sur_pair_part_2))

        def sample_shortest_paths(adj, pairs):
            G = nx.from_scipy_sparse_array(adj)
            return np.array([nx.shortest_path_length(G, source=u, target=v) for u, v in pairs])

        x = sample_shortest_paths(emp_adj, emp_pairs)
        y = sample_shortest_paths(sur_adj, sur_pairs)

    return wasserstein_distance(x, y)
