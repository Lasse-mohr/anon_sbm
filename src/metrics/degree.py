""" 
Implementation of the degree distribution distance metric.
This module computes the Earth-mover distance (1-D Wasserstein distance)
between the degree distributions in two graphs represented by
their adjacency matrices.
"""
# metrics/degree.py
from typing import Optional
import numpy as np
from scipy.stats import wasserstein_distance #  1-D EMD
from scipy.sparse import csr_array

def degree_distance(
        emp_adj: csr_array,
        sur_adj: csr_array,
        directed: Optional[bool] = False,
        in_degree: Optional[bool] = False,
        out_degree: Optional[bool] = False,
        rng: np.random.Generator = np.random.default_rng(1)
    ) -> float:
    """ 
    Earth-mover (1-D Wasserstein) distance between the degree distributions
    of two graphs.

    Parameters
    ----------
    emp_adj, sur_adj : scipy.sparse.csr_matrix
        Adjacency of empirical and surrogate graphs (directed or undirected).
    directed : bool, optional
        If True, compute directed degree distribution.
        If False, compute undirected degree distribution.
    in_degree : bool, optional
        If True, compute in-degree distribution (for directed graphs).
        Ignored if `directed` is False.
    out_degree : bool, optional
        If True, compute out-degree distribution (for directed graphs).
        Ignored if `directed` is False.
    rng : np.random.Generator, optional
        Random number generator for sampling (default: np.random.default_rng(1)).
    Returns
    -------
    float
        Distance (lower = more similar).
    """

    if directed:
        raise NotImplementedError(
            "Directed degree distribution is not implemented yet."
        )
    else:
        if in_degree or out_degree:
            raise Warning(
                "in_degree and out_degree are ignored for undirected graphs."
            )
        # Undirected degree distribution
        emp_degrees = np.asarray(emp_adj.sum(axis=0)).flatten()
        sur_degrees = np.asarray(sur_adj.sum(axis=0)).flatten()

        emp_dist = np.bincount(emp_degrees)
        sur_dist = np.bincount(sur_degrees)

        # Normalize distributions
        emp_dist = emp_dist / emp_dist.sum()
        sur_dist = sur_dist / sur_dist.sum()

        # Compute Earth-mover distance
        distance = wasserstein_distance(emp_dist, sur_dist)
        return distance

