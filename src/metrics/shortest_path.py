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

from sbm.utils.util import restrict_to_lcc

import networkx as nx
from typing import Iterable, Hashable, List, Union

Number = Union[int, float]

def all_unique_shortest_distances(
    G: nx.Graph,
    weight: str | None = None,
    cutoff: Number | None = None,
    directed: bool = False,
) -> List[Number]:
    """
    Return a list containing the length of every unique shortest path in *G*.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (directed or undirected, weighted or unweighted).
    weight : str or None, default=None
        Edge-attribute key to use as weight.  ``None`` ⇒ treat edges as unit-weight.
    cutoff : int | float | None, default=None
        Ignore paths longer than *cutoff* (same semantics as NetworkX).

    Returns
    -------
    distances : list[Number]
        One entry per unordered, connected node pair.  
        Unreachable pairs are silently skipped.
    """
    # 1.  Choose the correct all-pairs iterator
    if weight is None:
        # Unweighted ⇢ multi-source breadth-first search
        iterator = nx.all_pairs_shortest_path_length(G, cutoff=cutoff)  # :contentReference[oaicite:0]{index=0}
    else:
        # Weighted ⇢ repeated Dijkstra
        iterator = nx.all_pairs_dijkstra_path_length(G, cutoff=cutoff, weight=weight)  # :contentReference[oaicite:1]{index=1}

    # 2.  Collect unique unordered pairs
    seen: set[frozenset[Hashable]] = set()
    distances: List[Number] = []

    for u, length_dict in iterator:
        for v, d in length_dict.items():
            if u == v:                       # skip self-loops (distance 0)
                continue
            pair = frozenset((u, v))         # unordered representation
            if pair in seen:                 # already counted via (v, u)
                continue
            seen.add(pair)
            distances.append(d)

    return distances


def shortest_path_distance(
        emp_adj: csr_array,
        sur_adj: csr_array,
        n_samples: Optional[int]=10_000,
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

    emp_adj = restrict_to_lcc(emp_adj, directed=False)
    sur_adj = restrict_to_lcc(sur_adj, directed=False)

    emp_graph_size = emp_adj.shape[0] # type: ignore
    sur_graph_size = sur_adj.shape[0] # type: ignore

    if n_samples is None:
        # Use all pairs if n_samples is None, generated using networkx (returns iterator)
        x = all_unique_shortest_distances(
            nx.from_scipy_sparse_matrix(emp_adj),
            weight=None,
            cutoff=None
        )
        y = all_unique_shortest_distances(
            nx.from_scipy_sparse_matrix(sur_adj),
            weight=None,
            cutoff=None
        )
    else:
        n_samples_emp = int(min(
            n_samples,
            emp_graph_size * (emp_graph_size - 1) // 2,
        ))
        n_samples_sur = int(min(
            n_samples,
            sur_graph_size * (sur_graph_size - 1) // 2,
        ))

        emp_pair_part_1 = rng.choice(emp_graph_size, size=n_samples_emp, replace=True)
        emp_pair_part_2 = rng.choice(emp_graph_size-1, size=n_samples_emp, replace=True)

        emp_pair_part_2[emp_pair_part_2 >= emp_pair_part_1] += 1
        emp_pairs = np.column_stack((emp_pair_part_1, emp_pair_part_2))

        sur_pair_part_1 = rng.choice(sur_graph_size, size=n_samples_sur, replace=True)
        sur_pair_part_2 = rng.choice(sur_graph_size-1, size=n_samples_sur, replace=True)

        sur_pair_part_2[sur_pair_part_2 >= sur_pair_part_1] += 1
        sur_pairs = np.column_stack((sur_pair_part_1, sur_pair_part_2))

        def sample_shortest_paths(adj, pairs):
            G = nx.from_scipy_sparse_matrix(adj)
            path_lengths = []
            for pair in pairs:
                try:
                    length = nx.shortest_path_length(G, source=pair[0], target=pair[1])
                    path_lengths.append(length)
                except nx.exception.NodeNotFound:
                    raise Warning(
                        f"Node {pair[0]} or {pair[1]} not found in the graph."
                    )

            return path_lengths

        x = sample_shortest_paths(emp_adj, emp_pairs)
        y = sample_shortest_paths(sur_adj, sur_pairs)

    return wasserstein_distance(x, y)
