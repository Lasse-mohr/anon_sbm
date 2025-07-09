import numpy as np
from pathlib import Path
import networkx as nx
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_array

def set_random_seed(seed: int):
    return np.random.default_rng(seed)


def restrict_to_lcc(adj: csr_array, directed:bool) -> csr_array:
    """ 
    resricts adjacency matrix to the largest connected component (LCC).
    """

    if directed:
        n_components, labels = connected_components(adj, directed=True)
    else:
        n_components, labels = connected_components(adj, directed=False)

    if n_components == 1:
        return adj

    largest_component = np.argmax(np.bincount(labels))
    mask = labels == largest_component
    adj_lcc = csr_array(adj[mask][:, mask]) # type: ignore

    return adj_lcc

def _nx_graph(adj: csr_array, *, directed: bool = False) -> nx.Graph:
    """Convert *adj* to a NetworkX (di)graph, restricted to its LCC."""
    adj_lcc = restrict_to_lcc(adj, directed)
    return (
        nx.from_scipy_sparse_matrix(adj_lcc, create_using=nx.DiGraph() if directed else nx.Graph())
    )