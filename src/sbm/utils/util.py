import numpy as np
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