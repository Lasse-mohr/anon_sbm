import numpy as np, scipy.sparse as sp
from scipy.sparse import csr_array
from metrics import (
    shortest_path_distance,
    degree_distance,
    clustering_distance
)
### Helper functions to create graphs
def _line_graph(n=6) -> csr_array:
    rows = np.arange(n-1); cols = rows + 1
    A = sp.coo_matrix((np.ones(n-1), (rows, cols)), shape=(n, n))
    A = A + A.T
    A = csr_array(A, dtype=np.int8)
    return A

def _er_graph(n: int=10, p: float=0.1, *, seed: int=1) -> csr_array:
    """Undirected G(n,p) without self-loops, returned as CSR matrix."""
    rng = np.random.default_rng(seed)
    upper = rng.random((n, n)) < p                           # boolean mask
    upper = np.triu(upper, k=1)                              # keep strict upper
    adj = upper | upper.T                                    # symmetrise
    return csr_array(adj.astype(np.int8))

#### test functions
def test_shortest_path_identical_line():
    A = _line_graph()
    print(A.toarray())
    assert shortest_path_distance(A, A, n_samples=None) == 0.0

def test_degree_identical_line():
    A = _line_graph()
    assert degree_distance(A, A) == 0.0

def test_clustering_identical_line():
    A = _line_graph()
    assert clustering_distance(A, A) == 0.0

def test_shortest_path_different_line():
    A = _line_graph(n=20)
    B = _line_graph(n=200)

    assert shortest_path_distance(A, B, n_samples=100) != 0.0

def test_degree_different_line():
    A = _line_graph(n=20)
    A = _line_graph(n=200)
    assert degree_distance(A, A) == 0.0

def test_clustering_identical_er():
    A = _er_graph(n=100)
    assert clustering_distance(A, A) == 0.0

def test_shortest_path_identical_er():
    A = _er_graph(p=0.9)
    assert shortest_path_distance(A, A, n_samples=None) == 0.0

def test_shortest_path_different_er():
    A = _er_graph(n=20, p=0.9)
    B = _er_graph(n=200, p=0.1)

    assert shortest_path_distance(A, B, n_samples=100) != 0.0

def test_degree_different_er():
    A = _er_graph()
    assert degree_distance(A, A) == 0.0

def test_clustering_different_er():
    A = _er_graph()
    assert clustering_distance(A, A) == 0.0