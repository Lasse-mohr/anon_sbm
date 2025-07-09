import numpy as np, scipy.sparse as sp
import pytest
from scipy.sparse import csr_array

# ------------------------------------------------------------------ helpers
def _line_graph(n: int = 6) -> csr_array:
    rows = np.arange(n - 1)
    cols = rows + 1
    A = sp.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n, n))
    A = A + A.T
    return csr_array(A, dtype=np.int8)

def _complete_graph(n: int = 6) -> csr_array:
    A = np.ones((n, n), dtype=np.int8)
    np.fill_diagonal(A, 0)
    return csr_array(A)

def _er_graph(n: int = 10, p: float = .2, *, seed: int = 1) -> csr_array:
    rng = np.random.default_rng(seed)
    upper = rng.random((n, n)) < p
    upper = np.triu(upper, k=1)
    A = upper | upper.T
    return csr_array(A.astype(np.int8))

def _two_triangles() -> csr_array:
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    rows, cols = zip(*edges)
    A = sp.coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(6, 6))
    A = A + A.T
    return csr_array(A, dtype=np.int8)

# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)