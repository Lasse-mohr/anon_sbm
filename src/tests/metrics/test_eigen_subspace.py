import numpy as np, pytest
from scipy.sparse import csr_array
from metrics.spectral import (
    eigen_val_distance,
    eigen_vec_distance,
    centrality_distance
)
from tests.metrics.conftest import _line_graph, _complete_graph, _er_graph
# ------------------------------------------------------------------------- #
# Eigenval tests
# ------------------------------------------------------------------------- #
def test_val_identical_zero_int_input() -> None:
    # adjacency is *int* on purpose – function must up-cast internally
    A: csr_array = _line_graph(12)
    assert eigen_val_distance(A, A, k=3) == 0.0

def test_val_different_size_graphs() -> None:
    # adjacency is *int* on purpose – function must up-cast internally
    A: csr_array = _line_graph(12)
    B: csr_array = _line_graph(13)

    eigen_val_distance(A, B, k=3)

@pytest.mark.parametrize("k", [1, 3])
def test_val_line_vs_complete_positive(k: int) -> None:
    d = eigen_val_distance(_line_graph(12), _complete_graph(12), k=k)
    assert d > 0.0 and np.isfinite(d)

# ------------------------------------------------------------------------- #
# Eigenvec tests
# ------------------------------------------------------------------------- #
def test_vec_identical_zero_int_input() -> None:
    # adjacency is *int* on purpose – function must up-cast internally
    A: csr_array = _line_graph(12)
    assert eigen_vec_distance(A, A, k=3) == 0.0

def test_vec_line_vs_er_positive() -> None:
    # adjacency is *int* on purpose – function must up-cast internally
    A: csr_array = _line_graph(50)
    B: csr_array = _er_graph(50)
    assert eigen_vec_distance(A, B, k=3) > 0.0

def test_vec_different_size_graphs() -> None:
    # adjacency is *int* on purpose – function must up-cast internally
    A: csr_array = _line_graph(12)
    B: csr_array = _line_graph(13)

    # check that the function raises a ValueError
    # because the graphs have different sizes
    with pytest.raises(ValueError):
        eigen_vec_distance(A, B, k=3)


# ------------------------------------------------------------------------- #
# Eigen centrality tests
# ------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 3])
def test_vec_line_vs_complete_positive(k: int) -> None:
    d = centrality_distance(_line_graph(12), _complete_graph(12))
    assert d > 0.0 and np.isfinite(d)

def test_cent_identical_zero_int_input() -> None:
    # adjacency is *int* on purpose – function must up-cast internally
    A: csr_array = _line_graph(12)
    assert centrality_distance(A, A) == 0.0

def test_cent_different_size_graphs() -> None:
    # adjacency is *int* on purpose – function must up-cast internally
    A: csr_array = _line_graph(12)
    B: csr_array = _line_graph(20)

    assert centrality_distance(A, B) > 0.0

@pytest.mark.parametrize("k", [1, 3])
def test_cent_line_vs_complete_positive(k: int) -> None:
    d = centrality_distance(_line_graph(12), _complete_graph(12))
    assert d > 0.0 and np.isfinite(d)