import numpy as np, pytest
from scipy.sparse import csr_array
from metrics.shortest_path import avg_path_length_difference
from tests.metrics.conftest import _line_graph, _complete_graph

def test_identical_zero() -> None:
    A: csr_array = _line_graph()
    assert avg_path_length_difference(A, A, n_samples=None) == 0.0

@pytest.mark.parametrize(
    "emp,sur",
    [
        (_line_graph(8), _complete_graph(8)),
        (_line_graph(20), _line_graph(30)),
    ],
)
def test_difference_positive(emp: csr_array, sur: csr_array) -> None:
    d = avg_path_length_difference(emp, sur, n_samples=100)
    assert d > 0.0 and np.isfinite(d)
