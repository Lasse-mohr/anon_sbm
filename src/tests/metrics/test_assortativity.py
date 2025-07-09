from scipy.sparse import csr_array
from metrics.assortativity import assortativity_difference
from tests.metrics.conftest import _line_graph, _complete_graph, _er_graph

def test_identical_zero() -> None:
    A: csr_array = _line_graph()
    assert assortativity_difference(A, A) == 0.0

def test_line_vs_complete_positive() -> None:
    d = assortativity_difference(_line_graph(12), _er_graph(n=12))
    assert d > 0
