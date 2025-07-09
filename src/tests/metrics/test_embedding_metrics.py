import numpy as np, pytest
from scipy.sparse import csr_array
from metrics.embedding import (
    embedding_node2vec_ip_emd,
    embedding_prone_ip_emd,
)
from tests.metrics.conftest import _line_graph, _er_graph

@pytest.mark.parametrize("fn", [embedding_node2vec_ip_emd, embedding_prone_ip_emd])
def test_dissimilar_for_different_graphs(fn):
    A: csr_array = _line_graph(100)
    B: csr_array = _er_graph(100)

    d_AA = fn(A, A, dim=32, n_pairs=500)
    d_AB = fn(A, B, dim=32, n_pairs=500)
    d_BB = fn(B, B, dim=32, n_pairs=500)
    # for identical graphs we expect *almost* zero – allow tiny noise
    assert d_AA < d_AB
    assert d_BB < d_AB

@pytest.mark.parametrize("fn", [embedding_node2vec_ip_emd, embedding_prone_ip_emd])
def test_line_vs_er_positive(fn):
    d = fn(_line_graph(20), _er_graph(20, p=.3), dim=32, n_pairs=500)
    assert d > 0 and np.isfinite(d)
