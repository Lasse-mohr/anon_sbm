import numpy as np
from scipy.sparse import csr_array
from metrics.community import (
    infomap_codelength_difference,
    leiden_modularity_difference,
)
from tests.metrics.conftest import _two_triangles, _complete_graph

def test_infomap_identical_zero():
    A: csr_array = _two_triangles()
    assert infomap_codelength_difference(A, A) == 0.0

def test_leiden_identical_zero():
    A: csr_array = _two_triangles()
    assert leiden_modularity_difference(A, A) == 0.0

def test_infomap_vs_complete_positive():
    d = infomap_codelength_difference(_two_triangles(), _complete_graph(6))
    assert d > 0

def test_leiden_vs_complete_positive():
    d = leiden_modularity_difference(_two_triangles(), _complete_graph(6))
    assert d > 0
