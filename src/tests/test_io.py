# tests/test_io.py
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
import networkx as nx
import pytest

from sbm.io import SBMFit, SBMWriter, GraphLoader
# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _simple_adj(undirected: bool = True) -> sp.csr_array:
    """
    3-node graph:
        0 — 1   and  (optionally) 1 → 2
    """
    rows, cols = [0, 1], [1, 0]            # 0–1 edge
    if not undirected:
        rows.append(1); cols.append(2)     # add 1→2 (makes it directed)
    data = np.ones(len(rows), dtype=np.int8)

    return sp.csr_array(sp.coo_matrix((data, (rows, cols)), shape=(3, 3)))


def _assert_same_csr(a: sp.csr_array, b: sp.csr_array):
    a.sort_indices()
    b.sort_indices()

    assert np.array_equal(a.data, b.data), f'Data arrays differ: {a.data} != {b.data}'
    assert np.array_equal(a.indices, b.indices), f'Indices differ: {a.indices} != {b.indices}'
    assert a.shape == b.shape, f'Shape differs: {a.shape} != {b.shape}'

# ---------------------------------------------------------------------
# 1. SBMWriter round-trip
# ---------------------------------------------------------------------
def test_sbmwriter_roundtrip(tmp_path: Path):
    # --- build a tiny SBMFit ----------------------------------------
    #adj = _simple_adj()
    #blocks = {0: 0, 1: 0, 2: 1}
    fit = SBMFit(
        block_sizes   = [2, 1],
        block_conn    = sp.csr_array([[1, .2],[.2, .1]]),
        directed_graph= False,
        neg_loglike   = -12.34,
        metadata      = {"foo": "bar"},
    )

    # --- save & load ------------------------------------------------
    SBMWriter.save(tmp_path, fit)
    fit2 = SBMWriter.load(tmp_path)

    # basic checks
    assert fit2.block_sizes == [2, 1], f"Block sizes do not match: {fit2.block_sizes} != [2, 1]"
    _assert_same_csr(fit.block_conn, fit2.block_conn)
    assert fit2.neg_loglike == pytest.approx(fit.neg_loglike), "Negative log-likelihood does not match"
    assert fit2.metadata["foo"] == "bar", "Metadata does not match"

# ---------------------------------------------------------------------
# 2. GraphLoader built-in formats
# ---------------------------------------------------------------------
@pytest.mark.parametrize("undirected", [True, False])
def test_graphloader_npz(tmp_path: Path, undirected: bool):
    adj = _simple_adj(undirected)
    f = tmp_path / "g.npz"
    sp.save_npz(f, adj)
    g = GraphLoader.load(f)
    _assert_same_csr(adj, g.adjacency)
    assert g.directed == (not undirected)


def test_graphloader_edges(tmp_path: Path):
    # plain edge list (space-sep)
    f = tmp_path / "toy.edges"
    f.write_text("0 1\n1 2\n")  # unsymmetrised → directed
    g = GraphLoader.load(f)
    assert g.directed
    assert g.num_nodes == 3
    assert g.adjacency[1, 2] == 1


def test_graphloader_mtx(tmp_path: Path):
    adj = _simple_adj()
    f = tmp_path / "toy.mtx"
    mmwrite(str(f), adj)
    g = GraphLoader.load(f)
    _assert_same_csr(adj, g.adjacency)
    assert not g.directed


def test_graphloader_gml(tmp_path: Path):
    # build with networkx
    G = nx.Graph()
    G.add_edge(0, 1); G.add_edge(1, 2)
    f = tmp_path / "toy.gml"
    nx.write_gml(G, f)
    g = GraphLoader.load(f)
    assert not g.directed
    assert g.adjacency.nnz == 4      # undirected ⇒ 2 edges ×2


# ---------------------------------------------------------------------
# 3. Registry decorator sanity check
# ---------------------------------------------------------------------
def test_register_new_loader(tmp_path: Path):
    # create a fake extension ".foo"
    ext = ".foo"

    @GraphLoader.register(ext)
    def _load_foo(path: Path):
        # loader that ignores content, returns 2-node edge
        rows, cols = [0], [1]
        adj = sp.coo_matrix((np.ones(1, int), (rows, cols)), shape=(2, 2)).tocsr()
        return adj, True

    # create dummy file and load
    f = tmp_path / f"dummy{ext}"
    f.write_text("ignored")
    g = GraphLoader.load(f)
    assert g.directed
    assert g.adjacency[0, 1] == 1
    assert f.suffix.lower() in GraphLoader.registry
