from typing import Dict, Callable, Iterable, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json

import gzip

import numpy as np
from scipy.sparse import csr_array, csr_array, load_npz, coo_matrix
from scipy.io import mmread
import networkx as nx                                # light dep

import numpy as np
from scipy.sparse import load_npz, save_npz

from scipy.sparse import csr_array
from sbm.graph_data import GraphData

# src/sbm/io.py
@dataclass
class SBMFit:
    block_sizes: list[int]
    block_conn: csr_array
    directed_graph: bool # if undirected, only upper triangle is stored
    neg_loglike: float
    metadata: dict

class SBMWriter:
    @staticmethod
    def save(path: Path, fit: SBMFit) -> None:
        """ save SBM fit to file """
        path.mkdir(parents=True, exist_ok=True)
 
        clean_sizes  = [int(s) for s in fit.block_sizes]
        (path / "block_sizes.json").write_text(json.dumps(clean_sizes))

        # save sparce block connectivity matrix using scipy
        with open(path / "block_connectivity.npz", 'wb') as file:
            save_npz(file, fit.block_conn, compressed=True)

        (path / "directed_graph.txt").write_text(str(fit.directed_graph))
        (path / "neg_loglike.txt").write_text(str(fit.neg_loglike))
        with open(path / "metadata.json", 'w') as f:
            json.dump(fit.metadata, f)

    @staticmethod
    def load(path: Path, silence:bool=False) -> SBMFit:
        if not silence:
            print(f"Loading SBM fit from {path}")

        with open(path / "block_sizes.json", 'r') as sizes_file:
            block_sizes = json.load(sizes_file)
        block_sizes = [int(size) for size in block_sizes]

        with open(path / "block_connectivity.npz", 'rb') as conn_file:
            block_conn = load_npz(conn_file)

        directed_graph = path / "directed_graph.txt"
        directed_graph = (path / "directed_graph.txt").read_text().strip().lower() == 'true'
        neg_loglike = float((path / "neg_loglike.txt").read_text().strip())

        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        return SBMFit(
            #blocks=blocks,
            block_sizes=block_sizes,
            block_conn=csr_array(block_conn),
            directed_graph=directed_graph,
            neg_loglike=neg_loglike,
            metadata=metadata
        )

# ---------------------------------------------------------------------
#  GraphLoader
# ---------------------------------------------------------------------

class GraphLoader:
    """
    Factory that maps a file *extension* to a loader function and returns
    a `GraphData` object (CSR adjacency + directed flag).

    Register new loaders with the `@GraphLoader.register('.ext')`
    decorator.
    """

    # maps extension (lower-case, incl. leading dot) -> callable
    registry: Dict[str, Callable[[Path], Tuple[csr_array, bool]]] = {}

    # ----------------------- decorator -------------------------------
    @classmethod
    def register(cls, *exts: str):
        """
        Use as::

            @GraphLoader.register('.gml', '.graphml')
            def _load_graphml(path): ...
        """
        def decorator(fn: Callable[[Path], Tuple[csr_array, bool]]):
            for ext in exts:
                cls.registry[ext.lower()] = fn
            return fn
        return decorator

    # ----------------------- public API ------------------------------
    @staticmethod
    def load(
        path: Path,
        *,
        directed: Optional[bool] = None,
        force_undirected: Optional[bool] = None
    ) -> GraphData:
        """Load graph at *path* and return GraphData."""
        ext = path.suffix.lower()
        if ext not in GraphLoader.registry:
            raise ValueError(
                f"GraphLoader: no loader registered for extension '{ext}'."
            )
        adj, is_directed = GraphLoader.registry[ext](path)

        # allow caller to override detection
        if directed is not None:
            is_directed = bool(directed)

        # if caller wants undirected, symmetrise the adjacency matrix
        if force_undirected:
            if is_directed:
                adj = adj.maximum(adj.T)
            is_directed = False

        adj = csr_array(adj, dtype=np.int8)  # ensure type is int8
        return GraphData(adjacency_matrix=adj, directed=is_directed)

    # ---------------- default loaders -------------------------------

# 1. compressed / plain .npz containing a CSR adjacency ----------------
@GraphLoader.register(".npz")
def _load_npz(path: Path) -> Tuple[csr_array, bool]:
    adj = load_npz(path)
    directed = _is_directed(adj)
    return adj.tocsr(), directed


# 2. Matrix Market -----------------------------------------------------
@GraphLoader.register(".mtx")
def _load_mtx(path: Path) -> Tuple[csr_array, bool]:
    adj = mmread(str(path))
    adj = csr_array(adj, dtype=np.int8)

    directed = _is_directed(adj)
    return adj, directed


# 3. Plain edge list (.edges, .edgelist, .txt, optional .gz) -----------
@GraphLoader.register(".edges", ".edgelist", ".txt", ".gz")
def _load_edgelist(path: Path) -> Tuple[csr_array, bool]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows, cols = [], []
    if not path.exists():
        raise FileNotFoundError(f"GraphLoader: file {path} does not exist.")
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            u, v = map(int, line.split()[:2])
            rows.append(u)
            cols.append(v)
    n = max(rows + cols) + 1
    data = np.ones(len(rows), dtype=np.int8)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    directed = _is_directed(adj)
    if not directed:            # symmetrise if undirected
        adj = adj.maximum(adj.T)
    adj = csr_array(adj, dtype=np.int8)  # ensure type is int8
    return adj, directed


# 4. GML / GraphML via NetworkX ---------------------------------------
@GraphLoader.register(".gml", ".graphml")
def _load_graphml(path: Path) -> Tuple[csr_array, bool]:
    G = nx.read_gml(path) if path.suffix == ".gml" else nx.read_graphml(path)
    directed = G.is_directed()

    # new version of networkx
    #adj = nx.to_scipy_sparse_array(G, format="csr", dtype=np.int8)

    # old version of networkx
    adj = nx.to_scipy_sparse_matrix(G, format="csr", dtype=np.int8)
    if not directed:
        adj = adj.maximum(adj.T)
    return adj, directed

# ---------------- helper ----------------------------------------------
def _is_directed(adj:  csr_array, tol: int = 0) -> bool:
    """
    Quick symmetric test for an unweighted adjacency.
    `tol` is an integer threshold: if more than `tol` entries differ,
    we declare the graph directed.
    """
    diff = adj - adj.T
    return diff.count_nonzero() > tol
