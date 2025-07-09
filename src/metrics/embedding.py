"""
Metrics for comparing two adjacency matrices based on node embeddings
"""

from typing import Literal
import numpy as np
import networkx as nx
from scipy.sparse import csr_array
from scipy.stats import wasserstein_distance  # 1‑D Earth mover distance
from scipy.spatial.distance import cosine
from sbm.utils.util import restrict_to_lcc
from nodevectors import Node2Vec, ProNE  # type: ignore

### Aliases --------------------------------------------------------
EmbeddingMethods = Literal["node2vec", "prone"]

### Helper functions -----------------------------------------------

def _embed_and_sample(
        adj: csr_array,
        method: str,
        dim: int,
        n_pairs:int,
        rng:np.random.Generator
    ) -> np.ndarray:

    G = nx.from_scipy_sparse_matrix(adj)
    if method == "node2vec":
        model = Node2Vec(n_components=dim, walklen=80, return_weight=1.0, 
                            neighbor_weight=1.0, epochs=20, verbose=False)
    elif method == "prone":
        model = ProNE(n_components=dim)
    else:
        raise ValueError("Unknown method")
    emb = model.fit_transform(G)

    nodes = np.arange(emb.shape[0])  # node indices
    vecs = np.vstack([emb[n] for n in nodes])

    # sample pairs uniformly without replacement (if possible)
    m = nodes.size
    total_pairs = m * (m - 1) // 2

    n_samp = min(n_pairs, total_pairs)

    idx1 = rng.choice(m, size=n_samp, replace=True)
    idx2 = rng.choice(m - 1, size=n_samp, replace=True)
    idx2[idx2 >= idx1] += 1  # ensure idx2 ≠ idx1

    # compute inner products between sampled pairs
    #ip = np.array([cosine(vecs[idx1[i]], vecs[idx2[i]]) for i in range(n_samp)])
    ip = np.einsum("ij,ij->i", vecs[idx1], vecs[idx2])  # inner product

    return ip.ravel().astype(float)  # return as 1D array

def _embedding_ip_emd(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    dim: int = 128,
    n_pairs: int = 10_000,
    embedding_method: EmbeddingMethods = "node2vec",  # "node2vec" or "prone"
    rng: np.random.Generator = np.random.default_rng(1),
) -> float:
    """Compare Node2Vec *and* ProNE embeddings via inner‑product distributions.

    The returned distance is the mean of the two 1‑D Wasserstein distances.
    """

    if embedding_method == "node2vec":
        ip_emp_n2v = _embed_and_sample(emp_adj, "node2vec", dim=dim,
                                       n_pairs=n_pairs, rng=rng
                                    )
        ip_sur_n2v = _embed_and_sample(sur_adj, "node2vec", dim=dim,
                                       n_pairs=n_pairs, rng=rng
                                    )

        d = wasserstein_distance(ip_emp_n2v, ip_sur_n2v)
    elif embedding_method == "prone":
        ip_emp_prn = _embed_and_sample(emp_adj, "prone", dim=dim,
                                       n_pairs=n_pairs, rng=rng
                                    )
        ip_sur_prn = _embed_and_sample(sur_adj, "prone", dim=dim,
                                       n_pairs=n_pairs, rng=rng
                                    )

        d = wasserstein_distance(ip_emp_prn, ip_sur_prn)
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}")

    return float(d)

###############################################################################
# Embedding inner‑product EMD ---------------------------------------------
###############################################################################

def embedding_node2vec_ip_emd(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    dim: int = 128,
    n_pairs: int = 10_000,
    rng: np.random.Generator = np.random.default_rng(1),
) -> float:
    """Compare Node2Vec embeddings via inner‑product distributions."""
    return _embedding_ip_emd(emp_adj, sur_adj, dim=dim, n_pairs=n_pairs, 
                             embedding_method="node2vec", rng=rng)

def embedding_prone_ip_emd(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    dim: int = 128,
    n_pairs: int = 10_000,
    rng: np.random.Generator = np.random.default_rng(1),
) -> float:
    """Compare ProNE embeddings via inner‑product distributions."""
    return _embedding_ip_emd(emp_adj, sur_adj, dim=dim, n_pairs=n_pairs, 
                             embedding_method="prone", rng=rng)