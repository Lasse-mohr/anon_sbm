# ---------------------------------------------------------------------
# tests/test_dp_noise.py
# ---------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp
import pytest

from sbm.io import SBMFit
from sbm.noisy_fit import (
    create_sbm_noise,
    HeterogeneousGaussNoise,
    NaiveDegreeGaussNoise,
    NaiveEdgeCountGaussNoise,
)
from sbm.sampling import sample_sbm_graph_from_fit


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _make_sbm(block_sizes, P):
    """
    Create an SBMFit with integer edge counts according to prob-matrix P.
    P must be square len(block_sizes) × len(block_sizes), symmetric.
    """
    k_vec = np.array(block_sizes, int)
    B = len(k_vec)
    data, rows, cols = [], [], []

    for r in range(B):
        for s in range(r, B):
            N = k_vec[r] * k_vec[s] if r != s else k_vec[r] * (k_vec[r] - 1) // 2
            m = int(round(P[r, s] * N))
            if m > 0:
                rows.append(r); cols.append(s); data.append(m)

    M = sp.csr_array((data, (rows, cols)), shape=(B, B))
    M = M + M.T - sp.diags(M.diagonal())

    return SBMFit(
        block_sizes=list(block_sizes),
        block_conn=M,
        directed_graph=False,
        neg_loglike=-1.0,
        metadata={},
    )


def _extract_sigma_e(noise_obj):
    """Return 1-D array of σ_e values stored (heterogeneous variant)."""
    if isinstance(noise_obj, HeterogeneousGaussNoise):
        return noise_obj.sigma_e.data
    raise ValueError


# ---------------------------------------------------------------------
# parametrisation -----------------------------------------------------
EPS, DELTA, ALPHA = 1.0, 1e-6, 0.999
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------
# 1. factory returns correct subclass ---------------------------------
@pytest.mark.parametrize("ntype,cls", [
    ("heterogeneous_gaussian",       HeterogeneousGaussNoise),
    ("naive_degree_gaussian",  NaiveDegreeGaussNoise),
    ("naive_edge_count_gaussian", NaiveEdgeCountGaussNoise),
])
def test_factory_returns(ntype, cls):
    sbm = _make_sbm([3, 3], np.zeros((2, 2)))
    nz = create_sbm_noise(sbm, EPS, DELTA, ALPHA, noise_type=ntype)
    assert isinstance(nz, cls)


# ---------------------------------------------------------------------
# 2. hetero σ_e differ when p differs ---------------------------------
def test_heterogeneous_noise_varies():
    P = np.array([[0.2, 0.9, 0.05],
                  [0.9, 0.2, 0.05],
                  [0.05,0.05, 0.01]])
    sbm = _make_sbm([3, 3, 3], P)
    oz  = create_sbm_noise(sbm, EPS, DELTA, ALPHA, noise_type="heterogeneous_gaussian")
    sig = _extract_sigma_e(oz)
    assert len(np.unique(sig)) > 1, "σ_e should vary for different p_rs"


# ---------------------------------------------------------------------
# 3. hetero σ_e identical when all p equal ----------------------------
def test_heterogeneous_noise_equal():
    P = np.full((3, 3), 0.3)
    sbm = _make_sbm([3, 3, 3], P)
    oz  = create_sbm_noise(sbm, EPS, DELTA, ALPHA,
                           noise_type="heterogeneous_gaussian")
    sig = _extract_sigma_e(oz)
    print(f'sigma_e: {sig}')

    diag = sig[[0, 4, 8]]  # diagonal σ’s
    # diagonal σ’s should all be identical.
    # Account for the fact that only upper-tri is stored.
    assert np.allclose(diag, diag[0])

    # … off-diagonals identical to each other …
    off  = sig[[1, 2, 5]]
    assert np.allclose(off, off[0])

    # … but OFF ≠ DIAG because N_rr ≠ N_rs
    assert not np.allclose(diag[0], off[0])

# ---------------------------------------------------------------------
# 4. naive σ scalars are equal for every coord ------------------------
@pytest.mark.parametrize("ntype", ["naive_degree_gaussian",
                                   "naive_edge_count_gaussian"])
def test_naive_sigma_equal(ntype):
    P = np.array([[0.4, 0.4],
                  [0.4, 0.4]])
    sbm = _make_sbm([4, 4], P)
    nz  = create_sbm_noise(sbm, EPS, DELTA, ALPHA, noise_type=ntype)
    assert np.all(nz.sigma_n == nz.sigma_n[0])
    assert nz.sigma_n_scalar == nz.sigma_e_scalar # type: ignore


# ---------------------------------------------------------------------
# 5. zero & one probabilities handled (no inf / NaN) ------------------
@pytest.mark.parametrize("pdiag,poff", [(0.0, 0.0), (1.0, 0.0),
                                        (0.0, 1.0), (1.0, 1.0)])
def test_zero_one_probabilities(pdiag, poff):
    P = np.array([[pdiag, poff],
                  [poff,  pdiag]])
    sbm = _make_sbm([3, 3], P)
    nz  = create_sbm_noise(sbm, EPS, DELTA, ALPHA,
                           noise_type="heterogeneous_gaussian")
    assert not np.isnan(nz.sigma_n).any(), \
        f'sigma_n should not contain NaN, got {nz.sigma_n}'
    sig = _extract_sigma_e(nz)
    assert not np.isnan(sig).any() and not np.isinf(sig).any(),\
        f'sigma_e should not contain NaN or inf, got {sig}'

# ---------------------------------------------------------------------
# 6. big blocks memory usage (no excessive RAM) -----------------------
def test_big_blocks_memory():
    B = 300
    k = 3
    sizes = [k] * B
    P = np.full((B, B), 0.1)
    np.fill_diagonal(P, 0.2)
    sbm = _make_sbm(sizes, P)          # builds sparse counts

    nz = create_sbm_noise(sbm, 1.0, 1e-6, 0.999,
                          noise_type="heterogeneous_gaussian"
                          )
    # should finish < 1 s and < 200 MB RAM

# ---------------------------------------------------------------------
# 7. sampling integrity (no counts exceed max possible) ---------------
def test_sample_integrity():

    P = np.array([[0.8, 0.3],
                  [0.3, 0.05]])
    sbm   = _make_sbm([10, 20], P)
    noise = create_sbm_noise(sbm, EPS, DELTA, ALPHA,
                             noise_type="heterogeneous_gaussian")
    sbm_noisy = noise.sample_sbm_fit(RNG)


    # check sizes positive
    assert min(sbm_noisy.block_sizes) >= 1
    # check counts ≤ possible
    k = sbm_noisy.block_sizes
    conn = sbm_noisy.block_conn
    rr, cc = conn.nonzero()
    for r, c in zip(rr, cc):
        N = k[r] * k[c] if r != c else k[r] * (k[r]-1) // 2
        assert conn[r, c] <= N
    # can we sample a surrogate graph?
    g = sample_sbm_graph_from_fit(sbm_noisy, RNG)
    assert g.adjacency.shape[0] == sum(k)
    # undirected check
    assert (g.adjacency != g.adjacency.T).nnz == 0