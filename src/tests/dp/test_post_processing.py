"""
Unit‑tests for sbm.post_process.constrained_lasso
------------------------------------------------
The tests focus exclusively on the *post‑processing* step and therefore
construct tiny synthetic inputs instead of going through the full
`HeterogeneousGaussNoise.sample_sbm_fit` pipeline.

The invariants we check are:
  1. No negative edge counts are returned.
  2. All counts are *integers* and *symmetric* (upper‑triangle mirrored).
  3. Every count is <= the maximum possible edges
     N_rs = k_r*k_s for r≠s,  k_r*(k_r−1)/2 for r=s,  where every k_r == k_val.
  4. The routine overwrites all released block sizes with `k_val`.

Two deterministic corner‑case tests are followed by a stochastic
property test that runs 50 random scenarios; this should still complete
in well under 100 ms.
"""

import numpy as np
import scipy.sparse as sp
import pytest

# ---------------------------------------------------------------------
# system under test
# ---------------------------------------------------------------------
from sbm.post_process import constrained_lasso
from sbm.noisy_fit import n_possible  # helper already used by the library

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _csr_symmetric_from_dense(mat: np.ndarray) -> sp.csr_array:
    """Return *exactly* the upper‑triangle incl. diagonal as CSR."""
    r, c = np.triu_indices(mat.shape[0])
    data = mat[r, c]
    nz_mask = data != 0
    return sp.csr_array((data[nz_mask], (r[nz_mask], c[nz_mask])), shape=mat.shape)

def _make_sigma_rs(pattern: sp.csr_array, *, sigma: float = 1.0) -> sp.csr_array:
    """Return a symmetric σ matrix that shares *pattern*'s sparsity."""
    return sp.csr_array((np.full_like(pattern.data, sigma, dtype=float), pattern.indices, pattern.indptr), shape=pattern.shape)

RNG = np.random.default_rng(0)
SIGMA_ZERO = lambda N: 1.0  # simple constant σ for zero pairs

# ---------------------------------------------------------------------
# 1. deterministic corner‑cases
# ---------------------------------------------------------------------
def test_negative_and_too_large_values_are_clamped():
    """Negative counts become zero; overly large counts are capped at N_rs."""
    k_val = 3
    B = 2
    # construct noisy_conn with impossible values
    dense = np.array([
        [ -2, 10],   # −2 on diagonal 0, 10 off‑diag
        [  0,  5]    # 5 on diag 1 (will be ignored, only upper‑tri processed)
    ], dtype=float)
    noisy_conn = _csr_symmetric_from_dense(dense)
    sigma_rs   = _make_sigma_rs(noisy_conn, sigma=0.5)  # arbitrary σ
    n_noisy    = np.array([k_val, k_val]) + 1  # wrong on purpose

    conn_out, n_out = constrained_lasso(
        n_noisy=n_noisy,
        noisy_conn=noisy_conn,
        sigma_rs=sigma_rs,
        k_val=k_val,
        sigma_zero_fun=SIGMA_ZERO,
        rng=RNG,
        lam=1.0,
        n_possible_fn=n_possible,
    )

    conn_dense = conn_out.toarray()
    # invariants ------------------------------------------------------
    assert (conn_dense >= 0).all()
    assert np.issubdtype(conn_dense.dtype, np.integer)
    # diag 0: max possible = 3*2/2 = 3 ; off‑diag: 3*3 = 9
    assert conn_dense[0, 0] <= 3
    assert conn_dense[0, 1] <= 9
    # symmetry
    assert conn_dense[0, 1] == conn_dense[1, 0]
    # block sizes all set to k_val
    assert (n_out == k_val).all()

def test_zero_matrix_remains_valid():
    """All‑zero input should stay zero (aside from possible added edges)."""
    k_val = 4
    B = 3
    noisy_conn = sp.csr_array((B, B), dtype=float)
    sigma_rs   = sp.csr_array((B, B), dtype=float)
    n_noisy    = np.ones(B, dtype=int) * k_val

    conn_out, _ = constrained_lasso(
        n_noisy=n_noisy,
        noisy_conn=noisy_conn,
        sigma_rs=sigma_rs,
        k_val=k_val,
        sigma_zero_fun=SIGMA_ZERO,
        rng=RNG,
        lam=0.0,          # turn off shrinkage so only zero‑pair addition matters
        round_thresh=10**10,  # large threshold ⇒ keep zeros
        n_possible_fn=n_possible,
    )
    print(noisy_conn.toarray())
    print(conn_out.toarray())
    assert conn_out.nnz == 0, "No edges should be added when threshold is huge"
    assert conn_out.shape == (B, B)

# ---------------------------------------------------------------------
# 2. stochastic property test
# ---------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(50))
def test_randomised_invariants(seed: int):
    """For many random inputs the core invariants must always hold."""
    rng = np.random.default_rng(seed)

    B      = rng.integers(2, 6)            # 2 ≤ B ≤ 5
    k_val  = int(rng.integers(2, 6))       # 2 ≤ k ≤ 5

    # generate random noisy counts in [‑5, 2*N_rs]
    dense  = np.zeros((B, B), dtype=float)
    for r in range(B):
        for s in range(r, B):
            N_rs = n_possible(k_val, k_val, r == s)
            dense[r, s] = rng.integers(-5, 2*N_rs + 1)
    noisy_conn = _csr_symmetric_from_dense(dense)
    sigma_rs   = _make_sigma_rs(noisy_conn, sigma=1.0)
    n_noisy    = rng.integers(1, k_val + 3, size=B)  # arbitrary wrong values

    conn_out, n_out = constrained_lasso(
        n_noisy=n_noisy,
        noisy_conn=noisy_conn,
        sigma_rs=sigma_rs,
        k_val=k_val,
        sigma_zero_fun=SIGMA_ZERO,
        rng=rng,
        lam=1.0,
        n_possible_fn=n_possible,
    )

    # ----- checks ----------------------------------------------------
    conn_dense = conn_out.toarray()
    assert (conn_dense >= 0).all(), "negative counts returned"
    assert np.issubdtype(conn_dense.dtype, np.integer), "counts not integer"
    assert np.allclose(conn_dense, conn_dense.T), "matrix not symmetric"

    for r in range(B):
        for s in range(r, B):
            N_rs = n_possible(k_val, k_val, r == s)
            assert conn_dense[r, s] <= N_rs, "count exceeds N_rs"
    # every block size released as k

    assert (n_out == k_val).all()