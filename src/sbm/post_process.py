"""
Functions and classes for post-processing SBM fits to create noisy SBM fits


"""
# --------------------------------------------------------------
from typing import Callable, Tuple, Literal, Optional
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
from scipy.sparse import csr_array
from typing import Callable

# ------------------------------------------------------------------
# Aliases
PostProcessFnName = Literal['naive', 'lasso']

# ------------------------------------------------------------------

def naive_clamping(
    n_noisy: np.ndarray,
    noisy_conn: sp.csr_array,
    sigma_e: sp.csr_array,
    k_val: int,
    sigma_zero_fun: Callable[[int], float],
    rng: np.random.Generator,
    *,
    round_thresh: float = 0.5,
    n_possible_fn: Callable[[int, int, bool], int],
)-> Tuple[csr_array, np.ndarray]: 

    rr, cc = noisy_conn.nonzero()
    noise = rng.normal(loc=0, scale=sigma_e[rr, cc]) # type: ignore
    noisy_conn.data = np.maximum(0., noisy_conn.data + noise).astype(int)

    ### perform simple post-processing:
    #   We have to release k anyway, so we set all block sizes to k_val.
    #   This allows us to put strict bounds on the connectivity matrix.
    for r, c in zip(rr, cc):
        n_noisy[r] = k_val # type: ignore
        n_noisy[c] = k_val # type: ignore

        # ensure no negative counts
        if noisy_conn[r, c] < 0: # type: ignore
            noisy_conn[r, c] = 0 # type: ignore
            continue

        # ensure no noisy_conn-count is larger than the max possible
        N = n_possible_fn(n_noisy[r], n_noisy[c], same_block=(r == c)) # type: ignore
        if noisy_conn[r, c] > N: # type: ignore
            noisy_conn[r, c] = N # type: ignore
            continue
        # round conn to int
        noisy_conn[r, c] = np.round(noisy_conn[r, c]) # type: ignore
            
    ### add noise to zero pairs
    noisy_conn_lil = noisy_conn.tolil()

    B = len(n_noisy)
    total_zero_pair_edges = 0
    for r in range(B):
        present = set(noisy_conn_lil.rows[r])
        for s in range(r, B):
            # only add noise to zero pairs
            if s in present:
                continue

            N_rs = n_possible_fn(k_val, k_val, r == s)  # type: ignore
            z = rng.normal(0, sigma_zero_fun(N_rs))  # noise for zero pair (r,s)
            if z < round_thresh:  # round_thresh
                continue  # remain zero

            m_rs = int(round(z))  # symmetric, non-negative
            m_rs = min(m_rs, N_rs)
            if m_rs > 0:
                total_zero_pair_edges += m_rs
                noisy_conn_lil[r, s] = m_rs # type: ignore
                if r != s:
                    noisy_conn_lil[s, r] = m_rs # type: ignore
    print(f"[NAIVE]    Added {total_zero_pair_edges} edges to zero pairs.") 

    # ---------- finish   ------------------------------------------
    conn_csr = csr_array(noisy_conn_lil, dtype=int)
    conn_sym = sp.triu(conn_csr, k=0, format='csr')
    conn_sym = conn_sym + conn_sym.T - sp.diags(conn_sym.diagonal())
    conn_sym.data = conn_sym.data.astype(int)

    noisy_conn = csr_array(conn_sym, dtype=int)

    return noisy_conn, n_noisy

# ------------------------------------------------------------------
# utility functions for Lasso post-processing
def lambda_for_activation(rho: float) -> float:
    """λ so that exactly rho of zero cells survive the threshold."""
    lam = norm.isf(rho / 2.0) # two-sided tail
    return float(lam)

# ------------------------------------------------------------------
def constrained_lasso(
    n_noisy: np.ndarray,
    noisy_conn: sp.csr_array,
    sigma_rs: sp.csr_array,
    k_val: int,
    sigma_zero_fun: Callable[[int], float],
    rng: np.random.Generator,
    *,
    round_thresh: float = 0.5,
    lam: Optional[float] = None,
    n_possible_fn: Callable[[int, int, bool], int],
) -> Tuple[sp.csr_array, np.ndarray]:
    """
    L1-constrained projection row-by-row.

    Parameters
    ----------
    conn_ut        : noisy counts, CSR, **upper triangle only**.
    sigma_rs       : matching σ_rs for the same nnz pattern.
    k_val          : released block size k (all blocks are k or k+1).
    sigma_zero_fun : N_rs ↦ σ for a zero entry (uses new weight scheme).
    rng            : np.random.Generator.
    round_thresh   : magnitude that rounds to 1 (default .5).
    lam            : λ in soft-threshold  x ← sign(x)·max(|x|−λσ²,0).
    n_possible_fn  : callable (k_r,k_s,r==s) → N_rs.
    """

    B = noisy_conn.shape[0]
    noisy_conn_lil = noisy_conn.tolil(copy=True)          # efficient insertion
    sigma = sigma_rs.tolil(copy=False)

    if lam is None:
        # compute lambda based on desired proportion of active block pairs
        # assume that the average block connects to 5 others
        rho = .1 #5 * B / (B * (B - 1)/2)  
        lam = lambda_for_activation(rho)  # default λ for 5% of zero pairs

    total_added = 0
    for r in range(B):

        ### set block sizes to k_val
        #   We have to release k anyway, so we set all block sizes to k_val.
        #   This allows us to put strict bounds on the connectivity matrix.
        n_noisy[r] = k_val

        # check where non-zero counts
        present = {c: i for i, c in enumerate(noisy_conn_lil.rows[r])}
        # ---------- a) process existing nnz ------------------------
        for idx, c in enumerate(noisy_conn_lil.rows[r]):
            # account for upper-triangle only
            if c < r:
                continue

            val   = noisy_conn_lil.data[r][idx]
            sig   = float(sigma[r, c]) # type: ignore

            # lasso regression with soft thresholding and clamping
            N_rs  = n_possible_fn(k_val, k_val, r == c)
            lasso_shrink = np.sign(val) * max(abs(val) - lam * sig, 0)
            new   = min(max(round(lasso_shrink), 0), N_rs)

            noisy_conn_lil.data[r][idx] = int(new)

        # ---------- b) add noise to zero cells (upper-tri only) ----
        for s in range(r, B):
            if s in present or s < r:                 # already processed nnz
                continue

            N_rs = n_possible_fn(k_val, k_val, r == s)
            sig0 = sigma_zero_fun(N_rs)
            z = rng.normal(0.0, sig0)
            if abs(z) < round_thresh:
                continue

            # lasso regression with soft thresholding
            lasso_shrink = np.sign(z) * max(abs(z) - lam * sig0, 0)
            m = min(round(lasso_shrink), N_rs)
            if m <=0:
                continue

            noisy_conn_lil[r, s] = m
            if r != s:
                noisy_conn_lil[s, r] = m
            total_added += m

    # ---------- finish   ------------------------------------------
    conn_csr = csr_array(noisy_conn_lil, dtype=int)
    conn_sym = sp.triu(conn_csr, k=0, format='csr')
    conn_sym = conn_sym + conn_sym.T - sp.diags(conn_sym.diagonal())
    conn_sym.data = conn_sym.data.astype(int)

    print(f"[LASSO]    Added {total_added} edges to previously-zero pairs")
    return conn_sym, n_noisy
