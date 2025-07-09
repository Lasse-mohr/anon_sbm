###############################################################################
#  noisy_fit_refactored.py
#  ───────────────────────────────────────────────────────────────────────────
#  Differential-privacy noise generation for Stochastic-Block-Model (SBM) fits
#
#  This module implements three Gaussian–mechanism variants:
#
#    • **heterogeneous_gaussian**  – optimal per-coordinate σq   (Sec.-4, eq. 12)
#    • **naive_degree_gaussian**   – one common  σ  calibrated to global
#                                     Δmax-degree  (classic node-DP baseline)
#    • **naive_edge_count_gaussian** – one common  σ  calibrated to
#                                       Δnlk  (α-quantile “single–block change”)
#
#  All noise is added to the *SBM sufficient statistics*
#           n_r   … block sizes            (1 × B   coordinates)
#           m_rs  … upper-tri edge counts  (B(B+1)/2 coordinates)
#
#  Memory footprint is O(B + nnz(m)), never B².
#
#──────────────────────────────────────────────────────────────────────────────
#  SYMBOL TABLE  (matches notation in the derivation)
#──────────────────────────────────────────────────────────────────────────────
#
#  B           : number of blocks.
#  k_vec       : integer array [k₀,…,k_{B-1}]  — true block sizes (≈k or k+1).
#
#  m_rs        : true edge count between blocks r and s (upper triangle incl. diag).
#  N_rs        : number of *possible* edges, =
#                  k_r·(k_r-1)/2    if r=s
#                  k_r·k_s          if r≠s
#
#  p_rs        : edge probability  m_rs / N_rs.  For p=0 or 1 we clip to
#                clip_p   (default 1e-12) to keep weights finite.
#
#  Δ_nlk       : “single-block change” sensitivity — α-quantile of the
#                maximum number of neighbours a *single* node can have in the
#                *same* block (max Binomial), eq. (2).
#  delta_tail  : 1-α — contributes to δ in (ε,δ+δ_tail)-DP.
#
#  c_n         : sensitivity² for n_r.  Here Δ=1 ⇒  c_n = 1.
#  c_e_val     : sensitivity² for *all* edge counts m_rs = Δ_nlk².
#
#  w_n[r]      : utility weight for n_r
#  w_rs        : utility weight for m_rs
#  w_e         : list of w_rs for every stored upper-tri cell.
#
#  S_sum       : Σ_q √(c_q w_q)   (needed in eq. (10)–(12)).
#
#  R           : ε² / (2 ln(1.25/(δ+δ_tail)))   — Gaussian mech. constant.
#
#  σ_n[r]      : standard deviation for noise on n_r  (heterogeneous case).
#  σ_e[r,s]    : standard deviation for noise on m_rs (upper-tri CSR).
#  σ_common    : single σ used by the two naïve variants.
#
#  noise_type  : one of {"heterogeneous_gaussian", "naive_degree_gaussian",
#                        "naive_edge_count_gaussian"}.
#
#  clip_p      : lower/upper bound substituted for p_rs = 0 or 1.
#  weight_clip : optional upper bound on w to avoid extreme σ→0.
###############################################################################

# sbm/noisy_fit.py

from dataclasses import dataclass
from typing import List, Optional, Callable, Literal, Tuple

import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_array
from scipy.special import binom
from math import exp, sqrt
from scipy.stats import norm

from sbm.io import SBMFit
from sbm.post_process import (
    naive_clamping,
    constrained_lasso,
    PostProcessFnName,
)

# --------------------------------------------------------------
### aliasses
# --------------------------------------------------------------
NoiseType = Literal[
    "heterogeneous_gaussian",
    "naive_degree_gaussian",
    "naive_edge_count_gaussian"
]
# --------------------------------------------------------------
### Helper functions
# --------------------------------------------------------------

# ---------------------------------------------------------------------
#  analytic_gauss_K  (now with the 2-factor)
# ---------------------------------------------------------------------


def analytic_gauss_K(eps: float, delta: float, tol: float = 1e-12) -> float:
    """
    Return alpha* for the analytic Gaussian mechanism (Balle et al. 2018, Alg. 1).
    Works for any eps > 0 and 0 < delta < 1.
    """
    delta0 = norm.cdf(0.0) - exp(eps) * norm.cdf(-sqrt(2*eps))

    # ----------------------------------------------------------------
    # Branch A  (delta >= delta0)  ->  solve for v >= 0
    # ----------------------------------------------------------------
    if delta >= delta0:
        def B_plus(v: float) -> float:
            return float(norm.cdf(sqrt(eps)*v) - \
                   exp(eps) * norm.cdf(-sqrt(eps)*(v + 2.0)))

        lo, hi = 0.0, 1.0
        while B_plus(hi) < delta:      # B_plus increases in v
            hi *= 2.0
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if B_plus(mid) < delta:
                lo = mid
            else:
                hi = mid
        v_star = hi
        alpha  = sqrt(1.0 + v_star/2.0) - sqrt(v_star/2.0)

    # ----------------------------------------------------------------
    # Branch B  (delta < delta0)   ->  solve for u >= 0
    # ----------------------------------------------------------------
    else:
        def B_minus(u: float) -> float:
            return float(norm.cdf(-sqrt(eps)*u) - \
                   exp(eps) * norm.cdf(-sqrt(eps)*(u + 2.0)))

        lo, hi = 0.0, 1.0
        while B_minus(hi) > delta:     # B_minus decreases in u
            hi *= 2.0
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if B_minus(mid) > delta:
                lo = mid
            else:
                hi = mid
        u_star = hi
        alpha  = sqrt(1.0 + u_star/2.0) + sqrt(u_star/2.0)

    return alpha      # this is K

def _upper_tri_csr(rows, cols, data, B):
    """Return symmetric CSR from upper-tri lists."""
    coo = sp.coo_array((data, (rows, cols)), shape=(B, B))
    csr = csr_array(coo)
    return csr + csr.T - sp.diags(csr.diagonal())

def max_binom_quantile(k: int, ps: np.ndarray, alpha: float) -> int:
    """
    Small k (≤50) ⇒ brute-force exact cdf of max{Bin(k,p_s)}.

    Returns the smallest c s.t.   P(max ≤ c) ≥ α.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    # pre-compute per-block Binomial pmf for 0..k
    pmf = np.array([binom(k, j) * ps**j * (1 - ps)**(k - j)
                    for j in range(k + 1)])        # shape (k+1,B)
    cdf = pmf.cumsum(axis=0)                       # shape (k+1,B)

    # P(max ≤ c) = ∏_s F_s(c)
    prod = np.prod(cdf, axis=1)                    # length k+1
    for c in range(k + 1):
        if prod[c] >= alpha:
            return c
    return k  # should never happen if alpha<1

def n_possible(k_r: int, k_s: int, same_block: bool) -> int:
    if same_block:
        return k_r * (k_r - 1) // 2
    return k_r * k_s

def add_edge_noise(
        block_conn: sp.csr_array,
        sigma_e: sp.csr_array,
        rng: np.random.Generator
    ) -> sp.csr_array:
    noisy_conn = sp.triu(
        block_conn.copy(),
        k=0, # include diagonal
        format="csr"
    )

    # add noise to edge-counts (m_rs) upper traingle
    rr, cc = noisy_conn.nonzero() # coordinates of the stored upper-tri values
    sigma = np.asarray(sigma_e[rr, cc]).ravel() # shape the sigma_e vector
    data = np.array(noisy_conn.data, dtype=float)
    data += rng.normal(0.0, sigma, size=len(sigma))

    return sp.csr_array(
        (data, (rr, cc)),
        shape=noisy_conn.shape,
        dtype=float
    )

# ---------------------------------------------------------
#### Main class
# ---------------------------------------------------------

@dataclass
class Noise:
    """Base class for noise added to an SBM fit."""
    fit: SBMFit
    sigma_n: np.ndarray              # shape (B,)
    sigma_e: sp.csr_array            # upper triangle incl. diag
    eps: float
    delta: float
    delta_tail: float                # = 1 - alpha
    delta_nlk: int                   # sensitivity used for m_rs
    metadata: dict

    def sample_sbm_fit(self,
                       rng: np.random.Generator,
                       post: Optional[Callable[[sp.csr_array], sp.csr_array]] = None,
                       ) -> SBMFit:
        """
        Draw *one* noisy SBM draw.

        Returns a **new** `SBMFit` instance ready for `sample_sbm_graph`.
        """
        raise NotImplementedError("Must be implemented in subclass")

@dataclass
class HeterogeneousGaussNoise(Noise):
    """An `SBMFit` with optimal Gaussian noise already added."""
    fit: SBMFit
    sigma_n: np.ndarray              # shape (B,)
    sigma_e: sp.csr_array            # upper triangle incl. diag
    sigma_zero_fun: Callable[[int], float]  # function to compute σ for zero pairs
    eps: float
    delta: float
    delta_tail: float                # = 1 - alpha
    delta_nlk: int                   # sensitivity used for m_rs
    S_sum: float                     # Σ_q √(c_q w_q) used in σ formula
    #R: float
    metadata: dict

    # ---------- sampling ---------------------------------------------
    def sample_sbm_fit(self, # type: ignore
                     rng: np.random.Generator,
                     post: Optional[PostProcessFnName] = 'naive',
                     ) -> SBMFit:
        """
        Draw *one* noisy SBM draw, then (optionally) post-process the
        noisy counts.

        Returns a **new** `SBMFit` instance ready for `sample_sbm_graph`.
        """
        k_vec = np.array(self.fit.block_sizes, int)
        # find the most frequent block size (this will be k)
        k_val = int(np.bincount(k_vec).argmax())  # most frequent block size

        # 1) add Gaussian noise to non-zero elements --------------------
        n_noisy = k_vec + rng.normal(0, self.sigma_n)

        noisy_conn = add_edge_noise(
            block_conn=self.fit.block_conn,
            sigma_e=self.sigma_e,
            rng=rng
        )
        ### post-process the noisy counts ------------------------
        if str(post).lower() == 'lasso':
            conn, n_noisy = constrained_lasso(
                n_noisy=n_noisy,
                noisy_conn=noisy_conn,  # type: ignore
                sigma_rs=self.sigma_e,
                k_val=k_val,
                sigma_zero_fun=self.sigma_zero_fun,
                rng=rng,
                round_thresh=0.5,  # threshold for rounding
                lam=None,  # λ in soft-threshold (impute from noisy fit)
                n_possible_fn=n_possible,
            )

        else:
            ### perform simple post-processing:
            conn, n_noisy = naive_clamping(
                n_noisy=n_noisy,
                noisy_conn=noisy_conn, # type: ignore
                sigma_e=self.sigma_e,
                k_val=k_val,
                sigma_zero_fun=self.sigma_zero_fun,
                rng=rng,
                round_thresh=0.5,  # threshold for rounding
                n_possible_fn=n_possible,
                )

        # 2) re-symmetrise & cast to int -----------------------------
        block_sizes = n_noisy.astype(int).tolist()

        return SBMFit(
            block_sizes=block_sizes,
            block_conn=conn,
            directed_graph=self.fit.directed_graph,
            neg_loglike=float("nan"),          # unknown after noise
            metadata={
                **self.fit.metadata,
                "dp_eps": self.eps,
                "dp_delta": self.delta,
                "dp_delta_tail": self.delta_tail,
            },
        )

@dataclass
class _HomogGaussNoiseBase(Noise):
    """
    Base for 'naïve' variants with *single* σ for n_r, single σ for m_rs.
    Stores only two scalars:  sigma_n_scalar, sigma_e_scalar
    """
    sigma_n_scalar: float      # same σ for every n_r
    sigma_e_scalar: float      # same σ for every m_rs (non-zero counts)

    # ---- helper ----------------------------------------------------
    def _draw_noisy_counts(self,
                           rng: np.random.Generator,
    ) -> tuple[np.ndarray, sp.csr_array]:
        k_vec = np.array(self.fit.block_sizes, int)
        n_noisy = np.maximum( 1, k_vec + rng.normal(0, self.sigma_n_scalar, size=len(k_vec)))

        conn = self.fit.block_conn.copy().astype(float).tocsr()
        mask_nz = conn.data != 0          # we only stored non-zeros
        conn.data[mask_nz] += rng.normal(
            0, self.sigma_e_scalar, size=mask_nz.sum()
        )

        conn = sp.triu(conn, k=0, format="csr")
        conn = conn + conn.T - sp.diags(conn.diagonal())
        conn = conn.astype(int).tocsr()
        return n_noisy, conn

    # ---- public ----------------------------------------------------
    def sample_sbm_fit(self,
                       rng: np.random.Generator,
                       post: Optional[Callable[[sp.csr_array], sp.csr_array]] = None
                       ) -> SBMFit:
        n_noisy, conn = self._draw_noisy_counts(rng)

        if post is not None:
            conn = post(conn)
        else:
            ### perform simple post-processing:
            #   round conn and block_sizes to int and ensure
            #   no conn-count is larger than the max possible
            rr, cc = conn.nonzero()
            for r, c in zip(rr, cc):
                n_noisy[r] = np.round(n_noisy[r])
                n_noisy[c] = np.round(n_noisy[c])

                N = n_possible(n_noisy[r], n_noisy[c], same_block=(r == c)) # type: ignore
                conn[r, c] = np.round(conn[r, c]) # type: ignore
                if conn[r, c] > N: # type: ignore
                    conn[r, c] = N # type: ignore

        return SBMFit(
            block_sizes=n_noisy.tolist(),
            block_conn=conn,
            directed_graph=self.fit.directed_graph,
            neg_loglike=float("nan"),
            metadata={**self.fit.metadata, **self.metadata},
        )

@dataclass
class NaiveDegreeGaussNoise(_HomogGaussNoiseBase):
    """Homogeneous σ using global Δ_deg (α-quantile max-degree)."""
    pass

@dataclass
class NaiveEdgeCountGaussNoise(_HomogGaussNoiseBase):
    """Homogeneous σ using global Δ_nlk (α-quantile single block-change)."""
    pass

# --------------------------------------------------------------------
# 1.  p-matrix and Δ_{nlk}
# --------------------------------------------------------------------
def _prob_matrix_and_sens(
    conn: sp.csr_array,
    k_vec: np.ndarray,
    alpha: float,
    clip_p: float
) -> Tuple[sp.csr_array, int]:
    """
    Convert edge counts → probabilities **p_{rs}** and compute
    Δ_{nlk} = α–quantile of max neighbours in any single block.
    """

    B = len(k_vec)
    p = conn.copy().astype(float).tocoo()
    for i, (r, s) in enumerate(zip(p.row, p.col)):
        N = n_possible(k_vec[r], k_vec[s], r == s) # type: ignore
        p.data[i] = conn[r, s] / N

    p = p.tocsr()
    delta_nlk = 0
    for r in range(B):
        ps_row = np.zeros(B)
        start, end = p.indptr[r], p.indptr[r + 1]
        ps_row[p.indices[start:end]] = p.data[start:end]
        delta_nlk = max(delta_nlk,
                        max_binom_quantile(k_vec[r], ps_row, alpha))

    # Avoid p=0 or 1 in later calculations
    p.data[p.data == 0.0] = clip_p
    p.data[p.data == 1.0] = 1.0 - clip_p

    return p, delta_nlk

# --------------------------------------------------------------------
# 2.  Weight computation
# --------------------------------------------------------------------
def _weights_and_S(
    p_mat: sp.csr_array,
    k_vec: np.ndarray,
    delta_nlk: int,
    weight_clip: float,
    clip_p: float,
    c_n_val: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], float]:
    """
    Compute weights w_q for n_r and m_rs, and the sum S = Σ √(c_q w_q).

    Weights are used to compute the noise levels σ_q so as to minimize
    expected likelihood loss

    Compute:
      * w_n   : vector length B
      * w_e   : list for every stored upper-tri cell (row parity)
      * idx_e : corresponding *column* indices (for CSR builder)
      * S_sum = Σ sqrt(c_q w_q) used in σ formula (12)
    """

    B = len(k_vec)
    c_e_val = delta_nlk ** 2
    w_n = np.zeros(B)
    w_e = []
    col_e= []
    row_e = []

    S_sum = 0.0

    for r in range(B):
        # ---- diagonal r,r (zero and non-zero alike) ---------------
        N_rr = n_possible(k_vec[r], k_vec[r], True)
        p_rr = p_mat[r, r]
        if p_rr < clip_p:
            w_rr = weight_clip
        else:
            # MSE weights
            #w_rr = 1.0 / N_rr

            # KL-based weight  w_rr = N_rr / [2 p_rr (1-p_rr)]
            w_rr = N_rr / (2.0 * p_rr * (1.0 - p_rr))   ### ← NEW (KL)

        w_rr = min(w_rr, weight_clip)

        # store sparse diagonal weight
        w_e.append(w_rr)
        row_e.append(r)
        col_e.append(r)

        S_sum += math.sqrt(c_e_val * w_rr)

        # ---- off-diagonal non-zero pairs r,s (s>r) -------------------------------
        for s_ptr in range(p_mat.indptr[r], p_mat.indptr[r + 1]):
            s = p_mat.indices[s_ptr]
            if s <= r:
                continue

            p_rs = p_mat[r, s]
            if p_rs <= clip_p or p_rs >= 1.0 - clip_p:
                w_rs = weight_clip
            else:
                N_rs = n_possible(k_vec[r], k_vec[s], False) # type: ignore

                # MSE weights
                #w_rs = 1.0 / N_rs# (2 * N_rs * p_rs * (1 - p_rs))

                # KL-based weight
                w_rs = N_rs / (2.0 * p_rs * (1.0 - p_rs))  ### ← NEW (KL)

            w_rs = min(w_rs, weight_clip)

            # store sparse upper-tri weights
            w_e.append(w_rs)
            row_e.append(r)
            col_e.append(s)

            # add to normalization S
            S_sum += math.sqrt(c_e_val * w_rs)

        # ---- off-diagonal zero-pairs r>s ----------------------------
        # off diag
        present = set(p_mat.indices[p_mat.indptr[r] : p_mat.indptr[r+1]])
        for s in range(r+1, B):
            # only add terms from zero pairs upper diagonal pairs
            if (s in present) or s<=r:
                continue

            N_rs = n_possible(k_vec[r], k_vec[s], False)
            S_sum += math.sqrt(c_e_val * (1/N_rs))

        # ---- MSE block-size weight w_n[r] -----------------------------
        #inter = p_mat[[r]].toarray().ravel()
        #inter[r] = 0.0

        ## ---- block-size weight w_n[r]  (MSE version)
        #p_row = p_mat[[r]].toarray().ravel()
        #p_off = np.delete(p_row, r)

        #w_n[r]  = k_vec[r] * (p_off ** 2 @ np.delete(k_vec, r))
        #p_rr    = p_row[r]
        #if k_vec[r] > 1:
        #    w_n[r] += ((2 * k_vec[r] - 1) ** 2) / (2 * k_vec[r] * (k_vec[r] - 1)) \
        #      * (p_rr ** 2)
        #if w_n[r] > weight_clip:
        #    w_n[r] = weight_clip

        # ---- block-size weight w_n[r]  (KL version) ---------------
        p_row = p_mat[[r]].toarray().ravel()

        # First sum:  2 * Σ w_rs * p_rs^2 / k^2
        w_nr = 0.0
        for s in range(B):
            if s == r:
                # diagonal term already added
                continue

            p_rs = p_row[s]
            if p_rs <= clip_p or p_rs >= 1.0 - clip_p:
                continue

            N_rs = n_possible(k_vec[r], k_vec[s], False)
            w_rs = N_rs / (2.0 * p_rs * (1.0 - p_rs))
            w_nr += 2.0 * w_rs * (p_rs ** 2) / (k_vec[r] ** 2)

        # Second (diagonal) term: w_rr * c_k^2 * p_rr^2
        c_k = (2 * k_vec[r] - 4) / (k_vec[r] * (k_vec[r] - 1))
        w_nr += w_rr * (c_k ** 2) * (p_rr ** 2)

        if w_nr > weight_clip:
            w_nr = weight_clip

        w_n[r] = w_nr

        S_sum += math.sqrt(c_n_val * w_n[r])

    # backup in case all probs were clipped to zero 
    if S_sum <= 0:
        print("Warning: S_sum is zero, using small value to avoid division by zero.")
        S_sum = 1e-12

    return w_n, np.asarray(w_e), row_e, col_e, S_sum


# --------------------------------------------------------------------
# 3.  σ computation
# --------------------------------------------------------------------
def _compute_sigmas(
    w_n: np.ndarray,
    w_e: np.ndarray,
    S_sum: float,
    eps: float,
    delta: float,
    delta_tail: float,
    B: int,
    c_e_val: float,
    col_e: List[int],
    row_e: List[int],
) -> Tuple[np.ndarray, sp.csr_array, float]:
    """
    Return vector σ_n and sparse CSR σ_e (upper-tri inc. diag).

    Computes the noise levels for the heterogeneous Gaussian mechanism
    """
    ### old gaussian mechanism:
    #R = eps**2 / (2*math.log(1.25/(delta+delta_tail)))
    #sigma_n = np.sqrt(1/np.sqrt(w_n*R) * S_sum)
    #sigma_e_vals = np.sqrt(c_e_val * w_e) * S_sum / R
    #sigma_e_vals = np.sqrt(np.sqrt(c_e_val / (w_e*R)) * S_sum)

    ### New gaussian mechanism: Balle and Wang 2018
    R = analytic_gauss_K(eps, delta)

    factor = (2 * R) ** 2 / S_sum
    sigma_n       = np.sqrt( np.sqrt(1.0 * w_n) * factor )
    sigma_e_vals  = np.sqrt( np.sqrt(c_e_val * w_e) * factor )

    print(f'max σ_n: {sigma_n.max():.3f}, max σ_e: {sigma_e_vals.max():.3f}')

    sigma_e = _upper_tri_csr(row_e, col_e, sigma_e_vals, B)

    return sigma_n, sigma_e, factor


# --------------------------------------------------------------------
# 4.  Factory
# --------------------------------------------------------------------
def create_sbm_noise(
    sbm: SBMFit,
    eps: float,
    delta: float,
    alpha: float,
    *,
    clip_p: float = 1e-12,
    weight_clip: float = 1e12,
    noise_type: NoiseType = "heterogeneous_gaussian",
) -> "Noise":
    """
    Construct one of the three Noise objects.

    Steps:
      (1) convert counts→probabilities and compute Δ_{nlk}
      (2) weights  w_q  and  S = Σ√(c_q w_q)
      (3) σ_q  via eq. (12)
      (4) package into chosen Noise subclass
    """
    if sbm.directed_graph:
        raise NotImplementedError("undirected only")

    k_vec = np.asarray(sbm.block_sizes, int)
    B = len(k_vec)

    # 1) probabilities & sensitivity Δ_nlk --------------------------
    p_mat, delta_nlk = _prob_matrix_and_sens(
        sbm.block_conn, k_vec, alpha, clip_p
    )
    delta_tail = 1.0 - alpha
    c_e_val = delta_nlk ** 2

    # 2) weights & S -------------------------------------------------
    w_n, w_e, row_e, col_e, S_sum = _weights_and_S(
        p_mat, k_vec, delta_nlk, weight_clip, clip_p, c_n_val=1.0
    )

    # 3) σ’s  --------------------------------------------------------
    sigma_n, sigma_e, factor = _compute_sigmas(
        w_n=w_n, w_e=w_e, S_sum=S_sum,
        eps=eps, delta=delta, delta_tail=delta_tail,
        B=B, c_e_val=c_e_val, col_e=col_e, row_e=row_e
    )

    # -- select variant ---------------------------------------------
    if noise_type == "heterogeneous_gaussian":
        return HeterogeneousGaussNoise(
            fit=sbm,
            sigma_n=sigma_n,
            sigma_e=sigma_e,
            ### OLD gaussian mechanism:
            #sigma_zero_fun=lambda N_rs: np.sqrt(np.sqrt(c_e_val / (N_rs*R)) * S_sum),
            ### NEW gaussian mechanism:
            sigma_zero_fun=lambda N_rs: np.sqrt(
                np.sqrt(c_e_val / N_rs) * factor      # ← updated line
            ),
            eps=eps,
            delta=delta,
            delta_tail=delta_tail,
            delta_nlk=delta_nlk,
            S_sum=S_sum,
            #R=R,
            metadata={"noise": "heterogeneous_gaussian",
                      "alpha": alpha,
                      "Delta_nlk": delta_nlk,
                      "delta_tail": delta_tail,
                      "eps": eps,
                      **sbm.metadata,
                      },
        )

    # ----- homogeneous variants share one σ ------------------------
    total_c = B*1.0 + (B*(B+1)//2) * c_e_val   # all cells!
    sigma_common = math.sqrt(total_c /
                             (eps**2 / (2*math.log(1.25/(delta+delta_tail)))))

    if noise_type == "naive_edge_count_gaussian":
        return NaiveEdgeCountGaussNoise(
            fit=sbm,
            sigma_n_scalar=sigma_common,
            sigma_e_scalar=sigma_common,
            sigma_n=np.full(B, sigma_common),
            sigma_e=sp.csr_array((B, B), dtype=float),
            eps=eps,
            delta=delta,
            delta_tail=delta_tail,
            delta_nlk=delta_nlk,
            metadata={"noise": "naive_edge_count_gaussian",
                      "sigma_common": sigma_common, **sbm.metadata},
        )

    # compute 
    if noise_type == "naive_degree_gaussian":
        return NaiveDegreeGaussNoise(
            fit=sbm,
            sigma_n_scalar=sigma_common,
            sigma_e_scalar=sigma_common,
            sigma_n=np.full(B, sigma_common),
            sigma_e=sp.csr_array((B, B), dtype=float),
            eps=eps,
            delta=delta,
            delta_tail=0.,
            delta_nlk=0,
            metadata={"noise": "naive_degree_gaussian",
                      "sigma_common": sigma_common,
                      **sbm.metadata},
        )

    raise ValueError(f"unknown noise_type {noise_type!r}")