"""
Spectral-subspace distance metric.

This module provides :pyfunc:`eigen_subspace_distance`, which compares the
leading *k* eigenpairs of two graphs.  It is robust to

* integer or float adjacency matrices (we up-cast when needed);
* asking for *too many* eigenpairs (ARPACK requires ``k < n``);
* graphs of **different size** – we fall back to an eigenvalue-only distance;
* tiny numerical noise – any value below *tolerance* is rounded to exact 0.
"""
from typing import Tuple, Literal

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance

# --------------------------------------------------------------------------- #
# type aliases
# --------------------------------------------------------------------------- #
metric_type = Literal['eigen_val', 'eigen_vec']

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _eig_pairs(A: csr_array, k: int, which: str = "LA") -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the *k* largest-algebraic eigenvalues and eigenvectors of ``A``.

    Parameters
    ----------
    A
        Square symmetric matrix in CSR format.
    k
        Number of eigenpairs requested.  If ``k`` is ≥ *min(A.shape)`` it is
        reduced to ``n − 1`` to satisfy ARPACK.
    which
        Part of the spectrum to compute (``"LA"`` = largest algebraic).

    Returns
    -------
    λ, V
        ``λ`` is length-``k`` (descending order); ``V`` has shape (*n*, *k*).
    """
    n = min(A.shape)
    if k >= n:
        k = n - 1  # ARPACK cannot handle k == n

    A = A.astype(float, copy=False)  # ensure a float dtype for eigsh
    vals, vecs = eigsh(A, k=k, which=which)
    order = np.argsort(vals)[::-1]  # descending
    return vals[order], vecs[:, order]


def _eigen_distance(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    k: int = 10,
    tolerance: float = 1e-12,
    metric: metric_type = 'eigen_val',
) -> float:
    """
    Distance between the leading spectral subspaces of two graphs.

    If *both* graphs have the same node count the metric is::

        ||λ_emp − λ_sur||₂  +  mean_i(1 − cos(|v_emp,i|, |v_sur,i|))

    where the absolute value removes the ± sign ambiguity of eigenvectors.

    If node counts differ an eigenvector comparison is undefined.  In that
    case a `RuntimeWarning` is emitted and the metric degrades gracefully to
    **eigenvalue-only** distance ``||λ_emp − λ_sur||₂``.

    Any result whose magnitude is below *tolerance* is returned as *exact*
    0.0 so that identical graphs do not yield tiny floating-point residues.

    Parameters
    ----------
    emp_adj, sur_adj
        CSR adjacency matrices (weighted or unweighted, symmetric).
    k
        Number of leading eigenpairs to use (will be clamped so that
        ``k < min(n_emp, n_sur)``).
    tolerance
        Numerical zero threshold for the final distance.

    Returns
    -------
    float
        Non-negative spectral distance.
    """
    # ------------------------------------------------------------------ k clamp
    k_emp = min(k, emp_adj.shape[0] - 1)
    k_sur = min(k, sur_adj.shape[0] - 1)
    k_ = min(k_emp, k_sur)

    # ---------------------------------------------------------------- eigpairs
    λ_emp, V_emp = _eig_pairs(emp_adj, k_)
    λ_sur, V_sur = _eig_pairs(sur_adj, k_)

    
    if metric == 'eigen_val':
        val_dist = float(np.linalg.norm(λ_emp - λ_sur))
        return val_dist if abs(val_dist) >= tolerance else 0.0

    elif metric == 'eigen_vec':
        # ------------------------------------------------------------- vectors term
        if emp_adj.shape == sur_adj.shape:
            vec_dist = [
                cosine(np.abs(V_emp[:, i]), np.abs(V_sur[:, i]))
                for i in range(k_)
            ]
            vec_dist = float(np.mean(vec_dist))
            return vec_dist if abs(vec_dist) >= tolerance else 0.0
        else:
            raise ValueError(
                "Cannot compare eigenvectors of graphs with different node counts."
            )

    else:
        raise ValueError(
            f"Unknown metric type: {metric}. Use 'eigen_val' or 'eigen_vec'."
        )

# --------------------------------------------------------------------------- #
# main functions
# --------------------------------------------------------------------------- #

def eigen_val_distance(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    k: int = 10,
    tolerance: float = 1e-12,
) -> float:
    return _eigen_distance(emp_adj, sur_adj, k=k, tolerance=tolerance, metric='eigen_val')

def eigen_vec_distance(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    k: int = 10,
    tolerance: float = 1e-12,
) -> float:
    return _eigen_distance(emp_adj, sur_adj, k=k, tolerance=tolerance, metric='eigen_vec')

def centrality_distance(
    emp_adj: csr_array,
    sur_adj: csr_array,
    *,
    tolerance: float = 1e-12,
) -> float:

    _, V_emp = _eig_pairs(emp_adj, 1)
    _, V_sur = _eig_pairs(sur_adj, 1)

    d = wasserstein_distance(
        np.abs(V_emp.ravel()),
        np.abs(V_sur.ravel()),
    )
    return d if abs(d) >= tolerance else 0.0