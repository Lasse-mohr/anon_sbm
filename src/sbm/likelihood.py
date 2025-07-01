from typing import (
    Dict,
    Tuple,
    Literal,
)
from numba import jit
from scipy.sparse import coo_array

import numpy as np
from sbm.block_data import BlockData
from sbm.block_change_proposers import (
    EdgeDelta,
    CombinationDelta,
)

#### aliases ######
LikelihoodType = Literal['bernoulli']

# Bernoulli functions
@jit(nopython=True, cache=True, fastmath=True)
def _bernoulli_ll_block_pair(e: int, n: int, eps:float= 1e-6) -> float:
    """
    Profile log-likelihood for one block pair (constants dropped).
    e: number of edges between block pair.
    n: number of possible pairs between block pair.
    """
    if e == 0: # 0 · log 0 := 0   (limit)
        return 0.0
    elif n <= 0:
        raise ValueError("Number of possible pairs (n) must be greater than 0.")
    
    # clip to avoid overflow in lo
    pos = max(e/n, eps)
    neg = max(1 - e/n, eps)
    
    return e * np.log(pos) - (n-e) * np.log(neg)

@jit(fastmath=True, cache=True)
def _delta_ll_bernoulli_block_pair(
        e_old: int, e_new: int,
        n_old: int, n_new: int,
        eps: float = 1e-6
        ) -> float:
    """Δℓ for a single block pair.
    e_new: new number of edges between block pair.
    e_old: old number of edges between block pair.
    n_new: new number of possible pairs between block pair.
    n_old: old number of possible pairs between block pair.
    """

    new_ll = _bernoulli_ll_block_pair(e=e_new, n=n_new, eps=eps)
    old_ll = _bernoulli_ll_block_pair(e=e_old, n=n_old, eps=eps)

    return new_ll - old_ll

def compute_delta_ll_from_change_bernoulli(
        delta_e: EdgeDelta,
        delta_n: CombinationDelta,
        block_data: BlockData) -> float:
    """
    Incremental change in Bernoulli log-likelihood after a node-swap or move.
    Only the pairs present in `delta_e` or `delta_n` need to be visited.
    delta_e: changes in edge counts between affected blocks.
    delta_n: changes in possible pairs between affected blocks.
    block_data: BlockData object containing edge counts and possible pairs.

    :return: Tuple of (change in log-likelihood, edge counts changes of move delta).
    """
    upper_triangle_only = not block_data.directed

    delta_ll = 0.0

    for (r, s), delta in delta_e.items():
        if upper_triangle_only and s < r:
            continue
        e_old = int(block_data.block_connectivity[r, s]) # type: ignore
        n_old = block_data.get_possible_pairs(r, s)

        e_new = e_old + delta
        n_new = n_old + delta_n[r, s]

        delta_ll += _delta_ll_bernoulli_block_pair(
            e_old=e_old,
            e_new=e_new,
            n_old=n_old,
            n_new=n_new
        )

    return delta_ll

# ────────────────────────────────────────────────────────────────────
### Helpter function to vectorise the LL global computation
# ────────────────────────────────────────────────────────────────────
@jit(nopython=True, cache=True, fastmath=True)   # remove decorator if you dislike Numba
def _ll_vec(edges: np.ndarray, pairs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    edges  : e_rs   (int ≥ 0)
    pairs  : n_rs   (int ≥ 1)
    returns: ℓ_rs   (float)
    """
    p = edges / pairs
    p = np.where(p < eps, eps, p)
    p = np.where(p > 1.0 - eps, 1.0 - eps, p)
    return edges * np.log(p) - (pairs - edges) * np.log1p(-p)

def compute_global_bernoulli_ll_fast(block_data:BlockData) -> float:
    """
    Computes the global log-likelihood of the SBM using Bernoulli likelihood.
    Same semantics as the original `compute_global_bernoulli_ll`, but
    **O(nnz)** instead of O(B²).

    The trick: only block pairs with at least one edge (e_rs > 0) can
    change the profiled Bernoulli LL once the constants are dropped.
    """
    conn: coo_array = coo_array(block_data.block_connectivity)
    rows, cols, e = conn.row, conn.col, conn.data.astype(np.int64)

    # Undirected graphs: keep only upper-triangle to avoid double count
    if not block_data.directed:
        keep = rows <= cols
        rows, cols, e = rows[keep], cols[keep], e[keep]

    # Block sizes in matrix-index order
    sizes = np.fromiter(
        (block_data.block_sizes[block_data.inverse_block_indices[i]]
         for i in range(len(block_data.block_sizes))),
        dtype=np.int64,
        count=len(block_data.block_sizes)
    )

    # Possible pair counts n_rs (vectorised)
    n = np.where(
        rows == cols,
        sizes[rows] * (sizes[rows] - 1) // 2,   # diagonal blocks
        sizes[rows] * sizes[cols]               # off-diagonal
    )

    # Safety: skip singleton blocks (n = 0) to avoid /0 in n==1 corner
    valid = n > 0
    if not valid.all():
        rows, cols, e, n = rows[valid], cols[valid], e[valid], n[valid]

    # Vectorised LL and reduction
    return float(_ll_vec(e, n).sum())


def compute_global_bernoulli_ll(
        block_data: BlockData,
) -> float:
    """
    Compute the global log-likelihood of the SBM using Bernoulli likelihood.
    
    :param block_data: The BlockData object containing block connectivity and sizes.
    :param upper_triangle_only: If True, only compute for upper triangle of the connectivity matrix.
    :return: The global log-likelihood.
    """
    upper_triangle_only = not block_data.directed
    ll = 0.0
    for r in range(len(block_data.block_sizes)):

        # if block has less than 2 nodes, skip it: no possible pairs
        size_r = block_data.block_sizes[ block_data.inverse_block_indices[r] ]
        if size_r <= 1:
            continue 

        for s in range(r if upper_triangle_only else 0, len(block_data.block_sizes)):
            e = block_data.block_connectivity[r, s]
            #n = block_data.get_possible_pairs(r, s)
            if r == s:
                # If the same block, return the number of pairs within the block
                n = block_data.block_sizes[r] * (block_data.block_sizes[r] - 1) // 2

            # If different blocks, return the product of their sizes
            else:
                n = block_data.block_sizes[r] * block_data.block_sizes[s]


            if e < 0 or n < 0:
                raise ValueError(f"Invalid edge count {e} or possible pairs {n} for block pair ({r}, {s}).")
            if e > n:
                raise ValueError(f"Edge count {e} cannot be greater than possible pairs {n} for block pair ({r}, {s}).")

            ll += _bernoulli_ll_block_pair(e, n) # type: ignore

    return ll

#### LikelihoodCalculator class ######
class LikelihoodCalculator:
    def __init__(self,
                 block_data: BlockData,
                 likelihood_type: LikelihoodType = 'bernoulli',
                 ):
        self.block_data = block_data

        self.likelihood_type: LikelihoodType = 'bernoulli'
        self.nll = self.compute_nll()

    def compute_nll(self) -> float:
        """
        Compute the negative likelihood of the network given the current partition.

        :return: The negative log-likelihood of the SBM.
        """
        if self.likelihood_type.lower() == 'bernoulli':
            return -compute_global_bernoulli_ll_fast(block_data=self.block_data)
        else:
            raise NotImplementedError("Only Bernoulli likelihood is implemented.")
 
    def _compute_delta_nll_from_changes(self,
                                       delta_e: EdgeDelta,
                                       delta_n: CombinationDelta,
    ) ->float:
        """
        efficeintly compute the change in log-likelihood from changes in edge counts and possible pairs.
    
        :param delta_e: Changes in edge counts between blocks.
        :param delta_n: Changes in possible pairs between blocks.
        :param total_edges: Total number of edges in the graph.
        :return: The change in log-likelihood.
        """
        if self.likelihood_type.lower() == 'bernoulli':
            return -compute_delta_ll_from_change_bernoulli(
                delta_e=delta_e,
                delta_n=delta_n,
                block_data=self.block_data
            )
        else:
            raise NotImplementedError("Only Bernoulli likelihood is implemented.")

    def compute_delta_nll(self,
                        delta_e: EdgeDelta,
                        delta_n: CombinationDelta,
        ) -> float:
        """
        Compute the change in log-likelihood for a proposed swap of two nodes.

        :param proposed_moves: A list of tuples (node_i, node_j) representing the nodes to swap.
        :return: The change in log-likelihood.
        """
        return self._compute_delta_nll_from_changes(
            delta_e=delta_e,
            delta_n=delta_n
            )