from typing import (
    Dict,
    Tuple,
    Literal,
)
import numpy as np
from sbm.block_data import BlockData
from sbm.block_change_proposers import (
    EdgeDelta,
    CombinationDelta,
)

#### aliases ######
LikelihoodType = Literal['bernoulli']

# Bernoulli functions
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
        delta_e: Dict[Tuple[int, int], int],
        delta_n: Dict[Tuple[int, int], int],
        block_data: BlockData) -> float:
    """
    Incremental change in Bernoulli log-likelihood after a node-swap or move.
    Only the pairs present in `delta_e` or `delta_n` need to be visited.
    delta_e: changes in edge counts between affected blocks.
    delta_n: changes in possible pairs between affected blocks.
    block_data: BlockData object containing edge counts and possible pairs.

    :return: Tuple of (change in log-likelihood, edge counts changes of move delta).
    """
    assert set(delta_e.keys()) == set(delta_n.keys()), \
        "Changes in edge counts and possible edge counts should be passed between identical block set."
    upper_triangle_only = not block_data.directed

    delta_ll = 0.0
    for (r, s) in delta_e.keys() | delta_n.keys():
        if upper_triangle_only and s < r:
            continue
        e_old = int(block_data.block_connectivity[r, s]) # type: ignore
        n_old = block_data.get_possible_pairs(r, s)

        e_new = e_old + delta_e[r, s]
        n_new = n_old + delta_n[r, s]

        delta_ll += _delta_ll_bernoulli_block_pair(
            e_old=e_old,
            e_new=e_new,
            n_old=n_old,
            n_new=n_new
        )

    return delta_ll

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
            n = block_data.get_possible_pairs(r, s)

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
        self.ll = self.compute_likelihood()

    def compute_likelihood(self) -> float:
        """
        Compute the likelihood of the network given the current partition.

        :return: The log-likelihood of the SBM.
        """
        if self.likelihood_type.lower() == 'bernoulli':
            return compute_global_bernoulli_ll(block_data=self.block_data)
        else:
            raise NotImplementedError("Only Bernoulli likelihood is implemented.")
 
    def _compute_delta_ll_from_changes(self,
                                       delta_e: Dict[Tuple[int, int], int],
                                       delta_n: Dict[Tuple[int, int], int],
    ) ->float:
        """
        efficeintly compute the change in log-likelihood from changes in edge counts and possible pairs.
    
        :param delta_e: Changes in edge counts between blocks.
        :param delta_n: Changes in possible pairs between blocks.
        :param total_edges: Total number of edges in the graph.
        :return: The change in log-likelihood.
        """
        if self.likelihood_type.lower() == 'bernoulli':
            return compute_delta_ll_from_change_bernoulli(
                delta_e=delta_e,
                delta_n=delta_n,
                block_data=self.block_data
            )
        else:
            raise NotImplementedError("Only Bernoulli likelihood is implemented.")

    def compute_delta_ll(self,
                        delta_e: EdgeDelta,
                        delta_n: CombinationDelta,
        ) -> float:
        """
        Compute the change in log-likelihood for a proposed swap of two nodes.

        :param proposed_moves: A list of tuples (node_i, node_j) representing the nodes to swap.
        :return: The change in log-likelihood.
        """
        return self._compute_delta_ll_from_changes(
            delta_e=delta_e,
            delta_n=delta_n
            )