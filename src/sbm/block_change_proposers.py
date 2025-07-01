from typing import List, Optional, Tuple, DefaultDict, Literal
from collections import defaultdict, Counter

import numpy as np
from sbm.block_data import BlockData
from sbm.edge_delta import EdgeDelta, NumpyEdgeDelta

### Aliases 
CombinationDelta= DefaultDict[Tuple[int, int], int] # changes in possible pairs between blocks
ProposedValidChanges = List[Tuple[int, int]]  # list of proposed node-block pairs

ChangeProposerName = Literal["uniform_swap", "edge_based_swap", "triadic_swap"]
ChangeProposers = Literal["NodeSwapProposer", "EdgeBasedSwapProposer", "TriadicSwapProposer"]



### ChangeProposer classes for proposing block changes in the SBM
# These classes handle the logic of proposing valid changes to the block assignments
# and computing the resulting edge deltas for the block connectivity matrix.
class ChangeProposer:
    """ 
    Class to propose block-assignment changes for the MCMC algorithm.

    Handles min block size constraints. All functions return None
    if a  and ensures valid moves.

    Proposers shoudl always change block-id to block-adjacency idx before
    computing deltas.
    """
    def __init__(self,
                 block_data: BlockData,
                 rng: np.random.Generator=np.random.default_rng(1),
                 use_numpy: bool = False,
                 ):

        self.block_data = block_data
        self.rng = rng
        self.min_block_size = 1
        self.use_numpy = use_numpy

        # Direct CSR pointers for O(1) edge sampling
        self._indptr = self.block_data.graph_data.adjacency.indptr
        self._indices = self.block_data.graph_data.adjacency.indices
    
    def propose_change(self,
        changes: Optional[ProposedValidChanges] = None,
        )-> Tuple[ProposedValidChanges, EdgeDelta, CombinationDelta]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _compute_delta_edge_counts(self, proposed_changes: ProposedValidChanges) -> EdgeDelta:
        """
        Compute the edge deltas for the proposed change.

        :param change: Proposed change as a list of (node, target_block) tuples.
        :return: EdgeDelta containing the changes in edge counts between blocks.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    # -----------------------------------------------------------------------------
    def _compute_edge_counts_between_node_and_blocks(self,
                                               node: int,
                                               ) -> Counter[int]:
        """
        Compute the number of edges between a node and each affected block.
    
        :param node: The node to move.
        :param affected_blocks: The blocks affected by the move.
        :return: A dictionary mapping block IDs to edge counts with the node.

        k_i: Counter[int]: number of edges between node and each affected block.

        returns a Counter object where keys are block IDs and values are edge counts.
        """
        if self.block_data.directed:
            raise NotImplementedError("Directed graphs are not supported yet.")
        else:
            #neighbors = self.block_data.graph_data.adjacency[:, [node]].indices  # type: ignore
            neighbors = self.block_data.graph_data.adjacency[[node], :].indices  # type: ignore
            blocks_among_neighbors = [
                self.block_data.block_indices[
                    self.block_data.blocks[neighbor]
                ] for neighbor in neighbors
            ]
            k_i = Counter(blocks_among_neighbors)

            return k_i

class NodeSwapProposer(ChangeProposer):
    def propose_change(self,
        changes: Optional[ProposedValidChanges] = None,
    ) -> Tuple[ProposedValidChanges, EdgeDelta, CombinationDelta]:
        """
        Propose swapping two nodes between different blocks.

        :return: Tuple of (node1, node2) or None if no valid swap.
        """
        if changes is not None:
            if len(changes) != 2:
                raise ValueError("NodeSwapProposer requires exactly two nodes to swap.")
            proposed_changes = changes
        else:
            # Select two different blocks
            block1, block2 = self.rng.choice(
                self.block_data.block_connectivity.shape[0],
                #list(self.block_data.block_sizes.keys()),
                size=2,
                replace=False
            )

            # Select one node from each block
            # Note: changing to list is inefficient for large blocks.
            # However, having memberships being lists allow for fast 
            # membership updates.
            # Change if large blocks are common.
            node1 = self.rng.choice(
                list(self.block_data.block_members[block1])
            )
            node2 = self.rng.choice(
                list(self.block_data.block_members[block2])
            )

            proposed_changes :ProposedValidChanges = [(node1, block2), (node2, block1)]

        delta_e: EdgeDelta = self._compute_delta_edge_counts(
                proposed_changes=proposed_changes,
                use_numpy=self.use_numpy,
            )

        delta_n: CombinationDelta = defaultdict(int)

        return proposed_changes, delta_e, delta_n
    
    def _compute_delta_edge_counts(self,
            proposed_changes: ProposedValidChanges,
            use_numpy: bool = False,
        )-> EdgeDelta:
        """
        Compute the changes in edge counts between blocks due to swapping
        node i and node j.

        :param i: The index of the first node being swapped.
        :param j: The index of the second node being swapped.
        :param k_i: The edges between the moving node and its neighbor blocks.
        :param source_block: The block from which the node is moved.
        :param target_block: The block to which the node is moved.

        :return: A Counter mapping block pairs to changes in edge counts.
        """
        if self.block_data.directed:
            raise NotImplementedError("Directed graphs are not supported yet.")
        
        (i, old_block_j), (j, old_block_i) = proposed_changes

        if use_numpy:
            delta_e = NumpyEdgeDelta(
                n_blocks=len(self.block_data.block_sizes)
            )
        else:
            delta_e = EdgeDelta(
                n_blocks=len(self.block_data.block_sizes)
            )

        # compute the edge counts for the blocks of i and j
        # on block-adjacency idx level
        k_i = self._compute_edge_counts_between_node_and_blocks(i)
        k_j = self._compute_edge_counts_between_node_and_blocks(j)
        affected_blocks = set(k_i.keys()) | set(k_j.keys())

        # new implementation with combined increment function
        neighbor_blocks = affected_blocks - {old_block_i, old_block_j}
        # build increment lists for neighbor blocks
        counts = [
            -k_i[t] + k_j[t] for t in neighbor_blocks
        ] + [
            -k_j[t] + k_i[t] for t in neighbor_blocks
        ]
        blocks_i = [old_block_i] * len(neighbor_blocks) + [old_block_j] * len(neighbor_blocks)
        block_j = list(neighbor_blocks) + list(neighbor_blocks)

        delta_e.increment(
            counts = counts,
            blocks_i = blocks_i,
            blocks_j = block_j,
        ) 
        
        # Add the changes for the old blocks of i and j
        has_edge_ij = bool(self.block_data.graph_data.adjacency[i, j])
        delta_e.increment(
            counts=[
                k_i[old_block_i] - k_i[old_block_j] + k_j[old_block_j] - k_j[old_block_i] + 2 * has_edge_ij,
                k_j[old_block_i] - k_i[old_block_i] - has_edge_ij,
                k_i[old_block_j] - k_j[old_block_j] - has_edge_ij
            ],
            blocks_i=[old_block_i, old_block_i, old_block_j],
            blocks_j=[old_block_j, old_block_i, old_block_j]
        )

        return delta_e


# -----------------------------------------------------------------------------
#  Edge‑based swap proposer
# -----------------------------------------------------------------------------
class EdgeBasedSwapProposer(NodeSwapProposer):
    """A Peixoto‑style *edge‑conditioned* two‑vertex swap.

    1. Pick a **cross‑block edge** ``(i,j)`` uniformly at random.
    2. Swap the block labels of its end‑points.

    The proposal is *symmetric* (uniform over edges), so the Metropolis–
    Hastings acceptance probability is simply ``min(1, exp(Δℓ/T))``.
    """

    def __init__(
        self,
        block_data,
        rng: np.random.Generator = np.random.default_rng(1),
        use_numpy: bool = True,
        max_trials: int = 128,
    ) -> None:
        super().__init__(block_data=block_data, rng=rng, use_numpy=use_numpy)
        self.max_trials = max_trials

        # Direct CSR pointers for O(1) edge sampling
        self._indptr = self.block_data.graph_data.adjacency.indptr
        self._indices = self.block_data.graph_data.adjacency.indices

    # ------------------------------------------------------------------
    def propose_change(
        self,
        changes: Optional[ProposedValidChanges] = None,
    ) -> Tuple[ProposedValidChanges, EdgeDelta, CombinationDelta]:
        if changes is not None:
            return super().propose_change(changes=changes)

        n = self.block_data.graph_data.num_nodes  # type: ignore[attr-defined]
        blocks = self.block_data.blocks

        for _ in range(self.max_trials):
            i = int(self.rng.integers(n))

            # get i's neighbor index-range (adj is csr format)
            istart, iend = self._indptr[i], self._indptr[i + 1]
            if iend == istart:
                continue  # isolated vertex
            
            # pick a random neighbor j
            j = int(self.rng.choice(self._indices[istart:iend]))

            bi, bj = blocks[i], blocks[j]
            if bi == bj:
                continue  # need a cross‑block edge

            proposed_changes: ProposedValidChanges = [(i, bj), (j, bi)]
            break
        else:  # all trials failed – fall back to uniform swap
            return super().propose_change(changes=None)

        delta_e = self._compute_delta_edge_counts(
            proposed_changes=proposed_changes,
            use_numpy=self.use_numpy,
        )
        delta_n: CombinationDelta = defaultdict(int)  # block sizes unchanged
        return proposed_changes, delta_e, delta_n


# -----------------------------------------------------------------------------
#  Triadic informed swap   (new implementation)
# -----------------------------------------------------------------------------
class TriadicSwapProposer(NodeSwapProposer):
    """A *three‑vertex* informed swap.

    Strategy
    --------
    1. Pick a random vertex ``i`` (block *A*).
    2. Choose a random neighbour ``j`` with ``block(j) = B \neq A``.
    3. Search in block *B* for a vertex ``l \ne j`` that has **at least one**
       neighbour in block *A*.
    4. Swap the block labels of ``i`` and ``l``.

    Swapping these two vertices reduces the expected number of *cross* edges by
    converting:
    * all edges from ``i`` into *B* to *internal*, and
    * all edges from `l``` into *A* to *internal*,
    while typically adding fewer new cross edges because ``i`` and ``j'`` were
    originally “boundary” vertices.

    The proposal distribution is still *symmetric* because every triad is
    selected with the same probability in either direction, so the usual MH
    acceptance rule applies.
    """

    def __init__(
        self,
        block_data,
        rng: np.random.Generator = np.random.default_rng(1),
        use_numpy: bool = False,
        max_trials: int = 128,
        candidate_trials: int = 64,
    ) -> None:
        super().__init__(block_data=block_data, rng=rng, use_numpy=use_numpy)
        self.max_trials = max_trials            # attempts to find (i,j)
        self.candidate_trials = candidate_trials  # attempts to find j′ per (i,j)
        self._indptr = self.block_data.graph_data.adjacency.indptr
        self._indices = self.block_data.graph_data.adjacency.indices

    # ------------------------------------------------------------------
    def propose_change(
        self,
        changes: Optional[ProposedValidChanges] = None,
    ) -> Tuple[ProposedValidChanges, EdgeDelta, CombinationDelta]:
        # Explicit‑changes path used in unit tests
        if changes is not None:
            return super().propose_change(changes=changes)

        n = self.block_data.graph_data.num_nodes  # type: ignore[attr-defined]
        blocks = self.block_data.blocks

        for _ in range(self.max_trials):
            # ---- step 1: pick i ------------------------------------------------
            i = int(self.rng.integers(n))

            # find i's neighbour index-range (adj is csr format)
            istart, iend = self._indptr[i], self._indptr[i + 1]
            if iend == istart:
                continue  # isolated – try another

            # ---- step 2: pick neighbour j in a *different* block --------------
            neighs_i = self._indices[istart:iend]
            j = int(self.rng.choice(neighs_i))
            a, b = blocks[i], blocks[j]
            if a == b:
                continue  # need a cross edge i‑j

            # ---- step 3: find j′ in block b that touches block a --------------
            block_b_members = self.block_data.block_members[b]

            # change to list and randomlize order
            block_b_members = list(block_b_members)
            self.rng.shuffle(block_b_members)
            
            for l in block_b_members[:self.candidate_trials]:
                if l in (i, j):
                    continue
                
                # find neighbors of l in block a
                lstart, lend = self._indptr[l], self._indptr[l + 1]
                l_neighbors = self._indices[lstart:lend]

                l_neighbor_in_block_a = any(
                    blocks[neighbor] == a for neighbor in l_neighbors
                )
                if not l_neighbor_in_block_a:
                    continue  # l must touch block a

                proposed_changes: ProposedValidChanges = [(i, b), (l, a)]

                delta_e = self._compute_delta_edge_counts(
                    proposed_changes=proposed_changes,
                    use_numpy=self.use_numpy,
                )
                delta_n: CombinationDelta = defaultdict(int)
                return proposed_changes, delta_e, delta_n
            # could not find j′ – back to outer loop
            continue

        # ---- fallback --------------------------------------------------------
        # If every attempt failed (e.g. almost perfect partition), fall back to
        # a plain uniform swap to keep the chain ergodic.
        return super().propose_change(changes=None)
