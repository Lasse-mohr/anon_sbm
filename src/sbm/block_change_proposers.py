from typing import List, Optional, Tuple, DefaultDict, Literal
from collections import defaultdict, Counter

import line_profiler
import numpy as np
from sbm.block_data import BlockData
from sbm.edge_delta import EdgeDelta, NumpyEdgeDelta

### Aliases 
CombinationDelta= DefaultDict[Tuple[int, int], int] # changes in possible pairs between blocks
ProposedValidChanges = List[Tuple[int, int]]  # list of proposed node-block pairs

ChangeProposerName = Literal["swap"]
ChangeProposers = Literal["NodeSwapProposer"]

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
    
    @line_profiler.profile
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
    @line_profiler.profile
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

        if self.use_numpy:
            delta_e: EdgeDelta = NumpyEdgeDelta(
                n_blocks=len(self.block_data.block_sizes)
            )
        else:
            delta_e: EdgeDelta = self._compute_delta_edge_counts(
                proposed_changes=proposed_changes
            )

        delta_n: CombinationDelta = defaultdict(int)

        return proposed_changes, delta_e, delta_n
    
    @line_profiler.profile
    def _compute_delta_edge_counts(self,
            proposed_changes: ProposedValidChanges
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

        delta_e = EdgeDelta(
            n_blocks=len(self.block_data.block_sizes)
        )

        #block_i_idx = self.block_data.block_indices[old_block_i]
        #block_j_idx = self.block_data.block_indices[old_block_j]

        # compute the edge counts for the blocks of i and j
        # on block-adjacency idx level
        k_i = self._compute_edge_counts_between_node_and_blocks(i)
        k_j = self._compute_edge_counts_between_node_and_blocks(j)
        affected_blocks = set(k_i.keys()) | set(k_j.keys())

        # The following code is commented out because it is not used in the current implementation.
        # (Old implementation)
        #for i in affected_blocks - {old_block_i, old_block_j}:
            ## Δm_{r t}
            #delta_e = _increment_delta_e(
            #    count = -k_i[t] + k_j[t],
            #    block_i = old_block_i,
            #    block_j = t,
            #    delta_e = delta_e,
            #)
            ## Δm_{s t}
            #delta_e = _increment_delta_e(
            #    count = -k_j[t] + k_i[t],
            #    block_i = old_block_j,
            #    block_j = t,
            #    delta_e = delta_e,
            #)
        ## Add the changes for the old blocks of i and j
        #has_edge_ij = bool(self.block_data.graph_data.adjacency[i, j])

        #delta_e = _increment_delta_e(
        #    count = k_i[old_block_i] - k_i[old_block_j] \
        #            + k_j[old_block_j] - k_j[old_block_i] \
        #            + 2 * has_edge_ij,
        #    block_i=old_block_i,
        #    block_j=old_block_j,
        #    delta_e=delta_e
        #)

        #delta_e = _increment_delta_e(
        #    count = k_j[old_block_i] - k_i[old_block_i] - has_edge_ij,
        #    block_i=old_block_i,
        #    block_j=old_block_i,
        #    delta_e=delta_e
        #)
        #delta_e = _increment_delta_e(
        #    count = k_i[old_block_j] - k_j[old_block_j] - has_edge_ij,
        #    block_i=old_block_j,
        #    block_j=old_block_j,
        #    delta_e=delta_e
        #)

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