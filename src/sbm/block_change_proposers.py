from typing import List, Optional, Tuple, DefaultDict, Literal
from collections import defaultdict, Counter
import numpy as np
from sbm.block_data import BlockData

### Aliases 
EdgeDelta = DefaultDict[Tuple[int, int], int] # edge-count changes between blocks
CombinationDelta= DefaultDict[Tuple[int, int], int] # changes in possible pairs between blocks
ProposedValidChanges = List[Tuple[int, int]]  # list of proposed node-block pairs

ChangeProposerName = Literal["swap"]
ChangeProposers = Literal["NodeSwapProposer"]


#### Helper functions ######
def _increment_delta_e(count: int, block_i: int, block_j: int,
                        delta_e: EdgeDelta) -> EdgeDelta:
    """
    Increment the edge count delta for a pair of blocks.

    :param count: The change in edge count.
    :param block_i: The first block index.
    :param block_j: The second block index.
    :param delta_e: The current edge count delta.
    :return: Updated edge count delta.
    """
    if block_i < block_j:
        delta_e[(block_i, block_j)] = count
    else:
        delta_e[(block_j, block_i)] = count

    return delta_e


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
                 rng: np.random.Generator=np.random.default_rng(1)):

        self.block_data = block_data

        self.rng = rng
        self.min_block_size = 1
    
    def propose_change(self,
        changes: Optional[ProposedValidChanges] = None,
        )-> Tuple[ProposedValidChanges, EdgeDelta, CombinationDelta]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _compute_delta_edge_counts(self, proposed_changes: ProposedValidChanges) -> EdgeDelta:
        """
        Compute the edge deltas for the proposed change.

        :param change: Proposed change as a list of (node, target_block) tuples.
        :return: EdgeDelta dictionary with changes in edge counts.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
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
                list(self.block_data.block_sizes.keys()),
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
            proposed_changes=proposed_changes
        )

        delta_n: CombinationDelta = defaultdict(int, {key: 0 for key in delta_e.keys()})

        return proposed_changes, delta_e, delta_n
    
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

        delta_e: EdgeDelta = defaultdict(int)
        block_i_idx = self.block_data.block_indices[old_block_i]
        block_j_idx = self.block_data.block_indices[old_block_j]

        # compute the edge counts for the blocks of i and j
        # on block-adjacency idx level
        k_i = self._compute_edge_counts_between_node_and_blocks(i)
        k_j = self._compute_edge_counts_between_node_and_blocks(j)
        affected_blocks = set(k_i.keys()) | set(k_j.keys())

        # add the changes for the neighbor blocks of i and j
        for t in affected_blocks - {block_i_idx, block_j_idx}:
            # Δm_{r t}
            delta_e = _increment_delta_e(
                count = -k_i[t] + k_j[t],
                block_i = old_block_i,
                block_j = t,
                delta_e = delta_e,
            )
            # Δm_{s t}
            delta_e = _increment_delta_e(
                count = -k_j[t] + k_i[t],
                block_i = old_block_j,
                block_j = t,
                delta_e = delta_e,
            )
        
        # Add the changes for the old blocks of i and j
        has_edge_ij = bool(self.block_data.graph_data.adjacency[i, j])

        delta_e = _increment_delta_e(
            count = k_i[old_block_i] - k_i[old_block_j] \
                    + k_j[old_block_j] - k_j[old_block_i] \
                    + 2 * has_edge_ij,
            block_i=old_block_i,
            block_j=old_block_j,
            delta_e=delta_e
        )

        delta_e = _increment_delta_e(
            count = k_j[old_block_i] - k_i[old_block_i] - has_edge_ij,
            block_i=old_block_i,
            block_j=old_block_i,
            delta_e=delta_e
        )
        delta_e = _increment_delta_e(
            count = k_i[old_block_j] - k_j[old_block_j] - has_edge_ij,
            block_i=old_block_j,
            block_j=old_block_j,
            delta_e=delta_e
        )

        return delta_e