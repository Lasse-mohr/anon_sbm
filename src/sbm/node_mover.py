from typing import List, Tuple
import scipy.sparse as sp
import numpy as np
from sbm.block_data import BlockData, _BlockDataUpdater
from line_profiler import profile

from sbm.block_change_proposers import (
    ProposedValidChanges,
    EdgeDelta,
)

class NodeMover:
    """
    Class to move nodes between block assignment in the Stochastic Block Model (SBM).
    When performing a change, it updates the block sizes, connectivity matrix,
    block indices, and inverse block indices accordingly.
    All changes are performed in-place on the BlockData object by the _BlockDataUpdater.
    """
    def __init__(self, block_data: BlockData):
        self.block_data_updater = _BlockDataUpdater(block_data)

    def perform_change(self,
            proposed_changes: ProposedValidChanges,
            delta_e: EdgeDelta,
            ):
        """ 
        Change the block assignments of nodes according to the proposed change.
        Update:
        - block sizes
        - block connectivity matrix (edge counts between blocks)
        - block indices (node to block assignment)
        - inverse block indices (nodes in each block)

        Rely on increment_edge_count from BlockData to update edge counts.

        :param change: A list of tuples where each tuple contains a node and their new block.
        """

        (node_i, new_block_i), (node_j, new_block_j) = proposed_changes
        # update the block assignments, sizes, and memberships
        self.block_data_updater._move_node_to_block(node_i, new_block_i)
        self.block_data_updater._move_node_to_block(node_j, new_block_j)

        # update the edge counts between the blocks
        for (r, s), e_delta in delta_e.items():
            self.block_data_updater._increment_edge_count(r, s, e_delta)