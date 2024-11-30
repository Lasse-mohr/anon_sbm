from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from sbm.block_data import BlockData
from sbm.movers import NodeMover, BlockMerger, BlockSplitter

class MoveProposer:
    """ 
    Class to propose moves for the MCMC algorithm. By sampling.

    Handles min block size constraints. All functions return None
    if a  and ensures valid moves.
    """
    def __init__(self, block_data: BlockData, rng: np.random.Generator):
        self.block_data = block_data
        self.rng = rng

    def propose_move_node(self, min_block_size: int) -> Optional[Tuple[int, int, int]]:
        """
        Propose moving a node from one block to another.

        :param min_block_size: Minimum allowed size for any block.
        :return: Tuple of (node, source_block, target_block) or None if no valid move.
        """
        # Select a source block larger than min_block_size
        valid_blocks = [block_id for block_id, size in self.block_data.block_sizes.items() if size > min_block_size]
        if len(valid_blocks) == 0:
            return None

        source_block = self.rng.choice(valid_blocks)
        node = self.rng.choice(self.block_data.block_members[source_block])

        # Ensure moving the node will not reduce source_block below min_block_size
        if self.block_data.block_sizes[source_block] - 1 < min_block_size:
            return None

        # Select a target block different from the source block
        target_blocks = [block_id for block_id in self.block_data.block_members
                         if block_id != source_block]
        if len(target_blocks) == 0:
            return None

        target_block = self.rng.choice(target_blocks)

        return node, source_block, target_block

    def propose_split_block(self, min_block_size: int) -> Optional[Tuple[int, List[int], List[int]]]:
        """
        Propose splitting a block into two smaller blocks.

        :param min_block_size: Minimum allowed size for any block.
        :return: Tuple of (block_id, block1, block2) or None if no valid split.
        """
        # Select a block larger than twice the min_block_size
        valid_blocks = [block_id for block_id, size in self.block_data.block_sizes.items() if size > 2 * min_block_size]
        if len(valid_blocks) == 0:
            return None

        block_id = self.rng.choice(valid_blocks)
        block_members = self.block_data.block_members[block_id]

        # Randomly split the block into two
        self.rng.shuffle(block_members)
        split_idx = len(block_members) // 2

        # ensure that sampled blocks are not too small
        new_group_a = block_members[:split_idx]
        new_group_b = block_members[split_idx:]
        if len(new_group_a) < min_block_size or len(new_group_b) < min_block_size:
            return None
        
        return block_id, new_group_a, new_group_b

    def propose_merge_blocks(self) -> Optional[Tuple[int, int]]:
        """
        Propose merging two blocks into a single block.

        :return: Tuple of (block1, block2) or None if no valid merge.
        """
        if len(self.block_data.block_sizes) < 2:
            return None

        block1, block2 = self.rng.choice(
            list(self.block_data.block_sizes.keys()),
            size=2,
            replace=False
        )

        return block1, block2


class MoveExecutor:
    def __init__(self, block_data: BlockData):
        self.node_mover = NodeMover(block_data)
        self.block_merger = BlockMerger(block_data)
        self.block_splitter = BlockSplitter(block_data)
    
    def move_node(self, node: int, source_block: int, target_block: int):
        self.node_mover.move_node(node, source_block, target_block)
    
    def merge_blocks(self, block_a: int, block_b: int):
        self.block_merger.merge_blocks(block_a, block_b)

    def split_block(self, block_to_split: int, nodes_a: List[int], nodes_b: List[int]):
        self.block_splitter.split_block(block_to_split, nodes_a, nodes_b)

