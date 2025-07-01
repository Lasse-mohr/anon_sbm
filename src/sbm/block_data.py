from typing import Dict, Set, Optional

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sbm.graph_data import GraphData
from sbm.sampling import sample_sbm_graph

BlockConn = sp.dok_array
BlockMembership = Dict[int, Set[int]]  # Block ID to set of node indices

class _BlockDataUpdater:
    """
    Helper class to update edge counts and possible pairs in the block data.
    This class is used to hide bookkeeping of handling directed vs undirected graphs.

    Parameters
    ----------
    bd : BlockData
    """
    def __init__(self, block_data: "BlockData"):
        self.block_data = block_data # B × B integer matrix
    
    # block memberships
    def _move_node_to_block(self, node: int, block_id: int, update_sizes=True) -> None:
        # update block assignment
        old_block = self.block_data.blocks[node] # type: ignore

        if update_sizes: 
            # update block sizes
            self.block_data.block_sizes[block_id] += 1
            self.block_data.block_sizes[old_block] -= 1

        self.block_data.blocks[node] = block_id # type: ignore

        # update block membersets
        if block_id not in self.block_data.block_members:
            self.block_data.block_members[block_id] = set()

        self.block_data.block_members[block_id].add(node)
        self.block_data.block_members[old_block].remove(node)


    # ----- edge counts --------------------------------------------------
    def _increment_edge_count(self, idx_a: int, idx_b: int, e_delta: int) -> None:
        """ 
        Increment the edge count between two blocks.
        If the graph is undirected, increment both directions.

        e_delta can be negative to decrement the edge count.
        idx_a and idx_b are indices in the block_connectivity matrix.
        """

        self.block_data.block_connectivity[idx_a, idx_b] += e_delta

        if not self.block_data.directed and idx_a != idx_b:
            self.block_data.block_connectivity[idx_b, idx_a] += e_delta


class BlockData:
    """ 
    Class to store block data for the SBM.

    There are two ways to initialize this class:

    Attributes:
        graph_data: The graph data object.
        blocks: A dictionary mapping node indices to block indices.
        block_members: A dictionary mapping block indices to lists of node indices.
        block_sizes: A dictionary mapping block indices to the number of nodes in each block.
        directed: A boolean indicating whether the graph is directed or not.
        block_indices: A dictionary mapping block IDs to indices used in matrices.
        inverse_block_indices: A dictionary mapping indices used in matrices to block IDs.
        block_connectivity: A sparse matrix representing the block connectivity matrix.
    """

    def __init__(self,
                 initial_blocks: Dict[int, int],
                 graph_data: GraphData,
        ):

        self.blocks: Dict[int, int] = initial_blocks # Node to block mapping
        self.block_updater = _BlockDataUpdater(self)

        self.graph_data = graph_data
        self.directed = graph_data.directed

        self.block_members = self._initialize_block_members()
        self.block_sizes = {block: len(nodes) for block, nodes in self.block_members.items()}

        self._update_block_indices()

        self.block_connectivity: BlockConn = self._compute_block_connectivity()

        # Recompute block connectivity based on the new graph data
        self.block_connectivity = self._compute_block_connectivity()
        
    def increment_edge_count(self, block_a: int, block_b: int, e_delta: int) -> None:
        """ 
        Increment the edge count between two blocks.
        If the graph is undirected, increment both directions.

        e_delta can be negative to decrement the edge count.
        """
        idx_a = self.block_indices[block_a]
        idx_b = self.block_indices[block_b]
        self.block_updater._increment_edge_count(idx_a, idx_b, e_delta)
    
    def get_possible_pairs(self, block_a: int, block_b:int ) -> int:
        """ 
        Compute the possible number of edges between two blocks.
        """

        if block_a == block_b:
            # If the same block, return the number of pairs within the block
            return self.block_sizes[block_a] * (self.block_sizes[block_a] - 1) // 2

        # If different blocks, return the product of their sizes
        return self.block_sizes[block_a] * self.block_sizes[block_b]

    def _initialize_block_members(self) -> BlockMembership:
        """
        Initialize block members from the blocks mapping.

        :return: A dictionary mapping block indices to lists of node indices.
        """
        if self.blocks is None:
            raise ValueError("Blocks mapping is not provided to initialize block members.")

        block_members: BlockMembership = {}

        for node, block in self.blocks.items():
            if block not in block_members:
                block_members[block] = set()
            # Add node to the corresponding block
            block_members[block].add(node)

        return block_members

    def _update_block_indices(self):
        """
        Update mappings between block IDs and indices used in matrices.
        """
        # Sort block IDs to ensure consistent ordering
        sorted_block_ids = sorted(self.block_members.keys())
        self.block_indices = {
            block_id: idx for idx, block_id in enumerate(sorted_block_ids)
            }
        self.inverse_block_indices = {
            idx: block_id for block_id, idx in self.block_indices.items()
            }

    def _compute_block_connectivity(self) -> BlockConn:
        """
        Compute the block connectivity matrix.

        This matrix is a sparse matrix where the entry at (i, j) is number of edges 
        between block i and block j. If the graph is undirected, the matrix is symmetric.
        """

        if self.graph_data is None:
            raise ValueError("Graph data is not set. Cannot compute block connectivity.")
        if self.block_members is None:
            raise ValueError("Block members are not initialized. Cannot compute block connectivity.")
        
        num_blocks = len(self.block_members)
        self._update_block_indices()
        block_connectivity_dok = sp.dok_array((num_blocks, num_blocks), dtype=np.int64)

        if self.directed:
            raise ValueError("Block connectivity computation is not implemented for directed graphs.")
        else:
            for i_block_id, nodes_i in self.block_members.items():
                idx_i = self.block_indices[i_block_id]
                nodes_i = list(nodes_i)
                sub_adj_i = self.graph_data.adjacency[nodes_i]  # type: ignore

                for j_block_id, nodes_j in self.block_members.items():
                    idx_j = self.block_indices[j_block_id]
                    nodes_j = list(nodes_j)
                    # Sum of weights between block i and block j
                    weight = sub_adj_i[:, nodes_j].sum() # type: ignore

                    # If the blocks are the same, we only count pairs
                    if i_block_id == j_block_id:
                        weight = weight // 2

                    block_connectivity_dok[idx_i, idx_j] = weight

            return block_connectivity_dok

    def _remove_block_index(self, block_id: int):
        """
        Remove a block from block_indices and inverse_block_indices.

        Do not use directly, call remove_block instead.

        :param block_id: The block ID to remove.
        """
        idx = self.block_indices.pop(block_id)
        self.inverse_block_indices.pop(idx)

        # Adjust indices of remaining blocks
        for b_id, index in self.block_indices.items():
            if index > idx:
                self.block_indices[b_id] -= 1
                self.inverse_block_indices[self.block_indices[b_id]] = b_id

    def _remove_block_from_connectivity(self, block_id: int):
        """
        Remove the block's row and column from the block connectivity matrix.

        Do not use directly, call remove_block instead.

        :param block_id: The block ID to remove.
        """
        idx = self.block_indices[block_id]
        # Remove the row and column corresponding to idx

        slicable_array = self.block_connectivity.tocsr()
        non_slice_idx = np.arange(self.block_connectivity.shape[1]) != idx # type: ignore

        self.block_connectivity = slicable_array[:, non_slice_idx][non_slice_idx, :].todok()

    def remove_block(self, block_id: int):
        """
        Remove a block from the block data.
        Do not use directly, call remove_block instead.

        :param block_id: The block ID to remove.
        """
        del self.block_sizes[block_id]
        del self.block_members[block_id]
        self._remove_block_from_connectivity(block_id)
        self._remove_block_index(block_id)

    def _add_block_index(self, block_id: int):
        """
        Add a new block index for a new block.

        Do not use directly, call add_block instead.

        :param block_id: The block ID to add.
        """
        new_idx = len(self.block_indices)
        self.block_indices[block_id] = new_idx
        self.inverse_block_indices[new_idx] = block_id

    def _add_block_to_connectivity(self):
        """
        Add a new block to the block connectivity matrix.

        Do not use directly, call add_block instead.

        :param block_id: The block ID to add.
        """
        num_blocks = len(self.block_indices)
        connectivity_lil = sp.lil_matrix(self.block_connectivity)
        connectivity_lil.resize((num_blocks, num_blocks))
        self.block_connectivity = connectivity_lil.todok()
    
    def add_block(self, block_id: int, nodes=[]):
        """
        Add a new block to the block data.

        :param block_id: The block ID to add.
        """
        if self.blocks is None:
            raise ValueError("Blocks mapping is not initialized. Cannot add a block.")

        self.block_sizes[block_id] = len(nodes)
        self.block_members[block_id] = nodes
        for node in nodes:
            self.blocks[node] = block_id

        self._add_block_index(block_id)
        self._add_block_to_connectivity()