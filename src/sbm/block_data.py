import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Union
from sbm.sbm import GraphData

BlockConn = Union[sp.csr_array, sp.lil_array]

class BlockData:
    """ 
    Class to store block data for the SBM.
    
    Attributes:
        graph_data: The graph data object.
        blocks: A dictionary mapping node indices to block indices.
        block_members: A dictionary mapping block indices to lists of node indices.
        block_sizes: A dictionary mapping block indices to the number of nodes in each block.
        block_indices: A dictionary mapping block IDs to indices used in matrices.
        inverse_block_indices: A dictionary mapping indices used in matrices to block IDs.
        block_connectivity: A sparse matrix representing the block connectivity matrix.
    """
    def __init__(self, initial_blocks: Dict[int, int], graph_data: GraphData):
        self.graph_data = graph_data
        self.blocks = initial_blocks  # Node to block mapping
        self.block_members = self._initialize_block_members()
        self.block_sizes = {block: len(nodes) for block, nodes in self.block_members.items()}
        self.block_indices = {}
        self.inverse_block_indices = {}
        self._update_block_indices()
        block_connectivity = self._compute_block_connectivity()
        if block_connectivity is not None:
            self.block_connectivity: BlockConn = block_connectivity
        else:
            raise ValueError("Block connectivity matrix is empty")

    def _initialize_block_members(self) -> Dict[int, List[int]]:
        """
        Initialize block members from the blocks mapping.

        :return: A dictionary mapping block indices to lists of node indices.
        """
        block_members: Dict[int, List[int]] = {}

        for node, block in self.blocks.items():
            block_members.setdefault(block, []).append(node)

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
        """
        num_blocks = len(self.block_members)
        self._update_block_indices()
        block_connectivity_dok = sp.dok_matrix((num_blocks, num_blocks), dtype=float)

        for i_block_id, nodes_i in self.block_members.items():
            idx_i = self.block_indices[i_block_id]
            nodes_i = np.array(nodes_i)
            sub_adj_i = self.adjacency[nodes_i]  # type: ignore

            for j_block_id, nodes_j in self.block_members.items():
                idx_j = self.block_indices[j_block_id]
                nodes_j = np.array(nodes_j)
                # Sum of weights between block i and block j
                weight = sub_adj_i[:, nodes_j].sum()
                block_connectivity_dok[idx_i, idx_j] = weight

        return block_connectivity_dok.tocsr()

    def remove_block_index(self, block_id: int):
        """
        Remove a block from block_indices and inverse_block_indices.

        :param block_id: The block ID to remove.
        """
        idx = self.block_indices.pop(block_id)
        self.inverse_block_indices.pop(idx)

        # Adjust indices of remaining blocks
        for b_id, index in self.block_indices.items():
            if index > idx:
                self.block_indices[b_id] -= 1
                self.inverse_block_indices[self.block_indices[b_id]] = b_id

    def remove_block_from_connectivity(self, block_id: int):
        """
        Remove the block's row and column from the block connectivity matrix.

        :param block_id: The block ID to remove.
        """
        idx = self.block_indices[block_id]
        # Remove the row and column corresponding to idx
        self.block_connectivity = sp.lil_array(self.block_connectivity)

        non_slice_idx = np.arange(self.block_connectivity.shape[1]) != idx # type: ignore
        self.block_connectivity = sp.csr_array(
            self.block_connectivity[:, non_slice_idx][non_slice_idx, :]
            )

        self.block_connectivity = sp.csr_array(self.block_connectivity)

    def _add_block_index(self, block_id: int):
        """
        Add a new block index for a new block.

        :param block_id: The block ID to add.
        """
        new_idx = len(self.block_indices)
        self.block_indices[block_id] = new_idx
        self.inverse_block_indices[new_idx] = block_id

    def _remove_block_index(self, block_id: int):
        """
        Remove a block from block_indices and inverse_block_indices.

        :param block_id: The block ID to remove.
        """
        idx = self.block_indices.pop(block_id)
        self.inverse_block_indices.pop(idx)

        # Adjust indices of remaining blocks
        for b_id, index in self.block_indices.items():
            if index > idx:
                self.block_indices[b_id] -= 1
                self.inverse_block_indices[self.block_indices[b_id]] = b_id

    def _add_block_to_connectivity(self, block_id: int):
        """
        Add a new block to the block connectivity matrix.

        :param block_id: The block ID to add.
        """
        num_blocks = len(self.block_indices)
        connectivity_lil = sp.lil_matrix(self.block_connectivity)
        connectivity_lil.resize((num_blocks, num_blocks))
        self.block_connectivity = connectivity_lil.tocsr()
        self.block_connectivity.resize((num_blocks, num_blocks))