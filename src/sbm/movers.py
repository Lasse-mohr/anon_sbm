from typing import List
import scipy.sparse as sp
from sbm.block_data import BlockData

class NodeMover:
    """ 
    Class to move nodes between blocks in the SBM and update block data accordingly.
    """
    def __init__(self, block_data: BlockData):
        self.block_data = block_data

    def move_node(self, node: int, source_block: int, target_block: int):
        """
        Move a node from source_block to target_block.

        :param node: The node to move.
        :param source_block: The block from which the node is moved.
        :param target_block: The block to which the node is moved.
        """
        # Remove node from source block
        self.block_data.block_members[source_block].remove(node)
        # Add node to target block
        self.block_data.block_members[target_block].append(node)
        # Update blocks mapping
        self.block_data.blocks[node] = target_block

        # Update block sizes and connectivity
        self.update_blocks_after_move(node, source_block, target_block)

    def _update_block_connectivity_move_node(self, node: int, source_block: int, target_block: int):
        """
        Update the block connectivity matrix after moving a node.

        :param node: The node that was moved.
        :param source_block: The block from which the node was moved.
        :param target_block: The block to which the node was moved.
        """
        # Get indices of blocks
        idx_s = self.block_data.block_indices[source_block]
        idx_t = self.block_data.block_indices[target_block]

        # Edges connected to the moving node
        neighbors = self.graph_data.adjacency[node].indices # type: ignore
        for neighbor in neighbors:
            neighbor_block = self.block_data.blocks[neighbor]
            idx_k = self.block_data.block_indices[neighbor_block]

            if neighbor_block == source_block:
                # Update internal edge count of source_block
                # Edge is counted twice in undirected graphs
                self.block_data.block_connectivity[idx_s, idx_s] -= 2 # type: ignore
            elif neighbor_block == target_block:
                # Update internal edge count of target_block
                self.block_data.block_connectivity[idx_t, idx_t] += 2 # type: ignore
            else:
                # Update edge counts between blocks
                idx_min_s = min(idx_s, idx_k)
                idx_max_s = max(idx_s, idx_k)
                self.block_data.block_connectivity[idx_min_s, idx_max_s] -= 1 # type: ignore

                idx_min_t = min(idx_t, idx_k)
                idx_max_t = max(idx_t, idx_k)
                self.block_data.block_connectivity[idx_min_t, idx_max_t] += 1 # type: ignore

        # Update edge counts between source_block and target_block
        idx_min_st = min(idx_s, idx_t)
        idx_max_st = max(idx_s, idx_t)
        # Adjust for edges between source and target blocks
        # Since node is moved from source to target, any edges between node and nodes in target_block
        # become internal edges in target_block
        num_edges_node_target_block = len(
            set(neighbors).intersection(
                set(self.block_data.block_members[target_block])
                )
            )
        self.block_data.block_connectivity[idx_min_st, idx_max_st] -= num_edges_node_target_block # type: ignore
        self.block_data.block_connectivity[idx_t, idx_t] += num_edges_node_target_block * 2 # type: ignore

    def update_blocks_after_move(self, node: int, source_block: int, target_block: int):
        """
        Update block sizes and connectivity after moving a node.

        :param node: The node that was moved.
        :param source_block: The block from which the node was moved.
        :param target_block: The block to which the node was moved.
        """
        # Update block sizes
        self.block_data.block_sizes[source_block] -= 1
        self.block_data.block_sizes[target_block] += 1

        # If source block is empty, remove it
        if self.block_data.block_sizes[source_block] == 0:
            # Remove empty block
            del self.block_data.block_sizes[source_block]
            del self.block_data.block_members[source_block]
            # Remove block indices
            self.block_data._remove_block_index(source_block)
            # Update block connectivity matrix
            self.block_data.remove_block_from_connectivity(source_block)
        else:
            # Update block connectivity matrix incrementally
            self._update_block_connectivity_move_node(node, source_block, target_block)


class BlockMerger:
    """
    Class to merge blocks in the SBM and update block data accordingly.
    """
    def __init__(self, block_data: BlockData):
        self.block_data = block_data

    def merge_blocks(self, block_a: int, block_b: int):
        """
        Merge two blocks into one.

        :param block_a: The first block to merge.
        :param block_b: The second block to merge.
        """
        if block_a == block_b:
            return  # Cannot merge the same block

        # Merge nodes from both blocks
        nodes_a = self.block_data.block_members.get(block_a, [])
        nodes_b = self.block_data.block_members.get(block_b, [])
        merged_nodes = nodes_a + nodes_b

        # Update block members
        new_block_idx = min(block_a, block_b)
        del self.block_data.block_members[block_a]
        del self.block_data.block_members[block_b]
        self.block_data.block_members[new_block_idx] = merged_nodes

        # Update block sizes
        self.block_data.block_sizes[new_block_idx] = len(merged_nodes)
        del self.block_data.block_sizes[block_a]
        del self.block_data.block_sizes[block_b]

        # Update blocks mapping
        for node in merged_nodes:
            self.block_data.blocks[node] = new_block_idx

        # Update block indices and connectivity
        self.block_data._update_block_indices()
        #self.block_data._compute_block_connectivity()
        self._update_blocks_after_merge(block_a, block_b)

    def _update_blocks_after_merge(self, block_a: int, block_b: int):
        """
        Update block sizes and connectivity after merging two blocks.

        :param block_a: The first block that was merged.
        :param block_b: The second block that was merged.
        """
        # New block_c is the merged block
        block_c = min(block_a, block_b)

        # Update block sizes
        size_a = self.block_data.block_sizes.pop(block_a)
        size_b = self.block_data.block_sizes.pop(block_b)
        self.block_data.block_sizes[block_c] = size_a + size_b

        # Update block members
        nodes_a = self.block_data.block_members.pop(block_a)
        nodes_b = self.block_data.block_members.pop(block_b)
        self.block_data.block_members[block_c] = nodes_a + nodes_b

        # Update blocks mapping
        for node in nodes_a + nodes_b:
            self.block_data.blocks[node] = block_c

        # Update block indices
        self.block_data._remove_block_index(block_a)
        self.block_data._remove_block_index(block_b)
        self.block_data._add_block_index(block_c)

        # Update block connectivity matrix
        self._update_block_connectivity_merge_blocks(block_a, block_b)

    def _update_block_connectivity_merge_blocks(self, block_a: int, block_b: int):
        """
        Update the block connectivity matrix after merging two blocks.

        :param block_a: The first block that was merged.
        :param block_b: The second block that was merged.
        """
        block_c = min(block_a, block_b)
        idx_c = self.block_data.block_indices[block_c]

        # Initialize new row and column for block_c
        num_blocks = len(self.block_data.block_indices)
        new_connectivity = sp.lil_matrix((num_blocks, num_blocks))

        # Copy existing connectivity, excluding block_a and block_b
        idx_a = self.block_data.block_indices.get(block_a)
        idx_b = self.block_data.block_indices.get(block_b)

        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(self.block_data.block_indices.values()))}

        for i_old in range(self.block_data.block_connectivity.shape[0]): # type: ignore
            if i_old in {idx_a, idx_b}:
                continue
            i_new = idx_map[i_old]
            for j_old in self.block_data.block_connectivity[i_old].indices: # type: ignore
                if j_old in {idx_a, idx_b}:
                    continue
                j_new = idx_map[j_old]
                new_connectivity[i_new, j_new] = self.block_data.block_connectivity[i_old, j_old]

        # Calculate new connectivity for block_c
        # Sum the connections from block_a and block_b to other blocks
        for block_k in self.block_data.block_indices:
            if block_k in {block_a, block_b, block_c}:
                continue
            idx_k = self.block_data.block_indices[block_k]
            idx_k_new = idx_map[idx_k]

            e_ak = self.block_data.block_connectivity[idx_a, idx_k] if idx_a is not None else 0
            e_bk = self.block_data.block_connectivity[idx_b, idx_k] if idx_b is not None else 0

            e_ck = e_ak + e_bk # type: ignore
            idx_c_new = idx_map[idx_c]

            new_connectivity[idx_c_new, idx_k_new] = e_ck
            new_connectivity[idx_k_new, idx_c_new] = e_ck  # Symmetric

        # Update internal edges for block_c
        e_aa = self.block_data.block_connectivity[idx_a, idx_a] if idx_a is not None else 0
        e_bb = self.block_data.block_connectivity[idx_b, idx_b] if idx_b is not None else 0
        e_ab = self.block_data.block_connectivity[idx_a, idx_b] if idx_a is not None and idx_b is not None else 0

        e_cc = e_aa + e_bb + e_ab # type: ignore
        idx_c_new = idx_map[idx_c]
        new_connectivity[idx_c_new, idx_c_new] = e_cc

        # Assign the new connectivity matrix
        self.block_data.block_connectivity = new_connectivity.tocsr()


class BlockSplitter:
    """ 
    Class to split blocks in the SBM and update block data accordingly. 
    """
    def __init__(self, block_data: BlockData):
        self.block_data = block_data

    def split_block(self, block_to_split: int, nodes_a: List[int], nodes_b: List[int]):
        """
        Split a block into two new blocks.

        :param block_to_split: The index of the block to split.
        :param nodes_a: The nodes in the first new block.
        :param nodes_b: The nodes in the second new block.
        """
        # Remove the old block
        del self.block_data.block_members[block_to_split]
        del self.block_data.block_sizes[block_to_split]

        # Create two new blocks
        new_block_idx = max(self.block_data.block_members.keys(), default=-1) + 1
        self.block_data.block_members[block_to_split] = nodes_a
        self.block_data.block_members[new_block_idx] = nodes_b

        # Update block sizes
        self.block_data.block_sizes[block_to_split] = len(nodes_a)
        self.block_data.block_sizes[new_block_idx] = len(nodes_b)

        # Update blocks mapping
        for node in nodes_a:
            self.block_data.blocks[node] = block_to_split
        for node in nodes_b:
            self.block_data.blocks[node] = new_block_idx

        # Update block indices and connectivity
        self.block_data._update_block_indices()
        self.block_data._compute_block_connectivity()


    def _update_blocks_after_split(self, block_to_split: int, nodes_a: List[int], nodes_b: List[int]):
        """
        Update block sizes and connectivity after splitting a block.

        :param block_to_split: The index of the block that was split.
        :param nodes_a: The nodes in the first new block.
        :param nodes_b: The nodes in the second new block.
        """
        # Remove old block
        del self.block_data.block_sizes[block_to_split]
        del self.block_data.block_members[block_to_split]

        # Remove old block index
        self.block_data._remove_block_index(block_to_split)

        # Create two new blocks
        block_a = block_to_split  # Reuse the old block ID for block_a
        block_b = max(self.block_data.block_indices.keys(), default=-1) + 1  # New block ID

        # Update block sizes
        self.block_data.block_sizes[block_a] = len(nodes_a)
        self.block_data.block_sizes[block_b] = len(nodes_b)

        # Update block members
        self.block_data.block_members[block_a] = nodes_a
        self.block_data.block_members[block_b] = nodes_b

        # Update blocks mapping
        for node in nodes_a:
            self.block_data.blocks[node] = block_a
        for node in nodes_b:
            self.block_data.blocks[node] = block_b

        # Add new block indices
        self.block_data._add_block_index(block_a)
        self.block_data._add_block_index(block_b)

        # Update block connectivity matrix
        self._update_block_connectivity_split_block(block_to_split, nodes_a, nodes_b)

    def _update_block_connectivity_split_block(self, block_to_split: int, nodes_a: List[int], nodes_b: List[int]):
        """
        Update the block connectivity matrix after splitting a block.

        :param block_to_split: The index of the block that was split.
        :param nodes_a: The nodes in the first new block.
        :param nodes_b: The nodes in the second new block.
        """
        block_a = block_to_split
        block_b = max(self.block_data.block_indices.keys(), default=-1)  # New block ID
        idx_a = self.block_data.block_indices[block_a]
        idx_b = self.block_data.block_indices[block_b]

        # Initialize new connectivity matrix
        num_blocks = len(self.block_data.block_indices)
        new_connectivity = sp.lil_matrix((num_blocks, num_blocks))

        # Map old indices to new indices
        idx_s = self.block_data.block_indices.get(block_to_split)
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(self.block_data.block_indices.values()))}

        # Copy existing connectivity, excluding block_s
        for i_old in range(self.block_data.block_connectivity.shape[0]): # type: ignore
            if i_old == idx_s:
                continue
            i_new = idx_map[i_old]
            for j_old in self.block_data.block_connectivity[i_old].indices: # type: ignore
                if j_old == idx_s:
                    continue
                j_new = idx_map[j_old]
                new_connectivity[i_new, j_new] = self.block_data.block_connectivity[i_old, j_old]

        # Compute internal edges for block_a and block_b
        sub_adj_s = self.graph_data.adjacency[nodes_a + nodes_b][:, nodes_a + nodes_b] # type: ignore

        idx_nodes_a = list(range(len(nodes_a)))
        idx_nodes_b = list(range(len(nodes_a), len(nodes_a) + len(nodes_b)))

        e_aa = sub_adj_s[idx_nodes_a][:, idx_nodes_a].sum() / 2 # type: ignore
        e_bb = sub_adj_s[idx_nodes_b][:, idx_nodes_b].sum() / 2 # type: ignore
        e_ab = sub_adj_s[idx_nodes_a][:, idx_nodes_b].sum() # type: ignore

        idx_a_new = idx_map[idx_a]
        idx_b_new = idx_map[idx_b]

        new_connectivity[idx_a_new, idx_a_new] = e_aa
        new_connectivity[idx_b_new, idx_b_new] = e_bb
        new_connectivity[idx_a_new, idx_b_new] = e_ab
        new_connectivity[idx_b_new, idx_a_new] = e_ab  # Symmetric

        # Update connectivity between new blocks and other blocks
        for block_k in self.block_data.block_indices:
            if block_k in {block_a, block_b}:
                continue
            idx_k = self.block_data.block_indices[block_k]
            idx_k_new = idx_map[idx_k]

            # Edges between block_a and block_k
            nodes_k = self.block_data.block_members[block_k]
            e_ak = self.graph_data.adjacency[nodes_a][:, nodes_k].sum() # type: ignore
            new_connectivity[idx_a_new, idx_k_new] = e_ak
            new_connectivity[idx_k_new, idx_a_new] = e_ak  # Symmetric

            # Edges between block_b and block_k
            e_bk = self.graph_data.adjacency[nodes_b][:, nodes_k].sum() # type: ignore
            new_connectivity[idx_b_new, idx_k_new] = e_bk
            new_connectivity[idx_k_new, idx_b_new] = e_bk  # Symmetric

        # Assign the new connectivity matrix
        self.block_data.block_connectivity = new_connectivity.tocsr()