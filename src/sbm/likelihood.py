from typing import Dict, Tuple
from typing import List
import numpy as np
from src.sbm.block_data import BlockData

class LikelihoodCalculator:
    def __init__(self, block_data: BlockData):
        self.block_data = block_data
        self.ll = self.compute_likelihood()

    def compute_likelihood(self) -> float:
        """
        Compute the likelihood of the network given the current partition.

        :return: The log-likelihood of the SBM.
        """
        if self.block_data._compute_block_connectivity() is None:
            raise ValueError("""
                             Block connectivity matrix is not
                             computed. Run `_compute_block_connectivity` first.
                             """)

        ll = 0.0
        num_blocks = len(self.block_data.block_members)

        for idx_i in range(num_blocks):
            i_block_id = self.block_data.inverse_block_indices[idx_i]
            n_i = self.block_data.block_sizes.get(i_block_id, 0)
            if n_i == 0:
                continue

            for idx_j in range(num_blocks):
                j_block_id = self.block_data.inverse_block_indices[idx_j]
                n_j = self.block_data.block_sizes.get(j_block_id, 0)
                if n_j == 0:
                    continue

                e_ij = self.block_data.block_connectivity[idx_i, idx_j]

                if idx_i == idx_j:
                    n_pairs = n_i * (n_i - 1) / 2
                else:
                    n_pairs = n_i * n_j

                if self.block_data.graph_data.total_edges > 0:
                    p_ij = e_ij / self.block_data.graph_data.total_edges # type: ignore
                else:
                    p_ij = 0

                if e_ij > 0 and p_ij > 0 and n_pairs > 0:# type: ignore
                    ll += e_ij * np.log(p_ij) - n_pairs * p_ij

        return ll

    def _compute_edges_between_node_and_blocks(self,
                                               node: int,
                                               affected_blocks: set) -> Dict[int, int]:
        """
        Compute the number of edges between a node and each affected block.
    
        :param node: The node to move.
        :param affected_blocks: The blocks affected by the move.
        :return: A dictionary mapping block IDs to edge counts with the node.
        """
        neighbors = set(self.block_data.graph_data.adjacency[node].indices) # type: ignore
        m_vk = {}
        for block in affected_blocks:
            nodes_in_block = set(self.block_data.block_members[block])
            m_vk[block] = len(neighbors.intersection(nodes_in_block))

        return m_vk
    
    def _compute_delta_edge_counts(self, m_vk: Dict[int, int],
                                   source_block: int,
                                   target_block: int,
                                   affected_blocks: set) -> Dict[Tuple[int, int], int]:
        """
        Compute the changes in edge counts between blocks due to the move.
    
        :param m_vk: The edges between the moving node and each affected block.
        :param source_block: The block from which the node is moved.
        :param target_block: The block to which the node is moved.
        :param affected_blocks: The blocks affected by the move.

        :return: A dictionary mapping block pairs to changes in edge counts.
        """
        delta_e = {}
    
        for k in affected_blocks:
            if k != target_block:
                key_sk = (min(source_block, k), max(source_block, k))
                delta_e[key_sk] = delta_e.get(key_sk, 0) - m_vk.get(k, 0)
    
            if k != source_block:
                key_tk = (min(target_block, k), max(target_block, k))
                delta_e[key_tk] = delta_e.get(key_tk, 0) + m_vk.get(k, 0)
    
        delta_e[(source_block, source_block)] = \
              delta_e.get((source_block, source_block), 0) - 2 * m_vk.get(source_block, 0)

        delta_e[(target_block, target_block)] = \
              delta_e.get((target_block, target_block), 0) + 2 * m_vk.get(target_block, 0)
    
        return delta_e
    
    def _compute_delta_possible_pairs(self, source_block: int, target_block: int,
                                      affected_blocks: set) -> Dict[Tuple[int, int], int]:
        """
        Compute the changes in possible pairs between blocks due to the move.
    
        :param source_block: The block from which the node is moved.
        :param target_block: The block to which the node is moved.
        :param affected_blocks: The blocks affected by the move.
        :return: A dictionary mapping block pairs to changes in possible pairs.
        """
        delta_n = {}
        n_s_old = self.block_data.block_sizes[source_block]
        n_t_old = self.block_data.block_sizes[target_block]
        n_s_new = n_s_old - 1
        n_t_new = n_t_old + 1
    
        delta_n[(source_block, source_block)] = - (n_s_old - 1)
        delta_n[(target_block, target_block)] = + (n_t_new - 1)
        delta_n[(source_block, target_block)] = n_s_new * n_t_new - n_s_old * n_t_old
    
        for k in affected_blocks:
            if k != source_block and k != target_block:
                n_k = self.block_data.block_sizes[k]
                key_sk = (min(source_block, k), max(source_block, k))
                key_tk = (min(target_block, k), max(target_block, k))
                delta_n[key_sk] = delta_n.get(key_sk, 0) - n_k
                delta_n[key_tk] = delta_n.get(key_tk, 0) + n_k
    
        return delta_n
    
    def _compute_delta_ll_from_changes(self, delta_e: Dict[Tuple[int, int], int],
                                       delta_n: Dict[Tuple[int, int], int],
                                       total_edges: float) -> float:
        """
        Compute the change in log-likelihood from changes in edge counts and possible pairs.
    
        :param delta_e: Changes in edge counts between blocks.
        :param delta_n: Changes in possible pairs between blocks.
        :param total_edges: Total number of edges in the graph.
        :return: The change in log-likelihood.
        """
        delta_ll = 0.0
        for (i_block, j_block), delta_e_ij in delta_e.items():
            idx_i = self.block_data.block_indices[i_block]
            idx_j = self.block_data.block_indices[j_block]
    
            e_ij_old = self.block_data.block_connectivity[idx_i, idx_j]
            e_ij_new = e_ij_old + delta_e_ij # type: ignore
    
            if e_ij_new < 0:
                e_ij_new = 0
    
            delta_n_ij = delta_n.get((i_block, j_block), 0)
            n_ij_old = self.block_data.block_connectivity[i_block, j_block]
            n_ij_new = n_ij_old + delta_n_ij # type: ignore
    
            if n_ij_new <= 0:
                continue
            
            p_ij_old = e_ij_old / total_edges if total_edges > 0 else 0 # type: ignore
            p_ij_new = e_ij_new / total_edges if total_edges > 0 else 0
    
            if p_ij_old > 0 and p_ij_new > 0:
                delta_ll += e_ij_new * np.log(p_ij_new) - e_ij_old * np.log(p_ij_old)
                delta_ll -= n_ij_new * p_ij_new - n_ij_old * p_ij_old # type: ignore
    
        return delta_ll

    def _compute_delta_edge_counts_merge(self, block_a: int, block_b: int, affected_blocks: set) -> Dict[Tuple[int, int], int]:
        """
        Compute the changes in edge counts between blocks due to merging two blocks.

        :param block_a: The first block to merge.
        :param block_b: The second block to merge.
        :param affected_blocks: The blocks affected by the merge.
        :return: A dictionary mapping block pairs to changes in edge counts.
        """
        delta_e = {}

        idx_a = self.block_data.block_indices[block_a]
        idx_b = self.block_data.block_indices[block_b]

        # New block_c is the merged block
        block_c = min(block_a, block_b)  # Use the smaller block ID for consistency

        # Edge counts within block_c
        e_aa = self.block_data.block_connectivity[idx_a, idx_a]
        e_bb = self.block_data.block_connectivity[idx_b, idx_b]
        e_ab = self.block_data.block_connectivity[idx_a, idx_b]

        # Total internal edges in block_c
        e_cc_new = e_aa + e_bb + e_ab  # type: ignore # e_ab is already counted once in undirected graphs

        # Old internal edges
        e_aa_old = e_aa
        e_bb_old = e_bb
        e_ab_old = e_ab

        # Change in internal edges
        delta_e[(block_c, block_c)] = e_cc_new - (e_aa_old + e_bb_old + e_ab_old) # type: ignore

        # Remove old edge counts between block_a and block_b
        delta_e[(min(block_a, block_b), max(block_a, block_b))] = -e_ab_old # type: ignore

        # Update edge counts between block_c and other blocks
        for block_k in affected_blocks:
            if block_k == block_a or block_k == block_b:
                continue

            idx_k = self.block_data.block_indices[block_k]
        
            # Edge counts between block_a and block_k
            e_ak = self.block_data.block_connectivity[idx_a, idx_k]
            # Edge counts between block_b and block_k
            e_bk = self.block_data.block_connectivity[idx_b, idx_k]

            # Combined edge counts between block_c and block_k
            e_ck_new = e_ak + e_bk # type: ignore

            # Remove old edge counts
            key_ak = (min(block_a, block_k), max(block_a, block_k))
            key_bk = (min(block_b, block_k), max(block_b, block_k))
            delta_e[key_ak] = delta_e.get(key_ak, 0) - e_ak
            delta_e[key_bk] = delta_e.get(key_bk, 0) - e_bk

            # Add new edge counts
            key_ck = (min(block_c, block_k), max(block_c, block_k))
            delta_e[key_ck] = delta_e.get(key_ck, 0) + e_ck_new

        return delta_e

    def _compute_delta_possible_pairs_merge(self, block_a: int, block_b: int, affected_blocks: set) -> Dict[Tuple[int, int], int]:
        """
        Compute the changes in possible pairs between blocks due to merging two blocks.

        :param block_a: The first block to merge.
        :param block_b: The second block to merge.
        :param affected_blocks: The blocks affected by the merge.
        :return: A dictionary mapping block pairs to changes in possible pairs.
        """
        delta_n = {}

        n_a = self.block_data.block_sizes[block_a]
        n_b = self.block_data.block_sizes[block_b]
        n_c = n_a + n_b  # Size of the merged block

        # New block_c is the merged block
        block_c = min(block_a, block_b)

        # Change in internal possible pairs
        n_cc_new = n_c * (n_c - 1) // 2
        n_aa_old = n_a * (n_a - 1) // 2
        n_bb_old = n_b * (n_b - 1) // 2
        n_ab_old = n_a * n_b

        delta_n[(block_c, block_c)] = n_cc_new - (n_aa_old + n_bb_old + n_ab_old)

        # Remove old possible pairs between block_a and block_b
        delta_n[(min(block_a, block_b), max(block_a, block_b))] = -n_ab_old

        # Update possible pairs between block_c and other blocks
        for block_k in affected_blocks:
            if block_k == block_a or block_k == block_b:
                continue

            n_k = self.block_data.block_sizes[block_k]

            # Old possible pairs
            n_ak_old = n_a * n_k
            n_bk_old = n_b * n_k

            # New possible pairs
            n_ck_new = n_c * n_k

            # Remove old possible pairs
            key_ak = (min(block_a, block_k), max(block_a, block_k))
            key_bk = (min(block_b, block_k), max(block_b, block_k))
            delta_n[key_ak] = delta_n.get(key_ak, 0) - n_ak_old
            delta_n[key_bk] = delta_n.get(key_bk, 0) - n_bk_old

            # Add new possible pairs
            key_ck = (min(block_c, block_k), max(block_c, block_k))
            delta_n[key_ck] = delta_n.get(key_ck, 0) + n_ck_new

        return delta_n


    def _get_affected_blocks_merge(self, block_a: int, block_b: int) -> set:
        """
        Identify all blocks affected by merging two blocks.

        :param block_a: The first block to merge.
        :param block_b: The second block to merge.
        :return: A set of affected block IDs.
        """
        affected_blocks = set()

        idx_a = self.block_data.block_indices[block_a]
        idx_b = self.block_data.block_indices[block_b]

        # Get blocks connected to block_a
        connected_blocks_a = self.block_data.block_connectivity[idx_a].indices # type: ignore
        for idx in connected_blocks_a:
            block_id = self.block_data.inverse_block_indices[idx]
            affected_blocks.add(block_id)

        # Get blocks connected to block_b
        connected_blocks_b = self.block_data.block_connectivity[idx_b].indices # type: ignore
        for idx in connected_blocks_b:
            block_id = self.block_data.inverse_block_indices[idx]
            affected_blocks.add(block_id)

        # Add block_a and block_b themselves
        affected_blocks.add(block_a)
        affected_blocks.add(block_b)

        return affected_blocks

    def _compute_delta_ll_merge_blocks(self, block_a: int, block_b: int) -> float:
        """
        Compute the change in log-likelihood resulting from merging two blocks.

        :param block_a: The first block to merge.
        :param block_b: The second block to merge.
        :return: The change in log-likelihood.
        """
        total_edges = self.block_data.graph_data.total_edges

        # Step 1: Identify affected blocks
        affected_blocks = self._get_affected_blocks_merge(block_a, block_b)

        # Step 2: Compute changes in edge counts (delta_e)
        delta_e = self._compute_delta_edge_counts_merge(block_a, block_b, affected_blocks)

        # Step 3: Compute changes in possible pairs (delta_n)
        delta_n = self._compute_delta_possible_pairs_merge(block_a, block_b, affected_blocks)

        # Step 4: Compute delta log-likelihood
        delta_ll = self._compute_delta_ll_from_changes(
            delta_e=delta_e,
            delta_n=delta_n,
            total_edges=float(self.block_data.graph_data.total_edges)
            )

        return delta_ll

    def _get_affected_blocks(self, node: int, source_block: int, target_block: int) -> set:
        """
        Identify all blocks affected by moving a node.
    
        :param node: The node to move.
        :param source_block: The block from which the node is moved.
        :param target_block: The block to which the node is moved.
        :return: A set of affected block IDs.
        """
        neighbors = self.block_data.graph_data.adjacency[node].indices # type: ignore
        neighbor_blocks = {self.block_data.blocks[neighbor] for neighbor in neighbors}
        return neighbor_blocks.union({source_block, target_block})

    def _compute_delta_ll_move_node(self, node: int, source_block: int, target_block: int) -> float:
        """
        Compute the change in log-likelihood resulting from moving a node between blocks.

        :param node: The node to move.
        :param source_block: The block from which the node is moved.
        :param target_block: The block to which the node is moved.
        :return: The change in log-likelihood.
        """
        total_edges = self.block_data.graph_data.total_edges

        # Step 1: Identify affected blocks
        affected_blocks = self._get_affected_blocks(node, source_block, target_block)

        # Step 2: Compute m_vk for affected blocks
        m_vk = self._compute_edges_between_node_and_blocks(node, affected_blocks)

        # Step 3: Compute changes in edge counts (delta_e)
        delta_e = self._compute_delta_edge_counts(m_vk, source_block, target_block, affected_blocks)

        # Step 4: Compute changes in possible pairs (delta_n)
        delta_n = self._compute_delta_possible_pairs(source_block, target_block, affected_blocks)

        # Step 5: Compute delta log-likelihood
        delta_ll = self._compute_delta_ll_from_changes(
            delta_e=delta_e,
            delta_n=delta_n,
            total_edges=float(self.block_data.graph_data.total_edges),
        )

        return delta_ll

    def compute_delta_ll_split_block(self, block_to_split: int, nodes_a: List[int], nodes_b: List[int]) -> float:
        """
        Compute the change in log-likelihood resulting from splitting a block.

        :param block_to_split: The block to split.
        :param nodes_a: The nodes to move to the new block.
        :param nodes_b: The nodes to keep in the original block.
        :return: The change in log-likelihood.
        """
        total_edges = self.block_data.graph_data.total_edges

        # Step 1: Identify affected blocks
        affected_blocks = {block_to_split}

        # Step 2: Compute m_vk for affected blocks
        m_vk = {}
        for node in nodes_a:
            m_vk.update(self._compute_edges_between_node_and_blocks(node, affected_blocks))

        # Step 3: Compute changes in edge counts (delta_e)
        delta_e = {}
        for node in nodes_a:
            delta_e.update(self._compute_delta_edge_counts(m_vk, block_to_split, block_to_split, affected_blocks))

        # Step 4: Compute changes in possible pairs (delta_n)
        delta_n = {}
        for node in nodes_a:
            delta_n.update(self._compute_delta_possible_pairs(block_to_split, block_to_split, affected_blocks))

        # Step 5: Compute delta log-likelihood
        delta_ll = self._compute_delta_ll_from_changes(
            delta_e=delta_e,
            delta_n=delta_n,
            total_edges=float(self.block_data.graph_data.total_edges),
        )

        return delta_ll
