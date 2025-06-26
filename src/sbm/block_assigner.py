""" 
    Functions and classes for computing initial block assignment 
    used in the Stochastic Block Model (SBM). These are later updated
    using the MCMC algorithm.
"""
from typing import List, Dict, Optional, Tuple, Iterable
from collections import defaultdict

from time import time

import pymetis
import scipy.sparse as sp
import numpy as np
from k_means_constrained import KMeansConstrained
from nodevectors import ProNE

from sbm.block_data import BlockData
from sbm.graph_data import GraphData
from sbm.utils.pipeline_utils import InitMethodName

#  helper ---------------------------------------------------------------
def _block_sizes(blocks: Dict[int, int]) -> Dict[int, int]:
    sizes = defaultdict(int)
    for b in blocks.values():
        sizes[b] += 1
    return sizes

def _compute_acceptors(sizes, acceptor_size:int, max_increments: int = 1000):

    acceptors = {b for b, sz in sizes.items() if sz == acceptor_size}
    curr_increment = 0

    while len(acceptors) == 0:
        acceptor_size += 1
        curr_increment += 1

        acceptors = {b for b, sz in sizes.items() if sz == acceptor_size}

        if curr_increment >= max_increments:
            raise ValueError(
                "Cannot balance blocks to min_block_size."
            )

    return acceptors, acceptor_size


#  Mixin – functions for balancing block sizes of proposed block partition --
def rebalance_to_min_size(
    blocks: Dict[int, int],
    adjacency: sp.csr_array,
    min_block_size: Optional[int],
) -> Dict[int, int]:
    """
    Post-process a *proposed* partition so that every block has
    size `min_block_size` or `min_block_size+1`.

    The algorithm:

    1.  Collect all nodes that currently belong to a block of size
        ``< min_block_size``.
    2.  Carve out as many **new** full blocks of size
        ``min_block_size`` as possible from this pool.
    3.  For the *left-over* nodes (< `min_block_size`)  
        a) try to move a node to a neighboring block that is still
           of exact size ``min_block_size``;  
        b) if no such neighbour exists, pick a random block that
           is still at size ``min_block_size``.  
        Each block may accept **at most one** such extra node,
        reaching ``min_block_size+1``.
    4.  Return the new `node → block` mapping.

    Parameters
    ----------
    blocks : dict {node: block_id}
        Preliminary assignment.
    adjacency : scipy.sparse.csr_array
        Undirected adjacency for neighbourhood look-ups.
    min_block_size : int
        Target base size *k*.

    Returns
    -------
    dict
        Balanced `node → block` mapping.
    """
    if min_block_size is None:
        return blocks

    sizes = _block_sizes(blocks)
    undersized = {b for b, sz in sizes.items() if sz < min_block_size}
    if not undersized:
        return blocks  # already balanced

    # -----------------------------------------------------------------
    # 1. pool all nodes from undersized blocks
    # -----------------------------------------------------------------
    pool: List[int] = [v for v, b in blocks.items() if b in undersized]

    # remove those undersized blocks entirely
    for b in undersized:
        sizes.pop(b, None)

    # current max label so we can create fresh block IDs
    next_block_id = max(sizes.keys(), default=-1) + 1

    # -----------------------------------------------------------------
    # 2. carve as many full blocks (size == k) as possible
    # -----------------------------------------------------------------
    while len(pool) >= min_block_size:
        new_nodes = [pool.pop() for _ in range(min_block_size)]
        for v in new_nodes:
            blocks[v] = next_block_id
        sizes[next_block_id] = min_block_size
        next_block_id += 1

    # -----------------------------------------------------------------
    # 3. distribute remaining ( < k ) nodes
    # -----------------------------------------------------------------

    # keep a fast lookup of which *existing* blocks can still accept 1
    # first blocks of size `min_block_size` can accept 1 more node
    # if we run out, we increment size constraints on which blocks can accept
    acceptors, acceptor_size = _compute_acceptors(sizes, min_block_size)

    rng = np.random.default_rng(42)  # deterministic for tests

    for v in pool:
        # 3a. neighbour heuristic
        neigh_blocks = {
            blocks[u]
            for u in adjacency.indices[adjacency.indptr[v] : adjacency.indptr[v + 1]]
            if sizes.get(blocks[u], 0) == acceptor_size
        }
        target = None
        if neigh_blocks:
            target = rng.choice(list(neigh_blocks))
        elif acceptors:
            # 3b. random among remaining acceptors
            target = rng.choice(list(acceptors))
        else:
            # we have run out of acceptors, so we need to increase the size
            acceptors, acceptor_size = _compute_acceptors(sizes, acceptor_size)
            target = rng.choice(list(acceptors))



        # assign
        blocks[v] = target
        sizes[target] += 1
        if sizes[target] == min_block_size + 1:
            acceptors.discard(target)
        pool.remove(v)

    return blocks


### Base class for BlockAssigner
class BlockAssigner:
    """
    Base class for assigning nodes to blocks in the Stochastic Block Model (SBM).
    This class is intended to be subclassed for specific block assignment strategies.
    """
    def __init__(self,
                 graph_data: GraphData,
                 rng: np.random.Generator,
                 num_blocks: Optional[int] = None,
                 min_block_size: Optional[int] = None,
                 max_block_size: Optional[int] = None,
                 ):
        self.graph_data = graph_data

        # check if there exist a valid assignment
        # given num_blocks, min_block_size, max_block_size
        if num_blocks is not None and min_block_size is not None:
            if num_blocks * min_block_size > graph_data.num_nodes:
                raise ValueError("Invalid parameters: num_blocks * min_block_size exceeds total number of nodes.")
        if max_block_size is not None and min_block_size is not None:
            if max_block_size < min_block_size:
                raise ValueError("Invalid parameters: max_block_size cannot be less than min_block_size.")

        self.num_blocks = num_blocks
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.rng = rng
        self.min_size_balancers = rebalance_to_min_size

    def reindex_blocks(self, blocks: Dict[int, int]) -> Dict[int, int]:
        """ 
        Reindex block IDs to be consecutive integers starting from 0.
        """
        unique_blocks = sorted(set(blocks.values()))
        block_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_blocks)}
        return {node: block_mapping[block_id] for node, block_id in blocks.items()}


    def _compute_assignment(self) -> Dict[int, int]:
        raise NotImplementedError(
            "compute_assignment must be implemented by subclasses of BlockAssigner."
        )
    
    def compute_assignment(self) -> BlockData:
        raise NotImplementedError(
            "compute_assignment must be implemented by subclasses of BlockAssigner."
        )


class UniformSmallBlockAssigner(BlockAssigner):
    """ 
    Assigns nodes to blocks of size min_block_size uniformly at random. 
    Ignore num_blocks and max_block_size.
    """

    def _compute_assignment(self) -> Dict[int, int]:
        """
        Assign nodes to blocks uniformly at random, ensuring each block has at least min_block_size nodes.
        """
        if self.min_block_size is None:
            raise ValueError("min_block_size must be specified for UniformSmallBlockAssigner.")
        if self.min_block_size <= 0:
            raise ValueError("min_block_size must be a positive integer.")

        if self.max_block_size is not None:
            Warning("max_block_size is ignored in UniformSmallBlockAssigner.")
        if self.num_blocks is not None:
            Warning("num_blocks is ignored in UniformSmallBlockAssigner.")

        num_nodes = self.graph_data.num_nodes

        # create list of nodes in random order
        node_list = self.rng.permutation(np.arange(num_nodes))
        # assign nodes to blocks
        block_assignments = {
            node: node // self.min_block_size for node in node_list
        }

        return block_assignments
    
    def compute_assignment(self) -> BlockData:
        """
        Compute a block assignment based on the proposed assignment.
        Currently, this method only performs a min_size balancing step.
        """
        proposed_assignment = self._compute_assignment()
        balanced_assignment = self.min_size_balancers(
            blocks=proposed_assignment,
            adjacency=self.graph_data.adjacency,
            min_block_size=self.min_block_size,
        )
        reindexed_assignment = self.reindex_blocks(balanced_assignment)

        return BlockData(
            initial_blocks=reindexed_assignment,
            graph_data=self.graph_data
        )


class MetisBlockAssigner(BlockAssigner):
    """
    Use PyMetis to obtain a *balanced* `num_blocks`-way partition of the
    (undirected) graph.

    Parameters
    ----------
    graph_data : GraphData
        Graph wrapper holding the (sparse) adjacency matrix.
    num_blocks : int
        Desired number of blocks (≈ N // k where k is target block size).
    seed : int, optional
        Random seed forwarded to METIS.  If None, METIS uses its own seed.
    """

    def __init__(
        self,
        graph_data: GraphData,
        rng: np.random.Generator,
        num_blocks: Optional[int] = None,
        min_block_size: Optional[int] = None,
        max_block_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            graph_data=graph_data,
            rng=rng,
            num_blocks=num_blocks,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            )

        if graph_data.directed:
            raise NotImplementedError(
                "MetisBlockAssigner currently supports undirected graphs only."
            )

        if num_blocks is None and min_block_size is None:
            raise ValueError("Either num_blocks or min_block_size must be specified for MetisBlockAssigner.")
        
        if num_blocks is None:
            num_blocks = max(
                1, graph_data.num_nodes // min_block_size
            )

        self.num_blocks = int(num_blocks)
        self.seed = rng.integers(2**32)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------
    @staticmethod
    def _to_adj_lists(adj: sp.csr_array) -> list[list[int]]:
        """
        Convert a CSR adjacency matrix to the adjacency-list format PyMetis
        expects (no self-loops, undirected symmetry).
        """
        n = adj.shape[0] # type: ignore
        rows, cols = adj.nonzero() # type: ignore
        neigh = [[] for _ in range(n)]
        for u, v in zip(rows, cols):
            if u == v:
                continue  # ignore self-loops
            neigh[u].append(v)
        return neigh

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def _compute_assignment(self) -> Dict[int, int]:
        """Run METIS and return a BlockData with the resulting assignment."""
        n = self.graph_data.num_nodes
        adj_lists = self._to_adj_lists(self.graph_data.adjacency)

        # PyMetis returns (edgecuts, membership-array)
        _, parts = pymetis.part_graph(
            self.num_blocks,
            adjacency=adj_lists,
        )

        # PyMetis guarantees |parts| == n
        blocks: Dict[int, int] = {node: part for node, part in enumerate(parts)}

        # Wrap in BlockData so downstream code can use it directly
        return blocks

    def compute_assignment(self) -> BlockData:
        """
        Compute a balanced block assignment based on the proposed assignment.

        Currently, this method only performs a min_size balancing step.
        """
        proposed_assignment = self._compute_assignment()
        balanced_assignment = self.min_size_balancers(
            blocks=proposed_assignment,
            adjacency=self.graph_data.adjacency,
            min_block_size=self.min_block_size,
        )
        reindexed_assignment = self.reindex_blocks(balanced_assignment)

        return BlockData(
            initial_blocks=reindexed_assignment,
            graph_data=self.graph_data
        )


class EmbedAndConstrKMeansAssigner(BlockAssigner):
    """
    Assign nodes to blocks using a two-step process:
    1. use embed nodes into a low-dimensional space,
    2. use constrained KMeans to assign nodes to blocks of prespecified sizes.
    """ 

    def __init__(
        self,
        graph_data: GraphData,
        rng: np.random.Generator,
        num_blocks: Optional[int] = None,
        min_block_size: Optional[int] = None,
        max_block_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            graph_data=graph_data,
            rng=rng,
            num_blocks=num_blocks,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
        )

        if min_block_size is None:
            raise ValueError("num_blocks and min_block_size must be specified for ProneAndConstrKMeansAssigner.")
        if num_blocks is not None:
            Warning("num_blocks is ignored in ProneAndConstrKMeansAssigner. Only min_block_size is used.")
        if max_block_size is not None:
            Warning("max_block_size is ignored in ProneAndConstrKMeansAssigner. Only min_block_size is used.")

    def embed_nodes(self, adjacency:sp.csr_array, n_dimensions:int=128)->np.ndarray:
        """ 
        Method to perform node embedding. Subclasses should implement this method
        """
        raise NotImplementedError("This method should be overwritten by subclasses to provide specific embedding logic.")

    def _compute_assignment(self) -> Dict[int, int]:
        """
        Compute block assignments using constrained KMeans after embedding with Prone.
        """
        if self.graph_data.num_nodes < self.min_block_size:
            raise ValueError("Number of nodes in the graph is less than min_block_size.")
        if self.min_block_size is None:
            raise ValueError("min_block_size must be specified for ProneAndConstrKMeansAssigner.")

        # Step 1: Embed nodes using Prone
        embeddings = self.embed_nodes(
            adjacency=self.graph_data.adjacency,
            n_dimensions=128  # default embedding dimension
        )

        # compute how many blocks we need to only have blocks of
        #   size min_block_size and min_block_size+1
        number_of_clusters = self.graph_data.num_nodes // self.min_block_size
        
        # Step 2: Use constrained KMeans to assign nodes to blocks
        kmeans = KMeansConstrained(
                    n_clusters=number_of_clusters,
                    size_min=self.min_block_size,
                    size_max=self.min_block_size+1, # 
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    tol=1e-3,
                    verbose=False,
                    random_state=self.rng.choice(2**32), 
                    copy_x=True,
                    n_jobs=-2 # use all but one CPU core
                )
        tic = time() 
        labels = kmeans.fit_predict(embeddings)
        toc = time()
        print(f"KMeans with constraints took {toc - tic:.2f} seconds for {
            self.graph_data.num_nodes} nodes and {embeddings.shape[1]} dimensions."
        )

        # Create a mapping from node index to block ID
        blocks = {node: label for node, label in enumerate(labels)} # type: ignore

        return blocks
    
    def compute_assignment(self) -> BlockData:
        """
        Compute a block assignment based on the proposed assignment.
        Currently, this method only performs a min_size balancing step.
        """
        balanced_assignment = self._compute_assignment() # balanced from k-means w. size constraints
        reindexed_assignment = self.reindex_blocks(balanced_assignment)

        return BlockData(
            initial_blocks=reindexed_assignment,
            graph_data=self.graph_data
        )


class ProNEAndConstrKMeansAssigner(EmbedAndConstrKMeansAssigner):
    """
    Assign nodes to blocks using ProNE embedding followed by constrained KMeans.
    """

    def embed_nodes(self, adjacency: sp.csr_array, n_dimensions: int = 128) -> np.ndarray:
        """
        Embed nodes using ProNE.
        """
        if n_dimensions <= 0:
            raise ValueError("n_dimensions must be a positive integer.")

        # Create a ProNE instance and fit it to the adjacency matrix
        model = ProNE(
                    n_components=n_dimensions,
                    step=10,
                    mu=0.2,
                    theta=0.5, 
                    exponent=0.75,
                    verbose=True
                )
        tic = time()        
        embeddings = model.fit_transform(
            sp.csr_matrix(adjacency) # nodevectors expect a CSR matrix, and not array
            )
        toc = time()
        print(f"ProNE embedding took {toc - tic:.2f} seconds for {
            self.graph_data.num_nodes} nodes and {n_dimensions} dimensions."
        )

        return embeddings

class AssignerConstructor:
    """ 
    Factory class to construct block assigners based on configuration parameters. 

    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def create_assigner(self,
                        graph_data: GraphData,
                        init_method: InitMethodName = "metis", 
                        min_block_size: Optional[int] = None,
                        max_block_size: Optional[int] = None,
                        num_blocks: Optional[int] = None,
                    ) -> BlockAssigner:

        if init_method == "metis":
            return MetisBlockAssigner(
                graph_data=graph_data,
                rng=self.rng,
                min_block_size=min_block_size,
                max_block_size=max_block_size,
                num_blocks=num_blocks,
            )
        elif init_method == "random":
            return UniformSmallBlockAssigner(
                graph_data=graph_data,
                rng=self.rng,
                min_block_size=min_block_size,
                max_block_size=max_block_size,
                num_blocks=num_blocks,
            )
        elif init_method == "ProneKMeans":
            return ProNEAndConstrKMeansAssigner(
                graph_data=graph_data,
                rng=self.rng,
                min_block_size=min_block_size,
                max_block_size=max_block_size,
                num_blocks=num_blocks,
            )

