""" 
    Functions and classes for computing initial block assignment 
    used in the Stochastic Block Model (SBM). These are later updated
    using the MCMC algorithm.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Iterable, TypeAlias, Set
from collections import defaultdict, Counter

from time import time

import metis
import scipy.sparse as sp
from scipy.sparse import csr_array
import numpy as np
from k_means_constrained import KMeansConstrained
from nodevectors import ProNE

from sbm.block_data import BlockData
from sbm.graph_data import GraphData
from sbm.utils.pipeline_utils import InitMethodName
from ortools.sat.python import cp_model  # type: ignore

# ---------------------------------------------------------------------------
#  helper ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def _block_sizes(blocks: Dict[int, int]) -> Dict[int, int]:
    sizes = defaultdict(int)
    for b in blocks.values():
        sizes[b] += 1
    return sizes

def _boundary_vertices(block: int, members: Set[int], blocks: Dict[int, int],
                       indptr: np.ndarray, indices: np.ndarray) -> List[int]:
    """Return vertices in *block* that touch at least one different block."""
    out = []
    for v in members:
        row = slice(indptr[v], indptr[v + 1])
        if any(blocks[u] != block for u in indices[row]):
            out.append(v)
    return out


def _movable_vertex(src: int, dst_set: Set[int], *, rng: np.random.Generator,
                    blocks: Dict[int, int], members: Dict[int, Set[int]],
                    indptr: np.ndarray, indices: np.ndarray) -> Tuple[int, int] | None:
    """Pick (vertex, dst) with vertex in *src* boundary and dst in dst_set."""
    boundary = _boundary_vertices(src, members[src], blocks, indptr, indices)
    rng.shuffle(boundary)
    for v in boundary:
        row = slice(indptr[v], indptr[v + 1])

        neigh_blks = {blocks[u] for u in indices[row] if blocks[u] in dst_set}

        if neigh_blks:
            return v, rng.choice(list(neigh_blks))
    return None


def _move(v: int, src: int, dst: int, *, blocks: Dict[int, int],
          members: Dict[int, Set[int]], sizes: Counter
    ):
    """Execute the move and update bookkeeping structures."""
    blocks[v] = dst
    members[src].remove(v)
    members[dst].add(v)
    sizes[src] -= 1
    sizes[dst] += 1

    if sizes[src] == 0:
        # remove empty block
        del members[src]
        del sizes[src]
        del blocks[v]

def move_node_to_under(
        under: Set[int], # blocks of size < k
        over1: Set[int], # blocks of size k+1
        over2: Set[int], # blocks of size > k+1
        rng: np.random.Generator,
        sizes: Counter[int],
        k: int,
        members: Dict[int, Set[int]],
        blocks: Dict[int, int],
        indptr: np.ndarray,
        indices: np.ndarray
    ) -> None:
    """ 
    Move a node from an oversize block to an undersize block.
    """

    if len(under) == 0:
        # no undersize blocks available, skip
        return

    donors = list(over2 | over1) or list(b for b, s in sizes.items() if s > k)
    rng.shuffle(donors)
    moved = False
    if len(under) > 0:
        for b_src in donors:
            mv = _movable_vertex(b_src, under, rng=rng, blocks=blocks,
                                    members=members, indptr=indptr, indices=indices)
            if mv is not None:
                v, b_dst = mv
                _move(v, b_src, b_dst, blocks=blocks, members=members, sizes=sizes)
                moved = True
                return

    # If we reach here no boundary move could be found. Relax: pick random.
    if len(donors) == 0:
        # no oversize blocks available, pick random from all
        donors = list(blocks.keys())

    b_src = rng.choice(donors)
    if len(members[b_src]) == 0:
        # no members in the source block, skip
        return

    v = rng.choice(list(members[b_src]))
    b_dst = rng.choice(tuple(under))
    _move(v, b_src, b_dst, blocks=blocks, members=members, sizes=sizes)
    return

def move_node_from_over(
    under: Set[int], # blocks of size < k
    over1: Set[int], # blocks of size k+1
    over2: Set[int], # blocks of size > k+1
    rng: np.random.Generator,
    sizes: Counter[int],
    k: int,
    members: Dict[int, Set[int]],
    blocks: Dict[int, int],
    indptr: np.ndarray,
    indices: np.ndarray,
    r_target: int,
) -> None:
    """ 
        Nodes are moved from block with size > k to block with size either
        < k or <=k if there are fewer than r_target blocks with size k+1.
    """

    if len(over2) == 0:
        # no oversize blocks available, skip
        return

    b_src = rng.choice(tuple(over2))
    dests = under.copy()
    if len(over1) > r_target:
        dests |= {b for b, s in sizes.items() if s == k}
    if not dests:
        # no eligible destination respecting k‑lower‑bound → skip
        return
    mv = _movable_vertex(b_src, dests, rng=rng, blocks=blocks,
                            members=members, indptr=indptr, indices=indices)
    if mv is None:
        v = rng.choice(_boundary_vertices(b_src, members[b_src], blocks, indptr, indices))
        b_dst = rng.choice(tuple(dests))
    else:
        v, b_dst = mv

    _move(v, b_src, b_dst, blocks=blocks, members=members, sizes=sizes)
    return

def balance_k_plus_1_blocks(
    over1: Set[int], # blocks of size k+1
    over2: Set[int], # blocks of size >k+1
    rng: np.random.Generator,
    sizes: Counter[int],
    k: int,
    members: Dict[int, Set[int]],
    blocks: Dict[int, int],
    indptr: np.ndarray,
    indices: np.ndarray,
    r_target: int,
) -> None:
    """ 
    Balance the number of blocks with size k+1.

    If there are too many blocks with size k+1, shrink one of them
    by moving a vertex to a block with size k or smaller.

    If there are too few blocks with size k+1, enlarge one of the blocks
    with size k by moving a vertex from a block with size larger than k+1
    or smaller than k+1
    """

    if len(over1) == r_target:
        # already balanced, nothing to do
        return

    elif len(over1) > r_target: # need fewer k+1 blocks
        # shrink a k+1 block
        b_src = rng.choice(tuple(over1))
        dests = {b for b, s in sizes.items() if s <= k}

    else:  # need more k+1 blocks
        # enlarge a k block
        #dests = set()
        b_src = rng.choice(tuple(over2))
        dests = {b for b, s in sizes.items() if s == k}

    if len(dests) == 0:
        # no eligible destination respecting k‑upper‑bound → skip
        return

    # shrink case
    mv = _movable_vertex(b_src, dests, rng=rng, blocks=blocks,
                            members=members, indptr=indptr, indices=indices)
    if mv is not None:
        v, b_dst = mv
        _move(v, b_src, b_dst, blocks=blocks, members=members, sizes=sizes)
        return

    return

def categorize(
    sizes: Dict[int, int],
    k: int,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Return (oversize>k+1, oversize==k+1, undersize<k)."""
    over2 = {b for b, s in sizes.items() if s > k + 1}
    over1 = {b for b, s in sizes.items() if s == k + 1}
    under = {b for b, s in sizes.items() if s < k}
    return over2, over1, under

# ---------------------------------------------------------------------------
# Improved greedy balancer (split into helpers)
# ---------------------------------------------------------------------------

def _rebalance_to_min_size(
    blocks: Dict[int, int],
    adjacency: csr_array,
    k: int,
    rng: np.random.Generator | None = None,
    max_iter: int | None = None,
) -> Dict[int, int]:
    """Greedy boundary‑only balancing.

    Guarantees **no block ends smaller than *k***; tries to respect the stricter
    goal (sizes ∈ {k,k+1} & exactly *r* oversized) but will *sacrifice* that goal
    rather than leave an undersized block.
    """
    if rng is None:
        rng = np.random.default_rng(1)

    n = adjacency.shape[0]
    indptr, indices = adjacency.indptr, adjacency.indices

    sizes: Counter[int] = Counter(blocks.values())
    members: Dict[int, Set[int]] = defaultdict(set)
    for v, b in blocks.items():
        members[b].add(v)

    B = len(sizes)
    r_target = n - k * B  # blocks that *should* have k+1

    iter_limit = max_iter or 5 * n
    while iter_limit:
        iter_limit -= 1
        over2, over1, under = categorize(sizes=sizes, k=k)

        if len(under)==0 and len(over2)==0 and len(over1)==r_target:
            break  # fully balanced by strict rules

        # 1) fix undersized first ------------------------------------------------
        if len(under) > 0:
            move_node_to_under(
                under=under,
                over1=over1,
                over2=over2,
                rng=rng,
                sizes=sizes,
                k=k,
                members=members,
                blocks=blocks,
                indptr=indptr,
                indices=indices
            )
            continue

        # 2) shrink blocks > k+1 --------------------------------------------------
        if len(over2) > 0 :
            move_node_from_over(
                under=under,
                over1=over1,
                over2=over2,
                rng=rng,
                sizes=sizes,
                k=k,
                members=members,
                blocks=blocks,
                indptr=indptr,
                indices=indices,
                r_target=r_target
            )
            continue

        # 3) adjust number of k+1 blocks -----------------------------------------
        if len(over1) != r_target:
            balance_k_plus_1_blocks(
                over1=over1,
                over2=over2,
                rng=rng,
                sizes=sizes,
                k=k,
                members=members,
                blocks=blocks,
                indptr=indptr,
                indices=indices,
                r_target=r_target
            )
            continue

    # ---------------- final safety pass: remove any undersized ---------------
    over_blocks = [b for b, s in sizes.items() if s > k]
    under_blocks = [b for b, s in sizes.items() if s < k]
    for b_dst in under_blocks:
        while sizes[b_dst] < k and len(over_blocks) > 0:
            b_src = over_blocks[-1]  # take from the end for efficiency
            if sizes[b_src] == k: # have we taken all we can?
                over_blocks.pop() # discount this block from further consideration
                continue

            # move arbitrary boundary or any vertex as last resort
            mv = _movable_vertex(b_src, {b_dst}, rng=rng, blocks=blocks,
                                  members=members, indptr=indptr, indices=indices)
            if mv is None:
                v = rng.choice(tuple(members[b_src]))
            else:
                v, _ = mv
            _move(v, b_src, b_dst, blocks=blocks, members=members, sizes=sizes)

        # if there are no over_blocks left, spread nodes from under_block randomly among other blocks
        if len(over_blocks) == 0:
            # assign all nodes in under_block to a random block
            # we are changing the direction of movement, so destination becomes source
            b_src = b_dst
            for v in members[b_dst]:
                non_under_blocks = {b for b, s in sizes.items() if s >= k}

                # check if v touches a non_under block

                mv = _movable_vertex(v, non_under_blocks, rng=rng, blocks=blocks,
                                        members=members, indptr=indptr, indices=indices)
                if mv is not None:
                    v, b_dst = mv
                else:
                    b_dst= rng.choice(tuple(blocks.values()))
                _move(v, b_src, b_dst, blocks=blocks, members=members, sizes=sizes)
            # we should have no members left in the given block, and it should be removed
            assert len(members[b_src]) == 0, \
                "Rebalance failed: block {b_src} still has members after rebalancing."
            del sizes[b_dst]

    # final check
    _, _, under = categorize(sizes=sizes, k=k)
    assert len(under) == 0, \
        f"Rebalance failed: {len(under)} blocks are still undersized (<{k})."

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
    # -----------------------------------------------------------------
    def compute_assignment(self) -> BlockData:
        """
        Compute a block assignment based on the proposed assignment.
        Currently, this method only performs a min_size balancing step.
        """
        if self.min_block_size is None:
            raise ValueError("min_block_size must be specified for UniformSmallBlockAssigner.")

        assignment = self._compute_assignment()
        assignment= _rebalance_to_min_size(
            blocks=assignment,
            adjacency=self.graph_data.adjacency,
            k=self.min_block_size,
            rng=self.rng,
            max_iter=None,  # data-driven max_iter (10*num_nodes)
        )
        reindexed_assignment = self.reindex_blocks(assignment)

        return BlockData(
            initial_blocks=reindexed_assignment,
            graph_data=self.graph_data
        )


class MetisBlockAssigner(BlockAssigner):
    """
    Use Metis to obtain a *balanced* `num_blocks`-way partition of the
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
        Convert a CSR adjacency matrix to the adjacency-list format Metis
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
    def _compute_metis_assignment(self) -> Dict[int, int]:
        """
        Run METIS and return a BlockData with the resulting assignment.

        May results in blocks of size < min_block_size or > max_block_size.
        """
        n = self.graph_data.num_nodes
        adj_lists = self._to_adj_lists(self.graph_data.adjacency)

        # PyMetis returns (edgecuts, membership-array)
        _, parts = metis.part_graph(
            graph=adj_lists,
            nparts = self.num_blocks,
            ufactor = 30, # aggresively balance block
            tpwgts = [1/self.num_blocks]*self.num_blocks, # equal weights for each block
            ubvec = [1+0.05], # 5% imbalance allowed
            recursive=True
        )

        # PyMetis guarantees |parts| == n
        blocks: Dict[int, int] = {node: part for node, part in enumerate(parts)}

        # Wrap in BlockData so downstream code can use it directly
        return blocks
    # -----------------------------------------------------------------
    def compute_assignment(self) -> BlockData:
        """
        Compute a balanced block assignment based on the proposed assignment.

        Currently, this method only performs a min_size balancing step.
        """
        if self.min_block_size is None:
            raise ValueError("min_block_size must be specified for MetisBlockAssigner.")

        assignment = self._compute_metis_assignment()
        assignment = _rebalance_to_min_size(
            blocks=assignment,
            adjacency=self.graph_data.adjacency,
            k=self.min_block_size,
            rng=self.rng,
            max_iter=None,  # data-driven max_iter (10*num_nodes)
        )
        assignment = self.reindex_blocks(assignment)

        return BlockData(
            initial_blocks=assignment,
            graph_data=self.graph_data
        )


class RefinedMetisBlockAssigner(MetisBlockAssigner):
    """PyMETIS seed ➜ improved greedy rebalance ➜ optional CP‑SAT polish."""

    def __init__(
        self,
        graph_data: GraphData,
        rng: np.random.Generator,
        num_blocks: int | None = None,
        min_block_size: int | None = None,
        max_block_size: int | None = None,
        cpsat_time_limit: int | None = 5,
    ) -> None:
        super().__init__(
            graph_data=graph_data,
            rng=rng,
            num_blocks=num_blocks,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
        )
        self._rng = rng
        self._cpsat_limit = cpsat_time_limit
    # -----------------------------------------------------------------
    def compute_assignment(self) -> BlockData:  # noqa: D401 – keep signature
        # unbalanced
        if self.min_block_size is None:
            raise ValueError("min_block_size must be specified for RefinedMetisBlockAssigner.")

        blocks = super()._compute_metis_assignment()
        blocks = _rebalance_to_min_size(
            blocks,
            self.graph_data.adjacency,
            self.min_block_size,
            rng=self._rng,
        )
        # polish with CP‑SAT
        blocks = self._cpsat_polish(blocks)
        return BlockData(initial_blocks=self.reindex_blocks(blocks),
                         graph_data=self.graph_data)
    # -----------------------------------------------------------------
    def _compute_metis_assignment(self) -> Dict[int, int]:  # noqa: D401  – keep name
        return super()._compute_metis_assignment()
    # -----------------------------------------------------------------
    def _block_members(self, blk: int, blocks: Dict[int, int]) -> List[int]:
        return [v for v, b in blocks.items() if b == blk]
    # -----------------------------------------------------------------
    def _cpsat_polish(self, blocks: Dict[int, int]) -> Dict[int, int]:
        """ 
        Polish the block assignment to decrease the edge cut while ensuring
        block sizes are within min_block_size and min_block_size + 1.
        """
        if self.min_block_size is None:
            return blocks

        k = self.min_block_size
        sizes = _block_sizes(blocks)
        wrong = {b for b, s in sizes.items() if not (k <= s <= k + 1)}

        if not wrong:
            return blocks  # already good

        # collect *boundary* nodes of wrong blocks + their neighbours
        boundary: Set[int] = set()
        adj = self.graph_data.adjacency
        indptr, indices = adj.indptr, adj.indices

        for b in wrong:
            for v in self._block_members(b, blocks):
                row = slice(indptr[v], indptr[v + 1])
                if any(blocks[u] != b for u in indices[row]):
                    boundary.add(v)
                    boundary.update(indices[row])

        sub_nodes = sorted(boundary)
        idx_of: Dict[int, int] = {v: i for i, v in enumerate(sub_nodes)}
        sub_adj = adj[sub_nodes][:, sub_nodes]  # type: ignore[index]
        # blocks involved
        blks_sub = {blocks[v] for v in sub_nodes}

        # ------- build CP‑SAT model ----------------------------------
        model = cp_model.CpModel()
        x = {}
        for v in sub_nodes:
            for b in blks_sub:
                x[v, b] = model.NewBoolVar(f"x_{v}_{b}")
            # each vertex exactly one block (in sub‑problem)
            model.Add(sum(x[v, b] for b in blks_sub) == 1)
        # block‑size constraints & t_b variables
        t = {}
        r_target = (len(sub_nodes) + sum(sizes[b] for b in blks_sub) - k * len(blks_sub))  # local oversize quota
        for b in blks_sub:
            t[b] = model.NewBoolVar(f"t_{b}")
            size_expr = sum(x[v, b] for v in sub_nodes) + (sizes[b] - sum(blocks[v] == b for v in sub_nodes))
            model.Add(size_expr == k + t[b])
        model.Add(sum(t[b] for b in blks_sub) == r_target)

        # edge‑cut objective (linearised y/z eliminated – constant perimeter suffices in subgraph)
        rows, cols = sub_adj.nonzero()
        z = {}
        for v, u in zip(rows, cols):
            if v >= u:
                continue  # undirected upper triangle
            i, j = sub_nodes[v], sub_nodes[u]
            z[(i, j)] = model.NewBoolVar(f"z_{i}_{j}")
            # z = 1 if endpoints differ
            for b in blks_sub:
                model.AddBoolAnd([x[i, b], x[j, b]]).OnlyEnforceIf(z[(i, j)].Not())
            # if all same‑block conjunctions false → z=1

        model.Minimize(sum(z.values()))

        # solve
        solver = cp_model.CpSolver()
        if self._cpsat_limit:
            solver.parameters.max_time_in_seconds = float(self._cpsat_limit)
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for v in sub_nodes:
                for b in blks_sub:
                    if solver.BooleanValue(x[v, b]):
                        blocks[v] = b
                        break

        return blocks


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
                    n_init=1,
                    max_iter=10,
                    tol=1e-3,
                    verbose=False,
                    random_state=self.rng.choice(2**32), 
                    copy_x=False, # perform centering
                    # use all available CPU cores
                    n_jobs=-1
                )
        tic = time()
        labels = kmeans.fit_predict(embeddings)
        toc = time()
        print(f"KMeans with constraints took {toc - tic:.2f} seconds for {self.graph_data.num_nodes} nodes.")

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
                    verbose=False
                )
        tic = time()
        embeddings = model.fit_transform(
            sp.csr_matrix(adjacency) # nodevectors expect a CSR matrix, and not array
            )
        toc = time()
        print(f"ProNE embedding took {toc - tic:.2f} seconds for {adjacency.shape[0]} nodes.") # type: ignore

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

        if init_method == "uniform":
            return UniformSmallBlockAssigner(
                graph_data=graph_data,
                rng=self.rng,
                min_block_size=min_block_size,
                max_block_size=max_block_size,
                num_blocks=num_blocks,
            )
        elif init_method == "prone_and_kmeans":
            return ProNEAndConstrKMeansAssigner(
                graph_data=graph_data,
                rng=self.rng,
                min_block_size=min_block_size,
                max_block_size=max_block_size,
                num_blocks=num_blocks,
            )
        elif init_method == "metis":
            return MetisBlockAssigner(
                graph_data=graph_data,
                rng=self.rng,
                num_blocks=num_blocks,
                min_block_size=min_block_size,
                max_block_size=max_block_size,
            )
        elif init_method == "metis_refine":
            return RefinedMetisBlockAssigner(
                graph_data=graph_data,
                rng=self.rng,
                min_block_size=min_block_size,
                max_block_size=max_block_size,
                num_blocks=num_blocks,
                cpsat_time_limit=10
            )
        else:
            raise ValueError(f"Unknown initialization method: {init_method}. "
                "Available methods: 'metis', 'uniform', 'prone_and_kmeans', 'metis_refine'."
                )

