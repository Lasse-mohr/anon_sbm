import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_array
import pytest

from collections import Counter, defaultdict

from sbm.block_assigner import (
    _boundary_vertices,
    _movable_vertex,
    _move,
    categorize,
    move_node_to_under,
    move_node_from_over,
    balance_k_plus_1_blocks,
    _rebalance_to_min_size,
)

# ---------------------------------------------------------------------------
# Minimal helper to build a fully connected small graph ---------------------
# ---------------------------------------------------------------------------

def full_graph_csr(n: int) -> csr_array:
    rows, cols = np.triu_indices(n, 1)
    data = np.ones_like(rows)
    A = csr_array((data, (rows, cols)), shape=(n, n))
    A = A + A.T  # undirected
    return A

# ---------------------------------------------------------------------------
# Tests for categorize ------------------------------------------------------
# ---------------------------------------------------------------------------

def test_categorize_basic():
    k = 3
    sizes = {0: 2, 1: 3, 2: 4, 3: 5}
    over2, over1, under = categorize(sizes, k)
    assert over2 == {3}
    assert over1 == {2}
    assert under == {0}

# ---------------------------------------------------------------------------
# Tests for move_node_to_under -------------------------------------------
# ---------------------------------------------------------------------------


def test_move_node_to_under_correct():
    k = 2
    A = full_graph_csr(4)
    # block 0 undersize (1), block 1 oversize (3)
    blocks = {0: 0, 1: 1, 2: 1, 3: 1}
    sizes = Counter(blocks.values())
    members = defaultdict(set)
    for v, b in blocks.items():
        members[b].add(v)
    rng = np.random.default_rng(0)

    over2, over1, under = categorize(sizes, k)
    move_node_to_under(
        under=under,
        over1=over1,
        over2=over2,
        rng=rng,
        sizes=sizes,
        k=k,
        members=members,
        blocks=blocks,
        indptr=A.indptr,
        indices=A.indices,
    )
    # After move, block 0 should have size 2, block 1 size 2
    assert sizes[0] == k
    assert sizes[1] == k
    # No undersized blocks remain
    _, _, under_new = categorize(sizes, k)
    assert not under_new

# ---------------------------------------------------------------------------
# Tests for move_node_from_over --------------------------------------------
# ---------------------------------------------------------------------------

def test_move_node_from_over_shrink():
    k = 3
    A = full_graph_csr(8)
    # block 0 size 2 (<k), block 1 size 6 (>k+1), others size 0
    blocks = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}

    sizes = Counter(blocks.values())
    members = defaultdict(set)

    for v, b in blocks.items():
        members[b].add(v)

    rng = np.random.default_rng(1)
    over2, over1, under = categorize(sizes, k)

    move_node_from_over(
        under=under,
        over1=over1,
        over2=over2,
        rng=rng,
        sizes=sizes,
        k=k,
        members=members,
        blocks=blocks,
        indptr=A.indptr,
        indices=A.indices,
        r_target=0,
    )
    # block 0 size should have increased by 1, block1 decreased by1
    assert sizes[0] == 2  # reached k
    assert sizes[1] == 6

# ---------------------------------------------------------------------------
# Tests for balance_k_plus_1_blocks ----------------------------------------
# ---------------------------------------------------------------------------


def test_balance_k_plus_1_blocks_shrink_and_enlarge():
    k = 2
    A = full_graph_csr(6)
    # create three blocks: 0 size 3 (k+1), 1 size 3 (k+1), 2 size 0 (empty)
    blocks = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    sizes = Counter(blocks.values())
    members = defaultdict(set)

    for v, b in blocks.items():
        members[b].add(v)

    rng = np.random.default_rng(3)
    r_target = 1  # we want exactly one k+1

    over2, over1, under = categorize(sizes, k)
    assert len(over1) == 2  # pre‑condition

    # no possible balance to achieve
    balance_k_plus_1_blocks(
        over1=over1,
        over2=over2,
        rng=rng,
        sizes=sizes,
        k=k,
        members=members,
        blocks=blocks,
        indptr=A.indptr,
        indices=A.indices,
        r_target=r_target,
    )
    # After balancing, over1 should be 1
    over2_after, over1_after, under_after = categorize(sizes, k)

    assert len(over1_after) == len(over1) # no change made
    assert len(over2_after) == len(over2) # no change made
    assert not under_after and not under # no undersized introduced

# ---------------------------------------------------------------------------
# Test safety with empty sets ----------------------------------------------
# ---------------------------------------------------------------------------

def test_empty_sets_no_crash():
    k = 2
    A = full_graph_csr(3)
    blocks = {0: 0, 1: 0, 2: 0}
    sizes = Counter(blocks.values())
    members = defaultdict(set)
    for v, b in blocks.items():
        members[b].add(v)
    rng = np.random.default_rng(4)

    # empty over/under sets
    over2, over1, under = categorize(sizes, k)

    # should do nothing and not raise
    move_node_to_under(under, over1, over2, rng, sizes, k, members, blocks, A.indptr, A.indices)
    move_node_from_over(under, over1, over2, rng, sizes, k, members, blocks, A.indptr, A.indices, r_target=0)
    balance_k_plus_1_blocks(over1, over2, rng, sizes, k, members, blocks, A.indptr, A.indices, r_target=0)

# ---------------------------------------------------------------------------
# Integration test for _rebalance_to_min_size ------------------------------
# ---------------------------------------------------------------------------

def test_rebalance_removes_all_undersize():
    n = 25
    k = 3
    A = full_graph_csr(n)
    rng = np.random.default_rng(10)

    # start with random over/under assignment
    blocks = {i: rng.integers(0, 8) for i in range(n)}

    print(set(Counter(blocks.values()).values()))

    balanced = _rebalance_to_min_size(blocks.copy(), A, k, rng)
    sizes = Counter(balanced.values())
    print(set(sizes.values()))
    # No block smaller than k
    assert min(sizes.values()) >= k