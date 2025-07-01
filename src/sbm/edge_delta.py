"""
Classes to build and hold changes in edge counts between blocks in a Stochastic Block Model (SBM).
"""

from typing import DefaultDict, Tuple, List, Literal, Tuple, Iterator, Iterable, Literal
from collections import defaultdict, Counter

import numpy as np
from numba import jit

EdgeDeltas = Literal["PythonEdgeDelta", "NumpyEdgeDelta"]

#### Pure python class for edge deltas #######
class EdgeDelta: # edge-count changes between blocks
    def __init__(self, n_blocks: int):
        self._deltas: DefaultDict[Tuple[int, int], int] = defaultdict(int)

    def _increment(self, count: int, block_i: int, block_j: int,
     ) -> None:
        """
        Increment the edge count delta for a pair of blocks.

        :param count: The change in edge count.
        :param block_i: The first block index.
        :param block_j: The second block index.
        :return: Updated edge count delta.
        """
        if block_i < block_j:
            self._deltas[(block_i, block_j)] = count
        else:
            self._deltas[(block_j, block_i)] = count
    
    def __getitem__(self, pair: Tuple[int, int]) -> int:
        """
        Get the edge count delta for a pair of blocks.

        :param pair: A tuple containing the block indices (i, j).
        :return: The edge count delta for the pair.
        """
        if pair[0] < pair[1]:
            return self._deltas.get(pair, 0)
        else:
            return self._deltas.get((pair[1], pair[0]), 0)
    
    def __len__(self) -> int:
        """
        Return the number of non-zero edge count deltas.

        :return: The number of non-zero edge count deltas.
        """
        return len([v for v in self._deltas.values() if v != 0])
    
    def items(self) -> Iterator[Tuple[Tuple[int, int], int]]:
        """
        Yield tuple ((i, j), delta_e) for all stored pairs.

        :return: An iterator over tuples of (block_i, block_j, delta_e).
        """
        for (i, j), delta_e in self._deltas.items():
            yield (i, j), delta_e

    def increment(self,
                  counts: Iterable[int],
                  blocks_i: Iterable[int],
                  blocks_j: Iterable[int],
     ) -> None:
        """
        Increment the edge counts deltas for a list of block pairs.

        :param counts: List of changes in edge counts.
        :param blocks_i: List of first block indices.
        :param blocks_j: List of second block indices.
        """
        for count, block_i, block_j in zip(counts, blocks_i, blocks_j):
            self._increment(count, block_i, block_j)

##### NumPy class for edge deltas ######
class NumpyEdgeDelta(EdgeDelta):
    """Sparse, symmetric (i <= j) container for edge‑count deltas.

    Overwrites the pure‑Python :py:class:`EdgeDelta` class

    Internally stores three *contiguous* one‑dimensional NumPy arrays
    (`rows`, `cols`, `data`) in **COO** fashion as well as a Python
    ``dict`` that maps the linearised pair key ``i * n_blocks + j`` to the
    corresponding position in the arrays.  Only the *active* prefix
    (``self.size``) of the arrays is considered valid – this makes the
    structure friendly to Numba‐JIT’d consumers that expect fixed‑size
    buffers.
    
    The class focuses on *fast incremental updates* (``O(1)`` expected)
    and cheap vector export; memory usage is proportional to the number
    of *non‑zero* block pairs actually visited by the MCMC chain.
    """

    __slots__ = ("n_blocks", "rows", "cols", "data", "size", "_key2idx")

    def __init__(self,
                 n_blocks: int,
                 initial_capacity: int = 64
    ):
        self.n_blocks: int = int(n_blocks)
        cap = max(1, initial_capacity)
        self.rows: np.ndarray = np.empty(cap, dtype=np.int32)
        self.cols: np.ndarray = np.empty(cap, dtype=np.int32)
        self.data: np.ndarray = np.zeros(cap, dtype=np.int32)
        self.size: int = 0

        # auxiliary map for *O(1)* lookup – not accessed inside JIT code
        self._key2idx: dict[int, int] = {}

    ### function for printing the object
    def __repr__(self) -> str:
        """Return a string representation of the NumpyEdgeDelta object."""
        return (f"NumpyEdgeDelta(n_blocks={self.n_blocks}, "
                f"size={self.size}, "
                f"rows={self.rows[:self.size]}, "
                f"cols={self.cols[:self.size]}, "
                f"data={self.data[:self.size]})")
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _encode(self, i: int, j: int) -> int:
        """Encode an ordered pair (i ≤ j) into a unique scalar."""
        return i * self.n_blocks + j

    def _ensure_capacity(self):
        if self.size == len(self.rows):
            # double in‑place (amortised O(1))
            new_cap = len(self.rows) * 2
            self.rows = np.resize(self.rows, new_cap)
            self.cols = np.resize(self.cols, new_cap)
            self.data = np.resize(self.data, new_cap)

    def _increment(self, count: int, block_i: int, block_j: int):
        """Add *value* to entry (i, j) (symmetric pair)."""
        if block_i > block_j:
            block_i, block_j = block_j, block_i
        key = self._encode(block_i, block_j)
        idx = self._key2idx.get(key)

        if idx is None:
            self._ensure_capacity()
            idx = self.size
            self.size += 1
            self.rows[idx] = block_i
            self.cols[idx] = block_j
            self.data[idx] = count
            self._key2idx[key] = idx
        else:
            self.data[idx] += count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __getitem__(self, pair: Tuple[int, int]) -> int:
        i, j = pair
        if i > j:
            i, j = j, i
        idx = self._key2idx.get(self._encode(i, j))
        return 0 if idx is None else int(self.data[idx])
    
    def __len__(self) -> int:
        """Return the number of non-zero and *active* pairs."""
        #return self.size
        active_pairs = self.data[:self.size]
        print(f"Active pairs: {active_pairs}")
        return active_pairs[active_pairs != 0].shape[0]

    def __setitem__(self, pair: Tuple[int, int], value: int):
        i, j = pair
        current = self[pair]
        self._increment(i, j, value - current)

    def items(self) -> Iterator[Tuple[Tuple[int, int], int]]:
        """Yield triples ``(i, j, delta_e)`` for all stored pairs."""
        for k in range(self.size):
            yield (int(self.rows[k]), int(self.cols[k])), int(self.data[k])

    def to_coo(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the *active* COO view (no copying)."""
        return (self.rows[:self.size], self.cols[:self.size], self.data[:self.size])

    def increment(self,
            counts: Iterable[int],
            blocks_i: Iterable[int],
            blocks_j: Iterable[int],
     ) -> None:
        """Vectorised equivalent of ``increment`` for ``NumpyEdgeDelta``.

        Parameters
        ----------
        counts : 1‑D ``int`` array
            Changes in edge counts (positive or negative).
        blocks_i, blocks_j : 1‑D ``int`` arrays
            Block indices *parallel* to ``counts``.

        Notes
        -----
        The function works fully in **NumPy** space – no Python loops – by
        linearising the symmetric pair ``(i, j)`` into a *key* and then
        accumulating duplicate keys with :pyfunc:`numpy.add.at`.

        """
        # ------------------------------------------------------------------
        # Ensure ndarray inputs (copy=False promotes views)
        # ------------------------------------------------------------------
        assert isinstance(counts, (list, np.ndarray)) and \
                isinstance(blocks_i, (list, np.ndarray)) and \
                isinstance(blocks_j, (list, np.ndarray)), \
            "Counts and block indices must be list or ndarray."

        counts = np.asarray(counts, dtype=np.int32)
        blocks_i = np.asarray(blocks_i, dtype=np.int32)
        blocks_j = np.asarray(blocks_j, dtype=np.int32)

        # ------------------------------------------------------------------
        # Normalise the pair ordering so that i ≤ j
        # ------------------------------------------------------------------
        swap_mask = blocks_i > blocks_j
        if swap_mask.any():
            blocks_i, blocks_j = blocks_i.copy(), blocks_j.copy()  # avoid aliasing
            blocks_i[swap_mask], blocks_j[swap_mask] = blocks_j[swap_mask], blocks_i[swap_mask]

        # ------------------------------------------------------------------
        # Encode pairs → scalar keys and reduce duplicates in *one* pass
        # ------------------------------------------------------------------
        n_blocks = self.n_blocks
        keys = blocks_i.astype(np.int64) * n_blocks + blocks_j

        # ``np.unique`` already sorts – good for cache locality
        uniq_keys, inverse = np.unique(keys, return_inverse=True)
        reduced = np.zeros_like(uniq_keys, dtype=np.int32)
        np.add.at(reduced, inverse, counts)

        # ------------------------------------------------------------------
        # Decode unique keys and perform bulk update via the fast method
        # ------------------------------------------------------------------
        rows = (uniq_keys // n_blocks).astype(np.int32)
        cols = (uniq_keys %  n_blocks).astype(np.int32)

        for r, c, dv in zip(rows, cols, reduced):
            if dv != 0:
                self._increment(block_i=int(r), block_j=int(c), count=int(dv))