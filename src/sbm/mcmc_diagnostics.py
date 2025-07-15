
# sbm/mcmc_diagnostics.py
import numpy as np
import tensorflow_probability as tfp
from typing import Optional

class OnlineDiagnostics:
    """
    Keeps a rolling window of the last `window` draws *per chain* and
    computes
      • R̂  (potential scale‑reduction)  — via tfp.mcmc.potential_scale_reduction
      • ESS (effective sample size)     — via tfp.mcmc.effective_sample_size
    """

    def __init__(self, window: int = 4000, n_chain: int = 1):
        self.window   = int(window)
        self.n_chain  = int(n_chain)
        self._buf:    Optional[np.ndarray] = None   # (chain, window, stat)
        self._write   = 0
        self._filled  = False

    # ------------------------------------------------------------------
    def _ensure_buffer(self, n_stat: int):
        if self._buf is None:
            self._buf = np.empty(
                (self.n_chain, self.window, n_stat), dtype=np.float64)

    # ------------------------------------------------------------------
    def update(self, logp, m_diag, m_off, chain: int = 0):
        """
        Parameters
        ----------
        logp   : float
        m_diag : np.ndarray  shape (B,)
        m_off  : np.ndarray  shape (P,)
        chain  : int         which chain is calling (0 by default)
        """
        vec = np.concatenate([[logp], m_diag, m_off]).astype(np.float64)
        self._ensure_buffer(vec.size)

        if self._buf is None:
            raise RuntimeError("Buffer not initialized. Call `update` with valid data first.")
        self._buf[chain, self._write] = vec
        # advance circular pointer only once *all* chains have written
        if chain == self.n_chain - 1:
            self._write = (self._write + 1) % self.window
            if self._write == 0:
                self._filled = True

    # ------------------------------------------------------------------
    def _ordered_block(self):
        """Return draws in chronological order, shape (chain, draw, stat)."""
        if self._buf is None:
            raise RuntimeError("Buffer not initialized. Call `update` with valid data first.")

        if not self._filled:
            return self._buf[:, :self._write, :]
        a = self._buf[:, self._write:, :]
        b = self._buf[:, :self._write, :]
        return np.concatenate([a, b], axis=1)

    # ------------------------------------------------------------------
    def summary(self):
        """Return (max‑R̂, min‑ESS).  NaN until buffer is full."""
        if not self._filled or self._buf is None:
            return np.nan, np.nan

        block = self._ordered_block()          # (chain, draw, stat)
        rhat = tfp.mcmc.potential_scale_reduction(
            block, independent_chain_ndims=1
        ).numpy().max()

        ess  = tfp.mcmc.effective_sample_size(
            block, filter_beyond_positive_pairs=True
        ).numpy().min()

        return float(rhat), float(ess)
