"""
sbm.utils.logger
=================
Enhanced CSV logger that also records online convergence diagnostics
(R̂ and ESS) produced by :pyclass:`sbm.diagnostics.OnlineDiagnostics`.

* The monitoring class keeps only *running* summaries, so you need an
  external sink if you want to inspect those values later or feed them
  into your existing visualisation pipeline.

Row schema
----------
```
iteration, elapsed_seconds, neg_log_likelihood,
accept_rate_window, temperature,
rhat_max, ess_min
```
If you pass ``None`` for the diagnostic fields the columns are left
empty, so existing parsing code keeps working when the diagnostics are
switched off.
"""

import csv
import time
from pathlib import Path
from typing import Union, TextIO, Optional

__all__ = ["CSVLogger"]

class CSVLogger:
    """Light‑weight CSV logger for long‐running SBM fits.

    Parameters
    ----------
    file
        Path to a CSV file *or* an already opened file handle.  If a path
        is given and the file exists it will be **overwritten** so that
        every fit starts with a clean log.
    log_every
        Only every ``log_every``‑th call to :py:meth:`log` results in a
        new row.  Use this when your chain runs millions of sweeps but
        you only need a coarse‑grained trace on disk.
    """

    header = [
        "iteration",
        "elapsed_seconds",
        "neg_log_likelihood",
        "accept_rate_window",
        "temperature",
        "rhat_max",   # max split‑R̂ across monitored scalars
        "ess_min",    # min bulk/tail ESS across monitored scalars
    ]

    # ---------------------------------------------------------------------
    def __init__(
        self,
        file: Union[str, Path, TextIO],
        *,
        log_every: int = 1_000,
    ):
        self.log_every = int(log_every)
        self._start = time.time()

        # Ensure we own a fresh handle ------------------------------------------------
        if isinstance(file, (str, Path)):
            file = Path(file)
            if file.exists():
                file.unlink()  # start from scratch every run

        # Open handle + CSV writer ----------------------------------------------------
        if isinstance(file, (str, Path)):
            self._own_handle = True
            path = Path(file)
            path.parent.mkdir(parents=True, exist_ok=True)
            first = not path.exists()
            self._fh: TextIO = path.open("a", newline="")
            self._writer = csv.writer(self._fh)
            if first:
                self._writer.writerow(self.header)
        else:  # already a file‑like object
            self._own_handle = False
            self._fh = file
            self._writer = csv.writer(self._fh)
            # assume caller has written the header

        self._iteration_since_flush = 0

    # ------------------------------------------------------------------
    def log(
        self,
        iteration: int,
        neg_loglike: float,
        accept_rate_window: float,
        temperature: float,
        *,
        rhat_max: Optional[float] = None,
        ess_min: Optional[float] = None,
    ) -> None:
        """Append one new row if ``iteration`` meets the cadence.

        The diagnostics fields are *optional* so you can turn them off
        without touching call‑sites.
        """
        if iteration % self.log_every:
            return  # skip — not a checkpoint

        elapsed = time.time() - self._start
        self._writer.writerow([
            iteration,
            f"{elapsed:.3f}",
            f"{neg_loglike:.6f}",
            f"{accept_rate_window:.6f}",
            f"{temperature:.6f}",
            f"{rhat_max:.5f}" if rhat_max is not None else "",
            f"{ess_min:.1f}"   if ess_min  is not None else "",
        ])
        # Flush every ~10 rows to amortise disk writes.
        self._iteration_since_flush += 1
        if self._iteration_since_flush >= 10:
            self._fh.flush()
            self._iteration_since_flush = 0

    # ------------------------------------------------------------------
    def close(self):
        if self._own_handle:
            self._fh.close()

    # Context‑manager sugar ------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
