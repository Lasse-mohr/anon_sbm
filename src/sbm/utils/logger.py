# sbm/utils/logger.py

import csv
import time
from pathlib import Path
from typing import Union, TextIO


class CSVLogger:
    """
    Minimal CSV logger for long-running SBM fits.

    Each row contains:
        iteration, elapsed_seconds, neg_log_likelihood,
        accept_rate_window, temperature

    Parameters
    ----------
    file : str | pathlib.Path | TextIO
        Where to write.  If a path is given and the file does not yet
        exist, a header row is written automatically.
    log_every : int
        Only rows for which ``iteration % log_every == 0`` are written.
    """

    header = [
        "iteration",
        "elapsed_seconds",
        "neg_log_likelihood",
        "accept_rate_window",
        "temperature",
    ]

    def __init__(self,
                 file: Union[str, Path, TextIO],
                 *,
                 log_every: int = 1000,
                 ):
        self.log_every = int(log_every)
        self._start = time.time()

        # if prior log file exists, delete
        if isinstance(file, (str, Path)):
            file = Path(file)
            if file.exists():
                file.unlink()

        # open the handle
        if isinstance(file, (str, Path)):
            self._own_handle = True
            path = Path(file)
            path.parent.mkdir(parents=True, exist_ok=True)
            first = not path.exists()
            self._fh = path.open("a", newline="")
            self._writer = csv.writer(self._fh)
            if first:
                self._writer.writerow(self.header)
        else:                                  # file-like object supplied
            self._own_handle = False
            self._fh: TextIO = file
            self._writer = csv.writer(self._fh)
            # assume caller already wrote header

    # -----------------------------------------------------------------
    def log(self,
            iteration: int,
            neg_loglike: float,
            accept_rate_window: float,
            temperature: float,
            ) -> None:
        """
        Append a row
        """
        elapsed = time.time() - self._start
        self._writer.writerow([
            iteration,
            f"{elapsed:.3f}",
            f"{neg_loglike:.6f}",
            f"{accept_rate_window:.6f}",
            f"{temperature:.6f}",
        ])
        self._fh.flush()

    # -----------------------------------------------------------------
    def close(self):
        if self._own_handle:
            self._fh.close()

    # allow usage as a context manager -------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
