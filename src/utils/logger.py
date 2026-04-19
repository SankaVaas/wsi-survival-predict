"""
Structured logging with Rich console + optional TensorBoard / W&B.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                            SpinnerColumn, TextColumn, TimeElapsedColumn,
                            TimeRemainingColumn)

console = Console()


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Return a Rich-formatted logger, optionally writing to a file as well."""
    handlers: list[logging.Handler] = [
        RichHandler(console=console, rich_tracebacks=True, markup=True)
    ]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    return logging.getLogger(name)


def get_progress_bar() -> Progress:
    """Return a Rich progress bar for long-running loops."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


class MetricTracker:
    """Accumulate and summarise scalar metrics per epoch."""

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}
        self._epoch_start: float = time.time()

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._data.setdefault(k, []).append(float(v))

    def mean(self, key: str) -> float:
        vals = self._data.get(key, [])
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict[str, float]:
        elapsed = time.time() - self._epoch_start
        result = {k: self.mean(k) for k in self._data}
        result["epoch_time_s"] = elapsed
        return result

    def reset(self) -> None:
        self._data.clear()
        self._epoch_start = time.time()
