"""
PyTorch Geometric dataset for WSI survival graphs.

Reads pre-built .pt graph files (output of graph_builder.py) and
provides train/val/test splits stratified by event status.

Features:
  - Stratified split by event (ensures event rate balance across splits)
  - Optional on-the-fly data augmentation (node dropout, edge dropout)
  - Handles missing survival labels gracefully (inference mode)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from rich.console import Console

console = Console()


class WSISurvivalDataset(Dataset):
    """
    In-memory WSI graph survival dataset.

    Args:
        graph_paths:  List of paths to .pt graph files.
        augment:      Apply random node/edge dropout (training only).
        node_drop:    Fraction of nodes to randomly drop (augmentation).
        edge_drop:    Fraction of edges to randomly drop (augmentation).
    """

    def __init__(
        self,
        graph_paths: list[Path],
        augment: bool       = False,
        node_drop: float    = 0.1,
        edge_drop: float    = 0.1,
    ) -> None:
        super().__init__()
        self.graph_paths = [Path(p) for p in graph_paths]
        self.augment     = augment
        self.node_drop   = node_drop
        self.edge_drop   = edge_drop

        # Pre-load all graphs into memory (feasible for <1000 slides)
        # For very large cohorts, comment out and load lazily in get()
        console.print(f"  [cyan]Loading {len(self.graph_paths)} graphs into memory…[/cyan]")
        self._graphs: list[Data] = []
        missing = 0
        for p in self.graph_paths:
            try:
                g = torch.load(str(p), map_location="cpu")
                self._graphs.append(g)
            except Exception as e:
                console.print(f"[yellow]Could not load {p.name}: {e}[/yellow]")
                missing += 1
        if missing:
            console.print(f"[yellow]  {missing} graphs could not be loaded.[/yellow]")

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        data = self._graphs[idx]

        if self.augment:
            data = self._augment(data)

        return data

    def _augment(self, data: Data) -> Data:
        """Random node dropout and edge dropout for regularisation."""
        data = data.clone()

        # Node dropout: randomly zero out a fraction of node features
        if self.node_drop > 0 and data.x is not None:
            N = data.x.size(0)
            mask = torch.rand(N) > self.node_drop
            data.x = data.x * mask.float().unsqueeze(1)

        # Edge dropout: randomly remove edges
        if self.edge_drop > 0 and data.edge_index is not None:
            E = data.edge_index.size(1)
            keep = torch.rand(E) > self.edge_drop
            data.edge_index = data.edge_index[:, keep]
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[keep]

        return data

    def get_survival_arrays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (survival_months, events) arrays for all samples."""
        times  = torch.stack([g.survival_months.squeeze() for g in self._graphs])
        events = torch.stack([g.event.squeeze().float()   for g in self._graphs])
        return times, events

    def get_slide_ids(self) -> list[str]:
        return [g.slide_id if hasattr(g, "slide_id") else str(i)
                for i, g in enumerate(self._graphs)]


# ─── Dataset builder + data loaders ───────────────────────────────────────────

def build_datasets(
    graph_dir: str | Path,
    val_fraction: float   = 0.15,
    test_fraction: float  = 0.15,
    seed: int             = 42,
    node_drop: float      = 0.1,
    edge_drop: float      = 0.1,
) -> dict[str, WSISurvivalDataset]:
    """
    Load all .pt graphs from graph_dir, split into train/val/test
    stratified by event status.

    Returns:
        {"train": ..., "val": ..., "test": ...}
    """
    graph_paths = sorted(Path(graph_dir).glob("*.pt"))
    if not graph_paths:
        raise FileNotFoundError(f"No .pt graph files found in {graph_dir}")

    console.print(f"[cyan]Found {len(graph_paths)} graph files in {graph_dir}[/cyan]")

    # Load events for stratification (fast — only load labels)
    events = []
    valid_paths = []
    for p in graph_paths:
        try:
            g = torch.load(str(p), map_location="cpu")
            if hasattr(g, "event") and hasattr(g, "survival_months"):
                events.append(int(g.event.item()))
                valid_paths.append(p)
        except Exception:
            continue

    console.print(
        f"  Valid graphs: {len(valid_paths)} "
        f"(events: {sum(events)}, censored: {len(events)-sum(events)})"
    )

    # Stratified train/(val+test) split
    train_paths, valtest_paths, train_events, valtest_events = train_test_split(
        valid_paths, events,
        test_size=val_fraction + test_fraction,
        stratify=events,
        random_state=seed,
    )

    # Split val+test
    val_rel = val_fraction / (val_fraction + test_fraction)
    val_paths, test_paths = train_test_split(
        valtest_paths,
        test_size=1.0 - val_rel,
        stratify=valtest_events,
        random_state=seed,
    )

    console.print(
        f"  Split → train: {len(train_paths)}, "
        f"val: {len(val_paths)}, test: {len(test_paths)}"
    )

    return {
        "train": WSISurvivalDataset(train_paths, augment=True,  node_drop=node_drop, edge_drop=edge_drop),
        "val":   WSISurvivalDataset(val_paths,   augment=False),
        "test":  WSISurvivalDataset(test_paths,  augment=False),
    }


def get_dataloaders(
    datasets: dict[str, WSISurvivalDataset],
    batch_size: int = 1,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create PyG DataLoaders for each split."""
    return {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for split, ds in datasets.items()
    }
