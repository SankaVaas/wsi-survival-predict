"""
Graph construction from WSI patch features.

Each WSI becomes a graph G = (V, E) where:
  - V: patch nodes with HIPT feature vectors as node attributes
  - E: spatial edges from K-nearest neighbours in (x, y) coordinate space

Additionally, we compute per-node morphological heterogeneity features:
  - Local entropy of feature magnitudes within the K-neighbourhood
  - Cosine dissimilarity to neighbourhood centroid (how "outlier" is this patch?)
  - Nearest-neighbour feature distance std (morphological diversity signal)

These heterogeneity node features are concatenated to the HIPT embedding,
giving the graph a richer representation of intra-tumour heterogeneity —
the core architectural novelty of this project.

Output: PyTorch Geometric Data objects saved as .pt files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import track
from scipy.spatial import KDTree
from sklearn.preprocessing import normalize
from torch_geometric.data import Data

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)


# ─── Heterogeneity feature computation ────────────────────────────────────────

def _compute_heterogeneity_features(
    feats: np.ndarray,         # (N, D)
    coords: np.ndarray,        # (N, 2)
    k: int = 8,
    n_bins: int = 8,
) -> np.ndarray:
    """
    Compute per-node morphological heterogeneity features.

    For each node i, consider its K spatial neighbours. Compute:
      1. Entropy of L2-norm distribution in neighbourhood (richness of morphology)
      2. Mean cosine dissimilarity to neighbourhood centroid (isolation metric)
      3. Std of pairwise L2 distances in feature space (spread metric)

    Returns:
        het_feats: float32 (N, 3) — one row per node.
    """
    N = feats.shape[0]
    feats_norm = normalize(feats, norm="l2")         # L2-normalised rows

    # Build KD-tree on spatial coordinates for neighbourhood lookup
    tree = KDTree(coords.astype(float))
    _, nn_idx = tree.query(coords.astype(float), k=k + 1)  # +1 includes self
    nn_idx = nn_idx[:, 1:]  # exclude self → (N, k)

    entropy_vals    = np.zeros(N, dtype=np.float32)
    cosine_dissim   = np.zeros(N, dtype=np.float32)
    feat_spread     = np.zeros(N, dtype=np.float32)

    # L2 norms of all features
    l2_norms = np.linalg.norm(feats, axis=1)  # (N,)

    for i in range(N):
        nb_idx = nn_idx[i]                          # (k,)
        nb_feats_norm = feats_norm[nb_idx]          # (k, D)
        nb_norms      = l2_norms[nb_idx]            # (k,)

        # 1. Entropy of L2-norm histogram (morphological richness)
        hist, _ = np.histogram(nb_norms, bins=n_bins, density=False)
        hist    = hist.astype(float) + 1e-8
        hist   /= hist.sum()
        entropy_vals[i] = float(-np.sum(hist * np.log(hist + 1e-8)))

        # 2. Cosine dissimilarity to centroid
        centroid     = nb_feats_norm.mean(axis=0)
        centroid    /= (np.linalg.norm(centroid) + 1e-8)
        node_feat_n  = feats_norm[i]
        cos_sim      = float(np.dot(node_feat_n, centroid))
        cosine_dissim[i] = 1.0 - cos_sim  # dissimilarity ∈ [0, 2]

        # 3. Std of pairwise feature-space L2 distances among neighbours
        if k > 1:
            pairwise_dists = []
            for a in range(k):
                for b in range(a + 1, k):
                    d = float(np.linalg.norm(feats[nb_idx[a]] - feats[nb_idx[b]]))
                    pairwise_dists.append(d)
            feat_spread[i] = float(np.std(pairwise_dists)) if pairwise_dists else 0.0

    het_feats = np.stack([entropy_vals, cosine_dissim, feat_spread], axis=1)  # (N, 3)
    return het_feats


# ─── Graph construction ────────────────────────────────────────────────────────

def build_graph_for_slide(
    feat_h5_path: str | Path,
    out_dir: str | Path,
    clinical_row: Optional[dict] = None,
    k_neighbors: int = 8,
    n_het_bins: int  = 8,
) -> Optional[Path]:
    """
    Build a PyTorch Geometric graph for one slide.

    Node features: concat(HIPT_feats [D], heterogeneity_feats [3])
    Edge index:    spatial KNN edges (undirected, bidirectional)
    Edge attr:     Euclidean distance in coordinate space (normalised)
    Graph labels:  survival_months, event (from clinical_row)

    Saves to: out_dir/<slide_id>.pt

    Returns:
        Path to saved .pt file, or None if failed.
    """
    feat_h5_path = Path(feat_h5_path)
    out_dir      = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_id = feat_h5_path.stem
    out_path = out_dir / f"{slide_id}.pt"

    if out_path.exists():
        return out_path

    # ── Load features ──────────────────────────────────────────────────────
    with h5py.File(str(feat_h5_path), "r") as f:
        feats  = f["feats"][:]    # (N, D) float32
        coords = f["coords"][:]   # (N, 2) int32

    N = feats.shape[0]
    if N < k_neighbors + 1:
        console.print(f"[yellow]Skipping {slide_id}: only {N} patches (need >{k_neighbors})[/yellow]")
        return None

    # ── Heterogeneity node features ────────────────────────────────────────
    het_feats = _compute_heterogeneity_features(feats, coords, k=k_neighbors, n_bins=n_het_bins)

    # Normalise heterogeneity features to [0,1] per dimension
    het_min = het_feats.min(axis=0, keepdims=True)
    het_max = het_feats.max(axis=0, keepdims=True)
    het_feats = (het_feats - het_min) / (het_max - het_min + 1e-8)

    # Concatenate HIPT + heterogeneity → node feature matrix
    node_feats = np.concatenate([feats, het_feats], axis=1).astype(np.float32)  # (N, D+3)

    # ── Build KNN spatial edge index ───────────────────────────────────────
    tree = KDTree(coords.astype(float))
    _, nn_idx = tree.query(coords.astype(float), k=k_neighbors + 1)
    nn_idx = nn_idx[:, 1:]  # exclude self

    src_list, dst_list, dist_list = [], [], []
    coord_norms = np.linalg.norm(coords.astype(float), axis=1).max() + 1e-8

    for i in range(N):
        for j in nn_idx[i]:
            if j < N:
                src_list.append(i)
                dst_list.append(j)
                # Normalised spatial distance as edge attribute
                d = float(np.linalg.norm(coords[i].astype(float) - coords[j].astype(float)))
                dist_list.append(d / coord_norms)

    # Make undirected: add reverse edges
    src_all  = src_list + dst_list
    dst_all  = dst_list + src_list
    dist_all = dist_list + dist_list

    edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
    edge_attr  = torch.tensor(dist_all, dtype=torch.float32).unsqueeze(1)

    # ── Build PyG Data object ──────────────────────────────────────────────
    data = Data(
        x          = torch.from_numpy(node_feats),     # (N, D+3)
        edge_index = edge_index,                        # (2, E)
        edge_attr  = edge_attr,                         # (E, 1)
        coords     = torch.from_numpy(coords.astype(np.float32)),  # (N, 2)
        slide_id   = slide_id,
        n_patches  = N,
    )

    # ── Attach survival labels if provided ────────────────────────────────
    if clinical_row is not None:
        data.survival_months = torch.tensor(
            [float(clinical_row.get("survival_months", -1))], dtype=torch.float32
        )
        data.event = torch.tensor(
            [int(clinical_row.get("event", 0))], dtype=torch.long
        )
        data.case_id = clinical_row.get("case_id", "")

    torch.save(data, str(out_path))
    return out_path


@app.command()
def run(
    feat_dir:    Path  = typer.Argument(..., help="Directory of feature HDF5 files"),
    clinical_csv: Path = typer.Argument(..., help="Clinical CSV with case_id, survival_months, event"),
    out_dir:     Path  = typer.Option(Path("data/graphs"), help="Graph .pt output dir"),
    k_neighbors: int   = typer.Option(8,  help="Spatial KNN degree"),
    n_het_bins:  int   = typer.Option(8,  help="Histogram bins for entropy features"),
) -> None:
    """Build PyTorch Geometric graphs for all feature HDF5 files."""
    import pandas as pd

    clinical_df = pd.read_csv(str(clinical_csv))
    # Build lookup: file_name stem → clinical row
    if "file_name" in clinical_df.columns:
        clinical_df["slide_id"] = clinical_df["file_name"].str.replace(".svs", "", regex=False)
    clinical_lookup = clinical_df.set_index("slide_id").to_dict(orient="index")

    h5_files = sorted(Path(feat_dir).glob("*.h5"))
    console.rule(f"[bold blue]Graph Builder · {len(h5_files)} slides")

    success, skipped = 0, 0
    for h5_path in track(h5_files, description="Building graphs…"):
        slide_id    = h5_path.stem
        clinical_row = clinical_lookup.get(slide_id)
        if clinical_row is None:
            # Try to match by submitter prefix
            for key in clinical_lookup:
                if slide_id.startswith(key[:12]):
                    clinical_row = clinical_lookup[key]
                    break

        if clinical_row is None:
            skipped += 1
            continue

        result = build_graph_for_slide(
            feat_h5_path=h5_path,
            out_dir=out_dir,
            clinical_row=clinical_row,
            k_neighbors=k_neighbors,
            n_het_bins=n_het_bins,
        )
        if result:
            success += 1
        else:
            skipped += 1

    console.print(f"\n[green]Graphs built:[/green] {success}  [yellow]Skipped:[/yellow] {skipped}")


if __name__ == "__main__":
    app()
