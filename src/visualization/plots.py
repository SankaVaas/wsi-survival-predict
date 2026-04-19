"""
Visualisation utilities for WSI survival analysis results.

1. plot_km_curves         — Kaplan-Meier stratified survival curves
2. plot_training_history  — Loss and C-index curves across epochs
3. plot_attention_heatmap — Overlay patch attention weights on WSI thumbnail
4. plot_umap_features     — UMAP projection of patch features coloured by attention/survival
5. plot_heterogeneity_map — Spatial map of per-patch heterogeneity scores
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/Colab

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

PALETTE = {
    "high_risk":  "#D85A30",
    "low_risk":   "#1D9E75",
    "train":      "#378ADD",
    "val":        "#7F77DD",
    "attn_cmap":  "RdYlGn_r",
    "het_cmap":   "plasma",
}


def plot_km_curves(
    km_data: dict,
    save_path: Optional[str | Path] = None,
    title: str = "Kaplan-Meier Survival Curves",
) -> plt.Figure:
    """
    Plot Kaplan-Meier curves for high/low risk groups with 95% CI bands
    and log-rank p-value annotation.

    Args:
        km_data:    Output of metrics.km_stratification().
        save_path:  If provided, save figure to this path.
        title:      Figure title.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    kmf_high = km_data["kmf_high"]
    kmf_low  = km_data["kmf_low"]
    p_value  = km_data["logrank_p"]

    # High risk
    kmf_high.plot_survival_function(
        ax=ax, ci_show=True,
        color=PALETTE["high_risk"],
        ci_alpha=0.15,
        linewidth=2,
    )
    # Low risk
    kmf_low.plot_survival_function(
        ax=ax, ci_show=True,
        color=PALETTE["low_risk"],
        ci_alpha=0.15,
        linewidth=2,
    )

    # P-value annotation
    p_str = f"p = {p_value:.3e}" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(
        0.97, 0.95, p_str,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )

    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, linestyle="--")
    sns.despine(ax=ax)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_training_history(
    history: dict[str, list],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot training/validation loss and C-index curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], color=PALETTE["train"], label="Train", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"],   color=PALETTE["val"],   label="Val",   linewidth=1.8, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cox NLL Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2, linestyle="--")
    sns.despine(ax=axes[0])

    # C-index
    axes[1].plot(epochs, history["train_cindex"], color=PALETTE["train"], label="Train", linewidth=1.8)
    axes[1].plot(epochs, history["val_cindex"],   color=PALETTE["val"],   label="Val",   linewidth=1.8, linestyle="--")
    best_epoch = int(np.argmax(history["val_cindex"])) + 1
    axes[1].axvline(x=best_epoch, color="#888780", linestyle=":", linewidth=1.2, label=f"Best epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("C-index")
    axes[1].set_title("Concordance Index")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.2, linestyle="--")
    sns.despine(ax=axes[1])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_attention_heatmap(
    coords: np.ndarray,           # (N, 2)  patch (x, y) top-left in WSI pixels
    attention: np.ndarray,        # (N,)    attention weights [0, 1]
    thumbnail: Optional[np.ndarray] = None,  # (H, W, 3) WSI thumbnail
    patch_size: int = 256,
    save_path: Optional[str | Path] = None,
    title: str = "Attention Heatmap",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Overlay patch-level attention weights on the WSI thumbnail.

    Patches with high attention (morphologically heterogeneous) are
    highlighted in red/yellow; low-attention patches are shown in green/blue.

    Args:
        coords:     (N, 2) patch top-left coordinates in WSI pixel space.
        attention:  (N,)   normalised attention weights.
        thumbnail:  Optional WSI thumbnail as background.
        patch_size: Patch size in WSI pixel space.
        save_path:  Save path for figure.
        title:      Figure title.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if thumbnail is not None:
        ax.imshow(thumbnail, aspect="auto")
        wsi_w, wsi_h = thumbnail.shape[1], thumbnail.shape[0]
        # Scale coords to thumbnail dimensions
        x_max = coords[:, 0].max() + patch_size
        y_max = coords[:, 1].max() + patch_size
        scale_x = wsi_w / x_max
        scale_y = wsi_h / y_max
        coords_scaled = coords * np.array([scale_x, scale_y])
        patch_size_scaled_x = patch_size * scale_x
        patch_size_scaled_y = patch_size * scale_y
    else:
        coords_scaled = coords.copy()
        patch_size_scaled_x = patch_size
        patch_size_scaled_y = patch_size
        ax.set_facecolor("#f0f0f0")

    cmap = plt.get_cmap(PALETTE["attn_cmap"])
    attn_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    for i, (x, y) in enumerate(coords_scaled):
        color = cmap(float(attn_norm[i]))
        rect  = mpatches.Rectangle(
            (x, y), patch_size_scaled_x, patch_size_scaled_y,
            linewidth=0,
            edgecolor="none",
            facecolor=color,
            alpha=0.55,
        )
        ax.add_patch(rect)

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Attention weight", fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_umap_features(
    features: np.ndarray,          # (N, D) patch features
    colour_values: np.ndarray,     # (N,)   values to colour by (e.g. attention or heterogeneity)
    colour_label: str = "Attention",
    save_path: Optional[str | Path] = None,
    title: str = "UMAP of Patch Features",
    n_neighbours: int = 15,
    min_dist: float = 0.1,
) -> plt.Figure:
    """
    Project patch features to 2D with UMAP and colour by a scalar.
    Reveals morphological clusters in the tumour microenvironment.
    """
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Falling back to PCA.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(features)
    else:
        reducer = umap.UMAP(n_neighbors=n_neighbours, min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colour_values, cmap="plasma",
        s=4, alpha=0.7, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(colour_label, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_heterogeneity_map(
    coords: np.ndarray,       # (N, 2) — x, y patch positions
    entropy: np.ndarray,      # (N,)   — entropy heterogeneity score
    cosine_dissim: np.ndarray,# (N,)   — cosine dissimilarity score
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Spatial map of the two most interpretable heterogeneity scores:
    entropy and cosine dissimilarity. Side-by-side panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    titles  = ["Morphological Entropy",   "Cosine Dissimilarity to Neighbourhood"]
    values  = [entropy,                    cosine_dissim]
    cmaps   = ["magma",                    "viridis"]

    for ax, val, cmap_name, ttl in zip(axes, values, cmaps, titles):
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=val, cmap=cmap_name,
            s=6, alpha=0.8, linewidths=0,
        )
        ax.invert_yaxis()   # match image orientation
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        sns.despine(ax=ax)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
