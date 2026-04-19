"""
Tissue-aware patch extraction from whole-slide images (SVS/TIFF).

Pipeline:
  1. Load WSI thumbnail at low magnification.
  2. Run Otsu thresholding on grayscale to produce a tissue mask.
  3. Sample up to `patches_per_slide` non-overlapping patch coordinates
     from tissue-positive regions (using the mask at target magnification).
  4. Extract patches at the target magnification and save as HDF5
     (one HDF5 per slide: datasets "patches" [N,H,W,3] and "coords" [N,2]).

HDF5 format (per slide):
    /patches  : uint8  (N, patch_size, patch_size, 3)
    /coords   : int32  (N, 2)   — (x, y) top-left at target magnification
    /attrs    : slide_id, magnification, patch_size
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import typer
from PIL import Image
from rich.console import Console
from rich.progress import track

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    console.print("[yellow]openslide-python not found. Install openslide system library first.[/yellow]")


def _otsu_tissue_mask(thumbnail: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """
    Convert thumbnail to grayscale, run Otsu thresholding.
    Returns boolean mask (True = tissue).

    Args:
        thumbnail:  RGB numpy array (H, W, 3).
        threshold:  Fraction of non-white pixels to qualify as tissue.
    Returns:
        bool mask  (H_thumb, W_thumb)
    """
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    # Otsu on inverted (tissue is dark on white background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return (binary / 255).astype(bool)


def _sample_patch_coords(
    tissue_mask: np.ndarray,
    wsi_dims: tuple[int, int],
    patch_size: int,
    max_patches: int,
    stride: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample up to `max_patches` (x, y) coordinates at WSI resolution
    from tissue-positive regions.

    The tissue mask is at thumbnail scale; we project candidate coords
    back to WSI resolution for extraction.

    Returns:
        coords: int32 array (N, 2) of (x, y) top-left corners at WSI resolution.
    """
    wsi_w, wsi_h = wsi_dims
    mask_h, mask_w = tissue_mask.shape

    scale_x = wsi_w / mask_w
    scale_y = wsi_h / mask_h

    # Build grid of candidate patch coords at WSI resolution
    xs = np.arange(0, wsi_w - patch_size, stride)
    ys = np.arange(0, wsi_h - patch_size, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    # Map each candidate to mask coordinates
    mask_x = np.clip((grid_x / scale_x).astype(int), 0, mask_w - 1)
    mask_y = np.clip((grid_y / scale_y).astype(int), 0, mask_h - 1)

    # Keep only tissue candidates
    tissue_flag = tissue_mask[mask_y, mask_x]
    tissue_x    = grid_x[tissue_flag]
    tissue_y    = grid_y[tissue_flag]

    if len(tissue_x) == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Random subsample
    rng = np.random.default_rng(seed)
    n   = min(max_patches, len(tissue_x))
    idx = rng.choice(len(tissue_x), n, replace=False)
    coords = np.stack([tissue_x[idx], tissue_y[idx]], axis=1).astype(np.int32)
    return coords


def _get_level_for_magnification(slide: "openslide.OpenSlide", target_mag: int) -> int:
    """Find the WSI pyramid level closest to `target_mag`."""
    native_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
    downsample_needed = native_mag / target_mag
    downsample_factors = slide.level_downsamples
    diffs = [abs(ds - downsample_needed) for ds in downsample_factors]
    return int(np.argmin(diffs))


def extract_patches_from_wsi(
    wsi_path: str | Path,
    out_dir: str | Path,
    patch_size: int   = 256,
    target_mag: int   = 20,
    max_patches: int  = 2000,
    tissue_thresh: float = 0.6,
    seed: int         = 42,
) -> Optional[Path]:
    """
    Extract tissue patches from a single WSI and save as HDF5.

    Args:
        wsi_path:      Path to .svs / .tiff file.
        out_dir:       Directory to write HDF5 files.
        patch_size:    Patch size in pixels at target magnification.
        target_mag:    Target magnification (e.g. 20x).
        max_patches:   Maximum patches to extract.
        tissue_thresh: Otsu threshold for tissue detection.
        seed:          RNG seed.

    Returns:
        Path to written HDF5, or None if extraction failed.
    """
    if not OPENSLIDE_AVAILABLE:
        raise RuntimeError("openslide-python is required for WSI patch extraction.")

    wsi_path = Path(wsi_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_id = wsi_path.stem
    h5_path  = out_dir / f"{slide_id}.h5"

    if h5_path.exists():
        return h5_path  # already processed

    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        console.print(f"[red]Failed to open {wsi_path.name}: {e}[/red]")
        return None

    # ── Tissue mask from thumbnail ──────────────────────────────────────────
    thumb_size   = (512, 512)
    thumbnail    = np.array(slide.get_thumbnail(thumb_size).convert("RGB"))
    tissue_mask  = _otsu_tissue_mask(thumbnail, tissue_thresh)
    tissue_frac  = tissue_mask.mean()

    if tissue_frac < 0.01:
        console.print(f"[yellow]Skipping {slide_id}: <1% tissue ({tissue_frac:.2%})[/yellow]")
        slide.close()
        return None

    # ── Level selection ─────────────────────────────────────────────────────
    level       = _get_level_for_magnification(slide, target_mag)
    level_dims  = slide.level_dimensions[level]   # (W, H) at chosen level
    downsample  = slide.level_downsamples[level]
    stride      = patch_size  # non-overlapping

    # ── Sample coords ───────────────────────────────────────────────────────
    # Coords in level-0 (native) space
    native_dims = slide.dimensions  # (W, H)
    # Scale patch_size to level-0
    patch_size_l0 = int(patch_size * downsample)
    stride_l0     = patch_size_l0

    coords = _sample_patch_coords(
        tissue_mask=tissue_mask,
        wsi_dims=native_dims,
        patch_size=patch_size_l0,
        max_patches=max_patches,
        stride=stride_l0,
        seed=seed,
    )

    if len(coords) == 0:
        console.print(f"[yellow]No tissue patches found for {slide_id}[/yellow]")
        slide.close()
        return None

    # ── Extract patches ─────────────────────────────────────────────────────
    patches_list: list[np.ndarray] = []
    valid_coords: list[np.ndarray] = []

    for x_l0, y_l0 in coords:
        try:
            region = slide.read_region(
                location=(int(x_l0), int(y_l0)),
                level=level,
                size=(patch_size, patch_size),
            ).convert("RGB")
            patch = np.array(region)

            # Basic quality filter: discard near-white patches
            mean_val = patch.mean()
            if mean_val > 230:
                continue

            patches_list.append(patch)
            valid_coords.append(np.array([x_l0, y_l0], dtype=np.int32))
        except Exception:
            continue

    slide.close()

    if not patches_list:
        console.print(f"[yellow]All patches filtered for {slide_id}[/yellow]")
        return None

    patches_arr = np.stack(patches_list, axis=0).astype(np.uint8)
    coords_arr  = np.stack(valid_coords, axis=0).astype(np.int32)

    # ── Write HDF5 ───────────────────────────────────────────────────────────
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("patches", data=patches_arr,  compression="gzip", compression_opts=4)
        f.create_dataset("coords",  data=coords_arr,   compression="gzip")
        f.attrs["slide_id"]    = slide_id
        f.attrs["n_patches"]   = len(patches_arr)
        f.attrs["patch_size"]  = patch_size
        f.attrs["magnification"] = target_mag
        f.attrs["wsi_path"]    = str(wsi_path)

    return h5_path


@app.command()
def run(
    wsi_dir:       Path  = typer.Argument(...,                help="Directory of .svs files"),
    out_dir:       Path  = typer.Option(Path("data/processed/patches"), help="HDF5 output dir"),
    patch_size:    int   = typer.Option(256,  help="Patch size (pixels at target magnification)"),
    target_mag:    int   = typer.Option(20,   help="Target magnification"),
    max_patches:   int   = typer.Option(2000, help="Max patches per slide"),
    tissue_thresh: float = typer.Option(0.6,  help="Tissue fraction Otsu threshold"),
    seed:          int   = typer.Option(42,   help="RNG seed"),
) -> None:
    """Batch-extract patches from all SVS files in a directory."""
    wsi_files = sorted(Path(wsi_dir).rglob("*.svs"))
    if not wsi_files:
        wsi_files = sorted(Path(wsi_dir).rglob("*.tiff"))

    console.rule(f"[bold blue]Patch Extraction · {len(wsi_files)} slides")

    success, skipped = 0, 0
    for wsi_path in track(wsi_files, description="Extracting patches…"):
        result = extract_patches_from_wsi(
            wsi_path=wsi_path,
            out_dir=out_dir,
            patch_size=patch_size,
            target_mag=target_mag,
            max_patches=max_patches,
            tissue_thresh=tissue_thresh,
            seed=seed,
        )
        if result:
            success += 1
        else:
            skipped += 1

    console.print(f"\n[green]Done:[/green] {success} slides processed, {skipped} skipped.")
    console.print(f"Output → [cyan]{out_dir}[/cyan]")


if __name__ == "__main__":
    app()
