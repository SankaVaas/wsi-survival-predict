"""
HIPT-based patch feature extractor.

Hierarchical Image Pyramid Transformer (HIPT) encodes 256×256 patches
into 192-dim embeddings using a ViT-S/16 trained with DINO-style SSL on
pathology data. No slide-level or patch-level labels are needed.

Reference: Chen et al., "Scaling Vision Transformers to Gigapixel Images
via Hierarchical Self-Supervised Learning", CVPR 2022.

This module:
  - Loads the HIPT_4K pretrained weights (auto-downloaded from HuggingFace Hub).
  - Reads patch HDF5 files (output of patch_extractor.py).
  - Produces feature HDF5 files: /feats (N, D), /coords (N, 2).

If HIPT weights are unavailable, falls back to a ResNet-50 + ImageNet weights
pretrained on pathology (UNI / CONCH / ctranspath) — configurable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import timm
import torch
import torch.nn as nn
import typer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from rich.console import Console
from rich.progress import track

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)

# ─── HIPT model wrapper ────────────────────────────────────────────────────────

HIPT_HF_REPO = "MahmoodLab/HIPT"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# Pathology-specific normalization (stain normalised)
_PATHOLOGY_MEAN = (0.70322989, 0.53606487, 0.66096631)
_PATHOLOGY_STD  = (0.21716536, 0.26081574, 0.20723464)


def get_transform(pathology_norm: bool = True) -> transforms.Compose:
    mean = _PATHOLOGY_MEAN if pathology_norm else _IMAGENET_MEAN
    std  = _PATHOLOGY_STD  if pathology_norm else _IMAGENET_STD
    return transforms.Compose([
        transforms.Resize((224, 224)),      # HIPT uses 224×224 patches
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class HIPTEncoder(nn.Module):
    """
    HIPT patch-level encoder: ViT-S/16 pretrained on TCGA with DINO.

    Falls back to ResNet-50 (ImageNet) if HIPT weights cannot be loaded.
    Output dimension: 384 (ViT-S) or 2048 (ResNet-50).
    """

    def __init__(self, weights_path: Optional[str | Path] = None) -> None:
        super().__init__()
        self.embed_dim: int

        loaded = False
        if weights_path and Path(weights_path).exists():
            loaded = self._load_hipt(weights_path)

        if not loaded:
            loaded = self._try_hf_hipt()

        if not loaded:
            console.print(
                "[yellow]HIPT weights not available. "
                "Falling back to ResNet-50 ImageNet pretrained.[/yellow]"
            )
            self._load_resnet50_fallback()

    def _load_hipt(self, weights_path: str | Path) -> bool:
        try:
            from timm.models.vision_transformer import vit_small_patch16_224
            self.encoder = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
            state = torch.load(str(weights_path), map_location="cpu")
            # Handle various checkpoint formats
            state = state.get("teacher", state.get("model", state))
            state = {k.replace("backbone.", "").replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            self.embed_dim = self.encoder.embed_dim  # 384 for ViT-S
            console.print(f"[green]HIPT weights loaded from {weights_path}[/green]")
            console.print(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
            return True
        except Exception as e:
            console.print(f"[yellow]HIPT load failed: {e}[/yellow]")
            return False

    def _try_hf_hipt(self) -> bool:
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(repo_id=HIPT_HF_REPO, filename="HIPT_4K_feat_extractor.pth")
            return self._load_hipt(weights_path)
        except Exception as e:
            console.print(f"[yellow]HuggingFace HIPT download failed: {e}[/yellow]")
            return False

    def _load_resnet50_fallback(self) -> None:
        import torchvision.models as tv_models
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder  = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.embed_dim = 2048

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        if out.dim() == 4:            # ResNet: (B, C, 1, 1)
            out = out.flatten(1)
        return out                     # (B, D)


# ─── Patch Dataset ─────────────────────────────────────────────────────────────

class PatchH5Dataset(Dataset):
    """Reads patches from a single HDF5 file."""

    def __init__(self, h5_path: str | Path, transform: transforms.Compose) -> None:
        self.h5_path   = str(h5_path)
        self.transform = transform
        with h5py.File(self.h5_path, "r") as f:
            self.n_patches = f["patches"].shape[0]
            self.coords    = f["coords"][:]   # load all coords into memory

    def __len__(self) -> int:
        return self.n_patches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:
            patch = f["patches"][idx]          # uint8 HWC
        img   = Image.fromarray(patch)
        img_t = self.transform(img)
        return img_t, self.coords[idx]


# ─── Feature extraction loop ───────────────────────────────────────────────────

def extract_features_for_slide(
    h5_patch_path: str | Path,
    out_dir: str | Path,
    model: HIPTEncoder,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 64,
    fp16: bool = True,
) -> Optional[Path]:
    """
    Extract HIPT features for one slide's patches.

    Writes to: out_dir/<slide_id>.h5
        /feats  : float32  (N, D)
        /coords : int32    (N, 2)
    """
    h5_patch_path = Path(h5_patch_path)
    out_dir       = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_id = h5_patch_path.stem
    feat_h5  = out_dir / f"{slide_id}.h5"

    if feat_h5.exists():
        return feat_h5

    dataset = PatchH5Dataset(h5_patch_path, transform)
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    model.eval()
    all_feats: list[np.ndarray]  = []
    all_coords: list[np.ndarray] = []

    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16)

    with torch.no_grad(), autocast_ctx:
        for patches, coords in loader:
            patches = patches.to(device)
            feats   = model(patches).float().cpu().numpy()
            all_feats.append(feats)
            all_coords.append(coords.numpy() if isinstance(coords, torch.Tensor) else coords)

    feats_arr  = np.concatenate(all_feats,  axis=0).astype(np.float32)
    coords_arr = np.concatenate(all_coords, axis=0).astype(np.int32)

    with h5py.File(str(feat_h5), "w") as f:
        f.create_dataset("feats",  data=feats_arr,  compression="gzip", compression_opts=4)
        f.create_dataset("coords", data=coords_arr, compression="gzip")
        f.attrs["slide_id"]  = slide_id
        f.attrs["n_patches"] = len(feats_arr)
        f.attrs["embed_dim"] = feats_arr.shape[1]

    return feat_h5


@app.command()
def run(
    patch_dir:    Path  = typer.Argument(..., help="Directory of patch HDF5 files"),
    out_dir:      Path  = typer.Option(Path("data/processed/features"), help="Feature output dir"),
    weights_path: Optional[Path] = typer.Option(None, help="HIPT weights path (auto-downloads if None)"),
    batch_size:   int   = typer.Option(64,    help="Inference batch size"),
    fp16:         bool  = typer.Option(True,  help="Use AMP float16"),
) -> None:
    """Batch-extract HIPT features from all patch HDF5 files."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device:[/cyan] {device}")

    model     = HIPTEncoder(weights_path=weights_path).to(device)
    transform = get_transform(pathology_norm=True)

    h5_files = sorted(Path(patch_dir).glob("*.h5"))
    console.rule(f"[bold blue]Feature Extraction · {len(h5_files)} slides")

    success = 0
    for h5_path in track(h5_files, description="Extracting features…"):
        result = extract_features_for_slide(
            h5_patch_path=h5_path,
            out_dir=out_dir,
            model=model,
            transform=transform,
            device=device,
            batch_size=batch_size,
            fp16=fp16,
        )
        if result:
            success += 1

    console.print(f"\n[green]Done:[/green] {success} / {len(h5_files)} slides extracted.")


if __name__ == "__main__":
    app()
