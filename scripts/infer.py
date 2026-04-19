"""
Inference on new slides (no survival labels required).

End-to-end pipeline: .svs → patches → features → graph → risk score + attention map.

Usage:
    python scripts/infer.py \
        --wsi_path  /path/to/slide.svs \
        --checkpoint results/checkpoints/best_model.pt \
        --config    configs/config.yaml \
        --out_dir   results/inference
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile

import numpy as np
import torch
import typer
from rich.console import Console

from src.models.heterogeneity_gat import HeterogeneityGAT
from src.preprocessing.feature_extractor import HIPTEncoder, extract_features_for_slide, get_transform
from src.preprocessing.graph_builder import build_graph_for_slide
from src.preprocessing.patch_extractor import extract_patches_from_wsi
from src.utils.config import load_config, get_device
from src.visualization.plots import plot_attention_heatmap, plot_heterogeneity_map

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def infer(
    wsi_path:    Path = typer.Argument(...,                          help="Path to WSI .svs file"),
    checkpoint:  Path = typer.Option(...,                            help="Trained model checkpoint"),
    config:      Path = typer.Option(Path("configs/config.yaml"),   help="Config YAML"),
    out_dir:     Path = typer.Option(Path("results/inference"),     help="Output directory"),
    weights_path:Path = typer.Option(None,                          help="HIPT weights (auto if None)"),
) -> None:
    """Run end-to-end survival risk prediction on a single new WSI."""

    cfg    = load_config(config)
    device = get_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_id = Path(wsi_path).stem
    console.rule(f"[bold blue]Inference — {slide_id}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ── 1. Patch extraction ─────────────────────────────────────────────
        console.print("[cyan]Step 1/4: Extracting tissue patches…[/cyan]")
        patch_h5 = extract_patches_from_wsi(
            wsi_path=wsi_path,
            out_dir=tmp_path / "patches",
            patch_size=cfg.data.patch_size,
            target_mag=cfg.data.patch_mag,
            max_patches=cfg.data.patches_per_slide,
            tissue_thresh=cfg.data.tissue_threshold,
        )
        if patch_h5 is None:
            console.print("[red]Patch extraction failed — aborting.[/red]")
            raise typer.Exit(1)

        # ── 2. Feature extraction ───────────────────────────────────────────
        console.print("[cyan]Step 2/4: Extracting HIPT features…[/cyan]")
        model_enc = HIPTEncoder(weights_path=weights_path).to(device)
        transform = get_transform(pathology_norm=True)
        feat_h5   = extract_features_for_slide(
            h5_patch_path=patch_h5,
            out_dir=tmp_path / "features",
            model=model_enc,
            transform=transform,
            device=device,
            batch_size=cfg.feature_extractor.batch_size,
            fp16=cfg.feature_extractor.fp16,
        )
        if feat_h5 is None:
            console.print("[red]Feature extraction failed — aborting.[/red]")
            raise typer.Exit(1)

        # ── 3. Graph construction ───────────────────────────────────────────
        console.print("[cyan]Step 3/4: Building spatial graph…[/cyan]")
        graph_pt = build_graph_for_slide(
            feat_h5_path=feat_h5,
            out_dir=tmp_path / "graphs",
            clinical_row=None,    # no survival label — inference mode
            k_neighbors=cfg.graph.k_neighbors,
        )
        if graph_pt is None:
            console.print("[red]Graph construction failed — aborting.[/red]")
            raise typer.Exit(1)

        # ── 4. Model inference ──────────────────────────────────────────────
        console.print("[cyan]Step 4/4: Running model inference…[/cyan]")
        ckpt = torch.load(str(checkpoint), map_location=device)
        first_w = next(
            (v for k, v in ckpt["model_state"].items() if "input_proj" in k and "weight" in k),
            None,
        )
        in_dim  = first_w.shape[1] if first_w is not None else 387
        m_cfg   = ckpt.get("cfg", {}).get("model", {})

        model = HeterogeneityGAT(
            in_dim        = in_dim,
            hidden_dim    = m_cfg.get("gat_hidden_dim", 256),
            gat_heads     = m_cfg.get("gat_heads", 4),
            gat_layers    = m_cfg.get("gat_layers", 3),
            gat_dropout   = 0.0,
            survival_mode = m_cfg.get("survival_head", "cox"),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        graph_data = torch.load(str(graph_pt), map_location=device)
        # Add dummy batch vector
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            out  = model(graph_data, return_attention=True)
            risk = float(out["logits"].squeeze().cpu())
            attn = out["attention"].squeeze().cpu().numpy()

        # ── 5. Save outputs ──────────────────────────────────────────────────
        coords   = graph_data.coords.cpu().numpy()
        het_feats = graph_data.x.cpu().numpy()[:, -3:]

        result = {
            "slide_id":    slide_id,
            "risk_score":  risk,
            "risk_group":  "high" if risk > 0 else "low",
            "n_patches":   int(graph_data.x.size(0)),
        }
        console.print(
            f"\n[bold green]Result:[/bold green]  "
            f"slide={slide_id}  risk_score={risk:.4f}  group={result['risk_group']}"
        )

        json_out = out_dir / f"{slide_id}_result.json"
        json_out.write_text(json.dumps(result, indent=2))

        # Attention heatmap
        plot_attention_heatmap(
            coords=coords,
            attention=attn,
            save_path=out_dir / f"{slide_id}_attention.png",
            title=f"Attention Heatmap — {slide_id} | Risk: {risk:.3f}",
        )

        # Heterogeneity map
        plot_heterogeneity_map(
            coords=coords,
            entropy=het_feats[:, 0],
            cosine_dissim=het_feats[:, 1],
            save_path=out_dir / f"{slide_id}_heterogeneity.png",
        )

        console.print(f"\n[bold green]Outputs saved to:[/bold green] {out_dir}")


if __name__ == "__main__":
    app()
