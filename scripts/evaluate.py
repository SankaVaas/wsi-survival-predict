"""
Standalone evaluation script.

Loads a saved checkpoint and runs full evaluation on the test set,
generating all clinical-grade figures: KM curves, attention heatmaps,
UMAP projections, and heterogeneity spatial maps.

Usage:
    python scripts/evaluate.py \
        --checkpoint results/checkpoints/best_model.pt \
        --graph_dir  data/graphs \
        --config     configs/config.yaml \
        --out_dir    results/figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table

from src.evaluation.metrics import full_evaluation, bootstrap_cindex
from src.models.heterogeneity_gat import HeterogeneityGAT
from src.training.dataset import build_datasets, get_dataloaders
from src.utils.config import load_config, seed_everything, get_device
from src.visualization.plots import (
    plot_km_curves,
    plot_attention_heatmap,
    plot_umap_features,
    plot_heterogeneity_map,
    plot_training_history,
)

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(...,                            help="Path to .pt checkpoint"),
    config:     Path = typer.Option(Path("configs/config.yaml"),   help="Config YAML"),
    graph_dir:  Path = typer.Option(Path("data/graphs"),           help="Graph .pt directory"),
    out_dir:    Path = typer.Option(Path("results/figures"),       help="Output figure directory"),
    n_bootstrap:int  = typer.Option(1000,                          help="Bootstrap iterations for CI"),
    split:      str  = typer.Option("test",                        help="Split to evaluate: test | val | train"),
) -> None:

    cfg    = load_config(config)
    device = get_device()
    seed_everything(cfg.project.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────
    console.print(f"[cyan]Loading checkpoint:[/cyan] {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location=device)

    # Infer in_dim from checkpoint
    first_layer_weight = ckpt["model_state"].get("input_proj.0.weight")
    if first_layer_weight is None:
        # Try alternate key
        for k, v in ckpt["model_state"].items():
            if "input_proj" in k and "weight" in k:
                first_layer_weight = v
                break
    in_dim = first_layer_weight.shape[1] if first_layer_weight is not None else 387

    saved_cfg = ckpt.get("cfg", dict(cfg))
    m_cfg     = saved_cfg.get("model", {})

    model = HeterogeneityGAT(
        in_dim        = in_dim,
        hidden_dim    = m_cfg.get("gat_hidden_dim", 256),
        gat_heads     = m_cfg.get("gat_heads", 4),
        gat_layers    = m_cfg.get("gat_layers", 3),
        gat_dropout   = 0.0,   # no dropout at eval
        survival_mode = m_cfg.get("survival_head", "cox"),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    console.print(f"[green]Model loaded. Checkpoint val C-index: {ckpt.get('val_cindex', 'N/A')}[/green]")

    # ── Dataset ───────────────────────────────────────────────────────────
    datasets = build_datasets(
        graph_dir=graph_dir,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.project.seed,
    )
    loaders = get_dataloaders(datasets, batch_size=1, num_workers=0)
    loader  = loaders[split]

    # ── Inference ─────────────────────────────────────────────────────────
    all_risks:     list[float] = []
    all_times:     list[float] = []
    all_events:    list[int]   = []
    all_slide_ids: list[str]   = []

    # For visualisation: store per-slide attention weights + node data
    viz_data: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch, return_attention=True)

            risk  = out["logits"].squeeze().cpu().item()
            attn  = out["attention"].squeeze().cpu().numpy()       # (N,)
            x_cpu = batch.x.cpu().numpy()                          # (N, D+3)
            coords_cpu = batch.coords.cpu().numpy() if hasattr(batch, "coords") else None

            all_risks.append(risk)
            all_times.append(float(batch.survival_months.squeeze().cpu()))
            all_events.append(int(batch.event.squeeze().cpu()))
            sid = batch.slide_id if isinstance(batch.slide_id, str) else batch.slide_id[0]
            all_slide_ids.append(sid)

            viz_data.append({
                "slide_id": sid,
                "risk":     risk,
                "time":     float(batch.survival_months.squeeze().cpu()),
                "event":    int(batch.event.squeeze().cpu()),
                "attention": attn,
                "features":  x_cpu[:, :-3],   # HIPT features (drop het dims)
                "het_feats": x_cpu[:, -3:],    # heterogeneity dims
                "coords":    coords_cpu,
            })

    risks_arr  = np.array(all_risks)
    times_arr  = np.array(all_times)
    events_arr = np.array(all_events)

    # ── Metrics ───────────────────────────────────────────────────────────
    train_times, train_events = datasets["train"].get_survival_arrays()
    results = full_evaluation(
        survival_times=times_arr,
        events=events_arr,
        risk_scores=risks_arr,
        survival_times_train=train_times.numpy(),
        events_train=train_events.numpy(),
        n_bootstrap=n_bootstrap,
    )

    # Print table
    tbl = Table(title=f"Evaluation Results — {split.upper()} set", show_header=True)
    tbl.add_column("Metric",     style="cyan",  min_width=30)
    tbl.add_column("Value",      style="white", min_width=20)
    tbl.add_row("C-index",       f"{results['cindex']:.4f}")
    tbl.add_row("95% CI",        f"[{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    tbl.add_row("IBS",           f"{results['ibs']:.4f}" if not np.isnan(results['ibs']) else "N/A")
    tbl.add_row("Log-rank p",    f"{results['logrank_p']:.4e}")
    tbl.add_row("N slides",      str(len(all_risks)))
    tbl.add_row("N events",      str(int(events_arr.sum())))
    console.print(tbl)

    # ── Figures ───────────────────────────────────────────────────────────

    # 1. KM curves
    console.print("[cyan]Plotting KM curves…[/cyan]")
    plot_km_curves(
        results["km_data"],
        save_path=out_dir / f"km_curves_{split}.png",
        title=f"KM Curves — {split.upper()} set  (C-index={results['cindex']:.3f})",
    )

    # 2. Attention heatmaps for top-3 highest risk and top-3 lowest risk slides
    console.print("[cyan]Plotting attention heatmaps…[/cyan]")
    sorted_by_risk = sorted(viz_data, key=lambda d: d["risk"], reverse=True)
    selected = sorted_by_risk[:3] + sorted_by_risk[-3:]

    for d in selected:
        if d["coords"] is None:
            continue
        label = "high" if d["risk"] >= np.median(risks_arr) else "low"
        plot_attention_heatmap(
            coords=d["coords"],
            attention=d["attention"],
            thumbnail=None,
            save_path=out_dir / f"attn_{label}_risk_{d['slide_id'][:20]}.png",
            title=f"Attention — {d['slide_id'][:20]} | risk={d['risk']:.3f} | event={d['event']}",
        )

    # 3. UMAP of patch features (subsample for speed — use highest-risk slide)
    console.print("[cyan]Plotting UMAP of patch features…[/cyan]")
    if viz_data:
        top_slide = sorted_by_risk[0]
        feats_sub = top_slide["features"]
        attn_sub  = top_slide["attention"]
        # Subsample to 2000 if too large
        if len(feats_sub) > 2000:
            idx = np.random.choice(len(feats_sub), 2000, replace=False)
            feats_sub = feats_sub[idx]
            attn_sub  = attn_sub[idx]
        plot_umap_features(
            features=feats_sub,
            colour_values=attn_sub,
            colour_label="Attention weight",
            save_path=out_dir / f"umap_top_risk_{top_slide['slide_id'][:20]}.png",
            title=f"UMAP Patch Features — {top_slide['slide_id'][:20]}",
        )

    # 4. Heterogeneity spatial map
    console.print("[cyan]Plotting heterogeneity spatial maps…[/cyan]")
    if viz_data and sorted_by_risk[0]["coords"] is not None:
        top = sorted_by_risk[0]
        plot_heterogeneity_map(
            coords=top["coords"],
            entropy=top["het_feats"][:, 0],
            cosine_dissim=top["het_feats"][:, 1],
            save_path=out_dir / f"heterogeneity_map_{top['slide_id'][:20]}.png",
        )

    # ── Save results JSON ─────────────────────────────────────────────────
    out_json = {
        "split":      split,
        "cindex":     results["cindex"],
        "ci_lower":   results["ci_lower"],
        "ci_upper":   results["ci_upper"],
        "ibs":        float(results["ibs"]) if not np.isnan(results["ibs"]) else None,
        "logrank_p":  results["logrank_p"],
        "n_slides":   len(all_risks),
        "n_events":   int(events_arr.sum()),
    }
    json_path = out_dir / f"results_{split}.json"
    json_path.write_text(json.dumps(out_json, indent=2))

    console.print(f"\n[bold green]All figures and results saved to:[/bold green] {out_dir}")


if __name__ == "__main__":
    app()
