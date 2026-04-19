"""
Main training script.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml training.lr=1e-3 training.epochs=50
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
import torch
from rich.console import Console

from src.models.heterogeneity_gat import build_model
from src.training.dataset import build_datasets, get_dataloaders
from src.training.trainer import SurvivalTrainer
from src.utils.config import load_config, seed_everything, get_device, print_config
from src.evaluation.metrics import full_evaluation
from src.visualization.plots import plot_km_curves, plot_training_history

console = Console()
app     = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Path to YAML config"),
    overrides: list[str] = typer.Argument(default=None, help="OmegaConf dot-notation overrides"),
) -> None:

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = load_config(config, overrides)
    print_config(cfg)

    seed_everything(cfg.project.seed)
    device = get_device()
    console.print(f"[cyan]Device:[/cyan] {device}")

    cfg_dict = dict(cfg)

    # ── Data ────────────────────────────────────────────────────────────────
    graph_dir = cfg.data.graph_dir
    datasets  = build_datasets(
        graph_dir=graph_dir,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.project.seed,
    )
    loaders = get_dataloaders(
        datasets,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # ── Infer input dimension from first graph ──────────────────────────────
    sample_graph = datasets["train"].get(0)
    in_dim = sample_graph.x.shape[1]
    console.print(f"[cyan]Node feature dimension:[/cyan] {in_dim}")

    # ── Model ───────────────────────────────────────────────────────────────
    model_cfg = dict(cfg.model)
    model_cfg["survival_head"] = model_cfg.pop("survival_head", "cox")
    model = build_model({
        "model": model_cfg,
        "feature_extractor": {"embed_dim": in_dim - 3},
    })
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[cyan]Model parameters:[/cyan] {n_params:,}")

    # ── Train ───────────────────────────────────────────────────────────────
    trainer = SurvivalTrainer(
        model=model,
        cfg=cfg_dict,
        device=device,
        results_dir=cfg.logging.results_dir,
    )
    history = trainer.fit(
        loaders["train"],
        loaders["val"],
        epochs=cfg.training.epochs,
    )

    # ── Save training plots ──────────────────────────────────────────────────
    fig_dir = Path(cfg.logging.figure_dir)
    plot_training_history(history, save_path=fig_dir / "training_history.png")
    console.print(f"[green]Training plot saved to {fig_dir / 'training_history.png'}[/green]")

    # ── Test evaluation ──────────────────────────────────────────────────────
    console.rule("[bold blue]Test Evaluation")
    trainer.load_best()
    trainer.model.eval()

    all_risks, all_times, all_events = [], [], []

    with torch.no_grad():
        for batch in loaders["test"]:
            batch = batch.to(device)
            out   = trainer.model(batch)
            all_risks.extend(out["logits"].cpu().tolist())
            all_times.extend(batch.survival_months.squeeze().cpu().tolist())
            all_events.extend(batch.event.squeeze().cpu().int().tolist())

    import numpy as np
    times_train, events_train = datasets["train"].get_survival_arrays()

    results = full_evaluation(
        survival_times=np.array(all_times),
        events=np.array(all_events),
        risk_scores=np.array(all_risks),
        survival_times_train=times_train.numpy(),
        events_train=events_train.numpy(),
        n_bootstrap=cfg.evaluation.n_bootstrap,
    )

    console.print(
        f"\n[bold green]Test Results[/bold green]\n"
        f"  C-index:    {results['cindex']:.4f}  "
        f"95% CI [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]\n"
        f"  IBS:        {results['ibs']:.4f}\n"
        f"  Log-rank p: {results['logrank_p']:.4e}"
    )

    # ── KM curves ──────────────────────────────────────────────────────────
    plot_km_curves(results["km_data"], save_path=fig_dir / "km_curves_test.png")
    console.print(f"[green]KM curves saved to {fig_dir / 'km_curves_test.png'}[/green]")

    # ── Save results JSON ──────────────────────────────────────────────────
    import json
    results_out = {
        "cindex":     results["cindex"],
        "ci_lower":   results["ci_lower"],
        "ci_upper":   results["ci_upper"],
        "ibs":        results["ibs"],
        "logrank_p":  results["logrank_p"],
    }
    out_path = Path(cfg.logging.results_dir) / "test_results.json"
    out_path.write_text(json.dumps(results_out, indent=2))
    console.print(f"[green]Results saved to {out_path}[/green]")


if __name__ == "__main__":
    app()
