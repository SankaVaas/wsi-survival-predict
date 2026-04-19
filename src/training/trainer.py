"""
Training engine for HeterogeneityGAT survival model.

Features:
  - Mixed precision (AMP) with GradScaler
  - Gradient accumulation (effective batch > 1 slide)
  - CosineAnnealingLR with linear warmup
  - Early stopping on val C-index (not loss)
  - Best checkpoint saving
  - TensorBoard logging
  - Configurable L1 regularisation on survival head weights
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from rich.console import Console
from rich.table import Table

from src.evaluation.metrics import concordance_index_censored
from src.models.heterogeneity_gat import HeterogeneityGAT
from src.training.losses import CoxNLLLoss
from src.utils.logger import MetricTracker

console = Console()


class EarlyStopping:
    """Early stopping monitor (maximize metric)."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -float("inf")
        self.counter    = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class SurvivalTrainer:
    """
    Trainer for HeterogeneityGAT.

    Args:
        model:           Instantiated model.
        cfg:             Training config dict (from OmegaConf).
        device:          Torch device.
        results_dir:     Root directory for logs + checkpoints.
    """

    def __init__(
        self,
        model: HeterogeneityGAT,
        cfg: dict,
        device: torch.device,
        results_dir: str | Path = "results",
    ) -> None:
        self.model      = model.to(device)
        self.cfg        = cfg
        self.device     = device
        self.results_dir = Path(results_dir)

        t_cfg = cfg.get("training", cfg)

        # ── Loss ────────────────────────────────────────────────────────────
        self.criterion = CoxNLLLoss(l1_reg=t_cfg.get("l1_reg", 1e-4))

        # ── Optimiser ───────────────────────────────────────────────────────
        self.optimizer = AdamW(
            model.parameters(),
            lr=float(t_cfg.get("lr", 2e-4)),
            weight_decay=float(t_cfg.get("weight_decay", 1e-5)),
        )

        # ── Scheduler: linear warmup → cosine decay ─────────────────────────
        epochs         = int(t_cfg.get("epochs", 30))
        warmup_epochs  = int(t_cfg.get("warmup_epochs", 3))
        warmup_sched   = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched   = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )

        # ── AMP ─────────────────────────────────────────────────────────────
        self.amp       = t_cfg.get("amp", True) and device.type == "cuda"
        self.scaler    = GradScaler(enabled=self.amp)

        # ── Gradient accumulation ───────────────────────────────────────────
        self.accum_steps = int(t_cfg.get("accumulate_grad_batches", 8))

        # ── Early stopping ──────────────────────────────────────────────────
        self.early_stopping = EarlyStopping(patience=int(t_cfg.get("early_stopping_patience", 10)))

        # ── Logging ─────────────────────────────────────────────────────────
        self.log_dir  = self.results_dir / "logs"
        self.ckpt_dir = self.results_dir / "checkpoints"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=str(self.log_dir))

        self.best_cindex  = -1.0
        self.best_ckpt    = self.ckpt_dir / "best_model.pt"

    # ── Training epoch ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        tracker = MetricTracker()
        self.optimizer.zero_grad()

        all_risks: list[float] = []
        all_times: list[float] = []
        all_events: list[int]  = []

        for step, batch in enumerate(loader, 1):
            batch = batch.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                out        = self.model(batch)
                risk_scores = out["logits"]               # (B,)
                surv_times  = batch.survival_months.squeeze()
                events      = batch.event.squeeze().float()

                loss = self.criterion(risk_scores, surv_times, events)
                loss = loss / self.accum_steps

            self.scaler.scale(loss).backward()

            if step % self.accum_steps == 0 or step == len(loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            tracker.update({"loss": float(loss.item()) * self.accum_steps})
            all_risks.extend(risk_scores.detach().cpu().tolist())
            all_times.extend(surv_times.cpu().tolist())
            all_events.extend(events.cpu().int().tolist())

        # Compute C-index on training set (noisy but informative)
        try:
            cindex = concordance_index_censored(all_times, all_events, all_risks)
        except Exception:
            cindex = 0.5

        summary = tracker.summary()
        summary["cindex"] = cindex
        return summary

    # ── Validation epoch ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()

        all_risks: list[float]  = []
        all_times: list[float]  = []
        all_events: list[int]   = []
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                out         = self.model(batch)
                risk_scores = out["logits"]
                surv_times  = batch.survival_months.squeeze()
                events      = batch.event.squeeze().float()
                loss        = self.criterion(risk_scores, surv_times, events)

            total_loss += float(loss.item())
            all_risks.extend(risk_scores.cpu().tolist())
            all_times.extend(surv_times.cpu().tolist())
            all_events.extend(events.cpu().int().tolist())

        try:
            cindex = concordance_index_censored(all_times, all_events, all_risks)
        except Exception:
            cindex = 0.5

        return {
            "loss":    total_loss / max(len(loader), 1),
            "cindex":  cindex,
        }

    # ── Main training loop ─────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> dict[str, list]:
        epochs = epochs or int(self.cfg.get("training", {}).get("epochs", 30))

        history: dict[str, list] = {
            "train_loss": [], "train_cindex": [],
            "val_loss":   [], "val_cindex":   [], "lr": [],
        }

        console.rule("[bold blue]Training HeterogeneityGAT")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Train ──────────────────────────────────────────────────────
            train_metrics = self._train_epoch(train_loader, epoch)

            # ── Validate ───────────────────────────────────────────────────
            val_metrics = self._val_epoch(val_loader)

            # ── LR step ────────────────────────────────────────────────────
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Record ─────────────────────────────────────────────────────
            history["train_loss"].append(train_metrics["loss"])
            history["train_cindex"].append(train_metrics["cindex"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_cindex"].append(val_metrics["cindex"])
            history["lr"].append(current_lr)

            # ── TensorBoard ─────────────────────────────────────────────────
            self.writer.add_scalars("Loss",    {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
            self.writer.add_scalars("C-Index", {"train": train_metrics["cindex"], "val": val_metrics["cindex"]}, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            # ── Checkpoint if best ─────────────────────────────────────────
            if val_metrics["cindex"] > self.best_cindex:
                self.best_cindex = val_metrics["cindex"]
                torch.save({
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer":   self.optimizer.state_dict(),
                    "val_cindex":  self.best_cindex,
                    "cfg":         self.cfg,
                }, str(self.best_ckpt))

            # ── Console output ─────────────────────────────────────────────
            elapsed = time.time() - t0
            marker  = " ★" if val_metrics["cindex"] == self.best_cindex else ""
            console.print(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  train_C={train_metrics['cindex']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  val_C={val_metrics['cindex']:.4f}  "
                f"lr={current_lr:.2e}  [{elapsed:.1f}s]{marker}"
            )

            # ── Early stopping ─────────────────────────────────────────────
            if self.early_stopping(val_metrics["cindex"]):
                console.print(f"[yellow]Early stopping at epoch {epoch}. Best val C-index: {self.best_cindex:.4f}[/yellow]")
                break

        self.writer.close()

        # ── Final summary table ─────────────────────────────────────────────
        tbl = Table(title="Training Summary", show_header=True)
        tbl.add_column("Metric", style="cyan")
        tbl.add_column("Value",  style="white")
        tbl.add_row("Best val C-index", f"{self.best_cindex:.4f}")
        tbl.add_row("Best checkpoint",  str(self.best_ckpt))
        tbl.add_row("Total epochs",     str(len(history["train_loss"])))
        console.print(tbl)

        return history

    def load_best(self) -> None:
        """Reload best checkpoint into model."""
        ckpt = torch.load(str(self.best_ckpt), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        console.print(f"[green]Loaded best model (val C-index={ckpt['val_cindex']:.4f}) from {self.best_ckpt}[/green]")
