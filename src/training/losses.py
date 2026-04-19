"""
Survival analysis loss functions.

1. CoxNLLLoss
   Negative partial log-likelihood for the Cox proportional hazards model.
   Handles tied survival times via Breslow approximation.
   Includes optional L1 regularisation on the log-hazard output.

2. DiscreteHazardLoss
   Cross-entropy loss on discretised survival time bins.
   Based on NNET-survival / Transformer-survival paradigm.

Both losses operate on mini-batches of WSI graphs (batch_size ≥ 1).
For Cox, a larger effective batch (via gradient accumulation) is important
because the partial likelihood is computed across all pairs in the batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CoxNLLLoss(nn.Module):
    """
    Cox proportional hazards negative partial log-likelihood (Breslow tie correction).

    For a batch of n patients with:
        risk_scores:      (n,) log-hazard predictions
        survival_times:   (n,) observed time (months or days)
        events:           (n,) 1=event, 0=censored

    The Cox partial likelihood for patient i (uncensored) is:
        L_i = exp(h_i) / Σ_{j: T_j >= T_i} exp(h_j)

    The negative mean log partial likelihood:
        Loss = - (1/n_events) * Σ_{i: event_i=1} [h_i - log Σ_{j: T_j>=T_i} exp(h_j)]

    Breslow approximation handles tied event times by summing exp(h_j) over the
    entire risk set at each event time, including ties.

    Args:
        l1_reg: L1 regularisation weight on risk_scores (encourages sparsity).
    """

    def __init__(self, l1_reg: float = 1e-4) -> None:
        super().__init__()
        self.l1_reg = l1_reg

    def forward(
        self,
        risk_scores: Tensor,       # (B,)
        survival_times: Tensor,    # (B,)
        events: Tensor,            # (B,) int or float
    ) -> Tensor:
        """
        Compute Cox NLL loss.

        Returns:
            Scalar loss tensor.
        """
        # Ensure correct types
        risk_scores    = risk_scores.float()
        survival_times = survival_times.float()
        events         = events.float()

        B = risk_scores.shape[0]
        if B == 0 or events.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)

        # Sort by descending survival time (required for efficient risk set computation)
        sort_idx   = torch.argsort(survival_times, descending=True)
        risk_sorted = risk_scores[sort_idx]
        events_sorted = events[sort_idx]

        # Numerically stable log-sum-exp of cumulative risk set
        # log Σ_{j: T_j >= T_i} exp(h_j)  for each i
        # Since sorted descending, cumulative sum from index 0 gives the risk set
        log_risk_set = torch.logcumsumexp(risk_sorted, dim=0)

        # NLL: only over uncensored patients
        nll_per_patient = risk_sorted - log_risk_set   # (B,)
        nll = -nll_per_patient[events_sorted == 1].mean()

        # L1 regularisation
        if self.l1_reg > 0:
            nll = nll + self.l1_reg * risk_scores.abs().mean()

        return nll


class DiscreteHazardLoss(nn.Module):
    """
    Discrete-time survival loss (NNET-survival / Transformer-survival).

    Time is discretised into T bins. For each patient, the label is:
        - Which bin the event/censoring occurred in.
        - Whether the event was observed.

    Loss = - Σ_t [event_t * log(h_t) + (1-event_t) * log(1 - h_t)]
    where h_t = discrete hazard at bin t (sigmoid of logit).

    Reference: Gensheimer & Narasimhan, Scientific Reports 2019.

    Args:
        num_bins:   Number of discrete time intervals.
        reduction: "mean" or "sum".
    """

    def __init__(self, num_bins: int = 4, reduction: str = "mean") -> None:
        super().__init__()
        self.num_bins  = num_bins
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,           # (B, T) raw unnormalised hazard logits
        time_bins: Tensor,        # (B,)  which bin the event/censoring fell into
        events: Tensor,           # (B,)  1=event, 0=censored
    ) -> Tensor:
        B, T = logits.shape
        events   = events.float()
        hazards  = torch.sigmoid(logits)  # (B, T) discrete hazard probabilities

        # Build per-bin event labels
        # For uncensored patient i in bin k: event at bin k, survived all bins <k
        # For censored patient i in bin k:  survived all bins ≤k (partial label)
        device = logits.device
        loss   = torch.zeros(B, device=device)

        for b in range(B):
            t_bin   = int(time_bins[b].item())
            t_bin   = min(t_bin, T - 1)
            event_b = events[b]

            h_b = hazards[b]                             # (T,)
            # Log survival up to bin t_bin - 1
            log_surv_bins = torch.log(1.0 - h_b[:t_bin] + 1e-7).sum()

            if event_b == 1:
                # Observed: log hazard at event bin + log survival before
                log_loss = log_surv_bins + torch.log(h_b[t_bin] + 1e-7)
            else:
                # Censored: log survival up to and including t_bin
                log_loss = log_surv_bins + torch.log(1.0 - h_b[t_bin] + 1e-7)

            loss[b] = -log_loss

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def discretise_survival_times(
    survival_times: Tensor,
    n_bins: int = 4,
    strategy: str = "quantile",
) -> tuple[Tensor, list[float]]:
    """
    Convert continuous survival times to discrete bin indices.

    Args:
        survival_times: (N,) float tensor of survival times.
        n_bins:         Number of bins.
        strategy:       "quantile" (equal population) or "uniform" (equal width).

    Returns:
        bin_indices: (N,) long tensor of bin assignments.
        cut_points:  List of n_bins-1 cut-point values.
    """
    times_np = survival_times.cpu().numpy()

    if strategy == "quantile":
        import numpy as np
        quantiles  = np.linspace(0, 100, n_bins + 1)[1:-1]
        cut_points = np.percentile(times_np, quantiles).tolist()
    else:
        t_min, t_max = float(times_np.min()), float(times_np.max())
        import numpy as np
        cut_points = np.linspace(t_min, t_max, n_bins + 1)[1:-1].tolist()

    bin_indices = torch.bucketize(survival_times, torch.tensor(cut_points))
    return bin_indices, cut_points
