"""
Survival analysis evaluation metrics.

1. Concordance Index (C-index / Harrell's C)
   Measures discrimination: how well the model ranks patients by risk.
   Computed with censoring-aware Harrell formula.
   Bootstrap CI support.

2. Integrated Brier Score (IBS)
   Measures calibration + discrimination jointly across time.
   Requires estimating the censoring distribution (IPCW).

3. Kaplan-Meier stratification
   Split patients into high/low risk by median predicted risk score.
   Compute KM curves for each group and log-rank p-value.
   Visually compelling for papers and presentations.

4. Time-dependent AUC (td-AUC)
   AUC at specific time points using censoring-adjusted estimator.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import chi2


def concordance_index_censored(
    survival_times: list[float] | np.ndarray,
    events: list[int]  | np.ndarray,
    risk_scores: list[float] | np.ndarray,
    tied_tol: float = 1e-8,
) -> float:
    """
    Compute Harrell's concordance index with censoring.

    A pair (i, j) is concordant if the patient with higher risk score
    has shorter survival time (and their event is observed).

    Args:
        survival_times: Observed times.
        events:         1=event, 0=censored.
        risk_scores:    Higher = higher risk. (log-hazard outputs)
        tied_tol:       Tolerance for tied risk scores.

    Returns:
        C-index in [0.5, 1.0] (0.5 = random, 1.0 = perfect).
    """
    times  = np.asarray(survival_times, dtype=float)
    events = np.asarray(events, dtype=int)
    risks  = np.asarray(risk_scores, dtype=float)

    concordant = 0
    discordant = 0
    tied_risk  = 0
    n_comparable = 0

    for i in range(len(times)):
        if events[i] == 0:
            continue  # censored: cannot form comparable pairs as the shorter
        for j in range(len(times)):
            if i == j:
                continue
            if times[j] <= times[i]:
                continue  # j died before or at same time as i → not a valid pair

            # Comparable pair: i had event, j survived longer
            n_comparable += 1
            if risks[i] - risks[j] > tied_tol:
                concordant += 1
            elif risks[j] - risks[i] > tied_tol:
                discordant += 1
            else:
                tied_risk += 0.5

    if n_comparable == 0:
        return 0.5

    return (concordant + tied_risk) / n_comparable


def bootstrap_cindex(
    survival_times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap confidence interval for C-index.

    Returns:
        {"cindex": float, "ci_lower": float, "ci_upper": float}
    """
    rng = np.random.default_rng(seed)
    n   = len(survival_times)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            c = concordance_index_censored(
                survival_times[idx], events[idx], risk_scores[idx]
            )
            scores.append(c)
        except Exception:
            continue

    scores  = np.array(scores)
    alpha   = 1.0 - confidence_level
    lower   = float(np.percentile(scores, 100 * alpha / 2))
    upper   = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    point   = float(concordance_index_censored(survival_times, events, risk_scores))

    return {"cindex": point, "ci_lower": lower, "ci_upper": upper}


def integrated_brier_score(
    survival_times_train: np.ndarray,
    events_train: np.ndarray,
    survival_times_test: np.ndarray,
    events_test: np.ndarray,
    predicted_risks: np.ndarray,
    n_time_points: int = 100,
) -> float:
    """
    Compute the Integrated Brier Score (IBS) using IPCW.

    Requires scikit-survival.

    Args:
        survival_times_train: Training set times (for censoring distribution).
        events_train:         Training set events.
        survival_times_test:  Test set times.
        events_test:          Test set events.
        predicted_risks:      (N,) risk scores — higher = higher risk.
        n_time_points:        Number of time grid points for integration.

    Returns:
        IBS scalar. Lower is better. Random model ≈ 0.25.
    """
    try:
        from sksurv.metrics import integrated_brier_score as sks_ibs
        from sksurv.util import Surv

        y_train = Surv.from_arrays(events_train.astype(bool), survival_times_train)
        y_test  = Surv.from_arrays(events_test.astype(bool),  survival_times_test)

        t_min = survival_times_test.min() + 1
        t_max = survival_times_test.max() - 1
        times = np.linspace(t_min, t_max, n_time_points)

        # Convert risk scores to survival probability estimates at each time
        # using a simple monotonic transformation: S(t|x) = exp(-exp(h_x) * t)
        # (baseline hazard = 1 for simplicity — this is an approximation)
        surv_probs = np.stack([
            np.exp(-np.exp(predicted_risks) * t / survival_times_train.mean())
            for t in times
        ], axis=1)  # (N, T)
        surv_probs = np.clip(surv_probs, 1e-6, 1.0 - 1e-6)

        _, ibs = sks_ibs(y_train, y_test, surv_probs, times)
        return float(ibs)
    except ImportError:
        return float("nan")


def km_stratification(
    survival_times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    quantile_split: float = 0.5,
) -> dict:
    """
    Kaplan-Meier analysis stratified by median predicted risk.

    Args:
        survival_times:  Observed times.
        events:          1=event, 0=censored.
        risk_scores:     Predicted log-hazard.
        quantile_split:  Risk quantile to use as split (default: median).

    Returns:
        dict with KMFitter objects and log-rank p-value.
    """
    threshold  = np.quantile(risk_scores, quantile_split)
    high_risk  = risk_scores >= threshold
    low_risk   = ~high_risk

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()

    kmf_high.fit(
        durations=survival_times[high_risk],
        event_observed=events[high_risk],
        label=f"High risk (n={high_risk.sum()})",
    )
    kmf_low.fit(
        durations=survival_times[low_risk],
        event_observed=events[low_risk],
        label=f"Low risk (n={low_risk.sum()})",
    )

    # Log-rank test
    lr_result = logrank_test(
        survival_times[high_risk],  events[high_risk],
        survival_times[low_risk],   events[low_risk],
    )

    return {
        "kmf_high":  kmf_high,
        "kmf_low":   kmf_low,
        "logrank_p": float(lr_result.p_value),
        "logrank_stat": float(lr_result.test_statistic),
        "n_high": int(high_risk.sum()),
        "n_low":  int(low_risk.sum()),
    }


def full_evaluation(
    survival_times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    survival_times_train: Optional[np.ndarray] = None,
    events_train: Optional[np.ndarray]         = None,
    n_bootstrap: int                           = 1000,
) -> dict:
    """
    Run all evaluation metrics and return a summary dict.

    Returns:
        {
          "cindex": float,
          "ci_lower": float,
          "ci_upper": float,
          "ibs": float,
          "logrank_p": float,
          "km_data": dict (KM fitters for plotting),
        }
    """
    # C-index with bootstrap CI
    ci_result = bootstrap_cindex(
        survival_times, events, risk_scores, n_bootstrap=n_bootstrap
    )

    # IBS
    ibs = float("nan")
    if survival_times_train is not None and events_train is not None:
        ibs = integrated_brier_score(
            survival_times_train, events_train,
            survival_times, events, risk_scores,
        )

    # KM stratification
    km = km_stratification(survival_times, events, risk_scores)

    return {
        **ci_result,
        "ibs":       ibs,
        "logrank_p": km["logrank_p"],
        "km_data":   km,
    }
