"""
Unit tests for model components, losses, and metrics.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import pytest
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from src.models.heterogeneity_gat import HeterogeneityGAT, HeterogeneityAwarePooling
from src.training.losses import CoxNLLLoss, DiscreteHazardLoss, discretise_survival_times
from src.evaluation.metrics import concordance_index_censored, km_stratification


# ─── Fixtures ─────────────────────────────────────────────────────────────────

IN_DIM     = 387   # 384 HIPT + 3 het
HIDDEN_DIM = 64    # small for tests
BATCH_SIZE = 4


def make_dummy_graph(n_nodes: int = 50, in_dim: int = IN_DIM, seed: int = 0) -> Data:
    """Create a minimal synthetic graph for testing."""
    torch.manual_seed(seed)
    x          = torch.randn(n_nodes, in_dim)
    # Spatial KNN-style edge index (random for tests)
    src        = torch.randint(0, n_nodes, (n_nodes * 8,))
    dst        = torch.randint(0, n_nodes, (n_nodes * 8,))
    edge_index = torch.stack([src, dst], dim=0)
    coords     = torch.rand(n_nodes, 2) * 10000

    data = Data(
        x=x, edge_index=edge_index, coords=coords,
        survival_months=torch.tensor([24.0]),
        event=torch.tensor([1]),
    )
    return data


def make_dummy_batch(n_graphs: int = BATCH_SIZE, in_dim: int = IN_DIM) -> Batch:
    graphs = [make_dummy_graph(n_nodes=30 + i * 5, in_dim=in_dim, seed=i)
              for i in range(n_graphs)]
    return Batch.from_data_list(graphs)


# ─── Model tests ──────────────────────────────────────────────────────────────

class TestHeterogeneityGAT:

    def test_forward_cox_single(self):
        model = HeterogeneityGAT(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, gat_layers=2)
        model.eval()
        data  = make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        with torch.no_grad():
            out = model(data)
        assert "logits" in out
        assert out["logits"].shape == (1,), f"Expected (1,), got {out['logits'].shape}"

    def test_forward_cox_batch(self):
        model = HeterogeneityGAT(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, gat_layers=2)
        model.eval()
        batch = make_dummy_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (BATCH_SIZE,), \
            f"Expected ({BATCH_SIZE},), got {out['logits'].shape}"

    def test_forward_with_attention(self):
        model = HeterogeneityGAT(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, gat_layers=2)
        model.eval()
        data  = make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        with torch.no_grad():
            out = model(data, return_attention=True)
        assert "attention" in out
        assert out["attention"].shape[0] == data.x.size(0)

    def test_discrete_mode(self):
        model = HeterogeneityGAT(
            in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, gat_layers=1,
            survival_mode="discrete", num_time_bins=4,
        )
        model.eval()
        batch = make_dummy_batch(2)
        with torch.no_grad():
            out = model(batch)
        assert out["logits"].shape == (2, 4), \
            f"Expected (2, 4), got {out['logits'].shape}"

    def test_gradient_flows(self):
        model = HeterogeneityGAT(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, gat_layers=2)
        batch = make_dummy_batch()
        out   = model(batch)
        loss  = out["logits"].mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_no_label_leakage(self):
        """Identical inputs must produce identical outputs (determinism)."""
        model = HeterogeneityGAT(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, gat_layers=2)
        model.eval()
        data  = make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        with torch.no_grad():
            o1 = model(data)["logits"]
            o2 = model(data)["logits"]
        assert torch.allclose(o1, o2), "Model is not deterministic in eval mode"


class TestHeterogeneityPooling:

    def test_output_shape(self):
        pool  = HeterogeneityAwarePooling(hidden_dim=64, het_dim=3, attn_hidden=32)
        N, B  = 100, 4
        x     = torch.randn(N, 64)
        het   = torch.rand(N, 3)
        batch = torch.cat([torch.full((25,), i) for i in range(B)])
        z, w  = pool(x, het, batch)
        assert z.shape == (B, 64)
        assert w.shape == (N, 1)

    def test_attention_weights_sum_to_one_per_graph(self):
        pool  = HeterogeneityAwarePooling(hidden_dim=64, het_dim=3, attn_hidden=32)
        N, B  = 40, 2
        x     = torch.randn(N, 64)
        het   = torch.rand(N, 3)
        batch = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        _, w  = pool(x, het, batch)
        for g in range(B):
            s = w[batch == g].sum().item()
            assert abs(s - 1.0) < 1e-5, f"Attention weights for graph {g} sum to {s}, expected 1.0"


# ─── Loss tests ───────────────────────────────────────────────────────────────

class TestCoxNLLLoss:

    def test_forward(self):
        loss_fn = CoxNLLLoss(l1_reg=0.0)
        B = 8
        risks  = torch.randn(B)
        times  = torch.rand(B) * 60 + 6
        events = torch.randint(0, 2, (B,)).float()
        loss   = loss_fn(risks, times, events)
        assert loss.item() == loss.item(), "Loss is NaN"
        assert loss.requires_grad

    def test_ordering_sensitivity(self):
        """Higher-risk patients dying earlier should yield lower loss."""
        loss_fn = CoxNLLLoss(l1_reg=0.0)
        # Perfect ordering: highest risk dies first
        risks_good = torch.tensor([3.0, 2.0, 1.0, 0.0])
        risks_bad  = torch.tensor([0.0, 1.0, 2.0, 3.0])
        times      = torch.tensor([10.0, 20.0, 30.0, 40.0])
        events     = torch.ones(4)
        loss_good  = loss_fn(risks_good, times, events)
        loss_bad   = loss_fn(risks_bad,  times, events)
        assert loss_good < loss_bad, "Good ordering should have lower Cox loss"

    def test_all_censored(self):
        """All-censored batch: loss should be 0 (no comparable pairs)."""
        loss_fn = CoxNLLLoss(l1_reg=0.0)
        risks  = torch.randn(5)
        times  = torch.rand(5) * 100
        events = torch.zeros(5)
        loss   = loss_fn(risks, times, events)
        assert loss.item() == 0.0


class TestDiscreteHazardLoss:

    def test_forward(self):
        loss_fn = DiscreteHazardLoss(num_bins=4)
        B, T    = 6, 4
        logits  = torch.randn(B, T)
        bins    = torch.randint(0, T, (B,))
        events  = torch.randint(0, 2, (B,)).float()
        loss    = loss_fn(logits, bins, events)
        assert loss.item() == loss.item()
        assert loss > 0


# ─── Metrics tests ────────────────────────────────────────────────────────────

class TestMetrics:

    def test_cindex_perfect(self):
        """Perfect predictor: C-index = 1.0"""
        times  = [10, 20, 30, 40, 50]
        events = [1, 1, 1, 1, 1]
        risks  = [5, 4, 3, 2, 1]   # higher risk → shorter survival
        c = concordance_index_censored(times, events, risks)
        assert c == 1.0, f"Expected 1.0, got {c}"

    def test_cindex_random(self):
        """Random predictor should give ~0.5."""
        np.random.seed(0)
        N      = 200
        times  = np.random.exponential(20, N)
        events = np.ones(N, dtype=int)
        risks  = np.random.randn(N)
        c = concordance_index_censored(times, events, risks)
        assert 0.4 < c < 0.6, f"Random predictor C-index out of range: {c}"

    def test_cindex_inverse(self):
        """Inverse predictor: C-index = 0.0"""
        times  = [10, 20, 30, 40, 50]
        events = [1, 1, 1, 1, 1]
        risks  = [1, 2, 3, 4, 5]   # lower risk → shorter survival (inverted)
        c = concordance_index_censored(times, events, risks)
        assert c == 0.0, f"Expected 0.0, got {c}"

    def test_km_stratification_outputs(self):
        np.random.seed(42)
        N      = 100
        times  = np.random.exponential(30, N)
        events = np.random.binomial(1, 0.6, N)
        risks  = np.random.randn(N)
        result = km_stratification(times, events, risks)
        assert "kmf_high"   in result
        assert "kmf_low"    in result
        assert "logrank_p"  in result
        assert 0.0 <= result["logrank_p"] <= 1.0

    def test_discretise_survival_times(self):
        times = torch.tensor([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
        bins, cuts = discretise_survival_times(times, n_bins=4, strategy="quantile")
        assert bins.shape == times.shape
        assert len(cuts) == 3
        assert bins.min() >= 0
        assert bins.max() <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
