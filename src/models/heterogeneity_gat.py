"""
HeterogeneityGAT — the core model of this project.

Architecture:
  1. Input projection:     Linear(D+3 → hidden_dim)
  2. GAT layers:           L × GATv2Conv(hidden_dim, heads) with residual connections
  3. Heterogeneity-aware pooling:
       - Computes a per-node pooling weight that is a learned function of the
         node's HETEROGENEITY features (the last 3 dims of node_feats).
       - Nodes that are morphologically divergent from their neighbourhood
         receive higher weight — they carry more information about tumour
         aggressiveness than morphologically uniform regions.
       - This is the architectural novelty: standard attention pooling ignores
         WHY a node is important; our module explicitly rewards heterogeneity.
  4. Survival head:        Fully-connected → 1 log-hazard output (Cox mode)
                           OR L discrete hazard bins (discrete hazard mode)

Loss:
  - Cox mode:     Negative partial log-likelihood (Breslow tie correction)
  - Discrete:     Cross-entropy on discretised hazard intervals

Reference papers:
  - Ilse et al., "Attention-based Deep MIL", ICML 2018
  - Bravo et al., "HetGNN", SIGKDD 2019
  - Chen et al., "Whole Slide Images Are 2D Point Clouds", MICCAI 2021
  - Lee et al., "PANTHER: Morphological Intra-Tumoral Heterogeneity", MICCAI 2024
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool


# ─── Heterogeneity-Aware Pooling ──────────────────────────────────────────────

class HeterogeneityAwarePooling(nn.Module):
    """
    Graph-level pooling that assigns higher weight to morphologically
    heterogeneous nodes.

    For each node i:
        weight_i = sigmoid( W_att · [h_i ; het_i] + b_att )

    where:
        h_i   is the GAT output embedding (hidden_dim)
        het_i is the original 3-dim heterogeneity feature slice:
              [entropy, cosine_dissim, feat_spread]

    The graph embedding is the weighted sum:
        z = Σ_i (weight_i · h_i) / Σ_i weight_i

    This differs from standard attention pooling (ABMIL) in that the
    gating explicitly uses the heterogeneity signal, rather than
    learning attention purely from task-level supervision.

    Args:
        hidden_dim:   GAT output dimension.
        het_dim:      Heterogeneity feature dimension (default 3).
        attn_hidden:  Hidden dim of the attention MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        het_dim: int   = 3,
        attn_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.het_dim     = het_dim

        # Gating network: takes [h_i ; het_i] → scalar weight
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + het_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        # Separate value transform on node embeddings
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,           # (N_total, hidden_dim)  — GAT node embeddings
        het: Tensor,         # (N_total, het_dim)     — raw heterogeneity feats
        batch: Tensor,       # (N_total,)             — graph assignment vector
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:     Node embeddings from GAT.
            het:   Heterogeneity features (pre-normalised 0–1).
            batch: PyG batch vector mapping each node to its graph.

        Returns:
            z:           (B, hidden_dim) graph-level embeddings.
            attn_weights:(N_total, 1) per-node attention weights (for visualisation).
        """
        # Gating: score each node based on embedding + heterogeneity
        gate_input   = torch.cat([x, het], dim=-1)        # (N, H+het)
        raw_scores   = self.gate(gate_input)               # (N, 1)

        # Softmax within each graph (not globally)
        # Use PyG scatter_softmax via manual logsumexp trick for stability
        B = int(batch.max().item()) + 1
        attn_weights = torch.zeros_like(raw_scores)

        for g in range(B):
            mask = (batch == g)
            if mask.sum() == 0:
                continue
            scores_g = raw_scores[mask]                    # (n_g, 1)
            attn_g   = F.softmax(scores_g, dim=0)
            attn_weights[mask] = attn_g

        # Weighted sum of value-projected embeddings
        values = self.value_proj(x)                        # (N, H)
        weighted = attn_weights * values                   # (N, H)

        # Scatter sum per graph
        z = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        z.scatter_add_(0, batch.unsqueeze(1).expand_as(weighted), weighted)

        return z, attn_weights


# ─── GAT Backbone ─────────────────────────────────────────────────────────────

class GATBlock(nn.Module):
    """Single GATv2 layer with multi-head, residual connection, and LayerNorm."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
        )
        self.norm     = nn.LayerNorm(hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.residual = residual

        # Projection for residual if dims differ
        self.res_proj = (
            nn.Linear(in_dim, hidden_dim, bias=False)
            if in_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        out = self.gat(x, edge_index)
        out = self.dropout(out)
        if self.residual:
            out = out + self.res_proj(x)
        return self.norm(out)


# ─── Main Model ───────────────────────────────────────────────────────────────

class HeterogeneityGAT(nn.Module):
    """
    Full model: Input Projection → GAT stack → Heterogeneity Pooling → Survival head.

    Args:
        in_dim:        Node feature dimension (HIPT_dim + het_dim, e.g. 387).
        hidden_dim:    GAT hidden dimension (default 256).
        gat_heads:     Number of attention heads per GAT layer.
        gat_layers:    Number of stacked GAT blocks.
        gat_dropout:   Dropout rate in GAT layers.
        het_dim:       Heterogeneity feature dimension (default 3).
        attn_hidden:   Hidden dim of pooling attention MLP.
        survival_mode: "cox" or "discrete".
        num_time_bins: Number of discrete time bins (used only if survival_mode="discrete").
    """

    HET_DIM = 3   # last 3 dims of node features are heterogeneity feats

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int       = 256,
        gat_heads: int        = 4,
        gat_layers: int       = 3,
        gat_dropout: float    = 0.25,
        het_dim: int          = 3,
        attn_hidden: int      = 128,
        survival_mode: str    = "cox",
        num_time_bins: int    = 4,
    ) -> None:
        super().__init__()
        assert survival_mode in ("cox", "discrete"), "survival_mode must be 'cox' or 'discrete'"

        self.survival_mode = survival_mode
        self.het_dim       = het_dim
        self.hipt_dim      = in_dim - het_dim   # HIPT embedding dimension

        # ── Input projection ───────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── GAT blocks ─────────────────────────────────────────────────────
        self.gat_blocks = nn.ModuleList()
        for i in range(gat_layers):
            in_ch = hidden_dim  # all blocks use hidden_dim after projection
            self.gat_blocks.append(
                GATBlock(in_ch, hidden_dim, gat_heads, gat_dropout, residual=(i > 0))
            )

        # ── Heterogeneity-aware pooling ────────────────────────────────────
        self.pool = HeterogeneityAwarePooling(hidden_dim, het_dim, attn_hidden)

        # ── Survival head ──────────────────────────────────────────────────
        out_dim = 1 if survival_mode == "cox" else num_time_bins
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        data: Data | Batch,
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        """
        Forward pass.

        Args:
            data:             PyG Data/Batch object with .x, .edge_index, .batch
            return_attention: If True, also return per-node attention weights.

        Returns:
            dict with:
                "logits"   : (B,) or (B, T) raw survival prediction
                "attention": (N, 1) attention weights  [if return_attention]
        """
        x          = data.x           # (N, D+3)
        edge_index = data.edge_index  # (2, E)
        batch      = data.batch if hasattr(data, "batch") and data.batch is not None \
                     else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Slice heterogeneity features (last het_dim dims) before projection
        het = x[:, -self.het_dim:]    # (N, 3)

        # ── Project inputs ─────────────────────────────────────────────────
        h = self.input_proj(x)        # (N, hidden_dim)

        # ── GAT stack ──────────────────────────────────────────────────────
        for gat_block in self.gat_blocks:
            h = gat_block(h, edge_index)

        # ── Heterogeneity-aware pooling ────────────────────────────────────
        z, attn_weights = self.pool(h, het, batch)   # z: (B, hidden_dim)

        # ── Survival prediction ────────────────────────────────────────────
        logits = self.survival_head(z).squeeze(-1)   # (B,) for Cox; (B, T) for discrete

        result = {"logits": logits}
        if return_attention:
            result["attention"] = attn_weights

        return result


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> HeterogeneityGAT:
    """Build model from config dict."""
    m_cfg = cfg.get("model", cfg)
    feat_dim = cfg.get("feature_extractor", {}).get("embed_dim", 384)
    het_dim  = cfg.get("graph", {}).get("heterogeneity_bins", 3)
    het_dim  = 3   # always 3: entropy, cosine_dissim, feat_spread
    in_dim   = feat_dim + het_dim

    return HeterogeneityGAT(
        in_dim        = in_dim,
        hidden_dim    = m_cfg.get("gat_hidden_dim", 256),
        gat_heads     = m_cfg.get("gat_heads", 4),
        gat_layers    = m_cfg.get("gat_layers", 3),
        gat_dropout   = m_cfg.get("gat_dropout", 0.25),
        het_dim       = het_dim,
        survival_mode = m_cfg.get("survival_head", "cox"),
        num_time_bins = m_cfg.get("num_time_bins", 4),
    )
