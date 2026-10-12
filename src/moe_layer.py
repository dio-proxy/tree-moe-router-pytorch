"""Tree Mixture-of-Experts layer: a :class:`TreeRouter` plus a flat bank of experts."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .experts import Expert
from .router_tree import TreeRouter


class TreeMoELayer(nn.Module):
    """Sparse MoE layer with hierarchical (cluster -> expert) top-1 routing.

    Experts are stored in a flat ``nn.ModuleList`` where expert ``c * experts_per_cluster + e``
    is the ``e``-th expert of cluster ``c``, matching the router's global indices.

    Args:
        hidden_dim: Model dimension ``d_model``.
        intermediate_dim: Inner FFN dimension of each expert.
        num_clusters: Number of level-1 clusters.
        experts_per_cluster: Number of experts per cluster.
        activation: Expert activation, ``"swiglu"`` or ``"gelu"``.
        dropout: Dropout inside each expert.
        cluster_loss_coef: Weight of the cluster-level balancing loss.
        expert_loss_coef: Weight of the expert-level balancing loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_clusters: int = 4,
        experts_per_cluster: int = 4,
        activation: str = "swiglu",
        dropout: float = 0.0,
        cluster_loss_coef: float = 1.0,
        expert_loss_coef: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.num_experts = num_clusters * experts_per_cluster

        self.router = TreeRouter(
            hidden_dim,
            num_clusters,
            experts_per_cluster,
            cluster_loss_coef=cluster_loss_coef,
            expert_loss_coef=expert_loss_coef,
        )
        self.experts = nn.ModuleList(
            Expert(hidden_dim, intermediate_dim, activation=activation, dropout=dropout)
            for _ in range(self.num_experts)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to their experts and combine the weighted outputs.

        Args:
            x: Input of shape ``[batch_size, seq_len, hidden_dim]`` (any leading dims work).

        Returns:
            ``(output, aux_loss)`` where ``output`` has the same shape as ``x`` and
            ``aux_loss`` is the router's scalar load-balancing loss.
        """
        expert_indices, routing_weights, aux_loss = self.router(x)

        tokens = x.reshape(-1, self.hidden_dim)
        flat_indices = expert_indices.reshape(-1)
        flat_weights = routing_weights.reshape(-1, 1)
        output = torch.zeros_like(tokens)

        # Group tokens by expert once (a single sort) instead of masking per expert.
        sorted_indices, order = torch.sort(flat_indices, stable=True)
        counts = torch.bincount(sorted_indices, minlength=self.num_experts).tolist()

        start = 0
        for expert_id, count in enumerate(counts):
            if count == 0:
                continue
            token_ids = order[start : start + count]
            start += count
            expert_out = self.experts[expert_id](tokens[token_ids])
            output = output.index_add(0, token_ids, expert_out * flat_weights[token_ids])

        return output.reshape(x.shape), aux_loss

    @torch.no_grad()
    def expert_load(self, x: torch.Tensor) -> torch.Tensor:
        """Count how many tokens of ``x`` are routed to each expert.

        Args:
            x: Input of shape ``[..., hidden_dim]``.

        Returns:
            int64 tensor of shape ``[num_experts]`` with per-expert token counts.
        """
        expert_indices, _, _ = self.router(x)
        return torch.bincount(expert_indices.reshape(-1), minlength=self.num_experts)
