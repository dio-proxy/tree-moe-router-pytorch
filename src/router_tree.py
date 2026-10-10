"""Two-level hierarchical (tree) router.

Routing is factorised into two decisions:

1. **Level 1 (cluster):** ``P(c | x) = softmax(W_c x)`` over ``num_clusters`` clusters,
   and each token is sent to its arg-max cluster ``c*``.
2. **Level 2 (expert):** ``P(e | x, c*) = softmax(W_e^{(c*)} x)`` over the
   ``experts_per_cluster`` experts of the selected cluster only, and each token is
   sent to its arg-max expert ``e*``.

The combined gate value is ``P(c* | x) * P(e* | x, c*)`` and the flat (global) expert
id is ``c* * experts_per_cluster + e*``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeRouter(nn.Module):
    """Hierarchical 2-level top-1 router.

    Args:
        hidden_dim: Dimension of incoming token representations.
        num_clusters: Number of level-1 clusters ``C``.
        experts_per_cluster: Number of experts inside each cluster ``E``.
    """

    def __init__(self, hidden_dim: int, num_clusters: int, experts_per_cluster: int) -> None:
        super().__init__()
        if num_clusters < 1 or experts_per_cluster < 1:
            raise ValueError("num_clusters and experts_per_cluster must both be >= 1.")

        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.num_experts = num_clusters * experts_per_cluster

        # Level 1: one gate choosing among clusters.
        self.cluster_gate = nn.Linear(hidden_dim, num_clusters, bias=False)
        # Level 2: one gate per cluster choosing among that cluster's experts.
        self.expert_gates = nn.ModuleList(
            nn.Linear(hidden_dim, experts_per_cluster, bias=False) for _ in range(num_clusters)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route each token to one global expert.

        Args:
            x: Token representations of shape ``[..., hidden_dim]``.

        Returns:
            A tuple ``(global_expert_indices, routing_weights)`` where

            * ``global_expert_indices``: int64 tensor of shape ``[...]`` with values in
              ``[0, num_clusters * experts_per_cluster - 1]``.
            * ``routing_weights``: tensor of shape ``[...]`` holding
              ``P(cluster) * P(expert | cluster)`` for the selected path (differentiable).
        """
        leading_shape = x.shape[:-1]
        tokens = x.reshape(-1, self.hidden_dim)
        num_tokens = tokens.shape[0]

        # ---- Level 1: cluster routing (softmax in fp32 for numerical stability) ----
        cluster_probs = F.softmax(self.cluster_gate(tokens).float(), dim=-1)  # [T, C]
        cluster_weight, cluster_idx = cluster_probs.max(dim=-1)  # [T], [T]

        # ---- Level 2: expert routing within the selected cluster ----
        expert_weight = cluster_probs.new_zeros(num_tokens)
        local_expert_idx = torch.zeros(num_tokens, dtype=torch.long, device=x.device)

        for c, gate in enumerate(self.expert_gates):
            token_ids = torch.nonzero(cluster_idx == c, as_tuple=True)[0]
            if token_ids.numel() == 0:
                continue
            probs_c = F.softmax(gate(tokens[token_ids]).float(), dim=-1)  # [T_c, E]
            weight_c, idx_c = probs_c.max(dim=-1)
            expert_weight = expert_weight.index_put((token_ids,), weight_c)
            local_expert_idx[token_ids] = idx_c

        # ---- Combine: P(final expert) = P(cluster) * P(expert | cluster) ----
        routing_weights = (cluster_weight * expert_weight).to(x.dtype)
        global_expert_indices = cluster_idx * self.experts_per_cluster + local_expert_idx

        return global_expert_indices.reshape(leading_shape), routing_weights.reshape(leading_shape)

    def extra_repr(self) -> str:
        """Readable summary shown by ``print(module)``."""
        return (
            f"hidden_dim={self.hidden_dim}, num_clusters={self.num_clusters}, "
            f"experts_per_cluster={self.experts_per_cluster}"
        )
