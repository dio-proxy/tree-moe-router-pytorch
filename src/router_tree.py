"""Two-level hierarchical (tree) router.

Routing is factorised into two decisions:

1. **Level 1 (cluster):** ``P(c | x) = softmax(W_c x)`` over ``num_clusters`` clusters,
   and each token is sent to its arg-max cluster ``c*``.
2. **Level 2 (expert):** TODO - pick an expert inside the selected cluster.
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
        """Route each token to one cluster (level 1 only for now).

        Args:
            x: Token representations of shape ``[..., hidden_dim]``.

        Returns:
            ``(cluster_indices, cluster_weights)``, both of shape ``[...]``.
        """
        leading_shape = x.shape[:-1]
        tokens = x.reshape(-1, self.hidden_dim)

        cluster_probs = F.softmax(self.cluster_gate(tokens), dim=-1)  # [T, C]
        cluster_weight, cluster_idx = cluster_probs.max(dim=-1)

        return cluster_idx.reshape(leading_shape), cluster_weight.reshape(leading_shape)

    def extra_repr(self) -> str:
        """Readable summary shown by ``print(module)``."""
        return (
            f"hidden_dim={self.hidden_dim}, num_clusters={self.num_clusters}, "
            f"experts_per_cluster={self.experts_per_cluster}"
        )
