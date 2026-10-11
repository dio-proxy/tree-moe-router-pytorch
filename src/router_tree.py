"""Two-level hierarchical (tree) router with auxiliary load-balancing loss.

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


def load_balancing_loss(probs: torch.Tensor, indices: torch.Tensor, num_choices: int) -> torch.Tensor:
    """Switch-Transformer style load-balancing loss ``N * sum_i f_i * P_i``.

    ``f_i`` (fraction of tokens dispatched to choice ``i``) is computed from the hard
    arg-max assignment and is not differentiable; gradients flow through ``P_i`` (the
    mean router probability for choice ``i``). The loss equals ``1.0`` under a perfectly
    uniform distribution and grows towards ``num_choices`` as routing collapses.

    Args:
        probs: Router probabilities of shape ``[num_tokens, num_choices]``.
        indices: Selected choice per token, shape ``[num_tokens]`` (int64).
        num_choices: Number of choices ``N`` (clusters or experts).

    Returns:
        A scalar tensor.
    """
    if probs.shape[0] == 0:
        return probs.new_zeros(())
    dispatch_fraction = F.one_hot(indices, num_choices).to(probs.dtype).mean(dim=0)  # f_i
    mean_prob = probs.mean(dim=0)  # P_i
    return num_choices * torch.sum(dispatch_fraction * mean_prob)


class TreeRouter(nn.Module):
    """Hierarchical 2-level top-1 router.

    Args:
        hidden_dim: Dimension of incoming token representations.
        num_clusters: Number of level-1 clusters ``C``.
        experts_per_cluster: Number of experts inside each cluster ``E``.
        cluster_loss_coef: Weight of the cluster-level balancing loss in ``aux_loss``.
        expert_loss_coef: Weight of the expert-level balancing loss in ``aux_loss``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_clusters: int,
        experts_per_cluster: int,
        cluster_loss_coef: float = 1.0,
        expert_loss_coef: float = 1.0,
    ) -> None:
        super().__init__()
        if num_clusters < 1 or experts_per_cluster < 1:
            raise ValueError("num_clusters and experts_per_cluster must both be >= 1.")

        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.num_experts = num_clusters * experts_per_cluster
        self.cluster_loss_coef = cluster_loss_coef
        self.expert_loss_coef = expert_loss_coef

        # Level 1: one gate choosing among clusters.
        self.cluster_gate = nn.Linear(hidden_dim, num_clusters, bias=False)
        # Level 2: one gate per cluster choosing among that cluster's experts.
        self.expert_gates = nn.ModuleList(
            nn.Linear(hidden_dim, experts_per_cluster, bias=False) for _ in range(num_clusters)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route each token to one global expert.

        Args:
            x: Token representations of shape ``[..., hidden_dim]``.

        Returns:
            A tuple ``(global_expert_indices, routing_weights, aux_loss)`` where

            * ``global_expert_indices``: int64 tensor of shape ``[...]`` with values in
              ``[0, num_clusters * experts_per_cluster - 1]``.
            * ``routing_weights``: tensor of shape ``[...]`` holding
              ``P(cluster) * P(expert | cluster)`` for the selected path (differentiable).
            * ``aux_loss``: scalar ``cluster_loss_coef * L_cluster + expert_loss_coef * L_expert``.
        """
        leading_shape = x.shape[:-1]
        tokens = x.reshape(-1, self.hidden_dim)
        num_tokens = tokens.shape[0]

        # ---- Level 1: cluster routing (softmax in fp32 for numerical stability) ----
        cluster_probs = F.softmax(self.cluster_gate(tokens).float(), dim=-1)  # [T, C]
        cluster_weight, cluster_idx = cluster_probs.max(dim=-1)  # [T], [T]
        cluster_loss = load_balancing_loss(cluster_probs, cluster_idx, self.num_clusters)

        # ---- Level 2: expert routing within the selected cluster ----
        expert_weight = cluster_probs.new_zeros(num_tokens)
        local_expert_idx = torch.zeros(num_tokens, dtype=torch.long, device=x.device)
        expert_loss = cluster_probs.new_zeros(())

        for c, gate in enumerate(self.expert_gates):
            token_ids = torch.nonzero(cluster_idx == c, as_tuple=True)[0]
            if token_ids.numel() == 0:
                continue
            probs_c = F.softmax(gate(tokens[token_ids]).float(), dim=-1)  # [T_c, E]
            weight_c, idx_c = probs_c.max(dim=-1)
            expert_weight = expert_weight.index_put((token_ids,), weight_c)
            local_expert_idx[token_ids] = idx_c

            # Per-cluster balancing loss, weighted by the share of tokens the cluster received
            # so that the combined term also equals 1.0 under perfect balance.
            share = token_ids.numel() / num_tokens
            expert_loss = expert_loss + share * load_balancing_loss(
                probs_c, idx_c, self.experts_per_cluster
            )

        # ---- Combine: P(final expert) = P(cluster) * P(expert | cluster) ----
        routing_weights = (cluster_weight * expert_weight).to(x.dtype)
        global_expert_indices = cluster_idx * self.experts_per_cluster + local_expert_idx
        aux_loss = self.cluster_loss_coef * cluster_loss + self.expert_loss_coef * expert_loss

        return (
            global_expert_indices.reshape(leading_shape),
            routing_weights.reshape(leading_shape),
            aux_loss,
        )

    def extra_repr(self) -> str:
        """Readable summary shown by ``print(module)``."""
        return (
            f"hidden_dim={self.hidden_dim}, num_clusters={self.num_clusters}, "
            f"experts_per_cluster={self.experts_per_cluster}"
        )
