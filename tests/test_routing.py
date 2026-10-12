"""Unit tests for the tree router and Tree MoE layer."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src import Expert, TreeMoELayer, TreeRouter, load_balancing_loss

BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, INTERMEDIATE_DIM = 4, 16, 32, 64
NUM_CLUSTERS, EXPERTS_PER_CLUSTER = 4, 4


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Make every test deterministic."""
    torch.manual_seed(0)


@pytest.fixture
def layer() -> TreeMoELayer:
    """A small Tree MoE layer shared by the tests."""
    return TreeMoELayer(HIDDEN_DIM, INTERMEDIATE_DIM, NUM_CLUSTERS, EXPERTS_PER_CLUSTER)


@pytest.fixture
def inputs() -> torch.Tensor:
    """Random input of shape ``[batch_size, seq_len, hidden_dim]``."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)


# ---------------------------------------------------------------- (a) shapes


@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
def test_expert_preserves_shape(activation: str) -> None:
    """An expert maps ``[..., hidden_dim]`` to the same shape."""
    expert = Expert(HIDDEN_DIM, INTERMEDIATE_DIM, activation=activation)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    assert expert(x).shape == x.shape


def test_layer_output_shape_matches_input(layer: TreeMoELayer, inputs: torch.Tensor) -> None:
    """The MoE output has shape ``[batch_size, seq_len, hidden_dim]``."""
    output, _ = layer(inputs)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    assert output.dtype == inputs.dtype


def test_router_output_shapes(layer: TreeMoELayer, inputs: torch.Tensor) -> None:
    """Router indices and weights carry one entry per token."""
    indices, weights, aux_loss = layer.router(inputs)
    assert indices.shape == (BATCH_SIZE, SEQ_LEN)
    assert weights.shape == (BATCH_SIZE, SEQ_LEN)
    assert indices.dtype == torch.long
    assert aux_loss.dim() == 0


# ---------------------------------------------------------------- (b) aux loss and gradients


def test_aux_loss_is_nonzero_scalar(layer: TreeMoELayer, inputs: torch.Tensor) -> None:
    """The auxiliary loss is a finite, strictly positive scalar."""
    _, aux_loss = layer(inputs)
    assert aux_loss.dim() == 0
    assert torch.isfinite(aux_loss)
    assert aux_loss.item() > 0.0


def test_backward_succeeds_and_reaches_all_parameter_groups(
    layer: TreeMoELayer, inputs: torch.Tensor
) -> None:
    """Gradients flow to the cluster gate, the expert gates and the experts."""
    output, aux_loss = layer(inputs)
    loss = output.pow(2).mean() + aux_loss
    assert aux_loss.requires_grad
    loss.backward()

    assert layer.router.cluster_gate.weight.grad is not None
    assert layer.router.cluster_gate.weight.grad.abs().sum() > 0

    used_clusters = (layer.expert_load(inputs).view(NUM_CLUSTERS, -1).sum(-1) > 0).nonzero()
    for c in used_clusters.flatten().tolist():
        grad = layer.router.expert_gates[c].weight.grad
        assert grad is not None and grad.abs().sum() > 0

    used_experts = (layer.expert_load(inputs) > 0).nonzero().flatten().tolist()
    assert used_experts, "at least one expert should receive tokens"
    for e in used_experts:
        assert layer.experts[e].down_proj.weight.grad is not None


def test_aux_loss_alone_backpropagates_to_router(layer: TreeMoELayer, inputs: torch.Tensor) -> None:
    """``aux_loss.backward()`` alone (no task loss) produces router gradients."""
    _, aux_loss = layer(inputs)
    aux_loss.backward()
    assert layer.router.cluster_gate.weight.grad is not None
    assert layer.router.cluster_gate.weight.grad.abs().sum() > 0


def test_load_balancing_loss_bounds() -> None:
    """``N * sum f_i P_i`` equals 1 when uniform and N when fully collapsed."""
    n, tokens = 4, 400
    uniform_probs = torch.full((tokens, n), 1.0 / n)
    balanced_idx = torch.arange(tokens) % n
    assert load_balancing_loss(uniform_probs, balanced_idx, n).item() == pytest.approx(1.0)

    collapsed_probs = F.one_hot(torch.zeros(tokens, dtype=torch.long), n).float()
    collapsed_idx = torch.zeros(tokens, dtype=torch.long)
    assert load_balancing_loss(collapsed_probs, collapsed_idx, n).item() == pytest.approx(n)


# ---------------------------------------------------------------- (c) routing logic


def test_global_indices_are_bounded(layer: TreeMoELayer) -> None:
    """Global expert ids lie in ``[0, num_clusters * experts_per_cluster - 1]``."""
    x = torch.randn(64, 128, HIDDEN_DIM) * 5.0  # large batch, scaled for diverse routing
    indices, _, _ = layer.router(x)
    assert indices.min().item() >= 0
    assert indices.max().item() <= NUM_CLUSTERS * EXPERTS_PER_CLUSTER - 1


def test_routing_follows_the_tree(layer: TreeMoELayer, inputs: torch.Tensor) -> None:
    """Indices and weights match a manual recomputation of the hierarchical routing."""
    router: TreeRouter = layer.router
    indices, weights, _ = router(inputs)
    tokens = inputs.reshape(-1, HIDDEN_DIM)

    cluster_probs = F.softmax(router.cluster_gate(tokens), dim=-1)
    p_cluster, cluster_idx = cluster_probs.max(dim=-1)

    # The cluster encoded in the global index must be the arg-max cluster.
    assert torch.equal(indices.reshape(-1) // EXPERTS_PER_CLUSTER, cluster_idx)

    for t in range(tokens.shape[0]):
        c = cluster_idx[t].item()
        expert_probs = F.softmax(router.expert_gates[c](tokens[t]), dim=-1)
        p_expert, e = expert_probs.max(dim=-1)
        assert indices.reshape(-1)[t].item() == c * EXPERTS_PER_CLUSTER + e.item()
        torch.testing.assert_close(weights.reshape(-1)[t], p_cluster[t] * p_expert)


def test_layer_matches_reference_dispatch(layer: TreeMoELayer, inputs: torch.Tensor) -> None:
    """The batched dispatch equals a naive per-token loop over experts."""
    output, _ = layer(inputs)
    indices, weights, _ = layer.router(inputs)

    tokens = inputs.reshape(-1, HIDDEN_DIM)
    expected = torch.stack(
        [
            weights.reshape(-1)[t] * layer.experts[indices.reshape(-1)[t].item()](tokens[t])
            for t in range(tokens.shape[0])
        ]
    )
    torch.testing.assert_close(output.reshape(-1, HIDDEN_DIM), expected)
