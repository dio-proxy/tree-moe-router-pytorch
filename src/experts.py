"""Expert feed-forward networks used inside the Tree MoE layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single MoE expert: a position-wise feed-forward network.

    Two activation variants are supported:

    * ``"swiglu"`` (default): ``W_down( SiLU(W_gate x) * W_up x )`` as used in LLaMA / PaLM.
    * ``"gelu"``: the classic Transformer FFN ``W_down( GELU(W_up x) )``.

    Args:
        hidden_dim: Model (input and output) dimension ``d_model``.
        intermediate_dim: Inner dimension of the FFN ``d_ff``.
        activation: Either ``"swiglu"`` or ``"gelu"``.
        dropout: Dropout probability applied to the hidden activation.
        bias: Whether the linear projections use a bias term.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str = "swiglu",
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        activation = activation.lower()
        if activation not in ("swiglu", "gelu"):
            raise ValueError(f"Unsupported activation '{activation}'; expected 'swiglu' or 'gelu'.")

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation

        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.gate_proj = (
            nn.Linear(hidden_dim, intermediate_dim, bias=bias) if activation == "swiglu" else None
        )
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transformation.

        Args:
            x: Tensor of shape ``[..., hidden_dim]``.

        Returns:
            Tensor of shape ``[..., hidden_dim]``.
        """
        if self.gate_proj is not None:
            hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        else:
            hidden = F.gelu(self.up_proj(x))
        return self.down_proj(self.dropout(hidden))

    def extra_repr(self) -> str:
        """Readable summary shown by ``print(module)``."""
        return (
            f"hidden_dim={self.hidden_dim}, intermediate_dim={self.intermediate_dim}, "
            f"activation={self.activation}"
        )
