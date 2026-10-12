"""Visualise expert load before and after optimising the auxiliary balancing loss.

Usage::

    python tests/plot_distribution.py            # writes PNGs to ./assets
    python tests/plot_distribution.py --out-dir . --steps 50

Synthetic tokens are drawn from an *anisotropic* Gaussian (a shared non-zero mean
direction plus noise). Real transformer hidden states share such a common direction,
and it is exactly what makes an untrained router send most tokens to a few experts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import TreeMoELayer  # noqa: E402

NUM_CLUSTERS = 4
EXPERTS_PER_CLUSTER = 4
HIDDEN_DIM = 64
INTERMEDIATE_DIM = 128
NUM_EVAL_TOKENS = 20_000

# Categorical palette (fixed order) — one hue per cluster; text stays in neutral ink.
CLUSTER_COLORS: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


class SyntheticTokens:
    """Sampler for anisotropic synthetic hidden states ``x = mu + sigma * eps``.

    Args:
        hidden_dim: Token dimension.
        mean_scale: Norm of the shared mean direction ``mu`` relative to the noise.
        seed: Seed for the fixed mean direction.
    """

    def __init__(self, hidden_dim: int, mean_scale: float = 1.0, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        direction = torch.randn(hidden_dim, generator=generator)
        self.mean = mean_scale * direction / direction.norm() * hidden_dim**0.5
        self.hidden_dim = hidden_dim

    def sample(self, num_tokens: int) -> torch.Tensor:
        """Draw ``num_tokens`` tokens of shape ``[num_tokens, hidden_dim]``."""
        return self.mean + torch.randn(num_tokens, self.hidden_dim)


def load_stats(counts: np.ndarray) -> str:
    """Summarise a load vector as coefficient of variation and max/min ratio."""
    cv = counts.std() / counts.mean()
    idle = int((counts == 0).sum())
    return f"CV = {cv:.2f}   ·   busiest expert = {counts.max() / counts.mean():.1f}× fair share   ·   idle experts = {idle}"


def plot_distribution(counts: np.ndarray, title: str, path: Path, y_max: float) -> None:
    """Save a per-expert token-count bar chart, bars coloured by parent cluster.

    Args:
        counts: Token counts per global expert, shape ``[num_experts]``.
        title: Figure title.
        path: Output PNG path.
        y_max: Shared y-axis upper bound so before/after charts are comparable.
    """
    num_experts = counts.shape[0]
    fair_share = counts.sum() / num_experts
    colors = [CLUSTER_COLORS[e // EXPERTS_PER_CLUSTER] for e in range(num_experts)]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(
        np.arange(num_experts), counts, width=0.8, color=colors,
        edgecolor=SURFACE, linewidth=2, zorder=3,  # 2px surface gap between adjacent fills
    )
    ax.axhline(fair_share, color=TEXT_SECONDARY, linestyle="--", linewidth=1.2, zorder=4)
    ax.text(
        num_experts - 0.5, fair_share, f"uniform load ({fair_share:,.0f})",
        ha="right", va="bottom", fontsize=9, color=TEXT_SECONDARY, zorder=5,
    )

    ax.set_xticks(np.arange(num_experts))
    ax.set_xticklabels([f"E{e}" for e in range(num_experts)], fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlim(-0.6, num_experts - 0.4)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Global expert id", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Tokens routed", fontsize=10, color=TEXT_SECONDARY)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT_SECONDARY)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)

    fig.suptitle(title, x=0.06, ha="left", fontsize=13, color=TEXT_PRIMARY, fontweight="bold")
    ax.set_title(load_stats(counts), loc="left", fontsize=9.5, color=TEXT_SECONDARY, pad=10)
    ax.legend(
        handles=[Patch(color=c, label=f"Cluster {i}") for i, c in enumerate(CLUSTER_COLORS)],
        loc="upper left", ncol=4, frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """Run the before/after experiment and write both charts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    parser.add_argument("--steps", type=int, default=50, help="Adam steps on the aux loss")
    parser.add_argument("--batch-tokens", type=int, default=4096, help="tokens per training step")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    layer = TreeMoELayer(HIDDEN_DIM, INTERMEDIATE_DIM, NUM_CLUSTERS, EXPERTS_PER_CLUSTER)
    data = SyntheticTokens(HIDDEN_DIM, seed=args.seed)

    # (b) Untrained distribution.
    layer.eval()
    before = layer.expert_load(data.sample(NUM_EVAL_TOKENS)).numpy()
    with torch.no_grad():
        _, aux_before = layer(data.sample(NUM_EVAL_TOKENS))

    # (c) Optimise only the router's load-balancing objective.
    layer.train()
    optimizer = torch.optim.Adam(layer.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        _, aux_loss = layer(data.sample(args.batch_tokens))
        optimizer.zero_grad(set_to_none=True)
        aux_loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:3d}/{args.steps}  aux_loss = {aux_loss.item():.4f}")

    # (d) Trained distribution.
    layer.eval()
    after = layer.expert_load(data.sample(NUM_EVAL_TOKENS)).numpy()
    with torch.no_grad():
        _, aux_after = layer(data.sample(NUM_EVAL_TOKENS))

    y_max = 1.15 * max(before.max(), after.max())
    untrained_path = args.out_dir / "distribution_untrained.png"
    trained_path = args.out_dir / "distribution_trained.png"
    plot_distribution(before, f"Untrained tree router  ·  aux loss {aux_before.item():.3f}", untrained_path, y_max)
    plot_distribution(
        after, f"After {args.steps} Adam steps on aux loss  ·  aux loss {aux_after.item():.3f}", trained_path, y_max
    )

    print(f"\nbefore: {before.tolist()}\nafter:  {after.tolist()}")
    print(f"saved {untrained_path}\nsaved {trained_path}")


if __name__ == "__main__":
    main()
