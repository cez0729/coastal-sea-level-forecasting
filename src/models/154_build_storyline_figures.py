from __future__ import annotations

"""Build publication-ready visual summaries for the story-enhanced manuscript."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication_final" / "overleaf_sea_level_story_enhanced_submission" / "figures"


def box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    ax.add_patch(FancyBboxPatch((x, y + h - 0.15), w, 0.15, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=0, facecolor=color))
    ax.text(x + 0.02, y + h - 0.075, title, color="white", fontsize=10, fontweight="bold", va="center")
    ax.text(x + 0.025, y + h - 0.19, body, color="#24333D", fontsize=8.2, va="top", linespacing=1.35)


def storyline() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("1  Forecasting task", "7 stations, 34 inputs\n24-h residual forecast\nQuestion: does physics help?", "#315B6D"),
        ("2  Initial physics model", "GNN-BiGRU + ODE/loss\nGood vs weak baselines\nAttribution remained confounded", "#6E7F55"),
        ("3  Strong benchmark", "FS-GWN sequence R2 0.7261\nMultistate lead-24 R2 0.6153\nDirect physics delta -0.000232", "#9B6B43"),
        ("4  Horizon specialization", "HS-DT: eta + multistate experts\nSequence R2 0.7357\nKeeps lead-24 R2 0.6153", "#536C91"),
        ("5  Conditional correction", "ORC-HS-DT R2 0.7386 / 0.6191\nq95 R2 0.6378\nHigh-forcing lead-24 benefit", "#2F7D6D"),
        ("6  Boundary tests", "Shift and lag controls\nTiming is not uniquely identified\nPhysics benefit is conditional", "#805A74"),
    ]
    margin = 0.025
    gap = 0.018
    width = (1 - 2 * margin - 5 * gap) / 6
    y, height = 0.29, 0.50
    for i, (title, body, color) in enumerate(steps):
        x = margin + i * (width + gap)
        box(ax, x, y, width, height, title, body, color)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.003, y + height / 2), (x + width + gap - 0.003, y + height / 2), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#65727A"))
    ax.text(0.5, 0.93, "From 'Does physics help?' to 'When and where does physics help?'", ha="center", va="center", fontsize=17, fontweight="bold", color="#193B4D")
    ax.text(0.5, 0.13, "Main inference: stronger backbones reduce the value of a global physics penalty; horizon-aware and regime-conditioned placement is more defensible.", ha="center", va="center", fontsize=10.5, color="#314650")
    fig.tight_layout()
    fig.savefig(OUT / "storyline_overview.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def injection_comparison() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.94, "Three ways of using physical information", ha="center", fontsize=18, fontweight="bold", color="#193B4D")
    cards = [
        (0.04, "A  Coupled full configuration", "GNN-BiGRU\nmultistate + terminal weight\n+ physical residual", "Lead-24 R2 = 0.5553", "Useful diagnostic, but\nnot physics-only attribution", "#71804B"),
        (0.355, "B  Global training penalty", "Matched multistate FS-GWN\nchange only physics loss\nlocked lambda = 0.0002", "Lead-24 R2 delta = -0.000232", "No measurable benefit on\nthe strong backbone", "#A46B52"),
        (0.67, "C  Conditional residual correction", "HS-DT forecast first\nODE-conditioned adapter second\nhigh-forcing mechanism audit", "High-forcing Lead-24\nMSE reduction = 0.00082681", "5/5 seeds vs HS-DT, but\nadapter controls retain gains", "#2F7D6D"),
    ]
    for x, title, method, metric, caveat, color in cards:
        patch = FancyBboxPatch((x, 0.25), 0.285, 0.57, boxstyle="round,pad=0.018,rounding_size=0.025", linewidth=1.5, edgecolor=color, facecolor="#FBFCFC")
        ax.add_patch(patch)
        ax.text(x + 0.018, 0.76, title, fontsize=11, fontweight="bold", color=color, va="center")
        ax.text(x + 0.018, 0.66, method, fontsize=9.3, color="#2F3D45", va="top", linespacing=1.35)
        ax.text(x + 0.018, 0.46, metric, fontsize=10.2, fontweight="bold", color="#193B4D", va="top", linespacing=1.3)
        ax.plot([x + 0.018, x + 0.267], [0.39, 0.39], color="#C7D0D6", linewidth=0.8)
        ax.text(x + 0.018, 0.35, caveat, fontsize=8.8, color="#55636C", va="top", linespacing=1.35)
    ax.text(0.5, 0.11, "Placement and regime matter more than increasing the physics-loss weight.", ha="center", fontsize=13, fontweight="bold", color="#2F7D6D")
    fig.tight_layout()
    fig.savefig(OUT / "physics_injection_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    storyline()
    injection_comparison()


if __name__ == "__main__":
    main()
