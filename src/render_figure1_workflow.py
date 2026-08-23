"""Render the publication workflow diagram used as Figure 1."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "figures_classification" / "manuscript"

INK = "#1F2933"
MUTED = "#52606D"
BLUE = "#DCEAF7"
BLUE_EDGE = "#4F7CAC"
GREEN = "#DDEEDC"
GREEN_EDGE = "#4C8C62"
AMBER = "#FAE7B5"
AMBER_EDGE = "#C58A24"
CORAL = "#F7D7CF"
CORAL_EDGE = "#C96856"
GREY = "#F3F5F7"
GREY_EDGE = "#9AA5B1"


def rounded_box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolour: str,
    edgecolour: str,
    fontsize: float = 12.5,
    weight: str = "normal",
    radius: float = 0.12,
) -> None:
    """Draw one labelled rounded box in O(1) time and space."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        linewidth=1.25,
        edgecolor=edgecolour,
        facecolor=facecolour,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        linespacing=1.12,
        zorder=3,
    )


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = MUTED,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    """Draw one directed connector in O(1) time and space."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.35,
            color=colour,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


def band(
    ax: Axes,
    x: float,
    width: float,
    title: str,
    *,
    facecolour: str,
    edgecolour: str,
) -> None:
    """Draw one workflow band in O(1) time and space."""
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.45),
            width,
            8.65,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.15,
            edgecolor=edgecolour,
            facecolor=facecolour,
            alpha=0.42,
            zorder=0,
        )
    )
    ax.text(
        x + width / 2,
        8.78,
        title,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )


def metric_labels(ax: Axes, labels: Iterable[str], *, x: float, y: float) -> None:
    """Draw a horizontal metric row in O(k) time and O(1) auxiliary space."""
    for index, label in enumerate(labels):
        rounded_box(
            ax,
            (x + 0.80 * index, y),
            0.68,
            0.48,
            label,
            facecolour="white",
            edgecolour=BLUE_EDGE,
            fontsize=6.5,
            weight="bold",
            radius=0.08,
        )


def build_figure() -> plt.Figure:
    """Construct the complete workflow in O(1) time and space."""
    fig, ax = plt.subplots(figsize=(12.5, 6.82), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18.33)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        9.165,
        9.68,
        "ToxLens: Leakage-Aware Multi-Task Molecular Toxicity Prediction",
        ha="center",
        va="center",
        fontsize=17.0,
        fontweight="bold",
        color=INK,
    )

    band(ax, 0.25, 3.85, "Data and\nPartitioning", facecolour=GREY, edgecolour=GREY_EDGE)
    band(ax, 4.35, 5.05, "Parallel Model\nArchitecture", facecolour=BLUE, edgecolour=BLUE_EDGE)
    band(ax, 9.65, 4.10, "Reliability and\nInterpretation", facecolour=AMBER, edgecolour=AMBER_EDGE)
    band(ax, 14.00, 4.05, "Evaluation and\nOutput", facecolour=GREEN, edgecolour=GREEN_EDGE)

    rounded_box(
        ax,
        (0.58, 7.15),
        3.20,
        1.05,
        "SMILES and Labels\n11 Toxicity Tasks",
        facecolour="white",
        edgecolour=GREY_EDGE,
        fontsize=10.5,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.58, 5.55),
        3.20,
        1.05,
        "RDKit Standardisation\nDuplicate Resolution\nLabel-Conflict Resolution",
        facecolour="white",
        edgecolour=GREY_EDGE,
        fontsize=9.0,
    )
    rounded_box(
        ax,
        (0.58, 3.75),
        3.20,
        1.22,
        "Sphere Exclusion\nUMAP-HDBSCAN\nCluster Split",
        facecolour="white",
        edgecolour=GREY_EDGE,
        fontsize=9.5,
        weight="bold",
    )
    rounded_box(ax, (0.58, 1.80), 0.92, 0.88, "Train\n80%", facecolour=BLUE, edgecolour=BLUE_EDGE, fontsize=8.5, weight="bold")
    rounded_box(ax, (1.72, 1.80), 0.92, 0.88, "Validation\n10%", facecolour=GREEN, edgecolour=GREEN_EDGE, fontsize=7.5, weight="bold")
    rounded_box(ax, (2.86, 1.80), 0.92, 0.88, "Test\n10%", facecolour=AMBER, edgecolour=AMBER_EDGE, fontsize=8.5, weight="bold")
    ax.text(2.18, 1.20, "Fixed, cluster-disjoint folds", ha="center", va="center", fontsize=8.5, color=MUTED)
    arrow(ax, (2.18, 7.15), (2.18, 6.60))
    arrow(ax, (2.18, 5.55), (2.18, 4.98))
    arrow(ax, (2.18, 3.75), (2.18, 2.72))

    rounded_box(
        ax,
        (4.72, 6.47),
        2.05,
        1.63,
        "Graph Path\n134D Node Features\n35D Edge Features\nFive GINE Layers",
        facecolour="white",
        edgecolour=BLUE_EDGE,
        fontsize=8.0,
        weight="bold",
    )
    rounded_box(
        ax,
        (4.72, 3.82),
        2.05,
        1.90,
        "Global Path\nECFP4 and RDKit\nMolFormer and 3D\nMLP Encoder",
        facecolour="white",
        edgecolour=GREEN_EDGE,
        fontsize=8.5,
        weight="bold",
    )
    rounded_box(
        ax,
        (7.18, 5.45),
        1.78,
        1.25,
        "Late\nConcatenation",
        facecolour=AMBER,
        edgecolour=AMBER_EDGE,
        fontsize=8.5,
        weight="bold",
    )
    rounded_box(
        ax,
        (7.18, 3.10),
        1.78,
        1.55,
        "Shared Trunk\nGroup Towers\nTask Heads",
        facecolour="white",
        edgecolour=BLUE_EDGE,
        fontsize=8.0,
        weight="bold",
    )
    ax.text(6.84, 2.15, "Five seeds (42-46)\nBest validation-MCC checkpoint\nselected for each seed", ha="center", va="center", fontsize=8.5, color=MUTED)
    arrow(ax, (3.78, 6.20), (4.72, 7.28))
    arrow(ax, (3.78, 5.00), (4.72, 4.76))
    arrow(ax, (6.77, 7.28), (7.18, 6.30))
    arrow(ax, (6.77, 4.76), (7.18, 5.82))
    arrow(ax, (8.07, 5.45), (8.07, 4.65))

    rounded_box(
        ax,
        (10.02, 6.60),
        3.36,
        1.48,
        "Temperature Scaling\n30-Pass MC Dropout\nPredictive Mean and SD",
        facecolour="white",
        edgecolour=AMBER_EDGE,
        fontsize=9.5,
        weight="bold",
    )
    rounded_box(
        ax,
        (10.02, 4.62),
        3.36,
        1.38,
        "Split Conformal Calibration\nPredictive Mean\nalpha = 0.05\nSingleton or Two-Class Set",
        facecolour="white",
        edgecolour=AMBER_EDGE,
        fontsize=8.5,
        weight="bold",
    )
    rounded_box(
        ax,
        (10.02, 1.68),
        3.36,
        2.20,
        "GradientSHAP\nHeavy-Atom Aggregation\nGraph Occlusion\nConsensus Subgraph Mining",
        facecolour="white",
        edgecolour=CORAL_EDGE,
        fontsize=8.5,
        weight="bold",
    )
    arrow(ax, (8.96, 3.87), (10.02, 7.34), connectionstyle="arc3,rad=-0.13")
    arrow(ax, (11.70, 6.60), (11.70, 6.00))
    arrow(ax, (8.96, 3.87), (10.02, 2.78), connectionstyle="arc3,rad=0.10")

    rounded_box(
        ax,
        (14.35, 6.75),
        3.35,
        1.33,
        "Five-Seed Soft Voting\n11 Endpoint Scores\nFrozen Validation Thresholds",
        facecolour="white",
        edgecolour=GREEN_EDGE,
        fontsize=8.0,
        weight="bold",
    )
    rounded_box(
        ax,
        (14.35, 4.72),
        3.35,
        1.40,
        "Held-Out Test Evaluation\nBootstrap 95% Intervals\nRanking, Calibration,\nand Classification",
        facecolour="white",
        edgecolour=GREEN_EDGE,
        fontsize=8.5,
        weight="bold",
    )
    metric_labels(ax, ("MCC", "AUROC", "AUPRC", "Brier"), x=14.43, y=3.68)
    rounded_box(
        ax,
        (14.35, 1.60),
        3.35,
        1.35,
        "Applicability-Domain\nAudit\nConformal Prediction Sets\nStructural Hypotheses",
        facecolour="white",
        edgecolour=GREEN_EDGE,
        fontsize=8.5,
        weight="bold",
    )
    arrow(ax, (13.38, 5.30), (14.35, 5.92))
    arrow(ax, (13.38, 2.78), (14.35, 2.90))
    arrow(ax, (16.02, 6.75), (16.02, 6.12))
    arrow(ax, (16.02, 4.72), (16.02, 4.20))
    arrow(ax, (16.02, 3.68), (16.02, 2.95))

    ax.text(
        9.165,
        0.15,
        "Training and validation support model selection and calibration; the held-out test fold is used once for final evaluation.",
        ha="center",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )
    return fig


def main() -> None:
    """Render SVG, PNG, and TIFF outputs in O(P) time and space for P pixels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    stem = OUTPUT_DIR / "Figure1_ToxLens_Workflow"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.06)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    print(f"Saved Figure 1 exports to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
