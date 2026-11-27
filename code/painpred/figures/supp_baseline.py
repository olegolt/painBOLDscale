"""Supplementary figure: warmth-referenced vs implicit-baseline correlation distributions."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..config import Project
from ..supplementary import COMPARISONS
from .style import apply_style

WARMTH_COLOR, BASELINE_COLOR, DIFFERENCE_COLOR = "#3B6670", "#C45A4B", "#4A4A4A"
SIZES = dict(label=14, tick=11.5, title=14, stat=10.5)
KDE_MAX_POINTS = 50_000


def kde(values, grid, seed: int) -> np.ndarray:
    """Gaussian KDE on at most 50,000 randomly drawn voxels (descriptive only)."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) > KDE_MAX_POINTS:
        values = values[np.random.default_rng(seed).choice(len(values), size=KDE_MAX_POINTS, replace=False)]
    return stats.gaussian_kde(values)(grid)


def run(project: Project, result: dict) -> None:
    """``result`` is the dictionary returned by ``supplementary.run``."""
    import matplotlib.pyplot as plt
    apply_style()
    out = project.baseline / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = result["comparison_summary"]
    rho_grid, delta_grid = np.linspace(-0.65, 0.75, 500), np.linspace(-0.55, 0.55, 500)
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2))
    for col, comp in enumerate(COMPARISONS):
        data = result["plot_data"][comp.condition]
        ax = axes[0, col]
        ax.plot(rho_grid, kde(data["rho_warmth"], rho_grid, 100 + col), color=WARMTH_COLOR, linewidth=2.2, label="Heat - warmth")
        ax.plot(rho_grid, kde(data["rho_baseline"], rho_grid, 200 + col), color=BASELINE_COLOR, linewidth=2.2,
                label="Heat - implicit baseline")
        ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=0)
        ax.set_title(comp.condition.replace(" C", " °C"), fontsize=SIZES["title"], fontweight="bold", pad=7)
        ax.set_xlabel("Voxelwise Spearman rho", fontsize=SIZES["label"])
        ax = axes[1, col]
        density = kde(data["delta_rho"], delta_grid, 300 + col)
        ax.plot(delta_grid, density, color=DIFFERENCE_COLOR, linewidth=2.2)
        ax.fill_between(delta_grid, 0, density, color="#D9D9D9", alpha=0.55, linewidth=0)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, zorder=5)
        ax.set_xlabel("Delta rho: heat-warmth minus heat-baseline", fontsize=SIZES["label"])
        row = summary.loc[summary["condition"] == comp.condition].iloc[0]
        ax.text(0.03, 0.96, "Whole-brain max-stat FWE\n"
                f"Warmth higher: {int(row['n_voxels_warmth_higher_fwe']):,} voxels\n"
                f"Baseline higher: {int(row['n_voxels_implicit_baseline_higher_fwe']):,} voxels",
                transform=ax.transAxes, ha="left", va="top", fontsize=SIZES["stat"])
        if col == 0:
            axes[0, 0].set_ylabel("Density", fontsize=SIZES["label"])
            axes[0, 0].legend(frameon=False, fontsize=10.0, loc="upper left")
            ax.set_ylabel("Density", fontsize=SIZES["label"])
    for ax in axes.ravel():
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(axis="both", labelsize=SIZES["tick"], width=1.1, length=4.0)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.11, top=0.94, wspace=0.28, hspace=0.38)
    for ext, dpi in (("png", 600), ("pdf", None), ("svg", None)):
        path = out / f"warmth_vs_implicit_baseline.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05) if dpi else fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved {path}")
    plt.close(fig)
