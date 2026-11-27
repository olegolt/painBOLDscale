"""Figure S1: reference-condition variability attenuates brain-behaviour correlations."""

from __future__ import annotations

import numpy as np

from .. import simulation
from ..config import Project, SimulationSettings
from .style import LEG_COLOR, save_figure

LEG_GROUP_COLOR, CI_COLOR = "#963B32", "#DFDFDF"
LABEL, TICK, PANEL, LEGEND, LINE = 17, 14, 22, 11.5, 2.4


def blend_with_white(color: str, white_fraction: float) -> str:
    import matplotlib as mpl
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    return mpl.colors.to_hex((1.0 - white_fraction) * rgb + white_fraction)


def scenario_colors():
    return [blend_with_white(LEG_COLOR, f) for f in (0.68, 0.50, 0.30)] + [LEG_COLOR, LEG_GROUP_COLOR]


def format_axis(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_linewidth(1.30)
    ax.tick_params(axis="both", labelsize=TICK, width=1.15, length=4.2, pad=3)


def make_figure(panel_a, panel_b_kde, s: SimulationSettings):
    import matplotlib.pyplot as plt
    colors = scenario_colors()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 5.2))

    ax_a.axhline(s.true_r, linestyle="--", linewidth=1.8, color="black", label="Pain-condition response", zorder=2)
    ax_a.plot(panel_a["reference_variability_ratio"], panel_a["mean_r"], linewidth=LINE, color="black",
              label="Pain − reference contrast", zorder=3)
    ax_a.fill_between(panel_a["reference_variability_ratio"], panel_a["low_r"], panel_a["high_r"], color=CI_COLOR,
                      alpha=0.65, linewidth=0, zorder=1)
    for ratio, color in zip(s.whole_brain_ratios, colors):
        row = panel_a.loc[np.isclose(panel_a["reference_variability_ratio"], ratio)]
        if len(row) != 1:
            raise RuntimeError(f"Panel a has no unique row for ratio {ratio:g}.")
        ax_a.scatter(row["reference_variability_ratio"], row["mean_r"], s=90, color=color, edgecolor="black",
                     linewidth=0.85, zorder=5)
    ax_a.set_xlabel("Reference condition variability", fontsize=LABEL, labelpad=8)
    ax_a.set_ylabel("Correlation with pain ratings", fontsize=LABEL, labelpad=9)
    ax_a.set(xlim=(-0.05, 3.05), xticks=np.arange(0.0, 3.01, 0.5), ylim=(-0.02, 0.70))
    ax_a.legend(frameon=False, fontsize=LEGEND, loc="upper right", handlelength=2.1, borderaxespad=0.4)

    for ratio, color in zip(s.whole_brain_ratios, colors):
        d = panel_b_kde[panel_b_kde["reference_variability_ratio"] == ratio]
        ax_b.plot(d["r"], d["mean_density"], linewidth=LINE, color=color, label=f"{ratio:g}", zorder=4)
        ax_b.fill_between(d["r"], 0, d["mean_density"], color=color, alpha=0.11, linewidth=0, zorder=1)
    ax_b.axvline(s.true_r, color="black", linestyle="--", linewidth=2.0, label=f"Underlying r = {s.true_r:.2f}", zorder=20)
    ax_b.set_xlabel("Observed voxelwise correlation", fontsize=LABEL, labelpad=8)
    ax_b.set_ylabel("Density", fontsize=LABEL, labelpad=9)
    ax_b.set_xlim(simulation.KDE_MIN, simulation.KDE_MAX)
    ax_b.set_ylim(bottom=0)
    ax_b.legend(frameon=False, fontsize=LEGEND, loc="upper left", handlelength=2.1, borderaxespad=0.4)

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.text(-0.13, 1.045, letter, transform=ax.transAxes, fontsize=PANEL, fontweight="bold", va="top", ha="left")
        format_axis(ax)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.19, top=0.96, wspace=0.32)
    return fig


def run(project: Project, settings: SimulationSettings = SimulationSettings(), results: dict | None = None,
        n_jobs: int | None = None) -> dict:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})
    results = results or simulation.run(project, settings, n_jobs)
    fig = make_figure(results["panel_a"], results["panel_b_kde"], settings)
    save_figure(fig, [project.figures / "FigureS1.png"], bbox_inches="tight", pad_inches=0.06, dpi=600)
    plt.close(fig)
    print(results["panel_b_summary"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return results
