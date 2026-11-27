"""Figure 6: pre-experiment calibration vs NPS as predictors of pain (panels a-c) and cross-validated R² (panel d)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import combined
from ..config import CombinedModelSettings, Project
from ..data import MEAN_PAIN
from .style import (
    ARM_COLOR,
    LEG_COLOR,
    OVERALL_COLOR,
    POINT_COLOR,
    SIZES,
    add_panel_label,
    apply_style,
    padded_limits,
    save_figure,
    scatter_with_fit,
    style_axis,
)

PAIN_YLIM = (40.0, 95.0)
OUTCOME_COLORS = {"48 °C arm": ARM_COLOR, "46 °C arm": ARM_COLOR, "48 °C leg": LEG_COLOR, "46 °C leg": LEG_COLOR,
                  "Mean across all conditions": OVERALL_COLOR}
MODEL_MARKERS = {"Pre-experiment calibration": "o", "NPS": "s", "Calibration + NPS": "D"}
MODEL_OFFSETS = {"Pre-experiment calibration": -0.09, "NPS": 0.0, "Calibration + NPS": 0.09}


def regression_panel(ax, x, y, x_label, y_label, text, y_limits=None, hide_y_labels=False) -> None:
    scatter_with_fit(ax, x, y, POINT_COLOR, line_z=2, band_z=1)
    ax.text(0.04, 0.99, text, transform=ax.transAxes, ha="left", va="top", fontsize=SIZES["stat"], linespacing=1.18)
    ax.set_xlabel(x_label, fontsize=SIZES["axis_label"], labelpad=8)
    ax.set_ylabel(y_label, fontsize=SIZES["axis_label"], labelpad=8)
    ax.set_xlim(*padded_limits(x, 0.06))
    ax.set_ylim(*(y_limits or padded_limits(y, 0.08)))
    if hide_y_labels:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("")
    style_axis(ax)


def cv_panel(ax, summary: pd.DataFrame) -> None:
    """Mean ± 1 SD of CV R² across repeated partitions, per outcome and model."""
    lookup = summary.set_index(["outcome", "model"])
    outcomes = [label for label, _ in combined.OUTCOMES]
    lows, highs = [], []
    for i, outcome in enumerate(outcomes):
        for model in combined.MODELS:
            mean, sd = float(lookup.loc[(outcome, model), "mean_cv_r2"]), float(lookup.loc[(outcome, model), "sd_cv_r2_across_splits"])
            lows.append(mean - sd)
            highs.append(mean + sd)
            y = i + MODEL_OFFSETS[model]
            ax.errorbar(mean, y, xerr=[[sd], [sd]], fmt="none", ecolor="#777777", elinewidth=1.05, capsize=2.7, alpha=0.40, zorder=1)
            ax.scatter(mean, y, s=130, marker=MODEL_MARKERS[model], facecolors=OUTCOME_COLORS[outcome],
                       edgecolors="#111111", linewidths=1.15, zorder=3, clip_on=False)
    ax.axvline(0.0, color="#7A7A7A", linewidth=1.0, alpha=0.68, linestyle="--", zorder=0).set_dashes((4.0, 3.0))
    ax.set_ylim(len(outcomes) - 0.55, -0.55)
    ax.set_yticks(np.arange(len(outcomes)))
    ax.set_yticklabels(["48 °C arm", "46 °C arm", "48 °C leg", "46 °C leg", "Mean across\nall conditions"],
                       fontsize=SIZES["tick"], multialignment="right")
    ax.set_xlim(min(-0.02, min(lows) - 0.01), max(0.505, max(highs) + 0.01))
    ax.set_xlabel("Cross-validated $R^2$", fontsize=SIZES["axis_label"], labelpad=7)
    ax.tick_params(axis="y", length=0, pad=8)
    handles = [ax.scatter([], [], s=62, marker=MODEL_MARKERS[m], facecolors="white", edgecolors="#111111", linewidths=1.0, label=m)
               for m in combined.MODELS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False, fontsize=12.5,
              handletextpad=0.55, columnspacing=1.8, borderaxespad=0.0)
    style_axis(ax)


def make_figure(paired, full, stats, summary):
    import matplotlib.pyplot as plt
    row = {p: stats[stats["panel"] == p].iloc[0] for p in "abc"}
    fig = plt.figure(figsize=(15.0, 9.15), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 0.96), left=0.06, right=0.986, bottom=0.155, top=0.965, hspace=0.30)
    top = outer[0].subgridspec(1, 3, wspace=0.28)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(top[0, 2])
    ax_d = fig.add_subplot(outer[1])
    text = lambda r, second: f"Pearson $r$ = {r['pearson_r']:.2f}\n{second}"  # noqa: E731
    regression_panel(ax_a, paired[combined.CALIBRATION].to_numpy(float), paired[MEAN_PAIN].to_numpy(float),
                     "Mean calibration temperature (°C)", "Mean pain rating (VAS)",
                     text(row["a"], f"Spearman $\\rho$ = {row['a']['spearman_rho']:.2f}"), PAIN_YLIM)
    regression_panel(ax_b, full[combined.NPS].to_numpy(float), full[MEAN_PAIN].to_numpy(float), "NPS expression", "",
                     text(row["b"], f"Spearman $\\rho$ = {row['b']['spearman_rho']:.2f}"), PAIN_YLIM, hide_y_labels=True)
    regression_panel(ax_c, paired[combined.CALIBRATION].to_numpy(float), paired[combined.NPS].to_numpy(float),
                     "Mean calibration temperature (°C)", "NPS expression", text(row["c"], f"VIF = {row['c']['vif']:.2f}"))
    cv_panel(ax_d, summary)
    for letter, ax, x, y in (("a", ax_a, -0.14, 1.075), ("b", ax_b, -0.14, 1.075), ("c", ax_c, -0.14, 1.075), ("d", ax_d, -0.055, 1.045)):
        add_panel_label(ax, letter, x, y)
    return fig


def run(project: Project, settings: CombinedModelSettings = CombinedModelSettings(), results: dict | None = None) -> dict:
    """Run the Figure 6 analysis (unless ``results`` from ``combined.run`` are passed) and draw the figure."""
    import matplotlib.pyplot as plt
    apply_style()
    results = results or combined.run(project, settings)
    fig = make_figure(results["paired"], results["full"], results["display_stats"], results["cv_summary"])
    save_figure(fig, project.figure_files("Figure6"), bbox_inches="tight")
    plt.close(fig)
    print(results["model_comparisons"][["outcome", "model_a", "model_b", "observed_delta_aggregated_oof_r2",
                                        "participant_bootstrap_ci_2p5", "participant_bootstrap_ci_97p5",
                                        "prediction_swap_p_one_sided_a_greater", "prediction_swap_p_two_sided"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return results
