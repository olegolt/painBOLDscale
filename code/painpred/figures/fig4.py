"""Figure 4: NPS and LASSO-PCR signature weights, and their association with pain ratings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HEAT_CONDITIONS, PRIMARY, Project
from ..data import load_analysis_table
from ..imaging import load_3d
from ..signatures import fullsample_associations, nps_cross_validate, score_correspondence
from ..utils import DataError, MissingInputError, canonical_subject_id, find_subject_column, read_table, require_columns
from . import brain
from .style import (
    SIZES,
    add_panel_label,
    apply_style,
    padded_limits,
    rating_axis,
    save_figure,
    scatter_with_fit,
    site_color,
    style_axis,
)

WEIGHT_LIMIT_PERCENTILE = 99.0
DISPLAY_THRESHOLD_FRACTION = 0.02
SURFACE_DEPTHS = np.linspace(-1.5, 2.5, 33)
CEREBELLAR_DEPTHS = np.linspace(-0.5, 1.5, 25)
LAYOUT = dict(figsize=(16.0, 18.6), width_ratios=(1.30, 1.0, 1.0), block_hspace=0.34, block_wspace=0.23,
              nps_lasso_spacer=0.42, lasso_correspondence_spacer=0.50, bottom_wspace=0.30)


def display_limit(img, percentile: float = WEIGHT_LIMIT_PERCENTILE) -> float:
    data = np.asarray(img.get_fdata(), dtype=float)
    finite = data[np.isfinite(data) & (data != 0)]
    if finite.size == 0:
        raise DataError("Weight map contains no finite non-zero values.")
    limit = float(np.percentile(np.abs(finite), percentile))
    return limit if np.isfinite(limit) and limit > 0 else float(np.abs(finite).max())


def brain_panels(weight_img, name: str, limit: float):
    mesh = brain.load_fsaverage("fsaverage5")
    threshold = DISPLAY_THRESHOLD_FRACTION * limit
    left, right = brain.sample_to_surface(weight_img, mesh, SURFACE_DEPTHS)
    cmap = _cmap()
    surface = brain.render_surface_panel(left, right, mesh, cmap, -limit, limit, threshold)
    flat = brain.project_cerebellum(weight_img, CEREBELLAR_DEPTHS, name)
    return surface, brain.render_cerebellum_panel(flat, cmap, -limit, limit, threshold)


def _cmap():
    import matplotlib as mpl
    return mpl.colormaps["RdBu_r"]


def add_brain_block(ax, surface_img, cerebellum_img, limit, title, colorbar_label, metrics: pd.Series) -> None:
    ax.set_axis_off()
    ax.text(0.00, 1.015, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=SIZES["block_title"], fontweight="bold")
    if surface_img is None:  # weight map not available (the NPS weights cannot be redistributed)
        ax.text(0.5, 0.64, f"{title} weight map not shown:\nthe weights are not part of the shared data\n(see data/external/README.md)",
                transform=ax.transAxes, ha="center", va="center", fontsize=14, color="0.35", linespacing=1.4)
    else:
        for rect, image in (([0.00, 0.43, 1.00, 0.48], surface_img), ([0.08, 0.19, 0.84, 0.24], cerebellum_img)):
            inset = ax.inset_axes(rect)
            inset.imshow(image, interpolation="lanczos", aspect="equal")
            inset.set_axis_off()
        brain.add_scaled_colorbar(ax.inset_axes([0.15, 0.135, 0.70, 0.020]), _cmap(), limit, colorbar_label)
    ax.text(0.02, 0.025, f"Pearson r = {metrics['pearson_r']:.2f}   Spearman ρ = {metrics['spearman_rho']:.2f}   "
            f"CV R² = {metrics['cross_validated_r2']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=13, linespacing=1.22)


def regression_panel(ax, x, y, cond, text, x_label, y_label) -> None:
    scatter_with_fit(ax, x, y, site_color(cond.site))
    ax.text(0.04, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=SIZES["stat"])
    ax.set_title(cond.label, fontsize=SIZES["condition_title"], fontweight="bold", pad=7)
    ax.set_xlabel(x_label, fontsize=SIZES["axis_label"], labelpad=7)
    ax.set_ylabel(y_label, fontsize=SIZES["axis_label"], labelpad=8)
    ax.set_xlim(*padded_limits(x))
    style_axis(ax)


def build_data(project: Project) -> pd.DataFrame:
    """Ratings + NPS + LASSO-PCR scores of the 101 participants (written to figure4/subjects.tsv)."""
    data = load_analysis_table(project, nps_contrasts=tuple(c.contrast for c in HEAT_CONDITIONS), lasso=True)
    keep = ["subject_key", *[c.rating for c in HEAT_CONDITIONS], *[f"nps_con_{c.contrast}" for c in HEAT_CONDITIONS],
            *[f"signature_score_con{c.contrast}" for c in HEAT_CONDITIONS]]
    data = data[keep]
    data.to_csv(project.f4 / "subjects.tsv", sep="\t", index=False)
    return data


def run(project: Project) -> dict:
    project.require("figure4")
    apply_style()
    data = build_data(project)
    assignments = read_table(project.f4 / "lasso_cv_folds.tsv", "LASSO-PCR fold assignments", sep="\t",
                             hint="Run the LASSO-PCR analysis first.")
    # one row per participant and repeat, so the same participant appears several times
    assignments["subject_key"] = assignments[find_subject_column(assignments, "fold assignments")].map(canonical_subject_id)
    nps_repeat, nps_average, nps_metrics = nps_cross_validate(data, assignments)
    for table, name in ((nps_repeat, "nps_cv_repeat"), (nps_average, "nps_cv_average"), (nps_metrics, "nps_cv_metrics")):
        table.to_csv(project.f4 / f"{name}.tsv", sep="\t", index=False)
    lasso_metrics = read_table(project.f4 / "lasso_cv_metrics.tsv", "LASSO-PCR CV metrics", sep="\t")
    require_columns(lasso_metrics, ["prediction_scale"], "LASSO-PCR CV metrics")
    lasso_row = lasso_metrics[lasso_metrics["prediction_scale"] == "raw"].iloc[0]
    association = fullsample_associations(data)
    correspondence = score_correspondence(data)
    association.to_csv(project.f4 / "pattern_associations.tsv", sep="\t", index=False)
    correspondence.to_csv(project.f4 / "pattern_correspondence.tsv", sep="\t", index=False)

    import matplotlib.pyplot as plt
    lasso_img = load_3d(project.lasso_raw_weights, "LASSO-PCR weight map")
    lasso_limit = display_limit(lasso_img)
    lasso_panels = brain_panels(lasso_img, "LASSO-PCR", lasso_limit)
    if project.nps_weights is None:
        if not project.public:
            raise MissingInputError(f"NPS weight map not found: {' or '.join(str(p) for p in project.nps_weight_candidates)}")
        print("NPS weights not found: the NPS brain map is left empty (the weights cannot be redistributed; see data/external/README.md).")
        nps_limit, nps_panels = None, (None, None)
    else:
        nps_img = load_3d(project.nps_weights, "NPS weight map")
        nps_limit = display_limit(nps_img)
        nps_panels = brain_panels(nps_img, "NPS", nps_limit)
    print(f"Display limits (99th |weight| percentile): NPS {nps_limit}, LASSO-PCR {lasso_limit:.6g}")

    L = LAYOUT
    fig = plt.figure(figsize=L["figsize"], facecolor="white")
    outer = fig.add_gridspec(5, 1, height_ratios=(2.0, L["nps_lasso_spacer"], 2.0, L["lasso_correspondence_spacer"], 1.18),
                             left=0.052, right=0.992, bottom=0.060, top=0.975, hspace=0.0)
    blocks = [outer[i, 0].subgridspec(2, 3, width_ratios=L["width_ratios"], wspace=L["block_wspace"], hspace=L["block_hspace"])
              for i in (0, 2)]
    bottom = outer[4, 0].subgridspec(1, 4, wspace=L["bottom_wspace"])
    add_brain_block(fig.add_subplot(blocks[0][:, 0]), *nps_panels, nps_limit, "Neurologic Pain Signature (NPS)", "NPS voxel weight", nps_metrics.iloc[0])
    add_brain_block(fig.add_subplot(blocks[1][:, 0]), *lasso_panels, lasso_limit, "LASSO-PCR signature", "Standardized voxel weight", lasso_row)

    cells = ((0, 1), (0, 2), (1, 1), (1, 2))  # 48 arm, 46 arm, 48 leg, 46 leg
    panel_axes = []
    for block, model, column in zip(blocks, ("NPS", "LASSO-PCR"), (f"nps_con_{PRIMARY.contrast}", f"signature_score_con{PRIMARY.contrast}")):
        label = "NPS expression\n(48 °C arm contrast)" if model == "NPS" else "LASSO-PCR signature expression\n(48 °C arm contrast)"
        for k, (cond, cell) in enumerate(zip(HEAT_CONDITIONS, cells)):
            ax = fig.add_subplot(block[cell])
            row = association[(association["model"] == model) & (association["rating"] == cond.rating)].iloc[0]
            regression_panel(ax, data[column].to_numpy(float), data[cond.rating].to_numpy(float), cond,
                             f"Spearman ρ = {row['spearman_rho']:.2f}", label if k in (2, 3) else "",
                             "Mean pain rating (VAS)" if k in (0, 2) else "")
            rating_axis(ax, data[cond.rating].to_numpy(float))
            panel_axes.append(ax)
    for k, cond in enumerate(HEAT_CONDITIONS):
        ax = fig.add_subplot(bottom[0, k])
        x, y = data[f"nps_con_{cond.contrast}"].to_numpy(float), data[f"signature_score_con{cond.contrast}"].to_numpy(float)
        row = correspondence[correspondence["contrast"].astype(str).str.zfill(4) == cond.contrast].iloc[0]
        regression_panel(ax, x, y, cond, f"Pearson r = {row['pearson_r']:.2f}", "NPS expression",
                         "LASSO-PCR signature expression" if k == 0 else "")
        ax.set_ylim(*padded_limits(y))
        panel_axes.append(ax)
    for letter, ax in zip("abcdefghijkl", panel_axes):
        add_panel_label(ax, letter, x=-0.095, y=1.085)
    save_figure(fig, project.figure_files("Figure4"), bbox_inches=None)
    plt.close(fig)
    return {"association": association, "correspondence": correspondence, "nps_cv_metrics": nps_metrics}


__all__ = ["run", "MissingInputError"]
