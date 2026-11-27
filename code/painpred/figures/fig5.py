"""Figure 5: NPS and LASSO-PCR expression for pain and non-pain (sound, checkerboard) contrasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import HEAT_CONDITIONS, Project
from ..data import MEAN_PAIN, load_imaging_sample, load_lasso_scores, load_nps_scores, merge_unique
from ..imaging import MaskGeometry, load_3d
from ..utils import DataError, MissingInputError, association_stats, find_subject_column, read_table, require_columns
from .style import apply_style, save_figure

SOUND, CHECKERBOARD = "0048", "0049"
CONDITIONS = (
    {"contrast": "0053", "label": "48 °C\narm", "short_label": "48 °C arm", "kind": "arm"},
    {"contrast": "0054", "label": "46 °C\narm", "short_label": "46 °C arm", "kind": "arm"},
    {"contrast": "0055", "label": "48 °C\nleg", "short_label": "48 °C leg", "kind": "leg"},
    {"contrast": "0056", "label": "46 °C\nleg", "short_label": "46 °C leg", "kind": "leg"},
    {"contrast": SOUND, "label": "Sound", "short_label": "Sound", "kind": "sound"},
    {"contrast": CHECKERBOARD, "label": "Checker-\nboard", "short_label": "Checkerboard", "kind": "vision"},
)
COLORS = {"arm": ("#3B6670", "#234A53"), "leg": ("#C45A4B", "#963B32"),
          "sound": ("#D4B66A", "#9B762A"), "vision": ("#D4B66A", "#9B762A")}
T_THRESHOLD, T_VMAX, T_CMAP, SURFACE_MESH = 5.0, 50.0, "YlOrRd", "fsaverage7"
JITTER_SEED, JITTER_WIDTH = 20260812, 0.090
SIZES = dict(axis=16, tick=14, panel=21, stat=14)
NEW_CONTRASTS = (SOUND, CHECKERBOARD)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def derive_image_path(reference: str | Path, source: str, target: str) -> Path:
    """Path of the ``target`` contrast image next to the ``source`` one (e.g. s6wcon_0053.nii -> s6wcon_0048.nii)."""
    reference = Path(reference)
    name = reference.name.replace(source, target)
    if name == reference.name:
        raise DataError(f"Cannot derive the con{target} image from {reference}: the file name has no '{source}'.")
    return reference.with_name(name)


def expression_table(project: Project) -> pd.DataFrame:
    """Ratings, NPS expression (6 contrasts) and LASSO-PCR expression (6 contrasts) per participant."""
    contrasts = tuple(c["contrast"] for c in CONDITIONS)
    if project.public:  # scores of the non-pain contrasts were computed from the images before release
        table = read_table(project.f5 / "subjects.tsv", "Expression table", sep="\t")
        required = ["mean_pain_overall", *[f"nps_con_{c}" for c in contrasts], *[f"lassopcr_con_{c}" for c in contrasts]]
        require_columns(table, required, "Expression table")
        id_column = find_subject_column(table, "Expression table")
        return table.sort_values(id_column).reset_index(drop=True)  # pseudonyms keep the original order
    sample = load_imaging_sample(project)
    primary_column = f"image_con{HEAT_CONDITIONS[0].contrast}"
    table = merge_unique(sample[["subject_key", "sub_id", primary_column, *[c.rating for c in HEAT_CONDITIONS], MEAN_PAIN]],
                         load_nps_scores(project, contrasts))
    lasso = load_lasso_scores(project)
    heat = {f"lassopcr_con_{c.contrast}": f"signature_score_con{c.contrast}" for c in HEAT_CONDITIONS}
    missing = [src for src in heat.values() if src not in lasso.columns]
    if missing:
        raise DataError(f"{project.lasso_scores} lacks columns {missing}; re-run the LASSO-PCR analysis.")
    table = merge_unique(table, lasso[["subject_key", *heat.values()]].rename(columns={v: k for k, v in heat.items()}))

    geometry = MaskGeometry(load_3d(project.lasso_mask, "LASSO-PCR feature mask"))
    weights = geometry.load_vector(project.lasso_raw_weights, "LASSO-PCR weight map").astype(np.float64)
    for contrast in NEW_CONTRASTS:
        paths = [derive_image_path(p, HEAT_CONDITIONS[0].contrast, contrast) for p in table[primary_column]]
        absent = [str(p) for p in paths if not p.exists()]
        if absent:
            raise MissingInputError(f"{len(absent)} con{contrast} images not found, e.g. {absent[0]}")
        print(f"Scoring con{contrast} with the LASSO-PCR weights ...")
        table[f"lassopcr_con_{contrast}"] = geometry.load_matrix(paths, f"con{contrast} image").astype(np.float64) @ weights
    required = [MEAN_PAIN, *[f"nps_con_{c}" for c in contrasts], *[f"lassopcr_con_{c}" for c in contrasts]]
    table = table[np.isfinite(table[required].to_numpy(float)).all(axis=1)].sort_values("subject_key").reset_index(drop=True)
    if table.empty:
        raise DataError("No participant has complete ratings, NPS and LASSO-PCR scores.")
    print(f"Final complete combined sample: n={len(table)}")
    table = table.rename(columns={MEAN_PAIN: "mean_pain_overall"}).drop(columns=[primary_column])
    table.to_csv(project.f5 / "subjects.tsv", sep="\t", index=False)
    return table


def statistics(project: Project, data: pd.DataFrame):
    models = (("NPS", "nps_con_"), ("LASSO-PCR", "lassopcr_con_"))
    rows = []
    for model, prefix in models:
        for contrast, condition in ((CHECKERBOARD, "Checkerboard"), (SOUND, "Sound")):
            rows.append({"model": model, "contrast": contrast, "condition": condition,
                         **association_stats(data[f"{prefix}{contrast}"], data["mean_pain_overall"])})
    associations = pd.DataFrame(rows)
    associations.to_csv(project.f5 / "association_stats.tsv", sep="\t", index=False)
    groups = []
    for model, prefix in models:
        for spec in CONDITIONS:
            v = data[f"{prefix}{spec['contrast']}"].to_numpy(float)
            groups.append({"model": model, "contrast": spec["contrast"], "condition": spec["short_label"],
                           "kind": spec["kind"], "n": len(v), "mean_expression": float(v.mean()),
                           "sd_expression": float(v.std(ddof=1)), "sem_expression": float(v.std(ddof=1) / np.sqrt(len(v))),
                           "median_expression": float(np.median(v))})
    summary = pd.DataFrame(groups)
    summary.to_csv(project.f5 / "expression_summary.tsv", sep="\t", index=False)
    return associations, summary


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def padded_limits(values, fraction: float = 0.08, include_zero: bool = False) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or max(abs(hi), 1.0)
    return lo - fraction * span, hi + fraction * span


def style_axis(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_linewidth(1.30)
    ax.tick_params(axis="both", labelsize=SIZES["tick"], width=1.15, length=4.2, pad=3)


def distribution_panel(ax, data: pd.DataFrame, prefix: str, y_label: str) -> None:
    ax.axhline(0.0, color="#6B6B6B", linewidth=1.0, linestyle="--", alpha=0.55, zorder=0).set_dashes((4.0, 3.0))
    pooled = []
    for i, spec in enumerate(CONDITIONS):
        values = data[f"{prefix}{spec['contrast']}"].to_numpy(float)
        point, group = COLORS[spec["kind"]]
        jitter = np.random.default_rng(JITTER_SEED + i).uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(values))
        ax.scatter(i + jitter, values, s=26, color=point, alpha=0.72, edgecolors="none", zorder=3, clip_on=False)
        ax.errorbar(float(i), values.mean(), yerr=values.std(ddof=1), fmt="D", markersize=11.0, markerfacecolor=group,
                    markeredgecolor="black", markeredgewidth=1.0, ecolor="#000000", elinewidth=2.3, capsize=5.8,
                    capthick=2.3, zorder=10, clip_on=False)
        pooled.append(values)
    ax.set_ylim(*padded_limits(np.concatenate(pooled), include_zero=True))
    ax.set_xlim(-0.48, len(CONDITIONS) - 0.52)
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels([s["label"] for s in CONDITIONS], fontsize=SIZES["tick"])
    ax.set_ylabel(y_label, fontsize=SIZES["axis"], fontweight="bold", labelpad=9)
    style_axis(ax)


def regression_panel(ax, data: pd.DataFrame, column: str, x_label: str, kind: str, y_limits, show_ylabel: bool) -> None:
    from .style import regression_band
    point, group = COLORS[kind]
    x, y = data[column].to_numpy(float), data["mean_pain_overall"].to_numpy(float)
    ax.scatter(x, y, s=29, color="#000000", alpha=0.80, edgecolors="none", zorder=3, clip_on=False)
    x_line, y_line, lower, upper = regression_band(x, y)
    ax.plot(x_line, y_line, color=group, linewidth=2.2, zorder=4)
    if lower is not None:
        ax.fill_between(x_line, lower, upper, color=point, alpha=0.18, linewidth=0, zorder=2)
    ax.text(0.04, 0.99, f"Spearman ρ = {stats.spearmanr(x, y).statistic:.2f}", transform=ax.transAxes, ha="left",
            va="top", fontsize=SIZES["stat"])
    ax.set_xlabel(x_label, fontsize=SIZES["axis"], labelpad=8)
    ax.set_ylabel("Mean pain rating (VAS)" if show_ylabel else "", fontsize=SIZES["axis"], labelpad=9)
    ax.set_ylim(*y_limits)
    ax.set_xlim(*padded_limits(x))
    style_axis(ax)


def brain_panel(figure, parent_spec, path: Path, mesh, backgrounds):
    """Four cortical views of the positive suprathreshold T values of a group map."""
    from nilearn import image, plotting, surface
    img = image.load_img(str(path))
    data = np.asarray(img.get_fdata(), dtype=float)
    if not (np.isfinite(data) & (data >= T_THRESHOLD)).any():
        raise DataError(f"No positive voxels survive T >= {T_THRESHOLD:g}: {path}")
    img = image.new_img_like(img, np.where(np.isfinite(data) & (data >= T_THRESHOLD), data, 0.0), copy_header=True)
    spec = parent_spec.subgridspec(2, 2, hspace=0.0, wspace=0.0)
    textures, axes = {}, []
    for i, (hemi, view) in enumerate((("right", "lateral"), ("left", "lateral"), ("right", "medial"), ("left", "medial"))):
        ax = figure.add_subplot(spec[i // 2, i % 2], projection="3d")
        if hemi not in textures:
            textures[hemi] = surface.vol_to_surf(img, mesh[f"pial_{hemi}"], interpolation="linear")
        plotting.plot_surf_stat_map(mesh[f"infl_{hemi}"], textures[hemi], bg_map=backgrounds[hemi], hemi=hemi, view=view,
                                    cmap=T_CMAP, threshold=T_THRESHOLD, vmin=0, vmax=T_VMAX, colorbar=False,
                                    bg_on_data=False, axes=ax, figure=figure)
        axes.append(ax)
    return axes


def make_figure(project: Project, data: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from . import brain
    figure = plt.figure(figsize=(15.0, 15.8), facecolor="white")
    grid = figure.add_gridspec(2, 1, height_ratios=(1.8, 2.0), left=0.055, right=0.988, bottom=0.055, top=0.985, hspace=0.025)
    brain_grid = grid[0].subgridspec(1, 2, wspace=0.0)
    rows = grid[1].subgridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.34)
    widths = dict(width_ratios=(0.40, 0.065, 0.30, 0.30), wspace=0.08)
    middle, bottom = rows[0].subgridspec(1, 4, **widths), rows[1].subgridspec(1, 4, **widths)
    mesh = brain.load_fsaverage(SURFACE_MESH)
    backgrounds = {h: brain.sulcal_background(mesh[f"sulc_{h}"]) for h in ("left", "right")}
    sound_axes = brain_panel(figure, brain_grid[0, 0], project.sound_tmap, mesh, backgrounds)
    checker_axes = brain_panel(figure, brain_grid[0, 1], project.checkerboard_tmap, mesh, backgrounds)
    mappable = ScalarMappable(norm=Normalize(0, T_VMAX), cmap=T_CMAP)
    mappable.set_array([])
    figure.colorbar(mappable, ax=sound_axes + checker_axes, orientation="vertical", location="right", shrink=0.72,
                    fraction=0.035, pad=0.02, aspect=30, label="T value", ticks=(0, 10, 20, 30, 40, 50))
    ax_a, ax_b = figure.add_subplot(middle[0, 0]), figure.add_subplot(middle[0, 2])
    ax_c = figure.add_subplot(middle[0, 3], sharey=ax_b)
    ax_d, ax_e = figure.add_subplot(bottom[0, 0]), figure.add_subplot(bottom[0, 2], sharey=ax_b)
    ax_f = figure.add_subplot(bottom[0, 3], sharey=ax_b)
    low, high = padded_limits(data["mean_pain_overall"])
    y_limits = (max(0.0, low), min(104.0, high))
    distribution_panel(ax_a, data, "nps_con_", "NPS expression")
    regression_panel(ax_b, data, f"nps_con_{CHECKERBOARD}", "Checkerboard NPS expression", "vision", y_limits, True)
    regression_panel(ax_c, data, f"nps_con_{SOUND}", "Sound NPS expression", "sound", y_limits, False)
    ax_c.tick_params(labelleft=False)
    distribution_panel(ax_d, data, "lassopcr_con_", "LASSO-PCR signature expression")
    regression_panel(ax_e, data, f"lassopcr_con_{CHECKERBOARD}", "Checkerboard signature expression", "vision", y_limits, True)
    regression_panel(ax_f, data, f"lassopcr_con_{SOUND}", "Sound signature expression", "sound", y_limits, False)
    ax_f.tick_params(labelleft=False)
    for target, label in ((sound_axes, "a"), (checker_axes, "b"), (ax_a, "c"), (ax_b, "d"), (ax_c, "e"), (ax_d, "f"),
                          (ax_e, "g"), (ax_f, "h")):
        if isinstance(target, list):
            x0 = ax_a.get_position().x0 if label == "a" else min(a.get_position().x0 for a in target)
            y1 = max(a.get_position().y1 for a in target) - 0.045
        else:
            x0, y1 = target.get_position().x0, target.get_position().y1
        figure.text(x0 - 0.018, y1 + 0.008, label, fontsize=SIZES["panel"], fontweight="bold", ha="left", va="bottom")
    return figure


def run(project: Project) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    project.require("figure5")
    apply_style(14)
    data = expression_table(project)
    associations, summary = statistics(project, data)
    print("\nAssociation statistics:\n" + associations.to_string(index=False))
    print("\nGroup-expression summary:\n" + summary.to_string(index=False))
    figure = make_figure(project, data)
    save_figure(figure, project.figure_files("Figure5"))
    plt.close(figure)
    return data
