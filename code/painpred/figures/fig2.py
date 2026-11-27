"""Figure 2: voxelwise correlation between heat-evoked activity and pain ratings (con0053 x pain_48_arm)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.ndimage import binary_dilation, map_coordinates, maximum_filter, minimum_filter

from ..config import PRIMARY, Project
from ..imaging import load_3d, resample_to, same_grid
from ..utils import DataError, MissingInputError, read_table, require_columns
from . import brain
from .style import apply_style, save_figure

FIGSIZE = (16.0, 15.9)
RENDER_DPI = 450
RECOVERY_THRESHOLD = 0.10  # |r| below this is not displayed
NICE_TICK_STEPS = (0.01, 0.02, 0.025, 0.05, 0.10, 0.20)
TARGET_INTERVALS, MIN_INTERVALS, MAX_INTERVALS = 4, 3, 6
SURFACE_DEPTHS = np.linspace(-1.5, 2.5, 33)
SURFACE_DILATION_ITERATIONS = 3
CEREBELLAR_DEPTHS = np.linspace(-0.5, 1.5, 25)
SUBCORTICAL_ATLAS = "sub-maxprob-thr25-2mm"
SUBCORTICAL_KEYWORDS = ("thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala", "accumbens", "brain stem")
BG_RANGE = (120, 300)
LAYOUT = dict(outer_left=0.010, outer_right=0.947, top=0.995, gutter=0.004, row_gap=0.010, colorbar_left=0.962,
              colorbar_width=0.012, colorbar_height_fraction=0.34, regression_top_gap=0.024, regression_bottom=0.035,
              regression_side=0.030, regression_gap=0.155)
FIXED_PANELS = (
    {"key": "cortex", "panel": "right aMCC", "label": "Cortex", "xyz": (6.0, 15.0, 33.0)},
    {"key": "brainstem", "panel": "right midbrain", "label": "Brainstem", "xyz": (6.0, -30.0, -12.0)},
    {"key": "cerebellum", "panel": "left cerebellar lobule V", "label": "Cerebellum", "xyz": (-12.0, -48.0, -13.5)},
    {"key": "thalamus", "panel": "right thalamus", "label": "Thalamus", "xyz": (9.0, -21.0, -4.0)},
)
LOBULE_LABELS = (("I–IV", 0.50, 0.915), ("V", 0.50, 0.825), ("VI", 0.50, 0.715), ("Crus I", 0.815, 0.610),
                 ("Crus II", 0.815, 0.475), ("VIIb", 0.795, 0.365), ("VIIIa", 0.745, 0.275), ("VIIIb", 0.665, 0.195),
                 ("IX", 0.525, 0.125), ("X", 0.500, 0.040))
BLUE, BLUE_DARK, RED, RED_DARK = "#3B6670", "#234A53", "#C45A4B", "#963B32"
YELLOW, YELLOW_MAX, YELLOW_START = "#E3C27A", "#F1D98A", 0.6
POINT_COLOR, LINE_COLOR, CI_COLOR, EDGE_COLOR = "#2C2C2C", "#802C24", "#945852", "#3B3B3B"
FONT = dict(panel=21, coordinate=14, axis=18, tick=16, stat=16, colorbar_label=16, colorbar_tick=14)


# --------------------------------------------------------------------------- #
# Colour map and display scale
# --------------------------------------------------------------------------- #
def make_combined_map(vmax: float):
    """Diverging map over [-vmax, vmax]; r >= 0.6 is in the yellow range whatever vmax is."""
    if vmax <= 0:
        raise ValueError("vmax must be positive.")

    def position(r):
        return float(np.clip(0.5 + 0.5 * r / vmax, 0.0, 1.0))

    pos = np.maximum.accumulate(np.array([0.500001, position(0.12), position(0.34), position(0.46), position(0.55),
                                          position(YELLOW_START), 1.0]))
    pos[-1] = 1.0
    return LinearSegmentedColormap.from_list("CombinedMap_BlueWhiteRedYellow", (
        (0.000, BLUE_DARK), (0.180, BLUE), (0.340, "#7898A0"), (0.460, "#DCE7E9"), (0.500, "#F7F7F7"),
        (pos[1], "#F1DEDA"), (pos[2], RED), (pos[3], RED_DARK), (pos[4], "#C78466"), (pos[5], YELLOW), (pos[6], YELLOW_MAX)))


def display_scale(data: np.ndarray, name: str) -> dict:
    """Symmetric legend range on 'nice' tick steps, chosen from the displayed |r| range."""
    values = np.abs(data[np.isfinite(data) & (np.abs(data) >= RECOVERY_THRESHOLD)])
    if values.size == 0:
        raise DataError(f"No values survive |r| >= {RECOVERY_THRESHOLD:.2f} in {name}; nothing to display.")
    lo, hi = float(values.min()), float(values.max())
    options = []
    for step in NICE_TICK_STEPS:
        lower = max(RECOVERY_THRESHOLD, float(np.floor((lo + 1e-12) / step) * step))
        upper = float(np.ceil((hi - 1e-12) / step) * step)
        if upper <= lower:
            upper = lower + TARGET_INTERVALS * step
        n = int(round((upper - lower) / step))
        if MIN_INTERVALS <= n <= MAX_INTERVALS:
            options.append((abs(n - TARGET_INTERVALS), step, lower, upper, n))
    if options:
        _, step, lower, upper, n = min(options, key=lambda o: (o[0], o[1]))
    else:
        step = min(NICE_TICK_STEPS, key=lambda c: abs(c - max(hi - lo, 0.01) / TARGET_INTERVALS))
        lower = max(RECOVERY_THRESHOLD, float(np.floor(lo / step) * step))
        upper = float(np.ceil(hi / step) * step)
        if upper <= lower:
            upper = lower + TARGET_INTERVALS * step
        n = max(1, int(round((upper - lower) / step)))
    ticks = np.round(lower + np.arange(n + 1) * step, 6)
    print(f"Display scale | {name}: |r| {lo:.4f}-{hi:.4f}; legend {ticks[0]:.2f}-{ticks[-1]:.2f}")
    return {"threshold": lo, "vmin": float(ticks[0]), "vmax": float(ticks[-1]), "ticks": ticks.tolist()}


def threshold_map(img, threshold: float = RECOVERY_THRESHOLD):
    import nibabel as nib
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    cleaned = np.where(np.isfinite(data) & (np.abs(data) >= threshold), data, 0.0).astype(np.float32)
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(cleaned, img.affine, header)


# --------------------------------------------------------------------------- #
# Peak (fixed-coordinate) data
# --------------------------------------------------------------------------- #
def sample_at(img, xyz) -> float:
    """Trilinear sample of ``img`` at a world coordinate."""
    data = np.asarray(img.get_fdata(), dtype=float)
    ijk = np.linalg.inv(img.affine) @ np.append(np.asarray(xyz, float), 1.0)
    ijk = ijk[:3]
    if np.any(ijk < 0) or np.any(ijk > np.asarray(data.shape[:3]) - 1):
        raise DataError(f"MNI coordinate {tuple(xyz)} lies outside the image.")
    return float(map_coordinates(data, ijk[:, None], order=1, mode="nearest", prefilter=False)[0])


def fixed_panels(display_img, value_img, value_path) -> list[dict]:
    inverse = np.linalg.inv(display_img.affine)
    panels = []
    for panel in FIXED_PANELS:
        xyz = np.asarray(panel["xyz"], float)
        ijk = (inverse @ np.append(xyz, 1.0))[:3]
        nearest = np.rint(ijk).astype(int)
        r = sample_at(value_img, xyz)
        panels.append({**panel, "xyz": xyz, "voxel_coordinate": ijk, "voxel_ijk": tuple(int(v) for v in nearest),
                       "nearest_voxel_xyz": (display_img.affine @ np.append(nearest.astype(float), 1.0))[:3],
                       "r": r, "abs_r": abs(r), "selection_mode": "fixed MNI coordinate",
                       "source_map_type": "Unthresholded", "source_map": str(value_path)})
        print(f"Panel {panel['panel']} | {panel['label']}: MNI={tuple(xyz)}; r={r:.4f}")
    return panels


def extract_peak_data(table: pd.DataFrame, reference, peaks: list[dict], image_column: str, rating: str) -> pd.DataFrame:
    rows = []
    for _, subject in table.iterrows():
        img = load_3d(subject[image_column], f"contrast image of {subject['sub_id']}")
        img = img if same_grid(img, reference) else resample_to(img, reference, "continuous")
        rows.append({"sub_id": subject["sub_id"], rating: float(subject[rating]),
                     **{f"{p['key']}_value": sample_at(img, p["xyz"]) for p in peaks}})
    out = pd.DataFrame(rows)
    valid = np.isfinite(out[[rating, *[f"{p['key']}_value" for p in peaks]]]).all(axis=1)
    out = out[valid].reset_index(drop=True)
    if len(out) < 5:
        raise DataError(f"Only {len(out)} participants have finite values at all peak coordinates.")
    return out


def save_peak_tables(project: Project, peaks: list[dict], peak_data: pd.DataFrame) -> None:
    rows = [{"panel": p["panel"], "region": p["label"],
             **{f"voxel_{a}_nearest": p["voxel_ijk"][i] for i, a in enumerate("ijk")},
             **{f"voxel_{a}_continuous": p["voxel_coordinate"][i] for i, a in enumerate("ijk")},
             **{f"mni_{a}_requested": p["xyz"][i] for i, a in enumerate("xyz")},
             **{f"mni_{a}_nearest_voxel": p["nearest_voxel_xyz"][i] for i, a in enumerate("xyz")},
             "r_value_at_requested_coordinate": p["r"], "abs_r_value": p["abs_r"],
             "selection_mode": p["selection_mode"], "source_map_type": p["source_map_type"],
             "source_map": p["source_map"]} for p in peaks]
    pd.DataFrame(rows).to_csv(project.f2 / "figure_peaks.tsv", sep="\t", index=False)
    peak_data.to_csv(project.f2 / "figure_peak_data.tsv", sep="\t", index=False)


# --------------------------------------------------------------------------- #
# Brain panels
# --------------------------------------------------------------------------- #
def dilate_signed(img, iterations: int = SURFACE_DILATION_ITERATIONS, threshold: float = RECOVERY_THRESHOLD):
    """Grow the thresholded map outwards (largest local value of the stronger sign) so it reaches the cortical surface."""
    import nibabel as nib
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    valid = np.isfinite(data) & (np.abs(data) >= threshold)
    if not valid.any():
        return img
    support = binary_dilation(valid, iterations=iterations)
    size = 2 * iterations + 1
    pos = maximum_filter(np.where(valid & (data > 0), data, -np.inf), size=size, mode="constant", cval=-np.inf)
    neg = minimum_filter(np.where(valid & (data < 0), data, np.inf), size=size, mode="constant", cval=np.inf)
    has_pos, has_neg = np.isfinite(pos), np.isfinite(neg)
    out = np.zeros_like(data, dtype=np.float32)
    for select, source in ((support & has_pos & ~has_neg, pos), (support & has_neg & ~has_pos, neg)):
        out[select] = source[select]
    both = support & has_pos & has_neg
    use_pos = both & (np.abs(pos) >= np.abs(neg))
    out[use_pos], out[both & ~use_pos] = pos[use_pos], neg[both & ~use_pos]
    out[valid] = data[valid]
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(out, img.affine, header)


def plot_surface_views(left, right, cmap, vmin: float, vmax: float, mesh=None) -> np.ndarray:
    """Left/right lateral (top) and medial (bottom) inflated-surface views of two vertex textures."""
    import matplotlib.pyplot as plt
    from nilearn import plotting
    mesh = mesh or brain.load_fsaverage("fsaverage7")
    fig = plt.figure(figsize=(10.2, 7.4), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.01, right=0.99, bottom=0.025, top=0.955, wspace=-0.15, hspace=-0.24)
    for (row, col), hemi, view in zip(((0, 0), (0, 1), (1, 0), (1, 1)), ("left", "right", "left", "right"),
                                      ("lateral", "lateral", "medial", "medial")):
        plotting.plot_surf_stat_map(mesh[f"infl_{hemi}"], left if hemi == "left" else right, hemi=hemi, view=view,
                                    bg_map=brain.sulcal_background(mesh[f"sulc_{hemi}"]),
                                    axes=fig.add_subplot(grid[row, col], projection="3d"), cmap=cmap, vmin=vmin,
                                    vmax=vmax, threshold=None, colorbar=False, bg_on_data=False)
    return brain.figure_to_array(fig, RENDER_DPI)


def render_surface(img, threshold: float, vmax: float):
    mesh = brain.load_fsaverage("fsaverage7")
    left, right = brain.sample_to_surface(dilate_signed(img), mesh, SURFACE_DEPTHS, interpolations=("linear",))
    left[np.abs(left) < threshold] = np.nan
    right[np.abs(right) < threshold] = np.nan
    return brain.crop_white_border(plot_surface_views(left, right, make_combined_map(vmax), -vmax, vmax, mesh), 0.987, 12)


def render_cerebellum(img, threshold: float, vmax: float, cmap=None, vmin: float | None = None):
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import SUITPy as suit
    results = {}
    for space in brain.CEREBELLAR_SPACES:
        try:
            mapped = np.asarray(suit.flatmap.vol_to_surf(img, space=space, ignore_zeros=True, depths=CEREBELLAR_DEPTHS,
                                                         stats=brain.max_abs_keep_sign), dtype=float)
            mapped = (mapped[:, 0] if mapped.ndim == 2 else mapped).reshape(-1)
            results[space] = (int(np.sum(np.isfinite(mapped) & (np.abs(mapped) >= threshold))), mapped)
        except Exception as exc:  # noqa: BLE001 - SUITPy raises several types; try the next space
            print(f"Cerebellar projection failed for {space}: {exc!r}")
    if not results:
        raise DataError("No SUITPy cerebellar projection succeeded; check the SUITPy installation and the map's space.")
    best = max(results, key=lambda name: results[name][0])
    flat = results[best][1].copy()
    flat[~np.isfinite(flat) | (np.abs(flat) < threshold)] = np.nan
    print(f"Cerebellar surface space: {best}")
    fig = plt.figure(figsize=(8.8, 7.3), facecolor="white")
    ax = fig.add_subplot(111)
    plt.sca(ax)
    suit.flatmap.plot(flat, cmap=cmap or make_combined_map(vmax), cscale=[-vmax if vmin is None else vmin, vmax], colorbar=False, bordercolor="#303030",
                      bordersize=0.82, new_figure=False)
    for text, x, y in LOBULE_LABELS:
        label = ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center", fontsize=11, fontweight="bold",
                        color="#161616", zorder=20, clip_on=False)
        label.set_path_effects([path_effects.Stroke(linewidth=2.4, foreground="white"), path_effects.Normal()])
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return brain.crop_white_border(brain.figure_to_array(fig, RENDER_DPI), 0.987, 4)


def subcortical_mask(target):
    """Harvard-Oxford subcortical structures (incl. brain stem), nearest-neighbour resampled to ``target``."""
    import nibabel as nib
    from nilearn import datasets, image
    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name=SUBCORTICAL_ATLAS)
    resampled = image.resample_to_img(image.load_img(atlas.maps), target, interpolation="nearest",
                                      force_resample=True, copy_header=True)
    ids = [i for i, name in enumerate(atlas.labels) if any(k in " ".join(str(name).lower().split()) for k in SUBCORTICAL_KEYWORDS)]
    mask = np.isin(np.rint(np.asarray(resampled.get_fdata(), float)).astype(np.int16), np.asarray(ids, dtype=np.int16))
    header = target.header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(mask.astype(np.uint8), target.affine, header)


def strongest_voxel_xyz(rmap, mask) -> tuple[float, ...]:
    r = np.asarray(rmap.get_fdata(), dtype=float)
    inside = np.asarray(mask.get_fdata(), dtype=float) > 0.5
    valid = inside & np.isfinite(r) & (np.abs(r) >= RECOVERY_THRESHOLD)
    if not valid.any():
        valid = inside & np.isfinite(r) & (np.abs(r) > 0)
    if not valid.any():
        raise DataError("No subcortical voxel survives in the map, so the volume panel has nothing to show.")
    ijk = np.unravel_index(int(np.argmax(np.where(valid, np.abs(r), -np.inf))), r.shape)
    return tuple(float(v) for v in (rmap.affine @ np.append(np.asarray(ijk, float), 1.0))[:3])


def render_volume(img, background_path, threshold: float, vmax: float):
    import matplotlib.pyplot as plt
    import nibabel as nib
    from nilearn import plotting
    cut = strongest_voxel_xyz(img, subcortical_mask(img))
    bg = load_3d(background_path, "T1 background")
    on_bg = threshold_map(img if same_grid(img, bg) else resample_to(img, bg, "continuous"))
    mask = np.asarray(subcortical_mask(bg).get_fdata(), float) > 0
    overlay = np.asarray(on_bg.get_fdata(), float)
    overlay = nib.Nifti1Image(np.where(mask & (np.abs(overlay) > 0), overlay, np.nan).astype(np.float32), bg.affine)
    lo, hi = BG_RANGE
    bg_data = np.asarray(bg.get_fdata(), float)
    bg_plot = nib.Nifti1Image(np.where(bg_data <= 0, hi, np.clip(bg_data, lo, hi)).astype(np.float32), bg.affine)
    fig = plt.figure(figsize=(18.2, 4.4), facecolor="white")
    display = plotting.plot_anat(bg_plot, cut_coords=cut, display_mode="ortho", black_bg=False, annotate=True,
                                 draw_cross=False, dim=0, vmin=lo, vmax=hi, colorbar=False, figure=fig)
    for cut_ax in display.axes.values():
        if not hasattr(cut_ax, "ax"):
            continue
        for text in cut_ax.ax.texts:
            if text.get_text() in ("L", "R"):
                text.set_visible(False)
            else:
                text.set_fontsize(18)
                text.set_fontweight("bold")
                text.set_transform(cut_ax.ax.transAxes)
                text.set_x(0.5)
                text.set_horizontalalignment("center")
    display.add_overlay(overlay, cmap=make_combined_map(vmax), threshold=threshold, vmin=-vmax, vmax=vmax, colorbar=False)
    for cut_ax in display.axes.values():
        cut_ax.ax.set_facecolor("white")
    return brain.crop_white_border(brain.figure_to_array(fig, RENDER_DPI), 0.987, 4)


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def regression_panel(fig, spec, ratings, values, peak, show_ylabel: bool) -> None:
    from .style import regression_band
    grid = spec.subgridspec(2, 1, height_ratios=(0.095, 0.905), hspace=0.002)
    header, ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    header.set_axis_off()
    header.set_zorder(10)
    ax.set_zorder(10)
    ax.patch.set_facecolor("white")
    header.text(0.5, 0.68, peak["panel"], ha="center", va="center", fontsize=FONT["panel"], fontweight="bold",
                transform=header.transAxes)
    x_mni, y_mni, z_mni = peak["xyz"]
    header.text(0.5, 0.12, f"({x_mni:.0f}, {y_mni:.0f}, {z_mni:.0f})", ha="center", va="center",
                fontsize=FONT["coordinate"], transform=header.transAxes)
    valid = np.isfinite(ratings) & np.isfinite(values)
    x, y = ratings[valid], values[valid]
    x_line, y_line, lower, upper = regression_band(x, y)
    if lower is None:
        raise DataError("At least three participants with rating variance are needed for the regression panels.")
    ax.fill_between(x_line, lower, upper, color=CI_COLOR, alpha=0.18, linewidth=0, zorder=1)
    ax.scatter(x, y, s=35, alpha=0.84, color=POINT_COLOR, edgecolors="white", linewidths=0.45, zorder=3)
    ax.plot(x_line, y_line, color=LINE_COLOR, linewidth=2.35, zorder=4)
    rho = stats.spearmanr(x, y, nan_policy="omit")[0]
    ax.text(0.04, 0.95, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=FONT["stat"])
    ax.set_xlabel("pain rating", fontsize=FONT["axis"], labelpad=7)
    ax.set_ylabel("contrast estimate" if show_ylabel else "", fontsize=FONT["axis"], labelpad=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color(EDGE_COLOR)
        ax.spines[name].set_linewidth(1.65)
    ax.tick_params(axis="both", labelsize=FONT["tick"], width=1.35, length=4.5, color=EDGE_COLOR)


def add_legend(ax, scale: dict) -> None:
    import matplotlib as mpl
    vmax = float(scale["vmax"])
    bar = mpl.colorbar.ColorbarBase(ax, cmap=make_combined_map(vmax), norm=mpl.colors.Normalize(-vmax, vmax),
                                    orientation="vertical")
    top = np.floor(vmax / 0.20) * 0.20
    ticks = np.round(np.arange(-top, top + 0.10, 0.20), 6)
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{t:.1f}" for t in ticks])
    bar.set_label("spearman ρ", fontsize=FONT["colorbar_label"], fontweight="bold", labelpad=6)
    bar.ax.tick_params(labelsize=FONT["colorbar_tick"], width=1.0, length=3.5, pad=3)
    bar.ax.yaxis.set_ticks_position("left")
    bar.ax.yaxis.set_label_position("right")
    bar.outline.set_linewidth(1.0)


def make_figure(project: Project, peaks, peak_data, panels: dict, rating: str) -> None:
    import matplotlib.pyplot as plt
    L = LAYOUT
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    width_in, height_in = FIGSIZE
    aspect = {k: panels[k].shape[1] / panels[k].shape[0] for k in ("surface", "cerebellum", "volume")}
    block_width = L["outer_right"] - L["outer_left"]
    top_height = (block_width * width_in - L["gutter"] * width_in) / (aspect["surface"] + aspect["cerebellum"]) / height_in
    top_bottom = L["top"] - top_height
    surface_width = aspect["surface"] * top_height * height_in / width_in
    cerebellum_width = aspect["cerebellum"] * top_height * height_in / width_in
    for key, left, w in (("surface", L["outer_left"], surface_width),
                         ("cerebellum", L["outer_left"] + surface_width + L["gutter"], cerebellum_width)):
        ax = fig.add_axes([left, top_bottom, w, top_height], frameon=False)
        _show(ax, panels[key])
    volume_height = block_width * width_in / aspect["volume"] / height_in
    volume_bottom = top_bottom - L["row_gap"] - volume_height
    _show(fig.add_axes([L["outer_left"], volume_bottom, block_width, volume_height], frameon=False), panels["volume"])
    bar_height = (L["top"] - volume_bottom) * L["colorbar_height_fraction"]
    centre = 0.5 * (L["top"] + volume_bottom)
    add_legend(fig.add_axes([L["colorbar_left"], centre - bar_height / 2, L["colorbar_width"], bar_height]), panels["scale"])
    regression_top = volume_bottom - L["regression_top_gap"]
    if regression_top <= L["regression_bottom"] + 0.20:
        raise DataError("The brain rows leave too little room for the regression panels; increase FIGSIZE height.")
    grid = fig.add_gridspec(1, 4, left=L["regression_side"], right=1 - L["regression_side"], bottom=L["regression_bottom"],
                            top=regression_top, wspace=L["regression_gap"], hspace=0.0)
    ratings = peak_data[rating].to_numpy(float)
    for i, peak in enumerate(peaks):
        regression_panel(fig, grid[0, i], ratings, peak_data[f"{peak['key']}_value"].to_numpy(float), peak, i == 0)
    save_figure(fig, project.figure_files("Figure2"))
    plt.close(fig)


def _show(ax, img) -> None:
    ax.imshow(img, interpolation="lanczos", aspect="equal")
    ax.set_axis_off()
    ax.set_anchor("C")
    ax.margins(0)


def peak_table(project: Project, peaks: list[dict], rating: str) -> pd.DataFrame:
    """Per-participant values at the four peak coordinates.

    With the shared data this is read from ``figure2/figure_peak_data.tsv``; with the raw study files
    it is sampled from the participants' contrast images (and written to that same file).
    """
    if project.public:
        table = read_table(project.f2 / "figure_peak_data.tsv", "Peak values", sep="\t")
        require_columns(table, ["sub_id", rating, *[f"{p['key']}_value" for p in peaks]], "Peak values")
        return table
    image_column = f"image_con{PRIMARY.contrast}"
    table = read_table(project.subjects, "Imaging sample table", sep="\t")
    require_columns(table, ["sub_id", rating, image_column], "Imaging sample table")
    table[rating] = pd.to_numeric(table[rating], errors="coerce")
    table = table[np.isfinite(table[rating])].reset_index(drop=True)
    missing = [p for p in table[image_column] if not Path(p).exists()]
    if missing:
        raise MissingInputError("Participant contrast images are missing:\n  - " + "\n  - ".join(missing[:5]))
    reference = load_3d(project.primary_dir / "rho.nii.gz", "unthresholded r-map")
    return extract_peak_data(table, reference, peaks, image_column, rating)


def run(project: Project, overwrite: bool = False) -> None:
    project.require("figure2")
    outputs = [*project.figure_files("Figure2"), project.f2 / "figure_peaks.tsv"]
    if not overwrite and all(p.is_file() for p in outputs):
        print("Figure 2 already exists (overwrite=True to rebuild).")
        return
    apply_style()
    rating = PRIMARY.rating
    candidates = []
    for label, name in (("FWE-corrected", "rho_fwe"), ("Uncorrected |z| >= 3.1", "rho_z3p1"), ("Unthresholded", "rho")):
        path = project.primary_dir / f"{name}.nii.gz"
        if path.exists():
            img = load_3d(path, label)
            if np.any(np.isfinite(d := np.asarray(img.get_fdata(), float)) & (np.abs(d) > 0)):
                candidates.append({"label": label, "path": path, "img": img})
    if not candidates:
        raise MissingInputError(f"No r-map found in {project.primary_dir}; run the Figure 2 mass-univariate analysis.")
    reference = candidates[0]["img"]
    for c in candidates:
        c["img"] = c["img"] if same_grid(c["img"], reference) else resample_to(c["img"], reference, "continuous")
    value = next((c for c in candidates if c["label"] == "Unthresholded"), None)
    if value is None:
        raise MissingInputError(f"The unthresholded r-map is needed for the peak panels: {project.primary_dir / 'rho.nii.gz'}")
    peaks = fixed_panels(reference, value["img"], value["path"])
    peak_data = peak_table(project, peaks, rating)
    save_peak_tables(project, peaks, peak_data)
    shown = threshold_map(reference)
    scale = display_scale(np.asarray(shown.get_fdata(), float), f"con{PRIMARY.contrast}_x_{rating}")
    threshold, vmax = scale["threshold"], scale["vmax"]
    panels = {"surface": render_surface(shown, threshold, vmax), "cerebellum": render_cerebellum(shown, threshold, vmax),
              "volume": render_volume(shown, project.t1_background, threshold, vmax), "scale": scale}
    make_figure(project, peaks, peak_data, panels, rating)
