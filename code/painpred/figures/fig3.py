"""Figure 3: overlap of FWE-significant voxels across the four rating analyses, and NPS vs mean pain."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import stats

from ..config import Project
from ..data import MEAN_PAIN, load_analysis_table
from ..imaging import load_3d, resample_to, same_grid
from ..utils import DataError
from . import brain
from .style import apply_style, regression_band, save_figure

RENDER_DPI = 450
FIGSIZE = (16.0, 9.6)
AXES = {"surface": (0.018, 0.575, 0.326, 0.400), "cerebellum": (0.368, 0.555, 0.310, 0.425),
        "volume": (0.016, 0.160, 0.655, 0.365), "donut": (0.738, 0.615, 0.230, 0.345),
        "regression": (0.738, 0.130, 0.240, 0.405), "legend": (0.122, 0.073, 0.350, 0.020)}
CUT_COORDS = (6.0, -30.0, -12.0)
BG_RANGE = (120, 300)
SURFACE_DEPTHS = np.linspace(-1.5, 2.5, 33)
CEREBELLAR_DEPTHS = np.linspace(-0.5, 1.5, 25)
COUNT_COLORS = ["#DAE8E2", "#ACCDBC", "#6B9D84", "#2A6B4D"]
COUNT_CMAP = ListedColormap(COUNT_COLORS, name="overlap_count_green")
COUNT_BOUNDARIES = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
POINT_COLOR, LINE_COLOR, CI_COLOR, EDGE_COLOR = "#2A6B4D", "#171717", "#7A7A7A", "#3E3E3E"
NPS_COLUMN = "nps_con_0053"
GRAND_MEAN = "grand_mean_pain_rating"


def clean_counts(img):
    """Integer count map: values outside 1-4 (or non-finite) become 0."""
    import nibabel as nib
    data = np.rint(np.asarray(img.get_fdata(), dtype=np.float64))
    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(np.where(np.isfinite(data) & (data >= 1) & (data <= 4), data, 0).astype(np.uint8), img.affine, header)


def max_count(data) -> np.ndarray:
    """Maximum count across samples (cortical depths / SUIT depths); values < 1 become NaN."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr[None]
    finite = np.isfinite(arr)
    result = np.max(np.where(finite, arr, -np.inf), axis=0)
    result[~finite.any(axis=0)] = np.nan
    result[result < 1] = np.nan
    ok = np.isfinite(result)
    result[ok] = np.rint(result[ok])
    return result


def render_surface(counts) -> np.ndarray:
    import matplotlib.pyplot as plt
    from nilearn import plotting
    mesh = brain.load_fsaverage("fsaverage7")
    left, right = brain.sample_to_surface(counts, mesh, SURFACE_DEPTHS, reducer=max_count)
    texture = {"left": left, "right": right}
    fig = plt.figure(figsize=(9.1, 7.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0, right=1, bottom=0, top=1, wspace=-0.13, hspace=-0.27)
    for (row, col), (hemi, view) in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                                        (("right", "lateral"), ("left", "lateral"), ("right", "medial"), ("left", "medial"))):
        plotting.plot_surf_stat_map(mesh[f"infl_{hemi}"], texture[hemi], hemi=hemi, view=view,
                                    bg_map=brain.sulcal_background(mesh[f"sulc_{hemi}"]),
                                    axes=fig.add_subplot(grid[row, col], projection="3d"), cmap=COUNT_CMAP, vmin=0.5,
                                    vmax=4.5, threshold=0.5, colorbar=False, bg_on_data=False)
    return brain.crop_white_border(brain.figure_to_array(fig, RENDER_DPI), 0.987, 4)


def render_cerebellum(counts) -> np.ndarray:
    import matplotlib.pyplot as plt
    import SUITPy as suit
    results = {}
    for space in brain.CEREBELLAR_SPACES:
        try:
            mapped = np.asarray(suit.flatmap.vol_to_surf(counts, space=space, ignore_zeros=True, depths=CEREBELLAR_DEPTHS,
                                                         stats=max_count), dtype=float)
            mapped = (mapped[:, 0] if mapped.ndim == 2 else mapped).reshape(-1)
            mapped[~np.isfinite(mapped) | (mapped < 1)] = np.nan
            ok = np.isfinite(mapped)
            mapped[ok] = np.rint(mapped[ok])
            results[space] = mapped
            print(f"Cerebellum | {space}: vertices={int(ok.sum())}")
        except Exception as exc:  # noqa: BLE001 - SUITPy raises several types; try the next space
            print(f"Cerebellum | {space} failed: {exc!r}")
    if not results:
        raise DataError("No SUITPy projection space worked for the overlap map.")
    best = max(results, key=lambda s: int(np.isfinite(results[s]).sum()))
    print(f"Cerebellum | selected space: {best}")
    fig = plt.figure(figsize=(8.8, 7.3), facecolor="white")
    ax = fig.add_subplot(111)
    plt.sca(ax)
    suit.flatmap.plot(results[best], cmap=COUNT_CMAP, cscale=[0.5, 4.5], colorbar=False, bordersize=0.82, new_figure=False)
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return brain.crop_white_border(brain.figure_to_array(fig, RENDER_DPI), 0.987, 4)


def render_volume(counts, background_path) -> np.ndarray:
    import matplotlib.pyplot as plt
    import nibabel as nib
    from nilearn import plotting
    bg = load_3d(background_path, "T1 background")
    on_bg = clean_counts(counts if same_grid(counts, bg) else resample_to(counts, bg, "nearest"))
    overlay = np.asarray(on_bg.get_fdata(), np.float32)
    overlay = nib.Nifti1Image(np.where(overlay >= 1, np.rint(overlay), np.nan).astype(np.float32), bg.affine)
    lo, hi = BG_RANGE
    data = np.asarray(bg.get_fdata(), np.float32)
    bg_plot = nib.Nifti1Image(np.where(data <= 0, hi, np.clip(data, lo, hi)).astype(np.float32), bg.affine)
    fig = plt.figure(figsize=(15.8, 5.0), facecolor="white")
    display = plotting.plot_anat(bg_plot, cut_coords=CUT_COORDS, display_mode="ortho", black_bg=False, annotate=True,
                                 draw_cross=False, dim=0, vmin=lo, vmax=hi, colorbar=False, figure=fig)
    for cut_ax in display.axes.values():
        if not hasattr(cut_ax, "ax"):
            continue
        for text in cut_ax.ax.texts:
            if text.get_text() in ("L", "R"):
                text.set_visible(False)
            else:
                text.set_fontsize(19)
                text.set_fontweight("normal")
    display.add_overlay(overlay, cmap=COUNT_CMAP, threshold=0.5, vmin=0.5, vmax=4.5, colorbar=False)
    for cut_ax in display.axes.values():
        cut_ax.ax.set_facecolor("white")
    return brain.crop_white_border(brain.figure_to_array(fig, RENDER_DPI), 0.987, 4)


# --------------------------------------------------------------------------- #
# NPS vs mean pain
# --------------------------------------------------------------------------- #
def nps_regression(data: pd.DataFrame) -> dict:
    """OLS of mean pain on NPS expression (con0053) plus Spearman correlation."""
    x, y = data[NPS_COLUMN].to_numpy(float), data[GRAND_MEAN].to_numpy(float)
    df = len(x) - 2
    if df <= 0 or np.var(y) == 0 or np.var(x) == 0:
        raise DataError("The NPS regression needs at least 3 participants with variable NPS scores and ratings.")
    fit = stats.linregress(x, y)
    t = np.array([fit.intercept / fit.intercept_stderr, fit.slope / fit.stderr])
    se = np.array([fit.intercept_stderr, fit.stderr])
    p = 2 * stats.t.sf(np.abs(t), df)
    fitted = fit.intercept + fit.slope * x
    r2 = fit.rvalue ** 2
    rho, rho_p = stats.spearmanr(x, y, nan_policy="omit")
    return {"n": len(x), "degrees_of_freedom": df, "intercept": float(fit.intercept), "slope": float(fit.slope),
            "slope_p_value": float(p[1]), "fitted": fitted, "residuals": y - fitted, "r_squared": float(r2),
            "adjusted_r_squared": float(1 - (1 - r2) * (len(x) - 1) / df), "f_statistic": float(t[1] ** 2),
            "model_p_value": float(stats.f.sf(t[1] ** 2, 1, df)), "spearman_r": float(rho), "spearman_p": float(rho_p),
            "coefficients": pd.DataFrame({"term": ("intercept", NPS_COLUMN), "coefficient": [fit.intercept, fit.slope],
                                          "standard_error": se, "t": t, "p": p})}


def save_regression(project: Project, data: pd.DataFrame, fit: dict) -> None:
    out = data.assign(fitted_grand_mean_pain_rating=fit["fitted"], residual=fit["residuals"])
    out.to_csv(project.f3 / "nps_data.tsv", sep="\t", index=False)
    fit["coefficients"].to_csv(project.f3 / "nps_coefficients.tsv", sep="\t", index=False)
    keys = ("n", "degrees_of_freedom", "intercept", "slope", "slope_p_value", "r_squared", "adjusted_r_squared",
            "f_statistic", "model_p_value", "spearman_r", "spearman_p")
    pd.DataFrame([{"outcome": GRAND_MEAN, "predictor": NPS_COLUMN, **{k: fit[k] for k in keys}}]).to_csv(
        project.f3 / "nps_summary.tsv", sep="\t", index=False)


def draw_regression(ax, data: pd.DataFrame, fit: dict) -> None:
    x, y = data[NPS_COLUMN].to_numpy(float), data[GRAND_MEAN].to_numpy(float)
    x_line, y_line, lower, upper = regression_band(x, y)
    ax.fill_between(x_line, lower, upper, color=CI_COLOR, alpha=0.18, linewidth=0, zorder=1)
    ax.scatter(x, y, s=34, alpha=0.92, color=POINT_COLOR, edgecolors="white", linewidths=0.45, zorder=3)
    ax.plot(x_line, y_line, color=LINE_COLOR, linewidth=2.25, zorder=4)
    ax.text(0.045, 0.955, rf"Spearman $\rho$ = {fit['spearman_r']:.2f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=12, linespacing=1.12)
    ax.set_xlabel("Neurologic Pain Signature (NPS)", fontsize=17, labelpad=5)
    ax.set_ylabel("Mean pain rating", fontsize=17, labelpad=4)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color(EDGE_COLOR)
        ax.spines[name].set_linewidth(1.55)
    ax.tick_params(axis="both", labelsize=13, width=1.35, length=5, color=EDGE_COLOR, pad=4)
    pad = 0.055 * (x.max() - x.min()) if x.max() > x.min() else 1.0
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(40, 100)


def draw_donut(ax, counts: np.ndarray) -> None:
    n = np.array([np.sum(counts == k) for k in (1, 2, 3, 4)], dtype=int)
    if n.sum() == 0:
        raise DataError("The overlap-count image contains no voxels with values 1-4.")
    ax.pie(n, colors=COUNT_COLORS, startangle=90, counterclock=False, radius=1.0,
           wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 0.30})
    ax.text(0.0, 0.06, f"{int(n.sum()):,}", ha="center", va="center", fontsize=27, fontweight="bold")
    ax.text(0.0, -0.16, "significant\nvoxels", ha="center", va="center", fontsize=12, linespacing=1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_legend(ax) -> None:
    import matplotlib as mpl
    bar = mpl.colorbar.ColorbarBase(ax, cmap=COUNT_CMAP, norm=BoundaryNorm(COUNT_BOUNDARIES, COUNT_CMAP.N),
                                    boundaries=COUNT_BOUNDARIES, ticks=[1, 2, 3, 4], spacing="uniform",
                                    orientation="horizontal", drawedges=False)
    bar.set_label("Number of significant rating associations", fontsize=15, fontweight="bold", labelpad=5)
    bar.ax.tick_params(axis="x", labelsize=14, width=1.0, length=4, pad=4)
    bar.outline.set_linewidth(1.0)


def run(project: Project) -> None:
    import matplotlib.pyplot as plt
    project.require("figure3")
    count_path = project.f3 / "overlap_count.nii.gz"
    if not count_path.exists():
        raise DataError(f"Overlap count map missing: {count_path}. Run `overlap.count_overlap(project)` first.")
    apply_style(14)
    data = load_analysis_table(project, nps_contrasts=("0053",))[["subject_key", NPS_COLUMN, MEAN_PAIN]]
    data = data.rename(columns={MEAN_PAIN: GRAND_MEAN})
    fit = nps_regression(data)
    save_regression(project, data, fit)
    print(f"NPS regression | n={fit['n']}, R²={fit['r_squared']:.4f}, slope={fit['slope']:.6f}, "
          f"p={fit['slope_p_value']:.3g}, Spearman rho={fit['spearman_r']:.4f}")
    counts = clean_counts(load_3d(count_path, "overlap count map"))
    values = np.asarray(counts.get_fdata(), dtype=np.uint8)
    for k in range(5):
        print(f"Count {k}: {int(np.sum(values == k))} voxels")
    panels = {"surface": render_surface(counts), "cerebellum": render_cerebellum(counts),
              "volume": render_volume(counts, project.t1_background)}
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    for key in ("surface", "cerebellum", "volume"):
        ax = fig.add_axes(AXES[key], frameon=False)
        ax.imshow(panels[key], interpolation="lanczos", aspect="equal")
        ax.set_axis_off()
        ax.set_anchor("C")
        ax.margins(0)
    draw_legend(fig.add_axes(AXES["legend"]))
    draw_donut(fig.add_axes(AXES["donut"], frameon=False), values)
    draw_regression(fig.add_axes(AXES["regression"]), data, fit)
    fig.text(0.006, 0.982, "a", ha="left", va="top", fontsize=20, fontweight="bold")
    for label, key in (("b", "donut"), ("c", "regression")):
        x, y, _, h = AXES[key]
        fig.text(x - 0.05, y + h + 0.012, label, ha="left", va="top", fontsize=20, fontweight="bold")
    save_figure(fig, project.figure_files("Figure3"))
    plt.close(fig)
