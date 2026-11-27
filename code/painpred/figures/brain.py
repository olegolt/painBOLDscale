"""Brain-map rendering to raster panels: cortical surfaces (nilearn) and cerebellar flatmaps (SUITPy).

Each ``render_*`` function returns an RGBA image array that the figure scripts place with ``imshow``.
nilearn / SUITPy are imported lazily.
"""

from __future__ import annotations

import io

import numpy as np

PANEL_DPI = 420
CEREBELLAR_SPACES = ("SUIT", "FSL", "SPM", "MNISym", "MNI")
SURFACE_INTERPOLATIONS = ("nearest_most_frequent", "nearest")


def figure_to_array(fig, dpi: int = PANEL_DPI) -> np.ndarray:
    import matplotlib.pyplot as plt
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="white", edgecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return plt.imread(buffer)


def crop_white_border(img, white_threshold: float = 0.985, pad: int = 6) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        return arr
    rgb = arr[:, :, :3].astype(float)
    if rgb.max() > 1:
        rgb /= 255.0
    rows, cols = np.where(np.any(rgb < white_threshold, axis=2))
    if rows.size == 0:
        return arr
    return arr[max(rows.min() - pad, 0): min(rows.max() + pad + 1, arr.shape[0]),
               max(cols.min() - pad, 0): min(cols.max() + pad + 1, arr.shape[1]), :]


def max_abs_keep_sign(data) -> np.ndarray:
    """Along axis 0, keep the value with the largest |value| (sign preserved); all-NaN columns stay NaN."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr
    finite = np.isfinite(arr)
    picked = np.take_along_axis(arr, np.argmax(np.where(finite, np.abs(arr), -np.inf), axis=0)[None], axis=0)[0]
    picked[~finite.any(axis=0)] = np.nan
    return picked


def load_fsaverage(mesh: str = "fsaverage5"):
    from nilearn import datasets
    return datasets.fetch_surf_fsaverage(mesh=mesh)


def sulcal_background(sulc_map, lower: float = 2.0, upper: float = 98.0, power: float = 0.72) -> np.ndarray:
    from nilearn import surface
    data = np.asarray(surface.load_surf_data(sulc_map), dtype=np.float64)
    finite = np.isfinite(data)
    lo, hi = np.nanpercentile(data[finite], [lower, upper])
    scaled = 2.0 * (np.clip(data, lo, hi) - lo) / (hi - lo) - 1.0
    enhanced = np.sign(scaled) * np.abs(scaled) ** power
    enhanced[~finite] = 0.0
    return enhanced.astype(np.float32)


def sample_to_surface(vol_img, mesh, depths, reducer=max_abs_keep_sign, interpolations=SURFACE_INTERPOLATIONS):
    """Sample a volume onto both hemispheres at several cortical depths and reduce across depths."""
    from nilearn import surface

    def at_depth(pial, white, depth):
        for interpolation in interpolations:
            try:
                out = surface.vol_to_surf(vol_img, pial, inner_mesh=white, interpolation=interpolation,
                                          kind="depth", depth=[float(depth)])
                return np.asarray(out, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
        raise RuntimeError("This nilearn version supports neither 'nearest_most_frequent' nor 'nearest' interpolation.")

    left = reducer(np.asarray([at_depth(mesh.pial_left, mesh.white_left, d) for d in depths]))
    right = reducer(np.asarray([at_depth(mesh.pial_right, mesh.white_right, d) for d in depths]))
    return left, right


def render_surface_panel(left, right, mesh, cmap, vmin, vmax, threshold, figsize=(8.6, 6.0)) -> np.ndarray:
    """Lateral (top) and medial (bottom) views of the right and left hemispheres."""
    import matplotlib.pyplot as plt
    from nilearn import plotting
    bg = {"left": sulcal_background(mesh.sulc_left), "right": sulcal_background(mesh.sulc_right)}
    texture = {"left": left, "right": right}
    fig = plt.figure(figsize=figsize, facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0, right=1, bottom=0, top=1, wspace=-0.12, hspace=-0.28)
    for (row, col), (hemi, view) in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                                        (("right", "lateral"), ("left", "lateral"), ("right", "medial"), ("left", "medial"))):
        plotting.plot_surf_stat_map(mesh[f"infl_{hemi}"], texture[hemi], hemi=hemi, view=view, bg_map=bg[hemi],
                                    axes=fig.add_subplot(grid[row, col], projection="3d"), cmap=cmap, vmin=vmin,
                                    vmax=vmax, threshold=threshold, colorbar=False, bg_on_data=False)
    return crop_white_border(figure_to_array(fig))


def project_cerebellum(vol_img, depths, label: str = "", reducer=max_abs_keep_sign):
    """Project to a SUIT flatmap in the coordinate space that yields the most non-zero vertices."""
    import SUITPy as suit
    results, counts = {}, {}
    for space in CEREBELLAR_SPACES:
        try:
            mapped = np.asarray(suit.flatmap.vol_to_surf(vol_img, space=space, ignore_zeros=True, depths=depths,
                                                         stats=reducer), dtype=float)
            mapped = (mapped[:, 0] if mapped.ndim == 2 else mapped).reshape(-1)
            valid = np.isfinite(mapped) & (mapped != 0)
            mapped[~valid] = np.nan
            results[space], counts[space] = mapped, int(valid.sum())
        except Exception as exc:  # noqa: BLE001 - SUITPy raises several types; try the next space
            print(f"  cerebellum {label}: space {space} failed ({exc!r})")
    if not results:
        raise RuntimeError(f"No SUITPy space worked for the cerebellar projection of {label or 'the map'}.")
    best = max(results, key=counts.get)
    print(f"  cerebellum {label}: using space {best} ({counts[best]} vertices)")
    return results[best]


def render_cerebellum_panel(flat_data, cmap, vmin, vmax, threshold, border_size=0.82, figsize=(7.8, 4.7)) -> np.ndarray:
    import matplotlib.pyplot as plt
    import SUITPy as suit
    flat = flat_data.copy()
    flat[np.abs(flat) < threshold] = np.nan
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111)
    plt.sca(ax)
    suit.flatmap.plot(flat, cmap=cmap, cscale=[vmin, vmax], colorbar=False, bordersize=border_size, new_figure=False)
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return crop_white_border(figure_to_array(fig))


def compact_tick(value: float) -> str:
    if abs(value) < 5e-12:
        return "0"
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}" if abs(value) >= 1 else f"{value:.2f}"


def add_scaled_colorbar(ax, cmap, limit: float, label: str, n_ticks: int = 5, label_size=10.5, tick_size=10.0) -> None:
    """Horizontal symmetric colorbar with tick labels divided by a power of ten (shown in the label)."""
    import matplotlib as mpl
    exponent = int(np.floor(np.log10(limit)))
    ticks = np.linspace(-limit, limit, n_ticks)
    bar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=mpl.colors.Normalize(-limit, limit), orientation="horizontal")
    bar.set_ticks(ticks)
    bar.set_ticklabels([compact_tick(t / 10.0**exponent) for t in ticks])
    bar.ax.tick_params(labelsize=tick_size, length=3, width=0.8, pad=2)
    bar.set_label(rf"{label} ($\times 10^{{{exponent}}}$)", fontsize=label_size, labelpad=3)
    bar.outline.set_linewidth(0.8)
