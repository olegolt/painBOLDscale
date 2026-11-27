"""Figure S2: positive correlation maps of the three non-primary heat contrasts (cortex and cerebellum)."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import map_coordinates

from ..config import Project
from ..imaging import load_3d
from ..utils import MissingInputError
from . import brain, fig2
from .style import apply_style, save_figure

Z_THRESHOLD = 3.1
PANELS = (
    {"label": "a", "contrast": "0055", "rating": "pain_48_leg", "vmin": 0.3, "vmax": 0.5, "ticks": np.arange(0.30, 0.501, 0.05)},
    {"label": "b", "contrast": "0054", "rating": "pain_46_arm", "vmin": 0.3, "vmax": 0.4, "ticks": np.arange(0.30, 0.401, 0.025)},
    {"label": "c", "contrast": "0056", "rating": "pain_46_leg", "vmin": 0.3, "vmax": 0.4, "ticks": np.arange(0.30, 0.401, 0.025)},
)
CMAP = LinearSegmentedColormap.from_list("FigureS2_white_red_yellow",
                                         ((0.0, "#FFFFFF"), (0.52, fig2.RED), (1.0, fig2.YELLOW)))
GEOMETRY = dict(width=18.0, left=0.025, right=0.975, cbar_gap=0.10, cbar_slot=0.48, cbar_width=0.16, row_gap=0.20,
                label_height=0.30, top=0.20, bottom=0.16)


def sample_surface_nearest(img, threshold: float):
    """Nearest-voxel sampling at each surface depth (no dilation); the largest |r| across depths is kept."""
    from nilearn import surface
    mesh = brain.load_fsaverage("fsaverage7")
    volume = np.asarray(img.get_fdata(), dtype=float)
    inverse = np.linalg.inv(img.affine)

    def coordinates(path):
        loaded = surface.load_surf_mesh(path)
        return np.asarray(loaded.coordinates if hasattr(loaded, "coordinates") else loaded[0], dtype=float)

    def hemisphere(pial, white):
        samples = []
        for depth in fig2.SURFACE_DEPTHS:
            world = pial + float(depth) * (white - pial)
            voxel = world @ inverse[:3, :3].T + inverse[:3, 3]
            samples.append(map_coordinates(volume, voxel.T, order=0, mode="constant", cval=np.nan, prefilter=False))
        texture = brain.max_abs_keep_sign(np.asarray(samples))
        texture[~np.isfinite(texture) | (np.abs(texture) < threshold)] = np.nan
        return texture

    return (hemisphere(coordinates(mesh.pial_left), coordinates(mesh.white_left)),
            hemisphere(coordinates(mesh.pial_right), coordinates(mesh.white_right)))


def panel_images(project: Project, panel: dict) -> dict:
    import nibabel as nib
    folder = project.analysis_dir(panel["contrast"], panel["rating"])
    r_path, z_path = folder / "rho_z3p1.nii.gz", folder / "z_z3p1.nii.gz"
    for path in (r_path, z_path):
        if not path.is_file():
            raise MissingInputError(f"Missing map: {path}\nRun the Figure 2 mass-univariate analysis first.")
    r_img, z_img = load_3d(r_path, "r-map"), load_3d(z_path, "z-map")
    r, z = np.asarray(r_img.get_fdata(), float), np.asarray(z_img.get_fdata(), float)
    keep = np.isfinite(z) & (z > Z_THRESHOLD) & np.isfinite(r) & (r >= panel["vmin"]) & (r <= panel["vmax"])
    shown = nib.Nifti1Image(np.where(keep, r, 0.0).astype(np.float32), r_img.affine, r_img.header.copy())
    left, right = sample_surface_nearest(shown, panel["vmin"])
    surface_img = brain.crop_white_border(fig2.plot_surface_views(left, right, CMAP, panel["vmin"], panel["vmax"]), 0.987, 12)
    cerebellum_img = fig2.render_cerebellum(shown, panel["vmin"], panel["vmax"], cmap=CMAP, vmin=panel["vmin"])
    return {**panel, "surface": surface_img, "cerebellum": cerebellum_img}


def run(project: Project) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    project.require("figures2")
    apply_style()
    rows = [panel_images(project, p) for p in PANELS]
    g = GEOMETRY
    aspect = lambda img: img.shape[1] / img.shape[0]  # noqa: E731
    content_width = (g["right"] - g["left"]) * g["width"] - g["cbar_gap"] - g["cbar_slot"]
    row_height = content_width / (aspect(rows[0]["surface"]) + aspect(rows[0]["cerebellum"]))
    height = g["top"] + g["bottom"] + len(rows) * (g["label_height"] + row_height) + (len(rows) - 1) * g["row_gap"]
    fig = plt.figure(figsize=(g["width"], height), facecolor="white")
    y_top = height - g["top"]
    for row in rows:
        fig.text(g["left"], y_top / height, row["label"], ha="left", va="top", fontsize=21, fontweight="bold", color="#161616")
        bottom = y_top - g["label_height"] - row_height
        surface_width, cerebellum_width = row_height * aspect(row["surface"]), row_height * aspect(row["cerebellum"])
        cerebellum_left = g["left"] * g["width"] + surface_width
        for key, left, w in (("surface", g["left"] * g["width"], surface_width), ("cerebellum", cerebellum_left, cerebellum_width)):
            ax = fig.add_axes([left / g["width"], bottom / height, w / g["width"], row_height / height], frameon=False)
            ax.imshow(row[key])
            ax.set_axis_off()
        bar_height = 0.62 * row_height
        bar_left = cerebellum_left + cerebellum_width + g["cbar_gap"]
        bar_ax = fig.add_axes([bar_left / g["width"], (bottom + (row_height - bar_height) / 2) / height,
                               g["cbar_width"] / g["width"], bar_height / height])
        bar = mpl.colorbar.ColorbarBase(bar_ax, cmap=CMAP, norm=mpl.colors.Normalize(row["vmin"], row["vmax"]),
                                        orientation="vertical")
        bar.set_ticks(row["ticks"])
        bar.set_ticklabels([f"{t:.2f}" for t in row["ticks"]])
        bar.ax.tick_params(labelsize=10, width=0.8, length=2.5, pad=2)
        bar.set_label("Spearman ρ", fontsize=11, fontweight="bold", labelpad=3)
        bar.outline.set_linewidth(0.8)
        y_top = bottom - g["row_gap"]
    save_figure(fig, project.figure_files("FigureS2"))
    plt.close(fig)
