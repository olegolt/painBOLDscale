"""Local peaks of the voxel-FWE-significant correlation map (Figure 2 table)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import atlas
from .config import Project
from .imaging import load_3d
from .utils import DataError

MIN_PEAK_DISTANCE_MM = 8.0
LOCAL_EXTREMA_NEIGHBOURHOOD = 3
ATLAS_FALLBACK_RADIUS_MM = 5.0
STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)


def plateau_representatives(candidates: np.ndarray, magnitude: np.ndarray, z: np.ndarray):
    """One voxel per connected plateau of equal-valued candidates (largest |z| breaks ties)."""
    labelled, n = ndimage.label(candidates, structure=STRUCTURE_26)
    out = []
    for plateau in range(1, n + 1):
        voxels = np.argwhere(labelled == plateau)
        if len(voxels) > 1:
            values = magnitude[tuple(voxels.T)]
            tied = voxels[np.isclose(values, np.nanmax(values), rtol=0, atol=1e-12)]
            voxels = tied[[int(np.nanargmax(np.abs(z[tuple(tied.T)])))]] if len(tied) > 1 else tied
        out.append(tuple(int(v) for v in voxels[0]))
    return out


def select_separated(candidates, effect, z, affine, min_distance_mm: float):
    """Greedy selection of the strongest candidates that are at least ``min_distance_mm`` apart."""
    ordered = sorted(candidates, key=lambda ijk: (-abs(float(effect[ijk])), -abs(float(z[ijk])), ijk))
    chosen, chosen_xyz = [], []
    for ijk in ordered:
        xyz = atlas.apply_affine(affine, np.asarray(ijk, dtype=float))
        if chosen_xyz and min(np.linalg.norm(xyz - p) for p in chosen_xyz) < min_distance_mm:
            continue
        chosen.append(ijk)
        chosen_xyz.append(xyz)
    return chosen


def cluster_peaks(cluster: np.ndarray, rho: np.ndarray, z: np.ndarray, affine, sign: str):
    """Local maxima (positive) or minima (negative) of rho within one cluster."""
    magnitude = np.where(cluster, rho if sign == "positive" else -rho, -np.inf)
    local_max = ndimage.maximum_filter(magnitude, size=LOCAL_EXTREMA_NEIGHBOURHOOD, mode="constant", cval=-np.inf)
    candidates = cluster & np.isfinite(magnitude) & np.isclose(magnitude, local_max, rtol=0, atol=1e-12)
    return select_separated(plateau_representatives(candidates, magnitude, z), rho, z, affine, MIN_PEAK_DISTANCE_MM)


def local_peaks(rho: np.ndarray, z: np.ndarray, p: np.ndarray, affine, atlas_data, atlas_affine, labels) -> pd.DataFrame:
    voxel_volume = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))
    rows = []
    for sign, significant in (("positive", np.isfinite(rho) & (rho > 0)), ("negative", np.isfinite(rho) & (rho < 0))):
        clusters, n = ndimage.label(significant, structure=STRUCTURE_26)
        sizes = np.bincount(clusters.ravel())
        order = sorted(range(1, n + 1), key=lambda c: (-int(sizes[c]), c))
        counter = 0
        for rank, cluster_id in enumerate(order, start=1):
            for k, ijk in enumerate(cluster_peaks(clusters == cluster_id, rho, z, affine, sign), start=1):
                counter += 1
                xyz = atlas.apply_affine(affine, np.asarray(ijk, dtype=float))
                index, region, distance, method = atlas.label_at(xyz, atlas_data, atlas_affine, labels, ATLAS_FALLBACK_RADIUS_MM)
                rows.append({"effect_direction": sign, "peak_rank_within_direction": counter,
                             "cluster_rank_within_direction": rank, "cluster_id": int(cluster_id),
                             "peak_rank_within_cluster": k, "cluster_extent_voxels": int(sizes[cluster_id]),
                             "cluster_extent_mm3": int(sizes[cluster_id]) * voxel_volume,
                             "mni_x_mm": float(xyz[0]), "mni_y_mm": float(xyz[1]), "mni_z_mm": float(xyz[2]),
                             "spearman_rho": float(rho[ijk]), "z_value": float(z[ijk]), "p_fwe_voxelwise": float(p[ijk]),
                             "atlas_label_index": index, "neuromorphometrics_region": region,
                             "atlas_assignment_method": method, "distance_to_atlas_label_mm": distance,
                             "minimum_peak_distance_mm": MIN_PEAK_DISTANCE_MM})
    if not rows:
        raise DataError("The voxel-FWE r-map contains no positive or negative voxels, so there are no peaks to report.")
    table = pd.DataFrame(rows)
    table["_order"] = pd.Categorical(table["effect_direction"], categories=["positive", "negative"], ordered=True)
    table = table.sort_values(["_order", "cluster_rank_within_direction", "peak_rank_within_cluster"]).drop(columns="_order")
    decimals = {"cluster_extent_mm3": 1, "mni_x_mm": 1, "mni_y_mm": 1, "mni_z_mm": 1, "spearman_rho": 4, "z_value": 4,
                "p_fwe_voxelwise": 6, "distance_to_atlas_label_mm": 2, "minimum_peak_distance_mm": 1}
    return table.round(decimals).reset_index(drop=True)


def run(project: Project) -> pd.DataFrame:
    """Peak table of the primary (con0053 x 48 °C arm rating) map -> ``figure2/local_peaks.csv``."""
    project.require("peaks")
    rho_img = load_3d(project.primary_dir / "rho_fwe.nii.gz", "FWE r-map")
    others = {name: load_3d(project.primary_dir / f"{name}.nii.gz", name) for name in ("z_fwe", "p_fwe")}
    for name, img in others.items():
        if img.shape != rho_img.shape or not np.allclose(img.affine, rho_img.affine, atol=1e-4):
            raise DataError(f"{name} map is not on the same grid as the r-map.")
    atlas_img = load_3d(project.atlas_nii, "atlas")
    table = local_peaks(np.asarray(rho_img.get_fdata(), float), np.asarray(others["z_fwe"].get_fdata(), float),
                        np.asarray(others["p_fwe"].get_fdata(), float), rho_img.affine,
                        np.rint(np.asarray(atlas_img.get_fdata(), float)).astype(np.int32), atlas_img.affine,
                        atlas.read_labels(project.atlas_xml))
    table.to_csv(project.f2 / "local_peaks.csv", index=False)
    print(f"{len(table)} local peaks -> {project.f2 / 'local_peaks.csv'}")
    return table
