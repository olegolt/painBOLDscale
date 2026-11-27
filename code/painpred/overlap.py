"""Overlap analyses: Figure 2 (significant map vs reference maps) and Figure 3 (agreement across ratings)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import atlas
from .config import MASSUNI_RATINGS, PRIMARY, Project
from .imaging import load_3d, resample_to, same_grid, save_volume, voxel_volume_mm3
from .utils import DataError, MissingInputError

TARGET_COUNT = 4  # voxels significant in all four rating analyses


# --------------------------------------------------------------------------- #
# Figure 2: overlap with reference maps
# --------------------------------------------------------------------------- #
def overlap_metrics(target: np.ndarray, reference: np.ndarray, search: np.ndarray, voxel_mm3: float) -> dict:
    """Voxel-count overlap statistics inside the search mask (descriptive)."""
    target, reference = target & search, reference & search
    n_t, n_r, n_s = int(target.sum()), int(reference.sum()), int(search.sum())
    n_o, n_u = int((target & reference).sum()), int((target | reference).sum())
    expected = n_t * n_r / n_s if n_s else np.nan
    return {"voxel_volume_mm3": voxel_mm3, "search_mask_voxels": n_s, "search_mask_volume_mm3": n_s * voxel_mm3,
            "target_voxels": n_t, "target_volume_mm3": n_t * voxel_mm3, "reference_voxels": n_r,
            "reference_volume_mm3": n_r * voxel_mm3, "overlap_voxels": n_o, "overlap_volume_mm3": n_o * voxel_mm3,
            "target_overlap_percent": 100.0 * n_o / n_t if n_t else np.nan,
            "reference_overlap_percent": 100.0 * n_o / n_r if n_r else np.nan,
            "dice": 2.0 * n_o / (n_t + n_r) if (n_t + n_r) else np.nan, "jaccard": n_o / n_u if n_u else np.nan,
            "reference_search_coverage_percent": 100.0 * (n_r / n_s if n_s else np.nan),
            "expected_overlap_voxels_spatial_independence": expected,
            "descriptive_fold_enrichment": n_o / expected if np.isfinite(expected) and expected > 0 else np.nan}


def reference_maps(project: Project):
    """(key, label, path, independent, interpretation) of the reference maps."""
    return [
        ("same_study_48_voxelFWE05_positive", "Same-study 48 °C activation (voxelwise FWE P < 0.05)",
         project.same_study_48_mask, False,
         "Mean positive activation during 48 °C stimulation in the same sample. This comparison is descriptive rather than independent."),
        ("hoeppli_high_heat_union", "Hoeppli high-heat stimulus-responsive map", project.hoeppli_mask, True,
         "Independent high-heat stimulus-responsive map. Because the provided mask is a union of positive and negative "
         "responses, overlap indicates stimulus responsiveness but not response direction."),
    ]


def run_reference_overlap(project: Project) -> pd.DataFrame:
    """Overlap of the FWE-significant primary map with the reference maps -> ``figure2/overlap``."""
    project.require("overlap")
    out = project.f2_overlap
    (out / "resampled_reference_masks").mkdir(parents=True, exist_ok=True)
    (out / "overlap_masks").mkdir(parents=True, exist_ok=True)
    target_img = load_3d(project.primary_dir / "rho_fwe.nii.gz", "FWE r-map")
    data = np.asarray(target_img.get_fdata(), dtype=np.float64)
    voxel_mm3 = voxel_volume_mm3(target_img)
    search = np.asarray(resample_to(load_3d(project.group_mask, "group mask"), target_img).get_fdata(), float) > 0.5
    targets = {"positive_fwe_correlations": np.isfinite(data) & (data > 0) & search,
               "negative_fwe_correlations": np.isfinite(data) & (data < 0) & search}
    targets["all_fwe_correlations"] = targets["positive_fwe_correlations"] | targets["negative_fwe_correlations"]
    if not targets["all_fwe_correlations"].any():
        raise DataError("The FWE r-map contains no non-zero voxels.")
    for name in ("positive", "negative"):
        save_volume(targets[f"{name}_fwe_correlations"], target_img, out / f"target_{name}_fwe_mask.nii.gz",
                    f"{name.capitalize()} FWE-significant con0053 x pain_48_arm voxels", np.uint8)
    grid_rows, rows = [], []
    for key, label, path, independent, interpretation in reference_maps(project):
        if not path.exists():
            print(f"Skipping missing reference map: {path}")
            continue
        source = load_3d(path, label)
        grid_rows.append({"reference_key": key, "path": str(path), "same_grid_before_resampling": same_grid(source, target_img),
                          "original_shape": "x".join(map(str, source.shape[:3])), "target_shape": "x".join(map(str, target_img.shape[:3]))})
        values = np.asarray(resample_to(source, target_img).get_fdata(), dtype=np.float64)
        reference = np.isfinite(values) & (values != 0) & search
        save_volume(reference, target_img, out / "resampled_reference_masks" / f"{key}.nii.gz", label, np.uint8)
        for target_name, target in targets.items():
            save_volume(target & reference, target_img, out / "overlap_masks" / f"{target_name}_AND_{key}.nii.gz",
                        f"{target_name} overlap with {key}", np.uint8)
            rows.append({"target": target_name, "reference_key": key, "reference_label": label, "reference_path": str(path),
                         "reference_threshold_mode": "nonzero", "independent_reference": independent,
                         "interpretation": interpretation, **overlap_metrics(target, reference, search, voxel_mm3)})
    if not rows:
        raise MissingInputError("None of the reference maps exists: " + ", ".join(str(r[2]) for r in reference_maps(project)))
    pd.DataFrame(grid_rows).to_csv(out / "grid_qc.tsv", sep="\t", index=False)
    results = pd.DataFrame(rows)
    results.to_csv(out / "overlap_summary.tsv", sep="\t", index=False)
    primary = results[results["target"] == "positive_fwe_correlations"].sort_values("target_overlap_percent", ascending=False)
    primary.to_csv(out / "overlap_summary_positive_target.tsv", sep="\t", index=False)
    print(primary[["reference_label", "target_voxels", "reference_voxels", "overlap_voxels", "target_overlap_percent",
                   "dice", "descriptive_fold_enrichment"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Note: these overlap statistics are descriptive.")
    return results


# --------------------------------------------------------------------------- #
# Figure 3: overlap of the four rating analyses
# --------------------------------------------------------------------------- #
def count_overlap(project: Project) -> np.ndarray:
    """Per voxel: in how many of the four con0053 x rating analyses is the voxelwise-FWE map non-zero?"""
    out = project.f3
    out.mkdir(parents=True, exist_ok=True)
    reference, masks, records = None, [], []
    for rating in MASSUNI_RATINGS:
        path = project.analysis_dir(PRIMARY.contrast, rating) / "rho_fwe.nii.gz"
        img = load_3d(path, f"FWE map for {rating}")
        if reference is None:
            reference = img
        elif not same_grid(img, reference):
            raise DataError(f"The FWE maps do not share one voxel grid ({rating}: {img.shape} vs {reference.shape}); "
                            "significance maps are never resampled silently for overlap counts.")
        significant = np.isfinite(d := np.asarray(img.get_fdata(), dtype=np.float64)) & (np.abs(d) > 0)
        masks.append(significant)
        binary = out / f"binary_significant_con{PRIMARY.contrast}_{rating}.nii.gz"
        save_volume(significant, reference, binary, dtype=np.uint8)
        records.append({"contrast": f"con{PRIMARY.contrast}", "rating": rating, "significance_map": str(path),
                        "binary_map": str(binary), "n_significant_voxels": int(significant.sum())})
        print(f"{rating}: {int(significant.sum()):,} significant voxels")
    count = np.sum(np.stack(masks), axis=0, dtype=np.uint8)
    save_volume(count, reference, out / "overlap_count.nii.gz", dtype=np.uint8)
    save_volume(count >= 1, reference, out / "overlap_any.nii.gz", dtype=np.uint8)
    save_volume(count == len(MASSUNI_RATINGS), reference, out / "overlap_all4.nii.gz", dtype=np.uint8)
    union = int((count >= 1).sum())
    summary = pd.DataFrame({"number_of_significant_analyses": range(len(MASSUNI_RATINGS) + 1),
                            "n_voxels": [int((count == k).sum()) for k in range(len(MASSUNI_RATINGS) + 1)]})
    summary["percentage_of_significant_union"] = np.where(
        (summary["number_of_significant_analyses"] > 0) & (union > 0), 100.0 * summary["n_voxels"] / max(union, 1), np.nan)
    summary.to_csv(out / "overlap_summary.tsv", sep="\t", index=False)
    pd.DataFrame(records).to_csv(out / "overlap_inputs.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    return count


def count4_regions(project: Project) -> pd.DataFrame:
    """Neuromorphometrics regions containing voxels that are significant in all four analyses."""
    from nibabel.processing import resample_from_to
    project.require("atlas")
    count_img = load_3d(project.f3 / "overlap_count.nii.gz", "overlap count map")
    count = np.rint(np.asarray(count_img.get_fdata(), dtype=np.float64))
    mask = np.isfinite(count) & (count == TARGET_COUNT)
    total = int(mask.sum())
    if total == 0:
        raise DataError(f"No voxel is significant in {TARGET_COUNT} analyses; nothing to report.")
    voxel_mm3 = voxel_volume_mm3(count_img)
    atlas_img = load_3d(project.atlas_nii, "atlas")
    if not same_grid(atlas_img, count_img):
        atlas_img = resample_from_to(atlas_img, (count_img.shape[:3], count_img.affine), order=0, mode="constant", cval=0)
    atlas_data = np.rint(np.asarray(atlas_img.get_fdata(), dtype=np.float64)).astype(np.int32)
    labels = atlas.read_labels(project.atlas_xml)
    out = project.f3
    save_volume(atlas_data, count_img, out / "atlas_in_count_space.nii.gz", "Neuromorphometrics atlas in overlap-count map space", np.int16)
    save_volume(mask, count_img, out / "count4_mask.nii.gz", "Voxels significant for all four rating associations", np.uint8)
    save_volume(np.where(mask, atlas_data, 0), count_img, out / "count4_atlas_labels.nii.gz",
                "Neuromorphometrics labels restricted to count-4 voxels", np.int16)
    rows = []
    for index in np.unique(atlas_data[mask]).astype(int):
        region = mask & (atlas_data == index)
        voxels = np.argwhere(region)
        xyz = atlas.apply_affine(count_img.affine, voxels.astype(float))
        centroid = xyz.mean(axis=0)
        representative = xyz[int(np.argmin(np.linalg.norm(xyz - centroid[None, :], axis=1)))]
        label = "Unlabelled" if index == 0 else labels.get(index, f"Atlas label {index}")
        hemisphere, abbreviation, name = atlas.split_label(label)
        parcel = int((atlas_data == index).sum())
        n = int(region.sum())
        rows.append({"atlas_label_index": index, "neuromorphometrics_label": label, "hemisphere": hemisphere,
                     "atlas_abbreviation": abbreviation, "region_name": name, "count4_voxels": n,
                     "count4_volume_mm3": n * voxel_mm3, "percent_of_all_count4_voxels": 100.0 * n / total,
                     "atlas_parcel_voxels_in_count_grid": parcel,
                     "percent_of_atlas_parcel_with_count4": 100.0 * n / parcel if parcel else np.nan,
                     "centroid_mni_x_mm": centroid[0], "centroid_mni_y_mm": centroid[1], "centroid_mni_z_mm": centroid[2],
                     "representative_mni_x_mm": representative[0], "representative_mni_y_mm": representative[1],
                     "representative_mni_z_mm": representative[2], "target_overlap_count": TARGET_COUNT,
                     "target_total_voxels": total, "target_total_volume_mm3": total * voxel_mm3, "voxel_volume_mm3": voxel_mm3})
    table = pd.DataFrame(rows).sort_values(["count4_voxels", "hemisphere", "region_name"], ascending=[False, True, True]).reset_index(drop=True)
    table.insert(0, "region_rank", np.arange(1, len(table) + 1))
    table = table.round({"count4_volume_mm3": 1, "percent_of_all_count4_voxels": 3, "percent_of_atlas_parcel_with_count4": 3,
                         "centroid_mni_x_mm": 1, "centroid_mni_y_mm": 1, "centroid_mni_z_mm": 1,
                         "representative_mni_x_mm": 1, "representative_mni_y_mm": 1, "representative_mni_z_mm": 1,
                         "target_total_volume_mm3": 1, "voxel_volume_mm3": 4})
    table.to_csv(out / "count4_regions.csv", index=False)
    table.to_csv(out / "count4_regions.tsv", sep="\t", index=False)
    print(f"{total:,} count-4 voxels in {len(table)} atlas regions")
    print(table[["region_rank", "neuromorphometrics_label", "count4_voxels", "percent_of_all_count4_voxels"]].to_string(index=False))
    return table
