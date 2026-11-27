"""Hoeppli et al. high-heat meta-analytic maps -> study template space (``cb_Template_T1``).

The FSL-MNI template that the maps are defined on is registered to the study template
(ANTsPy, deterministic ``antsRegistrationSyNRepro[s]``). The z maps are warped with linear
interpolation, re-thresholded at |z| >= 3.1, and the positive and negative maps are united and
restricted to the brain mask. The result (``hoeppli_union_mask.nii.gz``) restricts the LASSO-PCR
signature to a priori pain-responsive voxels.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from .config import HOEPPLI_REGISTRATION, HOEPPLI_SEED, HOEPPLI_Z_THRESHOLD, Project
from .imaging import load_3d, same_grid
from .utils import DataError, MissingInputError


def _libraries():
    try:
        import ants
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("The Hoeppli step needs ANTsPy and nibabel: pip install antspyx nibabel") from exc
    return ants, nib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_on_grid(data, target, path: Path, dtype) -> None:
    """Write ``data`` with the exact geometry (affine, q/s-forms) of ``target``."""
    _, nib = _libraries()
    header = target.header.copy()
    header.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), target.affine, header)
    img.set_qform(target.affine, int(target.header["qform_code"]) or 1)
    img.set_sform(target.affine, int(target.header["sform_code"]) or 1)
    img.to_filename(str(path))


def save_ants_on_grid(image, target, path: Path) -> None:
    ants, nib = _libraries()
    tmp = path.with_name(path.name + ".tmp.nii.gz")
    ants.image_write(image, str(tmp))
    data = nib.load(str(tmp)).get_fdata(dtype=np.float32)
    tmp.unlink()
    save_on_grid(data, target, path, np.float32)


def threshold_rule(source_data: np.ndarray, z: float):
    """Which tail(s) to keep after warping, decided from the sign content of the *source* map."""
    has_pos, has_neg = np.any(source_data > 1e-6), np.any(source_data < -1e-6)
    if has_pos and not has_neg:
        return (lambda w: w >= z), f"z >= {z}"
    if has_neg and not has_pos:
        return (lambda w: w <= -z), f"z <= {-z}"
    return (lambda w: np.abs(w) >= z), f"|z| >= {z}"


def registration_mask(project: Project, target_img, out_path: Path):
    """Target-space brain mask (provided one if available, otherwise estimated from the template)."""
    ants, _ = _libraries()
    if project.target_mask.exists():
        mask = load_3d(project.target_mask, "target brain mask")
        if not same_grid(mask, target_img):
            from nibabel.processing import resample_from_to
            mask = resample_from_to(mask, (target_img.shape, target_img.affine), order=0)
        save_on_grid((mask.get_fdata() > 0).astype(np.uint8), target_img, out_path, np.uint8)
    else:
        print(f"Target brain mask not found ({project.target_mask}); estimating it from the template.")
        ants.image_write(ants.get_mask(ants.image_read(str(project.target_t1)), cleanup=2), str(out_path))
    return out_path


def register(project: Project, fixed, moving, fixed_mask_path: Path, force: bool = False):
    """Run (or reuse) the registration. Returns (forward transforms, inverse transforms, id, reused)."""
    ants, _ = _libraries()
    moving_mask = ants.get_mask(moving, cleanup=2)
    ants.image_write(moving_mask, str(project.shared / "bg_image_registration_mask.nii.gz"))
    spec = {"source_sha256": sha256(project.hoeppli_t1), "target_sha256": sha256(project.target_t1),
            "target_mask_sha256": sha256(fixed_mask_path), "registration_type": HOEPPLI_REGISTRATION,
            "random_seed": HOEPPLI_SEED, "antspy_version": getattr(ants, "__version__", "unknown")}
    registration_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    manifest_path = project.shared / "transforms" / "registration_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        paths = manifest.get("forward_transforms", []) + manifest.get("inverse_transforms", [])
        if manifest.get("registration_id") == registration_id and all(Path(p).exists() for p in paths):
            print(f"Reusing registration {registration_id}")
            return manifest["forward_transforms"], manifest["inverse_transforms"], registration_id, True
    print(f"Running {HOEPPLI_REGISTRATION} (id {registration_id}); this takes a while ...")
    kwargs = dict(fixed=fixed, moving=moving, type_of_transform=HOEPPLI_REGISTRATION,
                  outprefix=str(project.shared / "transforms" / "fslMNI_to_cbTemplate_"), mask=ants.image_read(str(fixed_mask_path)).clone("float"),
                  moving_mask=moving_mask, mask_all_stages=True, random_seed=HOEPPLI_SEED,
                  write_composite_transform=False, verbose=True)
    if "singleprecision" in inspect.signature(ants.registration).parameters:
        kwargs["singleprecision"] = False
    result = ants.registration(**kwargs)
    forward, inverse = [str(p) for p in result["fwdtransforms"]], [str(p) for p in result["invtransforms"]]
    manifest_path.write_text(json.dumps({"registration_id": registration_id, "specification": spec,
                                         "forward_transforms": forward, "inverse_transforms": inverse}, indent=2))
    return forward, inverse, registration_id, False


def warp_z_map(project: Project, source_path: Path, name: str, fixed, forward, target_img, z: float) -> dict:
    """Linear warp, then re-threshold (the thresholded map and its binary mask are written)."""
    ants, nib = _libraries()
    warped = ants.apply_transforms(fixed=fixed, moving=ants.image_read(str(source_path)).clone("float"),
                                   transformlist=forward, interpolator="linear", imagetype=0, defaultvalue=0,
                                   singleprecision=False)
    out = project.shared
    linear_path = out / f"{name}_in_cbTemplate_linear.nii.gz"
    save_ants_on_grid(warped, target_img, linear_path)
    source = nib.load(str(source_path)).get_fdata(dtype=np.float32)
    data = nib.load(str(linear_path)).get_fdata(dtype=np.float32)
    data[~np.isfinite(data)] = 0
    keep_fn, rule = threshold_rule(source, z)
    keep = keep_fn(data)
    thresholded, binary = (out / f"{name}_in_cbTemplate_rethresholded_z3p1{s}.nii.gz" for s in ("", "_mask"))
    save_on_grid(np.where(keep, data, 0), target_img, thresholded, np.float32)
    save_on_grid(keep.astype(np.uint8), target_img, binary, np.uint8)
    print(f"{name}: {rule}; {int(np.count_nonzero(source)):,} source voxels -> {int(keep.sum()):,} after warping")
    return {"source": str(source_path), "linear_warp": str(linear_path), "rethresholded_map": str(thresholded),
            "binary_mask": str(binary), "threshold_rule": rule, "source_nonzero": int(np.count_nonzero(source)),
            "rethresholded_nonzero": int(keep.sum())}


def combine_masks(positive: Path, negative: Path, anatomical: Path, output: Path, metadata: Path, z: float) -> dict:
    """(positive OR negative) AND brain mask; no further interpolation."""
    imgs = [load_3d(p, label) for p, label in ((positive, "positive mask"), (negative, "negative mask"), (anatomical, "brain mask"))]
    if not all(same_grid(imgs[0], other) for other in imgs[1:]):
        raise DataError("Hoeppli masks and the brain mask are not on the same grid.")
    pos, neg, brain = (np.isfinite(i.get_fdata()) & (i.get_fdata() > 0) for i in imgs)
    union = pos | neg
    combined = union & brain
    if not combined.any():
        raise DataError("The combined Hoeppli mask is empty.")
    save_on_grid(combined.astype(np.uint8), imgs[2], output, np.uint8)
    summary = {"operation": "(positive > 0 OR negative > 0) AND anatomical > 0", "postwarp_threshold": z,
               "n_positive_voxels": int(pos.sum()), "n_negative_voxels": int(neg.sum()),
               "n_positive_negative_overlap": int((pos & neg).sum()),
               "n_union_voxels_outside_anatomical_mask": int((union & ~brain).sum()), "n_combined_voxels": int(combined.sum()),
               "combined_mask": str(output)}
    metadata.write_text(json.dumps(summary, indent=2))
    print(f"Combined Hoeppli mask: {summary['n_combined_voxels']:,} voxels "
          f"(pos {summary['n_positive_voxels']:,}, neg {summary['n_negative_voxels']:,}, "
          f"overlap {summary['n_positive_negative_overlap']:,}) -> {output}")
    return summary


def qc_plots(project: Project, warped_source: Path, results: dict) -> None:
    """Edge overlay of the warped source template and the warped maps (quality control only)."""
    from nilearn import plotting
    qc = project.qc / "hoeppli"
    display = plotting.plot_anat(str(project.target_t1), display_mode="ortho", cut_coords=(0, -20, 10),
                                 title="cb_Template_T1 with warped FSL-MNI template edges")
    display.add_edges(str(warped_source))
    display.savefig(str(qc / "target_with_warped_source_edges.png"), dpi=180)
    display.close()
    for key, title in (("positive", "Positive high-stimulation map"), ("negative", "Negative high-stimulation map")):
        path = results[key]["rethresholded_map"]
        display = plotting.plot_stat_map(path, bg_img=str(project.target_t1), display_mode="ortho", cut_coords=(0, -20, 10),
                                         threshold=1e-8, title=title)
        display.savefig(str(qc / (Path(path).name.replace(".nii.gz", ".png"))), dpi=180)
        display.close()


def build_mask(project: Project, force: bool = False, n_threads: int = 8) -> Path:
    """Create ``project.hoeppli_mask`` (skipped if it already exists and ``force`` is False)."""
    if project.hoeppli_mask.exists() and not force:
        print(f"Hoeppli mask already exists: {project.hoeppli_mask} (use force=True to rebuild)")
        return project.hoeppli_mask
    project.require("hoeppli")
    import os
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(n_threads)
    ants, _ = _libraries()
    project.make_output_dirs()
    target_img = load_3d(project.target_t1, "target template")
    for path, label in ((project.hoeppli_pos, "positive map"), (project.hoeppli_neg, "negative map")):
        if not same_grid(load_3d(path, f"Hoeppli {label}"), load_3d(project.hoeppli_t1, "Hoeppli anatomy")):
            print(f"WARNING: Hoeppli {label} is not on the exact grid of the Hoeppli anatomy image.")
    fixed_mask_path = registration_mask(project, target_img, project.shared / "cb_Template_T1_registration_mask.nii.gz")
    fixed = ants.image_read(str(project.target_t1)).clone("float")
    moving = ants.image_read(str(project.hoeppli_t1)).clone("float")
    forward, inverse, registration_id, reused = register(project, fixed, moving, fixed_mask_path, force)
    warped_source = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=forward, interpolator="linear",
                                          imagetype=0, defaultvalue=0, singleprecision=False)
    warped_source_path = project.shared / "bg_image_in_cb_Template_T1_space.nii.gz"
    save_ants_on_grid(warped_source, target_img, warped_source_path)
    results = {key: warp_z_map(project, path, name, fixed, forward, target_img, HOEPPLI_Z_THRESHOLD)
               for key, path, name in (("positive", project.hoeppli_pos, "heat_highStim_pos"),
                                       ("negative", project.hoeppli_neg, "heat_highStim_neg"))}
    summary = combine_masks(Path(results["positive"]["binary_mask"]), Path(results["negative"]["binary_mask"]),
                            fixed_mask_path, project.hoeppli_mask, project.hoeppli_mask.with_suffix("").with_suffix(".json"),
                            HOEPPLI_Z_THRESHOLD)
    try:
        qc_plots(project, warped_source_path, results)
    except Exception as exc:  # noqa: BLE001 - QC images must never block the analysis
        print(f"Note: QC plots skipped ({exc!r})")
    (project.shared / "warping_summary.json").write_text(json.dumps(
        {"registration_id": registration_id, "registration_reused": reused, "registration_type": HOEPPLI_REGISTRATION,
         "forward_transforms": forward, "inverse_transforms": inverse, "map_interpolation": "linear",
         "postwarp_threshold": HOEPPLI_Z_THRESHOLD, **results, "combined_hoeppli_mask": summary}, indent=2))
    return project.hoeppli_mask


__all__ = ["build_mask", "MissingInputError"]
