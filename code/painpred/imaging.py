"""NIfTI input/output helpers. nibabel/nilearn are imported lazily so that the numeric
modules can be used (and tested) without them."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

from .utils import DataError, MissingInputError


def _nib():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("nibabel is required for image I/O: pip install nibabel nilearn") from exc
    return nib


def load_3d(path: Path, label: str = "image"):
    """Load a 3-D (or singleton-4-D) NIfTI with clear errors."""
    nib = _nib()
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(f"{label} not found: {path}")
    try:
        img = nib.load(str(path))
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"{label} is not a readable NIfTI file ({path}): {exc}") from exc
    if img.ndim == 4 and img.shape[3] == 1:
        img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine, img.header)
    elif img.ndim != 3:
        raise DataError(f"{label} must be 3-D, got shape {img.shape}: {path}")
    return img


def same_grid(a, b) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-4)


def resample_to(source, target, interpolation: str = "nearest"):
    """Resample onto the target grid (no-op if the grids already match)."""
    if same_grid(source, target):
        return source
    from nilearn import image
    return image.resample_to_img(source, target, interpolation=interpolation,
                                 force_resample=True, copy_header=True)


def voxel_volume_mm3(img) -> float:
    return float(np.prod(img.header.get_zooms()[:3]))


def ascii_description(text: str, max_bytes: int = 79) -> bytes:
    """NIfTI 'descrip' is fixed-width ASCII."""
    for old, new in {"°": " deg", "×": "x", "–": "-", "—": "-", "≥": ">=", "≤": "<=", "ρ": "rho"}.items():
        text = text.replace(old, new)
    return unicodedata.normalize("NFKD", text).encode("ascii", errors="replace")[:max_bytes]


def save_volume(data: np.ndarray, reference, path: Path, description: str = "", dtype=np.float32) -> None:
    nib = _nib()
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    if description:
        header["descrip"] = ascii_description(description)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(np.asarray(data).astype(dtype), reference.affine, header).to_filename(str(path))


class MaskGeometry:
    """Maps between voxel vectors (masked) and 3-D volumes (C-order, as NiftiMasker does)."""

    def __init__(self, mask_img):
        mask = np.asarray(mask_img.get_fdata()) > 0
        self.img = mask_img
        self.shape = mask.shape
        self.flat_indices = np.flatnonzero(mask.ravel(order="C"))
        self.affine = np.asarray(mask_img.affine, dtype=float)
        self.voxel_volume_mm3 = voxel_volume_mm3(mask_img)

    @property
    def n_voxels(self) -> int:
        return len(self.flat_indices)

    def to_volume(self, vector, dtype=float) -> np.ndarray:
        flat = np.zeros(int(np.prod(self.shape)), dtype=dtype)
        flat[self.flat_indices] = np.asarray(vector, dtype=dtype)
        return flat.reshape(self.shape, order="C")

    def to_vector(self, volume) -> np.ndarray:
        return np.asarray(volume).ravel(order="C")[self.flat_indices]

    def save(self, vector, path: Path, dtype=np.float32) -> None:
        save_volume(self.to_volume(vector, dtype), self.img, path, dtype=dtype)

    def load_vector(self, path: Path, label: str = "image") -> np.ndarray:
        img = load_3d(path, label)
        if img.shape[:3] != self.shape:
            raise DataError(f"{label} has shape {img.shape[:3]} but the group mask is {self.shape}: {path}")
        return self.to_vector(np.asarray(img.get_fdata(), dtype=np.float32))

    def load_matrix(self, paths, label: str = "contrast image") -> np.ndarray:
        """Stack images into (n_images, n_voxels) float32, failing on NaN/Inf inside the mask."""
        out = np.empty((len(paths), self.n_voxels), dtype=np.float32)
        for i, path in enumerate(paths):
            out[i] = self.load_vector(path, f"{label} {i + 1}/{len(paths)}")
        if not np.isfinite(out).all():
            bad = np.flatnonzero(~np.isfinite(out).all(axis=1))
            raise DataError(f"{label}s contain NaN/Inf inside the group mask: {[str(paths[i]) for i in bad[:5]]}")
        return out


def make_group_mask(mean_epi_path: Path, out_path: Path, fraction: float, percentile: float):
    """Brain mask: mean EPI above fraction*percentile, holes filled, largest component."""
    from scipy.ndimage import binary_fill_holes, label
    img = load_3d(mean_epi_path, "mean EPI")
    data = np.asarray(img.get_fdata(), dtype=float)
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        raise DataError(f"The mean EPI contains no positive finite values: {mean_epi_path}")
    threshold = fraction * np.percentile(positive, percentile)
    labelled, n = label(binary_fill_holes(data > threshold))
    if n == 0:
        raise DataError("No connected brain component found; check the mean EPI image.")
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    mask = labelled == int(np.argmax(sizes))
    save_volume(mask, img, out_path, dtype=np.uint8)
    print(f"Group mask: {int(mask.sum()):,} voxels, threshold {threshold:.4f} -> {out_path}")
    return load_3d(out_path, "group mask")
