"""Compare a new results folder with an earlier run (e.g. the output of the original notebook)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLE_SUFFIXES = {".tsv": "\t", ".csv": ","}


def _compare_tables(old: Path, new: Path, rtol: float, atol: float) -> str:
    sep = TABLE_SUFFIXES[old.suffix]
    a, b = pd.read_csv(old, sep=sep), pd.read_csv(new, sep=sep)
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    worst = 0.0
    for column in a.columns:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
            x, y = a[column].to_numpy(float), b[column].to_numpy(float)
            if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                worst = max(worst, float(np.nanmax(np.abs(x - y))))
                return f"column '{column}' differs (max abs diff {worst:.3g})"
        elif not a[column].astype(str).equals(b[column].astype(str)):
            return f"column '{column}' differs"
    return "ok"


def _compare_images(old: Path, new: Path, rtol: float, atol: float) -> str:
    import nibabel as nib
    a, b = nib.load(str(old)), nib.load(str(new))
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-4):
        return "grid differs"
    x, y = np.asarray(a.get_fdata()), np.asarray(b.get_fdata())
    return "ok" if np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True) else f"values differ (max abs diff {np.nanmax(np.abs(x - y)):.3g})"


def compare_results(old_root, new_root, rtol: float = 1e-5, atol: float = 1e-8, images: bool = True) -> pd.DataFrame:
    """Compare every table (and, optionally, NIfTI map) that exists in both folders; prints a summary."""
    old_root, new_root = Path(old_root), Path(new_root)
    for root in (old_root, new_root):
        if not root.is_dir():
            raise FileNotFoundError(f"Results folder not found: {root}")
    rows = []
    for old in sorted(p for p in old_root.rglob("*") if p.is_file()):
        suffix = "".join(old.suffixes[-2:]) if old.name.endswith(".nii.gz") else old.suffix
        new = new_root / old.relative_to(old_root)
        if suffix not in (*TABLE_SUFFIXES, ".nii.gz", ".nii") or not new.exists():
            continue
        try:
            result = (_compare_tables(old, new, rtol, atol) if suffix in TABLE_SUFFIXES
                      else _compare_images(old, new, rtol, atol) if images else None)
        except Exception as exc:  # noqa: BLE001 - report any unreadable file instead of aborting the comparison
            result = f"could not compare: {exc}"
        if result is not None:
            rows.append({"file": str(old.relative_to(old_root)), "result": result})
    report = pd.DataFrame(rows)
    n_bad = int((report["result"] != "ok").sum()) if len(report) else 0
    print(f"Compared {len(report)} files: {len(report) - n_bad} identical within tolerance, {n_bad} different.")
    if n_bad:
        print(report[report["result"] != "ok"].to_string(index=False))
    return report
