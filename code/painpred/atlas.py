"""Neuromorphometrics atlas helpers (label table, hemisphere split, coordinate lookup)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .utils import DataError, MissingInputError


def read_labels(xml_path: Path) -> dict[int, str]:
    """{atlas index: label} from the atlas XML."""
    if not Path(xml_path).is_file():
        raise MissingInputError(f"Atlas label file not found: {xml_path}")
    try:
        root = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise DataError(f"Atlas label file is not valid XML ({xml_path}): {exc}") from exc
    labels = {}
    for element in root.findall(".//label"):
        index, name = element.findtext("index"), element.findtext("name")
        if index is not None and name is not None:
            labels[int(index)] = " ".join(name.strip().split())
    if not labels:
        raise DataError(f"No <label> entries found in {xml_path}")
    return labels


def split_label(label: str) -> tuple[str, str, str]:
    """'Right MSFG superior frontal gyrus medial segment' -> ('Right', 'MSFG', 'superior frontal ...')."""
    cleaned = " ".join(str(label).strip().split())
    if match := re.match(r"^(Right|Left)\s+([A-Za-z0-9]+)\s+(.+)$", cleaned):
        return match.group(1), match.group(2), match.group(3)
    if match := re.match(r"^(Right|Left)\s+(.+)$", cleaned):
        return match.group(1), "", match.group(2)
    return "Midline/other", "", cleaned


def apply_affine(affine, points):
    points = np.asarray(points, dtype=float)
    return points @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]


def label_at(xyz, atlas_data, atlas_affine, labels, fallback_radius_mm: float):
    """Atlas (index, label, distance_mm, method) at a world coordinate.

    The exact voxel is used if labelled; otherwise the nearest labelled voxel within the fallback radius.
    """
    xyz = np.asarray(xyz, dtype=float)
    ijk = np.rint(apply_affine(np.linalg.inv(atlas_affine), xyz)).astype(int)
    shape = np.asarray(atlas_data.shape[:3])
    if np.all(ijk >= 0) and np.all(ijk < shape):
        exact = int(atlas_data[tuple(ijk)])
        if exact != 0:
            return exact, labels.get(exact, f"Atlas label {exact}"), 0.0, "exact"
    voxel_sizes = np.sqrt((np.asarray(atlas_affine)[:3, :3] ** 2).sum(axis=0))
    radius = np.ceil(fallback_radius_mm / voxel_sizes).astype(int)
    lower, upper = np.maximum(ijk - radius, 0), np.minimum(ijk + radius + 1, shape)
    if np.any(lower >= upper):
        return 0, "Unlabelled", np.nan, "none"
    block = atlas_data[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    local = np.argwhere(block != 0)
    if local.size == 0:
        return 0, "Unlabelled", np.nan, "none"
    glob = local + lower
    distances = np.linalg.norm(apply_affine(atlas_affine, glob) - xyz[None, :], axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] > fallback_radius_mm:
        return 0, "Unlabelled", np.nan, "none"
    index = int(atlas_data[tuple(glob[nearest])])
    return index, labels.get(index, f"Atlas label {index}"), float(distances[nearest]), "nearest_within_radius"
