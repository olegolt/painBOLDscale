"""Mass-univariate Spearman correlation with permutation FWE (voxelwise and cluster-extent).

Voxels (rank-transformed) are correlated with a rating across participants; Fisher z is
scaled by sqrt(n-3). The same ``n_permutations`` shuffles of the ranked ratings give the null
distribution of the maximum |z| (voxelwise FWE) and of the maximum cluster extent at
|z| >= ``cluster_forming_z`` (cluster FWE; positive and negative clusters formed separately,
26-connectivity).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import generate_binary_structure, label

from .config import EXCLUDED_SUBJECTS, FIRSTLEVEL_MODEL, HEAT_CONTRASTS, MASSUNI_RATINGS, MassUnivariateSettings, Project
from .utils import DataError, MissingInputError, add_subject_key, progress, read_table, require_columns

CLAMP = 0.999999


# --------------------------------------------------------------------------- #
# Numeric core
# --------------------------------------------------------------------------- #
def deterministic_seed(name: str, base_seed: int = 42) -> int:
    """Stable per-analysis seed (independent of run order and Python's hash randomisation)."""
    offset = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
    return int((base_seed + offset) % (2**32 - 1))


def rank_and_center(data: np.ndarray) -> np.ndarray:
    """Column-wise average ranks minus the column mean (float32)."""
    if not np.isfinite(data).all():
        raise DataError(f"Voxel matrix contains {int((~np.isfinite(data)).sum())} non-finite values.")
    ranked = stats.rankdata(data, axis=0, method="average").astype(np.float32)
    ranked -= ranked.mean(axis=0, keepdims=True)
    return ranked


def _centered_ranks(values) -> np.ndarray:
    ranks = stats.rankdata(values, method="average").astype(np.float32)
    return ranks - ranks.mean()


def spearman_maps(ranked_voxels: np.ndarray, ratings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spearman rho, Fisher-z (scaled by sqrt(n-3)) and two-sided p for every voxel."""
    n = ranked_voxels.shape[0]
    if n <= 3:
        raise DataError("At least four participants are required for the z transform.")
    r = _centered_ranks(ratings)
    denominator = np.sqrt(np.sum(ranked_voxels**2, axis=0, dtype=np.float64) * float(np.sum(r**2, dtype=np.float64)))
    rho = np.divide(ranked_voxels.T @ r, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    rho = np.clip(rho, -CLAMP, CLAMP)
    z = np.arctanh(rho) * np.sqrt(n - 3)
    return rho.astype(np.float32), z.astype(np.float32), (2.0 * stats.norm.sf(np.abs(z))).astype(np.float32)


def largest_cluster(z, threshold: float, geometry, structure) -> int:
    """Largest suprathreshold cluster over both tails (tails are labelled separately)."""
    best = 0
    for above in (z >= threshold, z <= -threshold):
        if above.any():
            sizes = np.bincount(label(geometry.to_volume(above, bool), structure=structure)[0].ravel())[1:]
            best = max(best, int(sizes.max(initial=0)))
    return best


def permutation_nulls(ranked_voxels, ratings, n_permutations, batch_size, seed, geometry,
                      cluster_forming_z, structure) -> tuple[np.ndarray, np.ndarray]:
    """Null distributions of max |z| and of the maximum cluster extent."""
    n = ranked_voxels.shape[0]
    rng = np.random.default_rng(seed)
    r = _centered_ranks(ratings)
    denominator = np.sqrt(np.sum(ranked_voxels**2, axis=0, dtype=np.float64)[None, :] * float(np.sum(r**2, dtype=np.float64)))
    null_z = np.empty(n_permutations, dtype=np.float32)
    null_extent = np.empty(n_permutations, dtype=np.int32)
    for start in progress(range(0, n_permutations, batch_size), desc="permutations", leave=False):
        stop = min(start + batch_size, n_permutations)
        permuted = np.stack([rng.permutation(r) for _ in range(stop - start)]).astype(np.float32)
        numerator = permuted @ ranked_voxels
        rho = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32), where=denominator > 0)
        z = np.arctanh(np.clip(rho, -CLAMP, CLAMP)) * np.sqrt(n - 3)
        null_z[start:stop] = np.abs(z).max(axis=1)
        for row, z_row in enumerate(z):
            null_extent[start + row] = largest_cluster(z_row, cluster_forming_z, geometry, structure)
    return null_z, null_extent


def max_stat_p(observed_z, null_max_z) -> np.ndarray:
    """FWE p-value: (#null >= |z| + 1) / (N + 1)."""
    null = np.sort(np.asarray(null_max_z, dtype=np.float64))
    n_ge = len(null) - np.searchsorted(null, np.abs(np.asarray(observed_z, dtype=np.float64)), side="left")
    return ((n_ge + 1) / (len(null) + 1)).astype(np.float32)


def cluster_p(extent: int, null_extent) -> float:
    null = np.asarray(null_extent, dtype=np.int64)
    return float((np.sum(null >= int(extent)) + 1) / (len(null) + 1))


def voxel_fwe_threshold(null_max_z, alpha: float) -> float:
    # NB: index ceil((1-alpha)(N+1))-1 as in the original analysis (one rank above the exact
    # 95th percentile); it only affects the *reported* threshold, not the p-value maps.
    null = np.sort(np.asarray(null_max_z, dtype=np.float64))
    index = int(np.clip(np.ceil((1.0 - alpha) * (len(null) + 1)) - 1, 0, len(null) - 1))
    return float(null[index])


def cluster_extent_threshold(null_extent, alpha: float) -> int:
    """Smallest extent whose cluster-FWE p is <= alpha."""
    null = np.asarray(null_extent, dtype=np.int64)
    top = int(null.max(initial=0))
    for extent in range(1, top + 2):
        if (np.sum(null >= extent) + 1) / (len(null) + 1) <= alpha:
            return extent
    return top + 1


def observed_clusters(z, rho, null_extent, geometry, structure, forming_z, alpha):
    """Cluster table plus per-voxel survival mask and cluster-FWE p vector."""
    survive = np.zeros(z.shape, dtype=bool)
    p_vector = np.ones(z.shape, dtype=np.float32)
    rows, cluster_id = [], 0
    for sign, above in (("positive", z >= forming_z), ("negative", z <= -forming_z)):
        if not above.any():
            continue
        labelled, n_clusters = label(geometry.to_volume(above, bool), structure=structure)
        flat_labels = labelled.ravel(order="C")
        for local in range(1, n_clusters + 1):
            cluster_id += 1
            members = np.searchsorted(geometry.flat_indices, np.flatnonzero(flat_labels == local))
            p = cluster_p(len(members), null_extent)
            p_vector[members] = p
            survive[members] = p <= alpha
            peak = members[np.argmax(z[members]) if sign == "positive" else np.argmin(z[members])]
            ijk = np.unravel_index(int(geometry.flat_indices[peak]), geometry.shape, order="C")
            xyz = geometry.affine @ np.append(np.asarray(ijk, dtype=float), 1.0)
            rows.append({
                "cluster_id": cluster_id, "sign": sign, "extent_voxels": len(members),
                "extent_cm3": float(len(members) * geometry.voxel_volume_mm3 / 1000.0),
                "cluster_p_fwe": p, "survives_cluster_fwe": bool(p <= alpha),
                "peak_z": float(z[peak]), "peak_rho": float(rho[peak]),
                "peak_i": int(ijk[0]), "peak_j": int(ijk[1]), "peak_k": int(ijk[2]),
                "peak_x_mni": float(xyz[0]), "peak_y_mni": float(xyz[1]), "peak_z_mni": float(xyz[2]),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["survives_cluster_fwe", "extent_voxels"], ascending=[False, False]).reset_index(drop=True)
    return table, survive, p_vector


# --------------------------------------------------------------------------- #
# One analysis (voxel matrix + rating -> maps/tables on disk)
# --------------------------------------------------------------------------- #
def run_analysis(contrast, rating, subjects: pd.DataFrame, ranked_voxels, geometry,
                 out_dir: Path, s: MassUnivariateSettings) -> dict:
    structure = generate_binary_structure(3, s.cluster_connectivity)
    name = f"con{contrast}_x_{rating}"
    out_dir.mkdir(parents=True, exist_ok=True)
    values = subjects[rating].to_numpy(dtype=np.float32)
    subjects[["sub_id", "sub_num", rating, f"image_con{contrast}"]].to_csv(out_dir / "included_subjects.tsv", sep="\t", index=False)
    print(f"\n{name}: n={len(subjects)}, rating mean={values.mean():.2f} SD={values.std(ddof=1):.2f}")

    rho, z, p_unc = spearman_maps(ranked_voxels, values)
    unc = np.abs(z) >= s.z_uncorrected
    seed = deterministic_seed(name, s.random_seed)
    started = time.time()
    null_z, null_extent = permutation_nulls(ranked_voxels, values, s.n_permutations, s.batch_size, seed,
                                            geometry, s.cluster_forming_z, structure)
    minutes = (time.time() - started) / 60.0

    p_fwe = max_stat_p(z, null_z)
    voxel_sig = p_fwe <= s.alpha_voxel_fwe
    table, cluster_sig, p_cluster = observed_clusters(z, rho, null_extent, geometry, structure,
                                                      s.cluster_forming_z, s.alpha_cluster_fwe)
    extent_threshold = cluster_extent_threshold(null_extent, s.alpha_cluster_fwe)
    significant = table[table["survives_cluster_fwe"]] if not table.empty else table

    maps = {"rho": rho, "z": z, "p_unc": p_unc,
            "rho_z3p1": np.where(unc, rho, 0.0), "z_z3p1": np.where(unc, z, 0.0),
            "p_fwe": p_fwe, "rho_fwe": np.where(voxel_sig, rho, 0.0), "z_fwe": np.where(voxel_sig, z, 0.0),
            "p_cluster_fwe": p_cluster, "rho_cluster_fwe": np.where(cluster_sig, rho, 0.0),
            "z_cluster_fwe": np.where(cluster_sig, z, 0.0)}
    for key, vector in maps.items():
        geometry.save(vector, out_dir / f"{key}.nii.gz")
    table.to_csv(out_dir / "clusters.tsv", sep="\t", index=False)
    np.save(out_dir / "null_max_z.npy", null_z)
    np.save(out_dir / "null_max_cluster.npy", null_extent)

    vol = geometry.voxel_volume_mm3 / 1000.0
    pos, neg = rho > 0, rho < 0
    n_sig = len(significant)
    result = {
        "analysis": name, "contrast": contrast, "rating": rating, "n_subjects": len(subjects),
        "n_voxels": int(ranked_voxels.shape[1]), "voxel_volume_mm3": float(geometry.voxel_volume_mm3),
        "z_uncorrected_threshold": float(s.z_uncorrected),
        "n_voxels_uncorrected": int(unc.sum()),
        "n_voxels_uncorrected_positive": int((unc & pos).sum()),
        "n_voxels_uncorrected_negative": int((unc & neg).sum()),
        "alpha_voxel_fwe": float(s.alpha_voxel_fwe),
        "z_voxel_fwe_threshold": voxel_fwe_threshold(null_z, s.alpha_voxel_fwe),
        "n_voxels_voxel_fwe": int(voxel_sig.sum()),
        "n_voxels_voxel_fwe_positive": int((voxel_sig & pos).sum()),
        "n_voxels_voxel_fwe_negative": int((voxel_sig & neg).sum()),
        "voxel_fwe_volume_cm3": float(voxel_sig.sum() * vol),
        "cluster_forming_z": float(s.cluster_forming_z), "alpha_cluster_fwe": float(s.alpha_cluster_fwe),
        "cluster_extent_fwe_threshold_voxels": extent_threshold,
        "cluster_extent_fwe_threshold_cm3": float(extent_threshold * vol),
        "n_clusters_cluster_fwe": n_sig,
        "n_clusters_cluster_fwe_positive": int((significant["sign"] == "positive").sum()) if n_sig else 0,
        "n_clusters_cluster_fwe_negative": int((significant["sign"] == "negative").sum()) if n_sig else 0,
        "n_voxels_cluster_fwe": int(cluster_sig.sum()),
        "n_voxels_cluster_fwe_positive": int((cluster_sig & pos).sum()),
        "n_voxels_cluster_fwe_negative": int((cluster_sig & neg).sum()),
        "cluster_fwe_volume_cm3": float(cluster_sig.sum() * vol),
        "largest_significant_cluster_voxels": int(significant["extent_voxels"].max()) if n_sig else 0,
        "n_permutations": int(s.n_permutations), "permutation_seed": seed, "permutation_minutes": minutes,
        "output_directory": str(out_dir),
    }
    pd.DataFrame([result]).to_csv(out_dir / "analysis_summary.tsv", sep="\t", index=False)
    print(f"  uncorrected |z|>={s.z_uncorrected}: {result['n_voxels_uncorrected']:,} vox | "
          f"voxel FWE: {result['n_voxels_voxel_fwe']:,} vox (|z|>={result['z_voxel_fwe_threshold']:.3f}) | "
          f"cluster FWE: {n_sig} clusters, {result['n_voxels_cluster_fwe']:,} vox "
          f"(extent >= {extent_threshold})")
    return result


# --------------------------------------------------------------------------- #
# Sample construction and full run (Figure 2)
# --------------------------------------------------------------------------- #
def contrast_path(subject_dir: Path, contrast: str) -> Path | None:
    model = subject_dir / FIRSTLEVEL_MODEL
    return next((p for p in (model / f"s6wcon_{contrast}.nii", model / f"s6wcon_{contrast}.nii.gz") if p.exists()), None)


def build_sample(project: Project, contrasts=HEAT_CONTRASTS, ratings=MASSUNI_RATINGS) -> pd.DataFrame:
    """Participants with all heat ratings and all heat first-level images (minus exclusions)."""
    table = read_table(project.mean_ratings, "Mean ratings table", sep="\t")
    require_columns(table, ["sub_id", *ratings], "Mean ratings table")
    table = add_subject_key(table, "Mean ratings table", "sub_id")
    by_id = table.set_index("subject_key")
    rows, notes = [], {"no ratings": [], "missing ratings": [], "missing images": []}
    subject_dirs = sorted(project.firstlevel_root.glob("sub-*"))
    if not subject_dirs:
        raise MissingInputError(f"No 'sub-*' folders in {project.firstlevel_root}")
    for directory in subject_dirs:
        key = directory.name.replace("sub-", "").lstrip("0") or "0"
        if key in EXCLUDED_SUBJECTS:
            print(f"Excluding {directory.name}: {EXCLUDED_SUBJECTS[key]}")
            continue
        if key not in by_id.index:
            notes["no ratings"].append(directory.name)
            continue
        values = pd.to_numeric(by_id.loc[key, list(ratings)], errors="coerce").astype(float)
        if not np.isfinite(values.to_numpy()).all():
            notes["missing ratings"].append(directory.name)
            continue
        paths = {c: contrast_path(directory, c) for c in contrasts}
        if any(p is None for p in paths.values()):
            notes["missing images"].append(f"{directory.name} ({[c for c, p in paths.items() if p is None]})")
            continue
        rows.append({"sub_id": directory.name, "sub_num": int(key), **values.to_dict(),
                     **{f"image_con{c}": str(p) for c, p in paths.items()}})
    for reason, who in notes.items():
        if who:
            print(f"Not included ({reason}): {who}")
    if len(rows) < 5:
        raise DataError(f"Only {len(rows)} participants have all heat ratings and images; see messages above.")
    return pd.DataFrame(rows).sort_values("sub_num").reset_index(drop=True)


def results_complete(project: Project) -> bool:
    return all((project.analysis_dir(c, r) / "analysis_summary.tsv").is_file()
               for c in HEAT_CONTRASTS for r in MASSUNI_RATINGS)


def run_all(project: Project, settings: MassUnivariateSettings = MassUnivariateSettings(),
            overwrite: bool = False) -> pd.DataFrame:
    """Figure 2: 4 contrasts x 4 ratings. Skipped if all 16 summaries already exist."""
    from .imaging import MaskGeometry, make_group_mask
    summary_path = project.f2 / "summary_all_analyses.tsv"
    if not overwrite and results_complete(project):
        print("Mass-univariate results already complete; pass overwrite=True to recompute.")
        return pd.read_csv(summary_path, sep="\t")
    project.require("massuni")
    project.make_output_dirs()
    (project.f2 / "settings.json").write_text(json.dumps(
        {**settings.__dict__, "contrasts": HEAT_CONTRASTS, "ratings": MASSUNI_RATINGS,
         "excluded_subjects": EXCLUDED_SUBJECTS}, indent=2, default=str), encoding="utf-8")
    subjects = build_sample(project)
    subjects.to_csv(project.subjects, sep="\t", index=False)
    print(f"Imaging sample: n={len(subjects)} -> {project.subjects}")
    mask_img = make_group_mask(project.mean_epi, project.group_mask, settings.mask_threshold_fraction,
                               settings.mask_reference_percentile)
    geometry = MaskGeometry(mask_img)
    rows = []
    for contrast in HEAT_CONTRASTS:
        print(f"\nLoading contrast {contrast} ...")
        ranked = rank_and_center(geometry.load_matrix(subjects[f"image_con{contrast}"].tolist(), f"con{contrast}"))
        for rating in MASSUNI_RATINGS:
            rows.append(run_analysis(contrast, rating, subjects, ranked, geometry,
                                     project.analysis_dir(contrast, rating), settings))
            pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)
    return pd.DataFrame(rows)
