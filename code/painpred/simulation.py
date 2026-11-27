"""Supplementary simulation (Figure S1): how noise in the reference condition attenuates a
brain-behaviour correlation measured on a contrast (pain - reference).

Panel a: many small simulations of a single contrast at a range of reference-to-pain variability ratios.
Panel b: whole-brain simulation (true-effect + null voxels) with FDR-corrected voxelwise tests.
Every task has its own ``SeedSequence``, so results do not depend on the number of workers.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import t as t_dist

from .config import Project, SimulationSettings
from .utils import progress

PANEL_A_RATIOS = np.round(np.arange(0.0, 3.0 + 0.05, 0.025), 2)
KDE_MIN, KDE_MAX, KDE_N_BINS = -0.35, 0.70, 300
KDE_EDGES = np.linspace(KDE_MIN, KDE_MAX, KDE_N_BINS + 1)
KDE_X = (KDE_EDGES[:-1] + KDE_EDGES[1:]) / 2.0
KDE_BIN_WIDTH = KDE_EDGES[1] - KDE_EDGES[0]


def parallel_map(function, argument_tuples, n_jobs: int | None, desc: str) -> list:
    """Run ``function(*args)`` for every tuple, in parallel if joblib is installed."""
    n_jobs = n_jobs or max(1, min(os.cpu_count() or 1, 16))
    argument_tuples = list(argument_tuples)
    if n_jobs == 1:
        return [function(*a) for a in progress(argument_tuples, desc=desc)]
    try:
        from joblib import Parallel, delayed
    except ImportError:  # pragma: no cover
        return [function(*a) for a in progress(argument_tuples, desc=desc)]
    print(f"{desc}: {len(argument_tuples)} tasks on {n_jobs} workers")
    return Parallel(n_jobs=n_jobs, batch_size=1, verbose=0)(delayed(function)(*a) for a in argument_tuples)


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


def zscore_rows(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)


def zscore_columns(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, ddof=1, keepdims=True)


def corr_rows(x, y):
    return np.sum(zscore_rows(x) * zscore_rows(y), axis=1) / (x.shape[1] - 1)


def corr_and_pvals(data, ratings):
    """Pearson r of every column of ``data`` with ``ratings`` and its two-sided t-test p-value."""
    n = data.shape[0]
    r = np.asarray((zscore(ratings) @ data) / ((n - 1) * data.std(axis=0, ddof=1)), dtype=float)
    clipped = np.clip(r, -0.999999, 0.999999)
    t = clipped * np.sqrt((n - 2) / (1.0 - clipped**2))
    return r, 2.0 * t_dist.sf(np.abs(t), df=n - 2)


def bh_reject(p, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg (independence) rejections; same rule as statsmodels' ``fdrcorrection``."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, len(p) + 1) / len(p)
    reject = np.zeros(len(p), dtype=bool)
    if passed.any():
        reject[order[: np.flatnonzero(passed).max() + 1]] = True
    return reject


def density(values) -> np.ndarray:
    """Histogram density smoothed with a Gaussian (Silverman-type bandwidth), unit area."""
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=KDE_EDGES)
    dens = counts.astype(float) / (len(values) * KDE_BIN_WIDTH)
    sigma = max(np.std(values, ddof=1) * len(values) ** (-0.2) / KDE_BIN_WIDTH, 0.5)
    dens = ndimage.gaussian_filter1d(dens, sigma=sigma, mode="constant")
    area = np.trapezoid(dens, KDE_X)
    return dens / area if area > 0 else dens


# --------------------------------------------------------------------------- #
# Panel a
# --------------------------------------------------------------------------- #
def panel_a_chunk(ratio_index, ratio, chunk_index, n_repetitions, seed, s: SimulationSettings):
    rng = np.random.default_rng(np.random.SeedSequence([seed, ratio_index, chunk_index]))
    shape = (n_repetitions, s.n_subjects)
    ratings = np.clip(rng.normal(50.0, 15.0, size=shape), 0.0, 100.0)
    noise = zscore_rows(rng.normal(size=shape))
    signal = zscore_rows(s.true_r * zscore_rows(ratings) + np.sqrt(1.0 - s.true_r**2) * noise)
    pain = s.response_mean + s.response_sd * signal
    reference = rng.normal(0.0, ratio * s.response_sd, size=shape)
    return ratio_index, chunk_index, corr_rows(pain - reference, ratings)


def panel_a(s: SimulationSettings = SimulationSettings(), ratios=PANEL_A_RATIOS, n_jobs=None) -> pd.DataFrame:
    n_chunks = int(np.ceil(s.n_panel_a_repetitions / s.panel_a_chunk_size))
    tasks = [(i, float(ratio), c, min(s.panel_a_chunk_size, s.n_panel_a_repetitions - c * s.panel_a_chunk_size), s.seed_a, s)
             for i, ratio in enumerate(ratios) for c in range(n_chunks)]
    by_ratio: dict[int, list] = {i: [] for i in range(len(ratios))}
    for i, c, values in parallel_map(panel_a_chunk, tasks, n_jobs, "panel a"):
        by_ratio[i].append((c, values))
    rows = []
    for i, ratio in enumerate(ratios):
        r = np.concatenate([v for _, v in sorted(by_ratio[i], key=lambda item: item[0])])
        if len(r) != s.n_panel_a_repetitions:
            raise RuntimeError(f"Panel a: {len(r)} simulations for ratio {ratio:g}, expected {s.n_panel_a_repetitions}.")
        rows.append({"reference_variability_ratio": float(ratio), "reference_sd_psc": float(ratio * s.response_sd),
                     "mean_r": float(r.mean()), "low_r": float(np.percentile(r, 2.5)),
                     "high_r": float(np.percentile(r, 97.5)), "sd_r": float(r.std(ddof=1))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Panel b
# --------------------------------------------------------------------------- #
def whole_brain_base_data(rng, s: SimulationSettings):
    n, n_true, n_null = s.n_subjects, s.n_true_voxels, s.n_null_voxels
    ratings = np.clip(rng.normal(50.0, 15.0, size=n), 0.0, 100.0)
    ratings_z = zscore(ratings).astype(np.float32)
    true_noise = zscore_columns(rng.normal(size=(n, n_true)).astype(np.float32))
    true_z = zscore_columns(s.true_r * ratings_z[:, None] + np.sqrt(1.0 - s.true_r**2) * true_noise)
    null_z = zscore_columns(rng.normal(size=(n, n_null)).astype(np.float32))
    pain = np.empty((n, n_true + n_null), dtype=np.float32)
    pain[:, :n_true] = s.response_mean + s.response_sd * true_z
    pain[:, n_true:] = s.response_mean + s.response_sd * null_z
    reference_z = zscore_columns(rng.normal(size=(n, n_true + n_null)).astype(np.float32))
    is_true = np.zeros(n_true + n_null, dtype=bool)
    is_true[:n_true] = True
    return ratings, pain, reference_z, is_true


def whole_brain_repetition(repetition, seed, s: SimulationSettings):
    rng = np.random.default_rng(np.random.SeedSequence([seed, repetition]))
    ratings, pain, reference_z, is_true = whole_brain_base_data(rng, s)
    rows, densities = [], {}
    for ratio in s.whole_brain_ratios:
        r, p = corr_and_pvals(pain - ratio * s.response_sd * reference_z, ratings)
        reject = bh_reject(p, 0.05)
        true_r, null_r = r[is_true], r[~is_true]
        densities[ratio] = density(true_r)
        n_true_hits, n_null_hits = int(reject[is_true].sum()), int(reject[~is_true].sum())
        rows.append({"repetition": repetition, "reference_variability_ratio": ratio,
                     "reference_sd_psc": ratio * s.response_sd, "mean_true_voxel_r": float(true_r.mean()),
                     "median_true_voxel_r": float(np.median(true_r)), "sd_true_voxel_r": float(true_r.std(ddof=1)),
                     "mean_null_voxel_r": float(null_r.mean()), "n_fdr_significant": int(reject.sum()),
                     "n_true_fdr_significant": n_true_hits, "n_null_fdr_significant": n_null_hits,
                     "true_voxel_detection_rate": n_true_hits / s.n_true_voxels})
    return rows, densities


def panel_b(s: SimulationSettings = SimulationSettings(), n_jobs=None):
    results = parallel_map(whole_brain_repetition, [(i, s.seed_b, s) for i in range(s.n_whole_brain_repetitions)],
                           n_jobs and min(n_jobs, 12) or 12, "panel b")
    repetitions = pd.DataFrame([row for rows, _ in results for row in rows])
    kde = []
    for ratio in s.whole_brain_ratios:
        matrix = np.vstack([d[ratio] for _, d in results])
        kde.append(pd.DataFrame({"reference_variability_ratio": ratio, "r": KDE_X, "mean_density": matrix.mean(axis=0),
                                 "low_density": np.percentile(matrix, 2.5, axis=0),
                                 "high_density": np.percentile(matrix, 97.5, axis=0)}))
    percentile = lambda q: (lambda x: np.percentile(x, q))  # noqa: E731
    summary = repetitions.groupby("reference_variability_ratio", as_index=False).agg(
        reference_sd_psc=("reference_sd_psc", "first"), mean_true_voxel_r=("mean_true_voxel_r", "mean"),
        low_mean_true_voxel_r=("mean_true_voxel_r", percentile(2.5)),
        high_mean_true_voxel_r=("mean_true_voxel_r", percentile(97.5)),
        mean_sd_true_voxel_r=("sd_true_voxel_r", "mean"), mean_null_voxel_r=("mean_null_voxel_r", "mean"),
        mean_n_fdr_significant=("n_fdr_significant", "mean"),
        mean_n_true_fdr_significant=("n_true_fdr_significant", "mean"),
        mean_n_null_fdr_significant=("n_null_fdr_significant", "mean"),
        mean_true_voxel_detection_rate=("true_voxel_detection_rate", "mean"))
    return repetitions, summary, pd.concat(kde, ignore_index=True)


def run(project: Project, s: SimulationSettings = SimulationSettings(), n_jobs: int | None = None) -> dict:
    """Run both panels and write the same CSVs as the original notebook."""
    project.make_output_dirs()
    a = panel_a(s, n_jobs=n_jobs)
    a.to_csv(project.simulation / "panelA_reference_variability_summary.csv", index=False)
    repetitions, summary, kde = panel_b(s, n_jobs)
    repetitions.to_csv(project.simulation / "panelB_whole_brain_repetitions.csv", index=False)
    summary.to_csv(project.simulation / "panelB_whole_brain_summary.csv", index=False)
    kde.to_csv(project.simulation / "panelB_whole_brain_average_kde.csv", index=False)
    return {"panel_a": a, "panel_b_repetitions": repetitions, "panel_b_summary": summary, "panel_b_kde": kde}
