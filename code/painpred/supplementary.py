"""Supplementary analysis (Figure S2): conventional implicit-baseline contrasts vs. heat-minus-warmth.

1. The Figure 2 mass-univariate analysis is repeated on the implicit-baseline heat contrasts.
2. For each condition, the voxelwise Spearman correlation with pain ratings of the warmth-referenced
   contrast is compared with that of the implicit-baseline contrast by a paired label-swap test:
   within each participant the two rank-score vectors are exchanged at random, and the maximum
   |delta rho| over the brain gives whole-brain FWE p-values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import massuni
from .config import BASELINE_CONTRAST, EXCLUDED_SUBJECTS, HEAT_CONDITIONS, MassUnivariateSettings, Project, SupplementSettings
from .imaging import MaskGeometry, load_3d
from .utils import progress


@dataclass(frozen=True)
class Comparison:
    condition: str
    baseline_contrast: str
    warmth_contrast: str
    rating: str


COMPARISONS = tuple(Comparison(c.label.replace("°C", "C"), BASELINE_CONTRAST[c.contrast], c.contrast, c.rating)
                    for c in HEAT_CONDITIONS)
BASELINE_CONTRASTS = tuple(c.baseline_contrast for c in COMPARISONS)
WARMTH_CONTRASTS = tuple(c.warmth_contrast for c in COMPARISONS)
RATINGS = tuple(c.rating for c in COMPARISONS)


# --------------------------------------------------------------------------- #
# Numeric core
# --------------------------------------------------------------------------- #
def _ranked_ratings(ratings) -> np.ndarray:
    ranked = stats.rankdata(np.asarray(ratings, dtype=float), method="average").astype(np.float32)
    return ranked - ranked.mean()


def _rho(ranked_voxels, ranked_ratings) -> np.ndarray:
    denominator = np.sqrt(np.sum(ranked_voxels**2, axis=0, dtype=np.float64) * float(np.sum(ranked_ratings**2, dtype=np.float64)))
    rho = np.divide(ranked_voxels.T @ ranked_ratings, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return np.clip(rho, -0.999999, 0.999999).astype(np.float32)


def paired_rank_swap_test(ranked_warmth, ranked_baseline, ratings, n_permutations, batch_size, seed, alpha, desc=""):
    """Whole-brain max-|delta rho| permutation test for rho(warmth) - rho(baseline).

    With midpoint M and half-difference H of the two (column-centred) rank matrices, warmth = M + H and
    baseline = M - H; swapping a participant's two values is a sign flip of that participant's H.
    This lets every permuted correlation be computed from a few matrix products.
    """
    if ranked_baseline.shape != ranked_warmth.shape:
        raise ValueError(f"Warmth and baseline matrices differ in shape: {ranked_warmth.shape} vs {ranked_baseline.shape}")
    n = ranked_warmth.shape[0]
    r = _ranked_ratings(ratings)
    rho_warmth, rho_baseline = _rho(ranked_warmth, r), _rho(ranked_baseline, r)
    observed = (rho_warmth - rho_baseline).astype(np.float32)

    mid = (0.5 * (ranked_warmth + ranked_baseline)).astype(np.float32)
    half = (0.5 * (ranked_warmth - ranked_baseline)).astype(np.float32)
    cross = (2.0 * mid * half).astype(np.float32)
    base_ss = np.sum(mid**2 + half**2, axis=0, dtype=np.float64)
    rating_ss = float(np.sum(r**2, dtype=np.float64))
    mid_numerator = (r @ mid).astype(np.float64)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float32)
    for start in progress(range(0, n_permutations, batch_size), desc=desc, leave=False):
        stop = min(start + batch_size, n_permutations)
        signs = rng.integers(0, 2, size=(stop - start, n), dtype=np.int8).astype(np.float32) * 2.0 - 1.0
        half_sum = (signs @ half).astype(np.float64)
        signed_cross = (signs @ cross).astype(np.float64)
        warmth_ss = base_ss[None, :] + signed_cross - half_sum**2 / n
        baseline_ss = base_ss[None, :] - signed_cross - half_sum**2 / n
        half_numerator = ((signs * r[None, :]) @ half).astype(np.float64)
        numerators = (mid_numerator[None, :] + half_numerator, mid_numerator[None, :] - half_numerator)
        denominators = (np.sqrt(np.maximum(warmth_ss, 0.0) * rating_ss), np.sqrt(np.maximum(baseline_ss, 0.0) * rating_ss))
        w, b = (np.divide(num, den, out=np.zeros_like(num), where=den > 0) for num, den in zip(numerators, denominators))
        null[start:stop] = np.max(np.abs(w - b), axis=1).astype(np.float32)

    p_fwe = massuni.max_stat_p(observed, null)
    significant = p_fwe <= alpha
    return {"rho_warmth": rho_warmth, "rho_baseline": rho_baseline, "delta_rho": observed, "p_fwe": p_fwe,
            "significant": significant, "warmth_higher": significant & (observed > 0),
            "baseline_higher": significant & (observed < 0), "null_max_abs_delta": null,
            "fwe_delta_threshold": massuni.voxel_fwe_threshold(null, alpha)}


def distribution_row(condition, variable, values) -> dict:
    v = np.asarray(values, dtype=float)
    q = np.percentile(v, [2.5, 25, 75, 97.5])
    return {"condition": condition, "variable": variable, "n_voxels": len(v), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "minimum": float(v.min()), "q025": float(q[0]), "q25": float(q[1]),
            "median": float(np.median(v)), "q75": float(q[2]), "q975": float(q[3]), "maximum": float(v.max())}


def direction_at_peak(delta: float, p: float, alpha: float) -> str:
    if p > alpha:
        return "no_FWE_significant_difference"
    return "warmth_higher" if delta > 0 else "implicit_baseline_higher" if delta < 0 else "equal"


def overall_direction(n_warmth: int, n_baseline: int) -> str:
    if n_warmth and not n_baseline:
        return "warmth_higher_somewhere"
    if n_baseline and not n_warmth:
        return "implicit_baseline_higher_somewhere"
    if n_warmth and n_baseline:
        return "bidirectional_differences"
    return "no_whole_brain_FWE_significant_difference"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(project: Project, mass: MassUnivariateSettings = MassUnivariateSettings(),
        s: SupplementSettings = SupplementSettings()) -> dict:
    """Both parts of the supplementary analysis; outputs go to ``project.baseline``."""
    project.require("supplement")
    project.make_output_dirs()
    out = project.baseline
    maps_dir, comparison_dir = out / "massunivariate", out / "warmth_vs_implicit_baseline"
    subjects = massuni.build_sample(project, contrasts=(*BASELINE_CONTRASTS, *WARMTH_CONTRASTS), ratings=RATINGS)
    subjects.to_csv(out / "subjects_paired_all8contrasts.tsv", sep="\t", index=False)
    print(f"Supplementary paired sample (all 8 contrasts): n={len(subjects)}")
    if project.subjects.exists():
        main = set(pd.read_csv(project.subjects, sep="\t")["sub_num"].astype(int))
        paired = set(subjects["sub_num"].astype(int))
        pd.DataFrame({"sub_num": sorted(main | paired), "in_main_figure2": [i in main for i in sorted(main | paired)],
                      "in_paired_supplement": [i in paired for i in sorted(main | paired)]}
                     ).to_csv(out / "sample_comparison_with_main_figure2.tsv", sep="\t", index=False)
        if main != paired:
            print(f"  differs from the Figure 2 sample: main-only {sorted(main - paired)}, supplement-only {sorted(paired - main)}")

    geometry = MaskGeometry(load_3d(project.group_mask, "group mask"))
    (out / "settings.json").write_text(json.dumps({
        "baseline_contrasts": BASELINE_CONTRASTS, "warmth_contrasts": WARMTH_CONTRASTS, "ratings": RATINGS,
        "paired_n": len(subjects), "massunivariate_n_permutations": mass.n_permutations,
        "difference_n_permutations": s.n_permutations, "difference_alpha_fwe": s.alpha_fwe,
        "difference_definition": "warmth-contrast rho minus implicit-baseline-contrast rho",
        "excluded_subjects": EXCLUDED_SUBJECTS}, indent=2), encoding="utf-8")

    # Part 1: conventional implicit-baseline contrasts x 4 ratings
    summary_rows = []
    for contrast in BASELINE_CONTRASTS:
        print(f"\nLoading con{contrast} ...")
        ranked = massuni.rank_and_center(geometry.load_matrix(subjects[f"image_con{contrast}"].tolist(), f"con{contrast}"))
        for rating in RATINGS:
            summary_rows.append(massuni.run_analysis(contrast, rating, subjects, ranked, geometry,
                                                     maps_dir / f"con{contrast}" / rating, mass))
            pd.DataFrame(summary_rows).to_csv(out / "massunivariate_summary.tsv", sep="\t", index=False)

    # Part 2: paired comparison of warmth-referenced vs. implicit-baseline correlation maps
    comparisons, peaks, distributions, plot_data = [], [], [], {}
    for index, comp in enumerate(COMPARISONS):
        print(f"\nPaired comparison {comp.condition}: con{comp.warmth_contrast} vs con{comp.baseline_contrast} ({comp.rating})")
        target = comparison_dir / f"con{comp.warmth_contrast}_vs_con{comp.baseline_contrast}" / comp.rating
        ranked_w = massuni.rank_and_center(geometry.load_matrix(subjects[f"image_con{comp.warmth_contrast}"].tolist(), "warmth image"))
        ranked_b = massuni.rank_and_center(geometry.load_matrix(subjects[f"image_con{comp.baseline_contrast}"].tolist(), "baseline image"))
        res = paired_rank_swap_test(ranked_w, ranked_b, subjects[comp.rating].to_numpy(np.float32), s.n_permutations,
                                    s.batch_size, s.random_seed + index, s.alpha_fwe, f"{comp.condition}: warmth-baseline")
        del ranked_w, ranked_b
        delta, p_fwe = res["delta_rho"], res["p_fwe"]
        for name, vector in {"rho_warmth_contrast": res["rho_warmth"], "rho_implicit_baseline_contrast": res["rho_baseline"],
                             "delta_rho_warmth_minus_baseline": delta, "p_fwe_difference": p_fwe,
                             "delta_rho_fwe05": np.where(p_fwe <= s.alpha_fwe, delta, 0.0),
                             "warmth_higher_fwe05": np.where(res["warmth_higher"], delta, 0.0),
                             "implicit_baseline_higher_fwe05": np.where(res["baseline_higher"], delta, 0.0)}.items():
            geometry.save(vector, target / f"{name}.nii.gz")
        np.save(target / "null_max_abs_delta_rho.npy", res["null_max_abs_delta"])
        distributions += [distribution_row(comp.condition, "warmth_rho", res["rho_warmth"]),
                          distribution_row(comp.condition, "implicit_baseline_rho", res["rho_baseline"]),
                          distribution_row(comp.condition, "delta_rho_warmth_minus_baseline", delta)]
        for source, peak in (("warmth_peak", int(np.argmax(res["rho_warmth"]))),
                             ("implicit_baseline_peak", int(np.argmax(res["rho_baseline"])))):
            ijk = np.unravel_index(int(geometry.flat_indices[peak]), geometry.shape, order="C")
            xyz = geometry.affine @ np.append(np.asarray(ijk, float), 1.0)
            peaks.append({"condition": comp.condition, "rating": comp.rating, "warmth_contrast": comp.warmth_contrast,
                          "implicit_baseline_contrast": comp.baseline_contrast, "peak_source": source,
                          "vector_index": peak, "i": int(ijk[0]), "j": int(ijk[1]), "k": int(ijk[2]),
                          "x_mni": float(xyz[0]), "y_mni": float(xyz[1]), "z_mni": float(xyz[2]),
                          "rho_warmth": float(res["rho_warmth"][peak]), "rho_implicit_baseline": float(res["rho_baseline"][peak]),
                          "delta_rho_warmth_minus_baseline": float(delta[peak]), "p_fwe_difference": float(p_fwe[peak]),
                          "difference_direction": direction_at_peak(float(delta[peak]), float(p_fwe[peak]), s.alpha_fwe)})
        n_w, n_b = int(res["warmth_higher"].sum()), int(res["baseline_higher"].sum())
        comparisons.append({
            "condition": comp.condition, "rating": comp.rating, "warmth_contrast": comp.warmth_contrast,
            "implicit_baseline_contrast": comp.baseline_contrast, "n_subjects": len(subjects), "n_voxels": len(delta),
            "mean_rho_warmth": float(res["rho_warmth"].mean()), "median_rho_warmth": float(np.median(res["rho_warmth"])),
            "peak_rho_warmth": float(res["rho_warmth"].max()),
            "mean_rho_implicit_baseline": float(res["rho_baseline"].mean()),
            "median_rho_implicit_baseline": float(np.median(res["rho_baseline"])),
            "peak_rho_implicit_baseline": float(res["rho_baseline"].max()),
            "mean_delta_rho": float(delta.mean()), "median_delta_rho": float(np.median(delta)),
            "max_delta_rho": float(delta.max()), "min_delta_rho": float(delta.min()),
            "fwe_delta_threshold": float(res["fwe_delta_threshold"]), "minimum_p_fwe_difference": float(p_fwe.min()),
            "n_voxels_difference_fwe": n_w + n_b, "n_voxels_warmth_higher_fwe": n_w,
            "n_voxels_implicit_baseline_higher_fwe": n_b, "whole_brain_result": overall_direction(n_w, n_b),
            "n_difference_permutations": s.n_permutations, "comparison_directory": str(target)})
        plot_data[comp.condition] = {"rho_warmth": res["rho_warmth"], "rho_baseline": res["rho_baseline"],
                                     "delta_rho": delta, "n_warmth_higher": n_w, "n_baseline_higher": n_b}
        print(f"  FWE |delta rho| threshold {res['fwe_delta_threshold']:.4f}; warmth higher: {n_w:,} vox; "
              f"baseline higher: {n_b:,} vox -> {overall_direction(n_w, n_b)}")
    tables = {"comparison_summary": pd.DataFrame(comparisons), "peak_comparisons": pd.DataFrame(peaks),
              "distribution_summary": pd.DataFrame(distributions)}
    filenames = {"distribution_summary": "whole_brain_distribution_summary"}
    for name, table in tables.items():
        table.to_csv(comparison_dir / f"{filenames.get(name, name)}.tsv", sep="\t", index=False)
    return {**tables, "plot_data": plot_data, "baseline_summary": pd.DataFrame(summary_rows)}


def load_saved(project: Project) -> dict:
    """Rebuild the inputs of the supplementary figure from saved maps and tables (no subject images needed)."""
    geometry = MaskGeometry(load_3d(project.group_mask, "group mask"))
    folder = project.baseline / "warmth_vs_implicit_baseline"
    tables = {name: pd.read_csv(folder / f"{name}.tsv", sep="\t")
              for name in ("comparison_summary", "peak_comparisons", "whole_brain_distribution_summary")}
    plot_data = {}
    for comp in COMPARISONS:
        target = folder / f"con{comp.warmth_contrast}_vs_con{comp.baseline_contrast}" / comp.rating
        plot_data[comp.condition] = {
            "rho_warmth": geometry.load_vector(target / "rho_warmth_contrast.nii.gz", "warmth rho map"),
            "rho_baseline": geometry.load_vector(target / "rho_implicit_baseline_contrast.nii.gz", "baseline rho map"),
            "delta_rho": geometry.load_vector(target / "delta_rho_warmth_minus_baseline.nii.gz", "delta rho map")}
    return {**tables, "plot_data": plot_data}
