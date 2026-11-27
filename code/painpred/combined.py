"""Figure 6: does the NPS add predictive information beyond pre-experiment calibration?

Three OLS models (calibration temperature, NPS, both) predict pain ratings; performance is the
R² of out-of-fold predictions from repeated K-fold CV. Model differences are tested on the
repeat-averaged out-of-fold predictions (participant bootstrap + prediction-swap permutation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HEAT_CONDITIONS, CombinedModelSettings, Project
from .data import MEAN_PAIN, load_analysis_table
from .utils import (
    canonical_subject_id,
    check_sample_size,
    drop_excluded,
    progress,
    read_table,
    require_columns,
    safe_pearson,
    safe_spearman,
)

CALIBRATION = "mean_calibration_temperature_c"
NPS = "nps_con_0053"
OUTCOMES = tuple((c.label, c.rating) for c in HEAT_CONDITIONS) + (("Mean across all conditions", MEAN_PAIN),)
MODELS = ("Pre-experiment calibration", "NPS", "Calibration + NPS")
COMPARISONS = (("NPS", "Pre-experiment calibration"),
               ("Calibration + NPS", "NPS"),
               ("Calibration + NPS", "Pre-experiment calibration"))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def calibration_temperatures(project: Project, s: CombinedModelSettings) -> pd.DataFrame:
    """Wide table of the temperature (°C) estimated to evoke each target VAS, per participant."""
    table = read_table(project.calibration, "Calibration summary")
    require_columns(table, ["subject_id", "target_vas", "estimated_temperature_c"], "Calibration summary")
    table = drop_excluded(table.assign(subject_key=table["subject_id"].map(canonical_subject_id)),
                          project.excluded)
    for column in ("target_vas", "estimated_temperature_c"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table[table["target_vas"].isin(s.calibration_targets)]
    wide = (table.groupby(["subject_key", "target_vas"], as_index=False)["estimated_temperature_c"].mean()
            .pivot(index="subject_key", columns="target_vas", values="estimated_temperature_c")
            .reindex(columns=list(s.calibration_targets)))
    wide.columns = [f"calibration_temp_vas_{int(v)}_c" for v in wide.columns]
    return wide.reset_index()


def build_paired_dataset(project: Project, s: CombinedModelSettings = CombinedModelSettings()) -> pd.DataFrame:
    """Participants with ratings, NPS and complete calibration, minus calibration outliers (IQR rule)."""
    project.require("figure6")
    base = load_analysis_table(project, nps_contrasts=("0053",), expected_n=s.expected_full_n)
    wide = calibration_temperatures(project, s)
    targets = [f"calibration_temp_vas_{v}_c" for v in s.calibration_targets]
    availability = wide.assign(n_valid_targets=wide[targets].notna().sum(axis=1))
    availability["complete_calibration"] = availability["n_valid_targets"] == len(targets)
    availability.to_csv(project.f6 / "calibration_availability.tsv", sep="\t", index=False)
    wide = wide[np.isfinite(wide[targets].to_numpy(float)).all(axis=1)].copy()
    wide[CALIBRATION] = wide[targets].mean(axis=1)
    data = base.merge(wide, on="subject_key", how="inner", validate="one_to_one")
    data = data[np.isfinite(data.drop(columns="subject_key").to_numpy(float)).all(axis=1)]
    data = data.sort_values("subject_key").reset_index(drop=True)
    data.to_csv(project.f6 / "paired_before_filter.tsv", sep="\t", index=False)

    values = data[CALIBRATION].to_numpy(float)
    q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
    lower, upper = q1 - s.iqr_multiplier * (q3 - q1), q3 + s.iqr_multiplier * (q3 - q1)
    outlier = (values < lower) | (values > upper)
    report = data[["subject_key", *targets, CALIBRATION]].assign(
        q1=q1, q3=q3, iqr=q3 - q1, multiplier=s.iqr_multiplier, lower_fence=lower, upper_fence=upper, excluded=outlier)
    report.to_csv(project.f6 / "calibration_iqr.tsv", sep="\t", index=False)
    paired = data[~outlier].reset_index(drop=True)
    paired.to_csv(project.f6 / "paired_subjects.tsv", sep="\t", index=False)
    print(f"Paired sample: {len(data)} with complete calibration; {int(outlier.sum())} calibration outlier(s) "
          f"excluded ({s.iqr_multiplier:g} x IQR) -> n={len(paired)}")
    check_sample_size(len(paired), s.expected_paired_n, "Figure 6 paired sample")
    return paired


# --------------------------------------------------------------------------- #
# Models and statistics
# --------------------------------------------------------------------------- #
def safe_name(text: str) -> str:
    return text.casefold().replace(" ", "_").replace("+", "plus").replace("°", "deg").replace("-", "_").replace("/", "_")


def prediction_column(model: str, outcome: str) -> str:
    return f"oof_{safe_name(model)}__{safe_name(outcome)}"


def ols_fit_predict(x_train, y_train, x_test) -> np.ndarray:
    design = lambda x: np.column_stack([np.ones(len(x)), np.asarray(x, float).reshape(len(x), -1)])  # noqa: E731
    beta = np.linalg.lstsq(design(x_train), np.asarray(y_train, float), rcond=None)[0]
    return design(x_test) @ beta


def r2(y, prediction) -> float:
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    total = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - prediction) ** 2) / total) if total > 0 else np.nan


def r2_rows(y, predictions) -> np.ndarray:
    """R² of each row of ``y`` (shape (B, n)) against the same row of ``predictions``."""
    total = np.sum((y - y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - np.sum((y - predictions) ** 2, axis=1) / total
    out[total <= 0] = np.nan
    return out


def _predictors(calibration, nps):
    return {"Pre-experiment calibration": calibration[:, None], "NPS": nps[:, None],
            "Calibration + NPS": np.column_stack([calibration, nps])}


def one_repeat(repeat: int, keys, calibration, nps, outcomes, s: CombinedModelSettings):
    """One K-fold partition; all three models are scored on exactly the same held-out participants."""
    from sklearn.model_selection import KFold
    seed = s.base_seed + repeat - 1
    predictors = _predictors(calibration, nps)
    oof = {(m, o): np.full(len(keys), np.nan) for m in MODELS for o, _ in OUTCOMES}
    assignments = []
    for fold, (train, test) in enumerate(KFold(s.n_folds, shuffle=True, random_state=seed).split(np.arange(len(keys))), 1):
        assignments += [{"repeat": repeat, "outer_fold": fold, "random_state": seed, "subject_key": str(keys[i])} for i in test]
        for outcome, _ in OUTCOMES:
            for model in MODELS:
                x = predictors[model]
                oof[(model, outcome)][test] = ols_fit_predict(x[train], outcomes[outcome][train], x[test])
    metrics, predictions = [], pd.DataFrame({"repeat": repeat, "subject_key": keys.astype(str)})
    for outcome, _ in OUTCOMES:
        for model in MODELS:
            p = oof[(model, outcome)]
            if not np.isfinite(p).all():
                raise RuntimeError(f"Missing out-of-fold prediction: repeat={repeat}, {outcome}, {model}")
            y = outcomes[outcome]
            r, pr = safe_pearson(y, p)
            metrics.append({"repeat": repeat, "outcome": outcome, "model": model, "n": len(y), "n_folds": s.n_folds,
                            "cv_r2": r2(y, p), "cv_pearson_r": r, "cv_pearson_p": pr})
            predictions[prediction_column(model, outcome)] = p
    return pd.DataFrame(metrics), predictions, pd.DataFrame(assignments)


def repeated_cv(data: pd.DataFrame, s: CombinedModelSettings):
    keys = data["subject_key"].astype(str).to_numpy()
    calibration, nps = data[CALIBRATION].to_numpy(float), data[NPS].to_numpy(float)
    outcomes = {label: data[column].to_numpy(float) for label, column in OUTCOMES}
    results = [one_repeat(r, keys, calibration, nps, outcomes, s)
               for r in progress(range(1, s.n_repeats + 1), desc="repeated CV")]
    return tuple(pd.concat(parts, ignore_index=True) for parts in zip(*results))


def summarize_cv(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (outcome, model), group in metrics.groupby(["outcome", "model"], sort=False):
        v = group["cv_r2"].to_numpy(float)
        rows.append({"outcome": outcome, "model": model, "n": int(group["n"].iloc[0]), "n_repeats": len(v),
                     "mean_cv_r2": float(v.mean()), "median_cv_r2": float(np.median(v)),
                     "sd_cv_r2_across_splits": float(v.std(ddof=1)),
                     "split_percentile_2p5": float(np.percentile(v, 2.5)),
                     "split_percentile_97p5": float(np.percentile(v, 97.5)),
                     "min_cv_r2": float(v.min()), "max_cv_r2": float(v.max()),
                     "fraction_cv_r2_gt_zero": float(np.mean(v > 0))})
    return pd.DataFrame(rows)


def average_predictions(data: pd.DataFrame, predictions: pd.DataFrame):
    """Mean out-of-fold prediction per participant over repeats, and its (aggregated) metrics."""
    cols = [c for c in predictions.columns if c.startswith("oof_")]
    mean = predictions.groupby("subject_key", as_index=False, sort=True)[cols].mean()
    merged = data.assign(subject_key=data["subject_key"].astype(str)).merge(mean, on="subject_key", how="inner", validate="one_to_one")
    rows = []
    for outcome, column in OUTCOMES:
        y = merged[column].to_numpy(float)
        for model in MODELS:
            p = merged[prediction_column(model, outcome)].to_numpy(float)
            r, pr = safe_pearson(y, p)
            rows.append({"outcome": outcome, "model": model, "n": len(y), "aggregated_oof_r2": r2(y, p),
                         "aggregated_oof_pearson_r": r, "aggregated_oof_pearson_p": pr})
    return merged, pd.DataFrame(rows)


def bootstrap_deltas(y, a, b, rng, n_bootstraps, chunk=2000):
    """Participant bootstrap of R²(a) - R²(b); b may be None for a single model's R²."""
    n, out = len(y), []
    for start in range(0, n_bootstraps, chunk):
        index = rng.integers(0, n, size=(min(chunk, n_bootstraps - start), n))
        value = r2_rows(y[index], a[index])
        out.append(value if b is None else value - r2_rows(y[index], b[index]))
    return np.concatenate(out)


def prediction_swap_p(y, a, b, rng, n_permutations, chunk=2000):
    """Randomly swap the two models' predictions per participant; p of the observed R² difference."""
    observed = r2(y, a) - r2(y, b)
    null = []
    for start in range(0, n_permutations, chunk):
        swap = rng.random((min(chunk, n_permutations - start), len(y))) < 0.5
        pa, pb = np.where(swap, b, a), np.where(swap, a, b)
        yy = np.broadcast_to(y, pa.shape)
        null.append(r2_rows(yy, pa) - r2_rows(yy, pb))
    null = np.concatenate(null)
    return (float((1 + np.sum(null >= observed)) / (n_permutations + 1)),
            float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1)))


def bootstrap_model_r2(averaged: pd.DataFrame, s: CombinedModelSettings) -> pd.DataFrame:
    rows, counter = [], 0
    for outcome, column in OUTCOMES:
        for model in MODELS:
            counter += 1
            y, p = averaged[column].to_numpy(float), averaged[prediction_column(model, outcome)].to_numpy(float)
            values = bootstrap_deltas(y, p, None, np.random.default_rng(s.inference_seed + counter), s.n_bootstraps)
            finite = values[np.isfinite(values)]
            low, high = np.percentile(finite, [2.5, 97.5])
            rows.append({"outcome": outcome, "model": model, "n": len(y), "aggregated_oof_r2": r2(y, p),
                         "participant_bootstrap_ci_2p5": float(low), "participant_bootstrap_ci_97p5": float(high),
                         "n_bootstraps": len(finite)})
    return pd.DataFrame(rows)


def model_comparisons(averaged: pd.DataFrame, s: CombinedModelSettings) -> pd.DataFrame:
    rows, counter = [], 0
    for outcome, column in OUTCOMES:
        y = averaged[column].to_numpy(float)
        for model_a, model_b in COMPARISONS:
            counter += 1
            seed = s.inference_seed + 10_000 + counter
            a = averaged[prediction_column(model_a, outcome)].to_numpy(float)
            b = averaged[prediction_column(model_b, outcome)].to_numpy(float)
            deltas = bootstrap_deltas(y, a, b, np.random.default_rng(seed), s.n_bootstraps)
            finite = deltas[np.isfinite(deltas)]
            low, high = np.percentile(finite, [2.5, 97.5])
            p_one, p_two = prediction_swap_p(y, a, b, np.random.default_rng(seed + 100_000), s.n_permutations)
            rows.append({"outcome": outcome, "model_a": model_a, "model_b": model_b, "n": len(y),
                         "aggregated_oof_r2_model_a": r2(y, a), "aggregated_oof_r2_model_b": r2(y, b),
                         "observed_delta_aggregated_oof_r2": r2(y, a) - r2(y, b),
                         "participant_bootstrap_ci_2p5": float(low), "participant_bootstrap_ci_97p5": float(high),
                         "bootstrap_probability_delta_le_zero": float(np.mean(finite <= 0)),
                         "prediction_swap_p_one_sided_a_greater": p_one, "prediction_swap_p_two_sided": p_two,
                         "n_bootstraps": len(finite), "n_permutations": s.n_permutations})
    return pd.DataFrame(rows)


def panel_statistics(paired: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics shown in panels a-c (a, c: paired sample; b: full NPS sample)."""
    def row(panel, sample, label, n, x, y, vif=False):
        r, p = safe_pearson(x, y)
        rho, rho_p = safe_spearman(x, y)
        return {"panel": panel, "sample": sample, "comparison": label, "n": n, "pearson_r": r, "pearson_p": p,
                "spearman_rho": rho, "spearman_p": rho_p, "r2": r**2, "vif": 1.0 / (1.0 - r**2) if vif else np.nan}
    return pd.DataFrame([
        row("a", "paired IQR-filtered", "Calibration vs mean pain", len(paired), paired[CALIBRATION], paired[MEAN_PAIN]),
        row("b", "full scanner/NPS", "NPS con0053 vs mean pain", len(full), full[NPS], full[MEAN_PAIN]),
        row("c", "paired IQR-filtered", "Calibration vs NPS con0053", len(paired), paired[CALIBRATION], paired[NPS], vif=True)])


def run(project: Project, s: CombinedModelSettings = CombinedModelSettings()) -> dict:
    """All Figure 6 analyses; tables are written to ``project.f6``. Returns them in a dict."""
    project.make_output_dirs()
    paired = build_paired_dataset(project, s)
    full = load_analysis_table(project, nps_contrasts=("0053",), expected_n=s.expected_full_n)
    display = panel_statistics(paired, full)
    metrics, predictions, folds = repeated_cv(paired, s)
    summary = summarize_cv(metrics)
    averaged, averaged_metrics = average_predictions(paired, predictions)
    boot, comparisons = bootstrap_model_r2(averaged, s), model_comparisons(averaged, s)
    out = {"display_stats": display, "cv_repeat_metrics": metrics, "cv_summary": summary,
           "averaged_oof_metrics": averaged_metrics, "bootstrap_r2": boot, "model_comparisons": comparisons}
    for name, table in {**out, "oof_average": averaged}.items():
        table.to_csv(project.f6 / {"cv_repeat_metrics": "cv_repeat_metrics.tsv", "averaged_oof_metrics": "oof_metrics.tsv",
                                   "bootstrap_r2": "bootstrap_r2.tsv"}.get(name, f"{name}.tsv"), sep="\t", index=False)
    predictions.to_csv(project.f6 / "oof_repeat.tsv.gz", sep="\t", index=False, compression="gzip")
    folds.to_csv(project.f6 / "cv_folds.tsv.gz", sep="\t", index=False, compression="gzip")
    return {**out, "paired": paired, "full": full, "averaged_oof": averaged}


__all__ = ["run", "build_paired_dataset", "OUTCOMES", "MODELS", "CALIBRATION", "NPS"]
