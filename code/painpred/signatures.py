"""LASSO-PCR pain signature (Figure 4) and the matched NPS cross-validation.

Pipeline, fitted on training data only: voxel z-score -> PCA (``max(pc_grid)`` components)
-> component z-score -> Lasso on the first k components. Hyper-parameters (k, alpha) are chosen
by an inner K-fold CV that re-fits the whole pipeline on every inner split. The final model is
back-projected to a raw-voxel weight vector so that the signature score of any contrast image is
``image . weights`` (no intercept).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler

from .config import HEAT_CONDITIONS, PRIMARY, LassoPcrSettings, Project
from .imaging import MaskGeometry, load_3d, resample_to, same_grid, save_volume
from .utils import DataError, association_stats, progress, r2_manual, read_table, safe_pearson, safe_spearman

BACK_PROJECTION_TOLERANCE = 1e-4


def alpha_grid(s: LassoPcrSettings) -> np.ndarray:
    """Descending alpha grid (the order ``lasso_path`` requires)."""
    return np.sort(np.logspace(s.alpha_min_log10, s.alpha_max_log10, s.n_alphas))[::-1]


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def pc_features(x_train, x_other, max_pcs: int, seed: int):
    """Voxel z-score -> PCA -> PC z-score, all fitted on ``x_train``.

    Returns (train_pc_z, other_pc_z) as float32.
    """
    voxel_scaler = StandardScaler().fit(x_train)
    pca = PCA(n_components=max_pcs, svd_solver="randomized", random_state=seed)
    train_pc = pca.fit_transform(voxel_scaler.transform(x_train).astype(np.float32, copy=False)).astype(np.float32, copy=False)
    other_pc = pca.transform(voxel_scaler.transform(x_other).astype(np.float32, copy=False)).astype(np.float32, copy=False)
    pc_scaler = StandardScaler().fit(train_pc)
    return (pc_scaler.transform(train_pc).astype(np.float32, copy=False),
            pc_scaler.transform(other_pc).astype(np.float32, copy=False))


@dataclass
class InnerFold:
    train: np.ndarray
    valid: np.ndarray
    train_pc: np.ndarray
    valid_pc: np.ndarray


def inner_folds(x, s: LassoPcrSettings, split_seed: int, pca_seed_base: int, pca_seed_step: int = 1):
    """Inner-CV feature sets of a training sample (PCA re-fitted inside every inner split)."""
    cv = KFold(n_splits=s.inner_folds, shuffle=True, random_state=split_seed)
    folds = []
    for k, (train, valid) in enumerate(cv.split(np.arange(len(x))), start=1):
        train_pc, valid_pc = pc_features(x[train], x[valid], max(s.pc_grid), pca_seed_base + k * pca_seed_step)
        folds.append(InnerFold(np.asarray(train, np.int32), np.asarray(valid, np.int32), train_pc, valid_pc))
    return folds


def check_pc_grid(n_train: int, s: LassoPcrSettings) -> None:
    smallest = n_train - math.ceil(n_train / s.inner_folds)
    if max(s.pc_grid) >= smallest:
        raise DataError(
            f"Largest PC count ({max(s.pc_grid)}) must be below the smallest inner-training sample "
            f"({smallest}) for n_train={n_train}. Reduce `pc_grid` or provide more participants.")


# --------------------------------------------------------------------------- #
# Hyper-parameter selection and fitting
# --------------------------------------------------------------------------- #
def _path_validation_mse(x_train, y_train, x_valid, y_valid, alphas, s) -> np.ndarray:
    y_mean = float(np.mean(y_train))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        out_alphas, coefs, _ = lasso_path(x_train, np.asarray(y_train, float) - y_mean, alphas=alphas,
                                          max_iter=s.max_iter, tol=s.tol)
    if not np.allclose(out_alphas, alphas):
        raise RuntimeError("lasso_path returned alphas in an unexpected order.")
    predictions = x_valid @ coefs + y_mean
    return np.mean((np.asarray(y_valid, float)[:, None] - predictions) ** 2, axis=0)


def select_hyperparameters(folds: list[InnerFold], y, s: LassoPcrSettings):
    """(n_pcs, alpha, inner MSE) minimising the size-weighted inner-CV MSE."""
    alphas = alpha_grid(s)
    error = np.zeros((len(s.pc_grid), len(alphas)))
    weight = np.zeros_like(error)
    for fold in folds:
        for i, n_pcs in enumerate(s.pc_grid):
            mse = _path_validation_mse(fold.train_pc[:, :n_pcs], y[fold.train], fold.valid_pc[:, :n_pcs],
                                       y[fold.valid], alphas, s)
            error[i] += mse * len(fold.valid)
            weight[i] += len(fold.valid)
    mean_mse = error / weight
    i, j = np.unravel_index(int(np.argmin(mean_mse)), mean_mse.shape)
    return int(s.pc_grid[i]), float(alphas[j]), float(mean_mse[i, j])


def fit_lasso(x_pc_z, y, n_pcs: int, alpha: float, s: LassoPcrSettings) -> Lasso:
    model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=s.max_iter, tol=s.tol, selection="cyclic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_pc_z[:, :n_pcs], y)
    return model


# --------------------------------------------------------------------------- #
# Repeated nested cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(x: np.ndarray, y: np.ndarray, subject_ids, s: LassoPcrSettings = LassoPcrSettings()):
    """Repeated outer K-fold with nested hyper-parameter tuning.

    Returns (per-repeat predictions, repeat-averaged predictions, tuning table, fold assignments).
    """
    n = len(y)
    ids = np.asarray(subject_ids)
    max_pcs = max(s.pc_grid)
    outer = RepeatedKFold(n_splits=s.outer_folds, n_repeats=s.outer_repeats, random_state=s.random_seed)
    predictions = np.full((s.outer_repeats, n), np.nan)
    tuning, assignments = [], []
    splits = list(outer.split(np.arange(n)))
    for zero, (train, test) in enumerate(progress(splits, desc="outer CV splits")):
        split, repeat, fold = zero + 1, zero // s.outer_folds + 1, zero % s.outer_folds + 1
        check_pc_grid(len(train), s)
        train_pc_z, test_pc_z = pc_features(x[train], x[test], max_pcs, s.random_seed + 1_000 + split)
        inner = inner_folds(x[train], s, s.random_seed + 10_000 + split, s.random_seed + 100_000 + split * 100)
        n_pcs, alpha, inner_mse = select_hyperparameters(inner, y[train], s)
        model = fit_lasso(train_pc_z, y[train], n_pcs, alpha, s)
        predictions[repeat - 1, test] = model.predict(test_pc_z[:, :n_pcs])
        print(f"  repeat {repeat} fold {fold}: PCs={n_pcs} alpha={alpha:.4g} inner MSE={inner_mse:.3f}")
        tuning.append({"split_number": split, "repeat": repeat, "outer_fold": fold, "n_train": len(train),
                       "n_test": len(test), "selected_n_pcs": n_pcs, "selected_alpha": alpha,
                       "inner_cv_mse": inner_mse})
        assignments += [{"sub_id": ids[i], "subject_index": int(i), "repeat": repeat, "outer_fold": fold,
                         "split_number": split} for i in test]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Some participants lack an out-of-fold prediction in at least one repeat.")
    lo, hi = s.rating_bounds
    rating = "pain_48_arm"
    repeat_table = pd.DataFrame([
        {"sub_id": ids[i], "repeat": r + 1, rating: float(y[i]), "pred_raw_con0053": float(predictions[r, i]),
         "pred_bounded_con0053": float(np.clip(predictions[r, i], lo, hi))}
        for r in range(s.outer_repeats) for i in range(n)])
    mean_raw = predictions.mean(axis=0)
    averaged = pd.DataFrame({"sub_id": ids, rating: y, "pred_raw_con0053": mean_raw,
                             "pred_bounded_con0053": np.clip(mean_raw, lo, hi)})
    return repeat_table, averaged, pd.DataFrame(tuning), pd.DataFrame(assignments)


def prediction_metrics(observed, predicted) -> dict:
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    r, p = safe_pearson(observed, predicted)
    rho, rho_p = safe_spearman(observed, predicted)
    return {"n": len(observed), "observed_mean": float(observed.mean()), "observed_sd": float(observed.std(ddof=1)),
            "predicted_mean": float(predicted.mean()), "predicted_sd": float(predicted.std(ddof=1)),
            "pearson_r": r, "pearson_p_parametric_descriptive": p,
            "spearman_rho": rho, "spearman_p_parametric_descriptive": rho_p,
            "mae": float(np.mean(np.abs(observed - predicted))),
            "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2)))}


def cv_metrics(averaged: pd.DataFrame, r2_name: str = "cross_validated_r2") -> pd.DataFrame:
    """Metrics of the repeat-averaged out-of-fold predictions (raw and rating-bounded)."""
    rows = []
    for scale in ("raw", "bounded"):
        pred = averaged[f"pred_{scale}_con0053"].to_numpy(float)
        obs = averaged["pain_48_arm"].to_numpy(float)
        rows.append({"contrast": PRIMARY.contrast, "condition": PRIMARY.label, "rating": PRIMARY.rating,
                     "prediction_scale": scale, **prediction_metrics(obs, pred), r2_name: r2_manual(obs, pred)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Final model and back-projection
# --------------------------------------------------------------------------- #
@dataclass
class FullModel:
    voxel_scaler: StandardScaler
    pca: PCA
    pc_scaler: StandardScaler
    lasso: Lasso
    n_pcs: int
    alpha: float
    inner_mse: float
    raw_weights: np.ndarray
    standardized_weights: np.ndarray
    raw_pc_weights: np.ndarray
    intercept: float

    def predict(self, x_raw) -> np.ndarray:
        """Prediction through the fitted pipeline (used to validate the back-projection)."""
        pc_z = self.pc_scaler.transform(self.pca.transform(self.voxel_scaler.transform(x_raw)))
        return self.lasso.predict(pc_z[:, : self.n_pcs])


def back_project(voxel_scaler, pca, pc_scaler, lasso, n_pcs):
    """Express the fitted pipeline as ``x_raw @ w + b``. Returns (w_raw, w_standardized, w_pc_raw, b)."""
    pc_weights = np.asarray(lasso.coef_, float) / np.asarray(pc_scaler.scale_[:n_pcs], float)
    standardized = np.asarray(pca.components_[:n_pcs], float).T @ pc_weights
    scale, mean = np.asarray(voxel_scaler.scale_, float), np.asarray(voxel_scaler.mean_, float)
    intercept = float(lasso.intercept_ - np.dot(np.asarray(pc_scaler.mean_[:n_pcs], float), pc_weights)
                      - np.dot(np.asarray(pca.mean_, float), standardized) - np.dot(mean / scale, standardized))
    return standardized / scale, standardized, pc_weights, intercept


def fit_full_model(x: np.ndarray, y: np.ndarray, s: LassoPcrSettings = LassoPcrSettings()) -> FullModel:
    """Tune on the full sample (inner CV only) and fit the final model."""
    check_pc_grid(len(x), s)
    max_pcs = max(s.pc_grid)
    voxel_scaler = StandardScaler().fit(x)
    pca = PCA(n_components=max_pcs, svd_solver="randomized", random_state=s.random_seed + 900_000)
    pc = pca.fit_transform(voxel_scaler.transform(x).astype(np.float32, copy=False)).astype(np.float32, copy=False)
    pc_scaler = StandardScaler().fit(pc)
    pc_z = pc_scaler.transform(pc).astype(np.float32, copy=False)
    folds = inner_folds(x, s, s.random_seed + 910_000, s.random_seed + 920_000)
    n_pcs, alpha, inner_mse = select_hyperparameters(folds, y, s)
    lasso = fit_lasso(pc_z, y, n_pcs, alpha, s)
    raw, standardized, pc_weights, intercept = back_project(voxel_scaler, pca, pc_scaler, lasso, n_pcs)
    model = FullModel(voxel_scaler, pca, pc_scaler, lasso, n_pcs, alpha, inner_mse,
                      raw.astype(np.float64), standardized.astype(np.float64), pc_weights.astype(np.float64), intercept)
    difference = float(np.max(np.abs(model.predict(x) - (x.astype(np.float64) @ model.raw_weights + intercept))))
    print(f"Full model: PCs={n_pcs}, alpha={alpha:.4g}; back-projection max |diff| = {difference:.3g}")
    if difference > BACK_PROJECTION_TOLERANCE:
        raise RuntimeError(f"Back-projected weights do not reproduce the fitted pipeline (max diff {difference:.3g}).")
    return model


def signature_scores(model: FullModel, matrices: dict[str, np.ndarray], table: pd.DataFrame) -> pd.DataFrame:
    """Raw score = image . raw weights, plus z-scored score and the linear prediction (with intercept)."""
    out = table[["sub_id", *[c.rating for c in HEAT_CONDITIONS]]].copy()
    for cond in HEAT_CONDITIONS:
        x = matrices[cond.contrast]
        score = x.astype(np.float64) @ model.raw_weights
        linear = score + model.intercept
        if np.max(np.abs(linear - model.predict(x))) > BACK_PROJECTION_TOLERANCE:
            raise RuntimeError(f"con{cond.contrast}: back-projected score + intercept differs from the pipeline prediction.")
        sd = float(np.std(score, ddof=1))
        out[f"signature_score_con{cond.contrast}"] = score
        out[f"signature_score_z_con{cond.contrast}"] = (score - score.mean()) / sd if sd > 0 else np.nan
        out[f"full_model_linear_prediction_con{cond.contrast}"] = linear
    return out


def insample_metrics(model: FullModel, x: np.ndarray, table: pd.DataFrame, bounds=(0.0, 100.0)):
    """Resubstitution fit of the final model (descriptive; NOT out-of-sample performance)."""
    obs = table[PRIMARY.rating].to_numpy(float)
    raw = model.predict(x)
    preds = table[["sub_id", PRIMARY.rating]].copy()
    preds["fullsample_pred_raw_con0053"] = raw
    preds["fullsample_pred_bounded_con0053"] = np.clip(raw, *bounds)
    rows = [{"evaluation": "fullsample_insample", "contrast": PRIMARY.contrast, "condition": PRIMARY.label,
             "rating": PRIMARY.rating, "prediction_scale": scale, **prediction_metrics(obs, p),
             "insample_r2": r2_manual(obs, p)}
            for scale, p in (("raw", raw), ("bounded", np.clip(raw, *bounds)))]
    return pd.DataFrame(rows), preds


# --------------------------------------------------------------------------- #
# Associations
# --------------------------------------------------------------------------- #
def transfer_regressions(scores: pd.DataFrame) -> pd.DataFrame:
    """Signature score of each non-training contrast vs. its own pain rating."""
    rows = []
    for cond in HEAT_CONDITIONS[1:]:
        column = f"signature_score_con{cond.contrast}"
        x, y = scores[column].to_numpy(float), scores[cond.rating].to_numpy(float)
        a = association_stats(x, y)
        rows.append({"contrast": cond.contrast, "condition": cond.label, "rating": cond.rating, "score": column,
                     "n": a.pop("n"), "score_mean": float(x.mean()), "score_sd": float(x.std(ddof=1)),
                     "rating_mean": float(y.mean()), "rating_sd": float(y.std(ddof=1)), **a})
    return pd.DataFrame(rows)


def fullsample_associations(data: pd.DataFrame) -> pd.DataFrame:
    """The primary-contrast score of NPS and LASSO-PCR against each of the four heat ratings."""
    rows = []
    for model, column in (("NPS", f"nps_con_{PRIMARY.contrast}"), ("LASSO-PCR", f"signature_score_con{PRIMARY.contrast}")):
        for cond in HEAT_CONDITIONS:
            a = association_stats(data[column], data[cond.rating])
            for key in ("ols_slope_t", "ols_slope_df"):
                a.pop(key)
            rows.append({"model": model, "score_contrast": PRIMARY.contrast, "score_condition": PRIMARY.label,
                         "rating_condition_contrast": cond.contrast, "condition": cond.label, "site": cond.site,
                         "rating": cond.rating, "score_column": column, **a})
    return pd.DataFrame(rows)


def score_correspondence(data: pd.DataFrame) -> pd.DataFrame:
    """NPS vs LASSO-PCR score, per condition."""
    rows = []
    for cond in HEAT_CONDITIONS:
        x_col, y_col = f"nps_con_{cond.contrast}", f"signature_score_con{cond.contrast}"
        a = association_stats(data[x_col], data[y_col])
        for key in ("ols_slope_t", "ols_slope_df"):
            a.pop(key)
        rows.append({"contrast": cond.contrast, "condition": cond.label, "site": cond.site,
                     "x_column": x_col, "y_column": y_col, **a})
    return pd.DataFrame(rows)


def nps_cross_validate(data: pd.DataFrame, assignments: pd.DataFrame):
    """NPS 'CV' on the LASSO-PCR outer folds: only a linear calibration (score -> rating) is fitted
    on the training folds. Returns (per-repeat predictions, averaged predictions, metrics)."""
    if set(data["subject_key"]) != set(assignments["subject_key"]):
        missing = sorted(set(data["subject_key"]) - set(assignments["subject_key"]))
        extra = sorted(set(assignments["subject_key"]) - set(data["subject_key"]))
        raise DataError(f"NPS sample differs from the saved LASSO-PCR fold assignments. "
                        f"Missing from assignments: {missing}; extra in assignments: {extra}")
    keys = data["subject_key"].tolist()
    position = {k: i for i, k in enumerate(keys)}
    x = data[f"nps_con_{PRIMARY.contrast}"].to_numpy(float)
    y = data[PRIMARY.rating].to_numpy(float)
    repeats = sorted(pd.unique(assignments["repeat"]))
    predictions = np.full((len(repeats), len(data)), np.nan)
    rows = []
    for r, repeat in enumerate(repeats):
        in_repeat = assignments[assignments["repeat"] == repeat]
        tested: set = set()
        for fold in sorted(pd.unique(assignments["outer_fold"])):
            test_keys = set(in_repeat.loc[in_repeat["outer_fold"] == fold, "subject_key"])
            if not test_keys or tested & test_keys:
                raise DataError(f"Invalid fold assignment in repeat {repeat}, fold {fold}.")
            tested |= test_keys
            test = np.array([position[k] for k in test_keys])
            train = np.array([i for i, k in enumerate(keys) if k not in test_keys])
            line = stats.linregress(x[train], y[train])
            predictions[r, test] = line.intercept + line.slope * x[test]
        if tested != set(keys):
            raise DataError(f"Not every participant was tested exactly once in repeat {repeat}.")
        rows += [{"subject_key": keys[i], "repeat": int(repeat), PRIMARY.rating: float(y[i]),
                  "nps_con0053": float(x[i]), "pred_raw_nps_con0053": float(predictions[r, i])} for i in range(len(keys))]
    mean_pred = predictions.mean(axis=0)
    averaged = data[["subject_key", PRIMARY.rating]].copy()
    averaged["nps_con0053"], averaged["pred_raw_nps_con0053"] = x, mean_pred
    m = prediction_metrics(y, mean_pred)
    metrics = pd.DataFrame([{"model": "NPS", "contrast": PRIMARY.contrast, "condition": PRIMARY.label,
                             "rating": PRIMARY.rating, "prediction_scale": "raw",
                             "cv_design": "same repeated 5x5 outer folds; foldwise OLS calibration only",
                             "n": m["n"], "pearson_r": m["pearson_r"],
                             "pearson_p_parametric_descriptive": m["pearson_p_parametric_descriptive"],
                             "spearman_rho": m["spearman_rho"],
                             "spearman_p_parametric_descriptive": m["spearman_p_parametric_descriptive"],
                             "cross_validated_r2": r2_manual(y, mean_pred), "mae": m["mae"], "rmse": m["rmse"]}])
    return pd.DataFrame(rows), averaged, metrics


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_effective_mask(project: Project, reference):
    """Feature mask = Hoeppli high-heat map ∩ group mask, on the grid of the contrast images."""
    maps = {}
    for label, path in (("Hoeppli mask", project.hoeppli_mask), ("group mask", project.group_mask)):
        img = load_3d(path, label)
        img = img if same_grid(img, reference) else resample_to(img, reference)
        maps[label] = np.asarray(img.get_fdata(), float) > 0
    effective = maps["Hoeppli mask"] & maps["group mask"]
    print(f"Mask voxels | Hoeppli={int(maps['Hoeppli mask'].sum()):,}; group={int(maps['group mask'].sum()):,}; "
          f"intersection={int(effective.sum()):,}")
    if int(effective.sum()) < 100:
        raise DataError("The Hoeppli x group feature mask has fewer than 100 voxels; check that both masks are in the same space.")
    save_volume(effective, reference, project.lasso_mask, dtype=np.uint8)
    return load_3d(project.lasso_mask, "effective mask")


def _stage_done(files, overwrite: bool) -> bool:
    return not overwrite and all(f.is_file() for f in files)


def run_lasso_pcr(project: Project, s: LassoPcrSettings = LassoPcrSettings(), overwrite: bool = False) -> pd.DataFrame:
    """Nested CV, final model, signature scores and transfer regressions of Figure 4.

    Stages are skipped when their outputs exist (``overwrite=True`` recomputes everything).
    """
    import joblib
    project.require("lasso")
    out = project.f4
    out.mkdir(parents=True, exist_ok=True)
    table = read_table(project.subjects, "Imaging sample table", sep="\t").sort_values("sub_id").reset_index(drop=True)
    image_columns = {c.contrast: f"image_con{c.contrast}" for c in HEAT_CONDITIONS}
    missing = [c for c in image_columns.values() if c not in table.columns]
    if missing:
        raise DataError(f"{project.subjects} lacks image-path columns {missing}; re-run the Figure 2 sample step.")
    cv_files = [out / f"lasso_cv_{n}.tsv" for n in ("repeat", "average", "metrics", "tuning", "folds")]
    model_files = [out / n for n in ("lasso_model.joblib", "lasso_weights_raw.nii.gz", "lasso_weights_standardized.nii.gz",
                                     "lasso_pc_weights.tsv", "lasso_model.json", "lasso_insample_metrics.tsv",
                                     "con0053_fullsample_insample_predictions.tsv", "lasso_scores_heat.tsv")]
    do_cv, do_model = not _stage_done(cv_files, overwrite), not _stage_done(model_files, overwrite)
    if do_cv or do_model:
        reference = load_3d(Path(table.loc[0, image_columns[PRIMARY.contrast]]), "first contrast image")
        mask = build_effective_mask(project, reference)
        geometry = MaskGeometry(mask)
        matrices = {}
        for cond in HEAT_CONDITIONS:
            print(f"Loading con{cond.contrast} ({cond.label}) ...")
            matrices[cond.contrast] = geometry.load_matrix([Path(p) for p in table[image_columns[cond.contrast]]],
                                                           f"con{cond.contrast} image")
        x, y = matrices[PRIMARY.contrast], table[PRIMARY.rating].to_numpy(float)
        if not np.isfinite(y).all():
            raise DataError(f"{PRIMARY.rating} contains missing values in {project.subjects}.")
        _save_settings(project, s, len(table), geometry.n_voxels)
    if do_cv:
        print("\nNested repeated CV: con0053 -> pain_48_arm")
        repeat, average, tuning, folds = cross_validate(x, y, table["sub_id"].to_numpy(), s)
        for name, frame in (("repeat", repeat), ("average", average), ("metrics", cv_metrics(average)),
                            ("tuning", tuning), ("folds", folds)):
            frame.to_csv(out / f"lasso_cv_{name}.tsv", sep="\t", index=False)
    if do_model:
        print("\nFinal full-sample model")
        model = fit_full_model(x, y, s)
        joblib.dump(model, out / "lasso_model.joblib", compress=3)
        geometry.save(model.raw_weights, project.lasso_raw_weights)
        geometry.save(model.standardized_weights, out / "lasso_weights_standardized.nii.gz")
        pd.DataFrame({"pc": np.arange(1, model.n_pcs + 1), "lasso_weight_standardized_pc": model.lasso.coef_,
                      "weight_raw_pc_score": model.raw_pc_weights,
                      "pca_explained_variance_ratio": model.pca.explained_variance_ratio_[: model.n_pcs]}
                     ).to_csv(out / "lasso_pc_weights.tsv", sep="\t", index=False)
        (out / "lasso_model.json").write_text(json.dumps({
            "n_subjects": len(table), "training_contrast": PRIMARY.contrast, "training_rating": PRIMARY.rating,
            "selected_n_pcs": int(model.n_pcs), "selected_alpha": float(model.alpha), "inner_cv_mse": float(model.inner_mse),
            "raw_space_intercept": model.intercept, "n_mask_voxels": geometry.n_voxels,
            "n_nonzero_raw_weights": int(np.count_nonzero(model.raw_weights))}, indent=2))
        metrics, predictions = insample_metrics(model, x, table, s.rating_bounds)
        metrics.to_csv(out / "lasso_insample_metrics.tsv", sep="\t", index=False)
        predictions.to_csv(out / "con0053_fullsample_insample_predictions.tsv", sep="\t", index=False)
        scores = signature_scores(model, matrices, table)
        scores.to_csv(project.lasso_scores, sep="\t", index=False)
    scores = read_table(project.lasso_scores, "LASSO-PCR scores", sep="\t")
    transfer = transfer_regressions(scores)
    transfer.to_csv(out / "lasso_transfer_stats.tsv", sep="\t", index=False)
    print("\nCV metrics:\n" + read_table(out / "lasso_cv_metrics.tsv", "CV metrics", sep="\t").to_string(index=False))
    print("\nTransfer regressions:\n" + transfer[["condition", "rating", "n", "pearson_r"]].to_string(index=False))
    return transfer


def _save_settings(project: Project, s: LassoPcrSettings, n: int, n_voxels: int) -> None:
    info = {"n_subjects": n, "n_mask_voxels": n_voxels, "training_contrast": PRIMARY.contrast,
            "training_rating": PRIMARY.rating, "outer_folds": s.outer_folds, "outer_repeats": s.outer_repeats,
            "inner_folds": s.inner_folds, "pc_grid": list(s.pc_grid), "alpha_grid": alpha_grid(s).tolist(),
            "lasso_max_iter": s.max_iter, "lasso_tol": s.tol, "random_seed": s.random_seed,
            "feature_mask": "Hoeppli high-heat union intersected with the group mask",
            "signature_score": "raw masked contrast vector . back-projected raw weights (no intercept)",
            "known_limitation": "The same participants contribute con0053 training data and the transfer-regression "
                                "images, so transfer regressions are not fully independent out-of-sample tests."}
    (project.f4 / "settings.json").write_text(json.dumps(info, indent=2))
