"""Errors, subject-ID handling, table reading and small statistics helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:  # progress bars are optional
    from tqdm.auto import tqdm as progress
except ImportError:  # pragma: no cover
    def progress(iterable=None, **_):
        return iterable


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class PipelineError(RuntimeError):
    """Base class for errors raised by this package."""


class MissingInputError(PipelineError, FileNotFoundError):
    """A required input file or directory does not exist."""


class DataError(PipelineError, ValueError):
    """An input exists but its content is not what the analysis expects."""


# --------------------------------------------------------------------------- #
# Subject IDs and tables
# --------------------------------------------------------------------------- #
SUBJECT_COLUMNS = ("sub_id", "subject_id", "subject", "participant_id", "participant", "sub_num", "sub")


def canonical_subject_id(value) -> str | float:
    """'sub-7', 7, 7.0 and '007' all become '7'; non-numeric IDs are lower-cased."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.casefold().replace("sub-", "").replace("sub_", "").strip()
    return str(int(text)) if text.isdigit() else text


def subject_number(value) -> int:
    return int(canonical_subject_id(value))


def _squash(text) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


def find_subject_column(table: pd.DataFrame, label: str) -> str:
    for column in SUBJECT_COLUMNS:
        if column in table.columns:
            return column
    squashed = {_squash(c): c for c in table.columns}
    for candidate in SUBJECT_COLUMNS:
        if _squash(candidate) in squashed:
            return squashed[_squash(candidate)]
    raise DataError(f"{label}: no subject-ID column found. Columns: {list(table.columns)}")


def read_table(path: Path, label: str, sep: str | None = "auto", hint: str = "") -> pd.DataFrame:
    """Read a delimited file with a clear error if it is missing or unreadable.

    ``sep='auto'`` lets pandas sniff the delimiter; otherwise pass ``'\\t'`` or ``','``.
    """
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(f"{label} not found: {path}" + (f"\n  {hint}" if hint else ""))
    try:
        if sep == "auto":
            return pd.read_csv(path, sep=None, engine="python")
        return pd.read_csv(path, sep=sep)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise DataError(f"{label} could not be parsed ({path}): {exc}") from exc


def require_columns(table: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise DataError(
            f"{label} is missing column(s) {missing}. Available: {list(table.columns)}"
        )


def add_subject_key(table: pd.DataFrame, label: str, column: str | None = None) -> pd.DataFrame:
    """Return a copy with a normalised ``subject_key``; duplicates raise an error."""
    table = table.copy()
    column = column or find_subject_column(table, label)
    table["subject_key"] = table[column].map(canonical_subject_id)
    duplicated = table.loc[table["subject_key"].duplicated(keep=False), "subject_key"]
    if not duplicated.empty:
        raise DataError(
            f"{label} has duplicate subjects after ID normalisation: "
            f"{sorted(duplicated.unique())[:20]}"
        )
    return table


def drop_excluded(table: pd.DataFrame, excluded: Iterable[str]) -> pd.DataFrame:
    return table.loc[~table["subject_key"].isin(set(excluded))].copy()


def complete_cases(table: pd.DataFrame, columns: Sequence[str], label: str = "table") -> pd.DataFrame:
    """Keep rows that are finite in all ``columns`` and report who was dropped."""
    values = table[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(values).all(axis=1)
    if not keep.all():
        dropped = table.loc[~keep, "subject_key"].tolist() if "subject_key" in table else "?"
        print(f"{label}: dropped {int((~keep).sum())} participant(s) with missing values: {dropped}")
    return table.loc[keep].reset_index(drop=True)


def resolve_nps_column(table: pd.DataFrame, contrast: str) -> str:
    """Find the NPS column of a contrast, tolerating naming variants (nps_con_0053, NPS_con0053...)."""
    contrast = str(int(contrast)).zfill(4)
    for name in (f"nps_con_{contrast}", f"nps_con{contrast}", f"NPS_con_{contrast}",
                 f"NPS_con{contrast}", f"con_{contrast}_nps", f"con{contrast}_nps"):
        if name in table.columns:
            return name
    matches = [c for c in table.columns if "nps" in _squash(c) and contrast in _squash(c)]
    if len(matches) == 1:
        return matches[0]
    raise DataError(
        f"Cannot identify the NPS column for con{contrast} "
        f"({len(matches)} candidates: {matches}). Columns: {list(table.columns)}"
    )


def all_exist(paths: Iterable[Path]) -> bool:
    return all(Path(p).exists() for p in paths)


def check_sample_size(n: int, expected: int | None, label: str) -> None:
    if expected is not None and n != expected:
        raise DataError(
            f"{label}: expected n={expected} participants but found n={n}. "
            "Check the exclusions and missing data; if the new sample size is intended, "
            "change the expected value in config.py."
        )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _finite_pairs(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def safe_pearson(x, y) -> tuple[float, float]:
    x, y = _finite_pairs(x, y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def safe_spearman(x, y) -> tuple[float, float]:
    x, y = _finite_pairs(x, y)
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def r2_manual(y, prediction) -> float:
    """1 - SSE/SST (can be negative for out-of-sample predictions)."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    total = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - prediction) ** 2) / total) if total > 0 else np.nan


def association_stats(x, y) -> dict:
    """Pearson, Spearman and simple OLS of y on x (the row format used in several tables)."""
    x, y = _finite_pairs(x, y)
    pearson, spearman, linear = stats.pearsonr(x, y), stats.spearmanr(x, y), stats.linregress(x, y)
    slope_t = linear.slope / linear.stderr if np.isfinite(linear.stderr) and linear.stderr > 0 else np.nan
    return {
        "n": int(len(x)),
        "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
        "ols_r2": float(linear.rvalue**2), "ols_intercept": float(linear.intercept),
        "ols_slope": float(linear.slope), "ols_slope_se": float(linear.stderr),
        "ols_slope_t": float(slope_t), "ols_slope_df": int(len(x) - 2),
        "ols_slope_p": float(linear.pvalue),
    }
