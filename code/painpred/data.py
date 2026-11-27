"""Assembling analysis tables from ratings, NPS scores and LASSO-PCR scores.

All analyses use the same participants (n = 101, see ``config.EXCLUDED_SUBJECTS``).
The imaging sample table written by Figure 2 (``<results>/figure2/subjects.tsv``) is the
single source of truth for who is in the sample and for each participant's ratings and
contrast-image paths.
"""

from __future__ import annotations

import pandas as pd

from .config import EXPECTED_N_IMAGING, HEAT_CONTRASTS, MASSUNI_RATINGS, RATING_COLUMNS, Project
from .utils import (
    add_subject_key,
    check_sample_size,
    complete_cases,
    drop_excluded,
    read_table,
    require_columns,
    resolve_nps_column,
)

MEAN_PAIN = "pain_mean_all_conditions"


def load_imaging_sample(project: Project) -> pd.DataFrame:
    """The common imaging sample (ratings + image paths), with the overall mean pain rating."""
    table = read_table(project.subjects, "Imaging sample table", sep="\t",
                       hint="Run the Figure 2 mass-univariate analysis first.")
    table = drop_excluded(add_subject_key(table, "Imaging sample table"), project.excluded)
    require_columns(table, RATING_COLUMNS, "Imaging sample table")
    table[MEAN_PAIN] = table[list(RATING_COLUMNS)].mean(axis=1, skipna=False)
    return complete_cases(table, [*RATING_COLUMNS, MEAN_PAIN], "Imaging sample")


def load_nps_scores(project: Project, contrasts: tuple[str, ...]) -> pd.DataFrame:
    """Fixed-NPS expression per participant, columns ``nps_con_<contrast>``."""
    nps = read_table(project.nps_scores, "NPS score table")
    nps = drop_excluded(add_subject_key(nps, "NPS score table"), project.excluded)
    out = nps[["subject_key"]].copy()
    for contrast in contrasts:
        source = resolve_nps_column(nps, contrast)
        out[f"nps_con_{contrast}"] = pd.to_numeric(nps[source], errors="coerce")
    return out


def load_lasso_scores(project: Project) -> pd.DataFrame:
    scores = read_table(project.lasso_scores, "LASSO-PCR score table", sep="\t",
                        hint="Run the LASSO-PCR analysis (Figure 4) first.")
    return add_subject_key(scores, "LASSO-PCR score table")


def merge_unique(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Inner join on ``subject_key`` that fails if the join is not one-to-one."""
    return left.merge(right, on="subject_key", how="inner", validate="one_to_one")


def load_analysis_table(project: Project, nps_contrasts: tuple[str, ...] = (),
                        lasso: bool = False, expected_n: int | None = EXPECTED_N_IMAGING
                        ) -> pd.DataFrame:
    """Ratings (+ NPS scores, + LASSO-PCR scores) for the complete-case imaging sample."""
    sample = load_imaging_sample(project)
    table = sample[["subject_key", *RATING_COLUMNS, MEAN_PAIN]]
    if nps_contrasts:
        table = merge_unique(table, load_nps_scores(project, nps_contrasts))
    if lasso:
        scores = load_lasso_scores(project)
        keep = ["subject_key"] + [c for c in scores.columns if c.startswith("signature_score_con")]
        table = merge_unique(table, scores[keep])
    table = complete_cases(table, [c for c in table.columns if c != "subject_key"], "Analysis table")
    check_sample_size(len(table), expected_n, "Analysis table")
    return table.sort_values("subject_key").reset_index(drop=True)


__all__ = ["MEAN_PAIN", "HEAT_CONTRASTS", "MASSUNI_RATINGS", "load_imaging_sample",
           "load_nps_scores", "load_lasso_scores", "merge_unique", "load_analysis_table"]
