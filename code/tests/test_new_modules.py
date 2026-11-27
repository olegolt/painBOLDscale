"""Tests of pure helpers that need neither imaging libraries nor the original notebook."""

import numpy as np
import pandas as pd
import pytest

from painpred import atlas, overlap, peaks
from painpred.config import Project
from painpred.figures import fig3
from painpred.pipeline import STEPS, run_steps
from painpred.utils import MissingInputError, PipelineError


def test_project_without_input_root_explains_how_to_set_it(monkeypatch):
    monkeypatch.delenv("PAINPRED_INPUT_ROOT", raising=False)
    with pytest.raises(MissingInputError, match="PAINPRED_INPUT_ROOT"):
        Project()


def test_missing_inputs_are_listed_together(tmp_path):
    project = Project(tmp_path, tmp_path / "out")
    with pytest.raises(MissingInputError) as error:
        project.require("figure1")
    message = str(error.value)
    assert "mean ratings" in message and "trial ratings" in message and "calibration" in message


def test_unknown_step_is_rejected(tmp_path):
    with pytest.raises(PipelineError, match="Unknown step"):
        run_steps(Project(tmp_path, tmp_path / "out"), ["nope"])
    assert "figure1" in STEPS


def test_split_label():
    assert atlas.split_label("Right MSFG superior frontal gyrus medial segment") == (
        "Right", "MSFG", "superior frontal gyrus medial segment")
    assert atlas.split_label("Optic Chiasm")[0] == "Midline/other"


def test_overlap_metrics_counts():
    t, r = np.zeros(10, bool), np.zeros(10, bool)
    t[:4], r[2:6] = True, True
    m = overlap.overlap_metrics(t, r, np.ones(10, bool), 8.0)
    assert (m["overlap_voxels"], m["target_voxels"], m["reference_voxels"]) == (2, 4, 4)
    assert m["dice"] == pytest.approx(0.5) and m["jaccard"] == pytest.approx(2 / 6)
    assert m["descriptive_fold_enrichment"] == pytest.approx(2 / (4 * 4 / 10))


def test_separated_peaks_keep_strongest_and_respect_distance():
    effect = np.zeros((10, 1, 1))
    effect[1, 0, 0], effect[3, 0, 0], effect[9, 0, 0] = 0.9, 0.8, 0.5
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    chosen = peaks.select_separated([(1, 0, 0), (3, 0, 0), (9, 0, 0)], effect, effect, affine, min_distance_mm=8.0)
    assert chosen == [(1, 0, 0), (9, 0, 0)]  # (3,0,0) is only 4 mm from the stronger peak


def test_nps_regression_matches_normal_equations():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = 60 + 5 * x + rng.normal(size=40) * 8
    fit = fig3.nps_regression(pd.DataFrame({fig3.NPS_COLUMN: x, fig3.GRAND_MEAN: y}))
    design = np.column_stack([np.ones(40), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.allclose(fit["coefficients"]["coefficient"], beta)
    assert fit["r_squared"] == pytest.approx(1 - np.sum((y - design @ beta) ** 2) / np.sum((y - y.mean()) ** 2))


def test_max_count_reducer():
    samples = np.array([[0, 2, np.nan], [1, 4, np.nan]])
    out = fig3.max_count(samples)
    assert out[0] == 1 and out[1] == 4 and np.isnan(out[2])
