"""End-to-end smoke test of Figure 1 on synthetic input files (no imaging libraries needed)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
from painpred.config import Project
from painpred.figures import fig1


def make_inputs(root):
    rng = np.random.default_rng(0)
    n = 40
    ids = [f"sub-{i:03d}" for i in range(1, n + 1)]
    means = pd.DataFrame({"sub_id": ids})
    trials = pd.DataFrame({"sub_num": range(1, n + 1)})
    for t in (46, 48):
        for site in ("arm", "leg"):
            base = 40 + (t - 46) * 8 + (5 if site == "leg" else 0) + rng.normal(0, 8, n)
            means[f"pain_{t}_{site}"] = base
            for r in range(1, 13):
                trials[f"pain_{t}_{site}_rep{r:02d}"] = base + rng.normal(0, 6, n)
    means.to_csv(root / "sub_by_condition_mean_ratings.tsv", sep="\t", index=False)
    (root / "bids/behav").mkdir(parents=True)
    trials.to_csv(root / "bids/behav/pain_rating_wide_complete.csv", index=False)
    cal = pd.DataFrame([{"subject_id": i, "target_vas": v, "estimated_temperature_c": 43 + v / 10 + rng.normal(0, 1)}
                        for i in ids for v in (50, 60, 70, 80)])
    cal.to_csv(root / "bids/behav/revisitpain_calib_summary.csv", index=False)


def test_figure1(tmp_path):
    make_inputs(tmp_path)
    project = Project(tmp_path, tmp_path / "out")
    data = fig1.run(project)
    assert len(data) == 40
    assert (tmp_path / "out/figures/Figure1.png").exists()


def test_missing_inputs_give_helpful_error(tmp_path):
    import pytest

    from painpred.utils import MissingInputError
    with pytest.raises(MissingInputError, match="trial ratings"):
        Project(tmp_path, tmp_path / "out").require("figure1")
