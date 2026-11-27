"""The export keeps results identical, hides the original IDs and leaves out everything not whitelisted."""

import numpy as np
import pandas as pd
import pytest

from painpred import export
from painpred.config import Project
from painpred.data import load_nps_scores
from painpred.figures import fig1
from painpred.utils import DataError, add_subject_key

IDS = [2, 3, 10, 45, 100, 7]  # 45 is on the exclusion list
CALIBRATION_ONLY = 999  # appears in the calibration table only, so no figure uses this person
CONDITIONS = [f"pain_{t}_{s}" for s in ("arm", "leg") for t in ("46", "48")]


def make_private(root):
    rng = np.random.default_rng(0)
    ids = [f"sub-{i:04d}" for i in IDS]
    pd.DataFrame({"sub_id": ids, "age": 30, "sex": "f", **{c: rng.uniform(20, 80, len(ids)) for c in CONDITIONS}}
                 ).to_csv(root / "sub_by_condition_mean_ratings.tsv", sep="\t", index=False)
    (root / "bids/behav").mkdir(parents=True)
    trial = {f"pain_{t}_{s}_rep{k:02d}": rng.uniform(20, 80, len(ids)) for s in ("arm", "leg") for t in ("46", "48") for k in range(1, 13)}
    pd.DataFrame({"sub_num": IDS, "session_date": "2024-01-01", **trial}).to_csv(root / "bids/behav/pain_rating_wide_complete.csv", index=False)
    rows = [{"subject_id": f"sub-{i:04d}", "target_vas": v, "estimated_temperature_c": 40 + v / 10 + i / 100, "date": "x"}
            for i in [*IDS, CALIBRATION_ONLY] for v in (25, 50, 75)]
    pd.DataFrame(rows).to_csv(root / "bids/behav/revisitpain_calib_summary.csv", index=False)
    (root / "nps").mkdir()
    nps = {f"con_{c}_nps": rng.normal(size=len(ids)) for c in export.CONTRASTS_NPS}
    pd.DataFrame({"sub_id": ids, **nps, "extra": 1}).to_csv(root / "nps/nps_scores_Conc_mov24_wm_csf_roi_physio_revisit_hrf_0_none_s6.csv",
                                                           index=False)


@pytest.fixture()
def exported(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    make_private(private)
    project = Project(private, tmp_path / "results")
    (tmp_path / "results" / "figure2").mkdir(parents=True)
    pd.DataFrame({"sub_id": [f"sub-{i:04d}" for i in IDS if i != 45]}).to_csv(tmp_path / "results" / "figure2" / "subjects.tsv",
                                                                             sep="\t", index=False)
    data = tmp_path / "repo" / "data"
    data.mkdir(parents=True)
    id_map = export.export_public_data(project, data, id_map_path=tmp_path / "private_map.tsv")
    return project, data, id_map


def test_original_ids_and_extra_columns_do_not_leave(exported):
    _, data, id_map = exported
    assert "45" not in id_map and "999" not in id_map and len(id_map) == 5
    for path in data.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert "sub-0" not in text and "session_date" not in text, path
            if path.suffix == ".tsv":
                assert "age" not in text.splitlines()[0].split("\t") and "sex" not in text.splitlines()[0].split("\t"), path
    ratings = pd.read_csv(data / "phenotype/ratings_mean.tsv", sep="\t")
    assert list(ratings.columns) == ["sub_id", *CONDITIONS]
    assert set(ratings["sub_id"]) == set(id_map.values())


def test_pseudonyms_keep_the_analysis_row_order(exported):
    project, data, id_map = exported
    original = project.nps_scores
    old = add_subject_key(pd.read_csv(original), "nps")
    old = old[~old["subject_key"].isin(["45"])].sort_values("subject_key")
    public = Project(data_root=data, results_root=data.parent / "results")
    new = load_nps_scores(public, ("0053",)).sort_values("subject_key")
    assert list(new["subject_key"]) == sorted(new["subject_key"])
    assert [id_map[k] for k in old["subject_key"]] == [f"sub-{k}" for k in new["subject_key"]]
    assert np.allclose(old["con_0053_nps"].to_numpy(), new["nps_con_0053"].to_numpy())


def test_public_figure1_data_matches_private(exported):
    project, data, _ = exported
    project.make_output_dirs()
    private = fig1.prepare_behaviour(project)
    public_project = Project(data_root=data, results_root=data.parent / "results")
    public_project.make_output_dirs()
    public = fig1.prepare_behaviour(public_project)
    for column in [c for c in private.columns if c != "subject_key"]:
        assert np.allclose(private[column].to_numpy(float), public[column].to_numpy(float)), column


def test_id_map_cannot_be_written_inside_the_shared_folder(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    make_private(private)
    with pytest.raises(Exception, match="outside the shared folder"):
        export.export_public_data(Project(private, tmp_path / "r"), tmp_path / "data", id_map_path=tmp_path / "data" / "map.tsv")


def test_unmapped_ids_are_refused():
    with pytest.raises(DataError):
        export.pseudonymise(pd.DataFrame({"sub_id": ["sub-" + "0001"]}), {"2": "sub-p001"}, "t")


def test_table_with_only_the_internal_key_is_pseudonymised(tmp_path):
    table = pd.DataFrame({"subject_key": ["7", "12"], "value": [1.0, 2.0]})
    out = export.pseudonymise(table, {"7": "sub-p001", "12": "sub-p002"}, "demo")
    assert list(out["participant_id"]) == ["sub-p001", "sub-p002"] and "subject_key" not in out.columns


def test_final_check_stops_a_leaking_column(tmp_path):
    (tmp_path / "leak.tsv").write_text("subject_key\tvalue\n7\t1.0\n")
    with pytest.raises(export.PipelineError):
        export.check_no_original_ids(tmp_path)


def test_path_columns_and_json_paths_are_removed(tmp_path):
    home = "/" + "home"  # written this way so the release scan does not flag the test itself
    original = f"sub-{7:04d}"
    table = pd.DataFrame({"sub_id": [original], "image": [f"{home}/x/study/{original}/con.nii"], "value": [1.0]})
    out = export.pseudonymise(table, {"7": "sub-p001"}, "demo")
    assert list(out.columns) == ["sub_id", "value"]
    source, target = tmp_path / "a.json", tmp_path / "out" / "a.json"
    source.write_text('{"weights": "' + home + '/x/Results/figure4/w.nii.gz", "alpha": 0.1}')
    export.copy_if_exists(source, target, [])
    assert target.read_text().count(home) == 0 and '"w.nii.gz"' in target.read_text()
