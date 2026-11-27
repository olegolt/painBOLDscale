"""Build the shareable ``data/`` folder from the private results.

Run this once, privately. It writes only group-level maps and tables with one row per participant, with
pseudonymised IDs. Contrast images, original IDs, file paths and settings that list excluded participants are
never copied. A fixed whitelist of files and columns decides what leaves; anything else is left behind.

The pseudonym of participant *i* is ``sub-p<rank>``, where the rank follows the sort order of the original IDs
(as strings, the order the analysis code uses). Row order, and therefore every seeded cross-validation split,
is then the same as with the original IDs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .config import HEAT_CONDITIONS, MASSUNI_RATINGS, RATING_COLUMNS, Project
from .utils import (
    SUBJECT_COLUMNS,
    DataError,
    MissingInputError,
    PipelineError,
    canonical_subject_id,
    read_table,
    require_columns,
    resolve_nps_column,
)

CONTRASTS_NPS = ("0053", "0054", "0055", "0056", "0048", "0049")
MAP_FILES = ("rho", "z", "p_fwe", "rho_fwe", "z_fwe", "rho_z3p1", "z_z3p1", "rho_cluster_fwe", "z_cluster_fwe")
MAP_TABLES = ("clusters.tsv", "analysis_summary.tsv")
LASSO_FILES = ("lasso_cv_repeat.tsv", "lasso_cv_average.tsv", "lasso_cv_metrics.tsv", "lasso_cv_tuning.tsv", "lasso_cv_folds.tsv",
               "lasso_insample_metrics.tsv", "lasso_pc_weights.tsv", "lasso_model.json", "lasso_scores_heat.tsv",
               "lasso_transfer_stats.tsv", "con0053_fullsample_insample_predictions.tsv", "settings.json",
               "lasso_weights_raw.nii.gz", "lasso_weights_standardized.nii.gz", "lasso_mask.nii.gz")
PATH_COLUMNS = ("output_directory", "comparison_directory", "path", "source_map", "significance_map", "binary_map")
DROP_COLUMNS = ("subject_key", "sub_num_int")


# --------------------------------------------------------------------------- #
# Pseudonyms
# --------------------------------------------------------------------------- #
def build_id_map(original_ids) -> dict[str, str]:
    """{canonical original ID: 'sub-pNNN'}, ranked by the string order of the canonical IDs."""
    canonical = sorted({str(canonical_subject_id(i)) for i in original_ids if pd.notna(i)})
    width = max(3, len(str(len(canonical))))
    return {orig: f"sub-p{rank:0{width}d}" for rank, orig in enumerate(canonical, start=1)}


def looks_like_path(value) -> bool:
    return isinstance(value, str) and (value.startswith(("/", "~")) or ":\\" in value or "/home/" in value)


def drop_path_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Remove every column that holds file-system paths; they reveal folder names and original IDs."""
    paths = [c for c in table.columns if table[c].map(looks_like_path).any()]
    return table.drop(columns=paths)


def pseudonymise(table: pd.DataFrame, id_map: dict[str, str], label: str) -> pd.DataFrame:
    """Replace every subject-ID column; drop path and helper columns; refuse unmapped IDs."""
    table = table.copy()
    if "subject_key" in table.columns and not any(c in table.columns for c in SUBJECT_COLUMNS):
        table.insert(0, "participant_id", table["subject_key"])  # tables that carry only the internal key
    table = drop_path_columns(table.drop(columns=[c for c in (*PATH_COLUMNS, *DROP_COLUMNS) if c in table.columns]))
    found = [c for c in SUBJECT_COLUMNS if c in table.columns]
    if not found:
        raise DataError(f"{label}: no subject-ID column to pseudonymise. Columns: {list(table.columns)}")
    for column in found:
        keys = table[column].map(lambda v: str(canonical_subject_id(v)))
        unknown = sorted(set(keys) - set(id_map))
        if unknown:
            raise DataError(f"{label}: IDs {unknown[:5]} are not in the participant list.")
        table[column] = keys.map(id_map)
    return table.sort_values(found[0]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, float_format="%.10g")


def copy_image(source: Path, target: Path) -> None:
    """Copy a NIfTI with a fresh header (no description, auxiliary-file or other free-text fields)."""
    import nibabel as nib
    img = nib.load(str(source))
    clean = nib.Nifti1Image(img.get_fdata(dtype="float64").astype(img.get_data_dtype()), img.affine)
    clean.set_data_dtype(img.get_data_dtype())
    clean.header.set_xyzt_units("mm")
    clean.header["descrip"] = b""
    target.parent.mkdir(parents=True, exist_ok=True)
    name = target.name if target.name.endswith(".nii.gz") else target.name + ".gz"
    nib.save(clean, str(target.with_name(name)))


def strip_paths(value):
    """Recursively replace absolute paths inside JSON content by the bare file name."""
    if isinstance(value, dict):
        return {k: strip_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_paths(v) for v in value]
    return Path(value).name if looks_like_path(value) else value


def copy_if_exists(source: Path, target: Path, missing: list[str]) -> None:
    if not source.exists():
        missing.append(str(source))
    elif source.suffix == ".json":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(strip_paths(json.loads(source.read_text(encoding="utf-8"))), indent=2), encoding="utf-8")
    elif source.name.endswith((".nii", ".nii.gz")):
        copy_image(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def sanitized_table(source: Path, target: Path, id_map: dict[str, str], missing: list[str], keep=None, sep="\t") -> None:
    if not source.exists():
        missing.append(str(source))
        return
    table = read_table(source, source.name, sep=sep)
    if any(c in table.columns for c in (*SUBJECT_COLUMNS, "subject_key")):
        table = pseudonymise(table, id_map, source.name)
    else:
        table = drop_path_columns(table.drop(columns=[c for c in PATH_COLUMNS if c in table.columns]))
    if keep:
        table = table[[c for c in table.columns if keep(c)]]
    write_table(table, target)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def phenotype_tables(project: Project) -> dict[str, pd.DataFrame]:
    """The four participant-level input tables, whitelisted and with excluded participants removed."""
    sites, temperatures = ("arm", "leg"), ("46", "48")
    rating_cols = [f"pain_{t}_{s}" for s in sites for t in temperatures]
    means = read_table(project.mean_ratings, "Mean-rating table", sep="\t")
    require_columns(means, ["sub_id", *rating_cols], "Mean-rating table")
    trials = read_table(project.trial_ratings, "Trial-rating table")
    require_columns(trials, ["sub_num"], "Trial-rating table")
    trial_cols = [c for c in trials.columns if any(c.startswith(f"pain_{t}_{s}_rep") for s in sites for t in temperatures)]
    calibration = read_table(project.calibration, "Calibration table")
    require_columns(calibration, ["subject_id", "target_vas", "estimated_temperature_c"], "Calibration table")
    nps = read_table(project.nps_scores, "NPS score table")
    nps_cols = {c: resolve_nps_column(nps, c) for c in CONTRASTS_NPS}
    nps_id = next(c for c in SUBJECT_COLUMNS if c in nps.columns)
    tables = {
        "ratings_mean": means[["sub_id", *rating_cols]],
        "ratings_trial": trials[["sub_num", *trial_cols]],
        "calibration": calibration[["subject_id", "target_vas", "estimated_temperature_c"]],
        "nps_scores": nps[[nps_id, *nps_cols.values()]].rename(columns={v: f"nps_con_{k}" for k, v in nps_cols.items()}).rename(
            columns={nps_id: "sub_id"}),
    }
    used = analysed_participants(project)
    for name, table in tables.items():
        column = next(c for c in SUBJECT_COLUMNS if c in table.columns)
        keys = table[column].map(lambda v: str(canonical_subject_id(v)))
        kept = table[keys.isin(used) & ~keys.isin(set(project.excluded))].reset_index(drop=True)
        print(f"{name}: {len(kept)} of {len(table)} rows kept (participants used in the paper)")
        tables[name] = kept
    return tables


def analysed_participants(project: Project) -> set[str]:
    """Canonical IDs of everyone any figure uses: the imaging sample and the Figure 1 behavioural sample."""
    from .figures.fig1 import behaviour_table

    imaging = read_table(project.subjects, "Imaging sample table", sep="\t")
    require_columns(imaging, ["sub_id"], "Imaging sample table")
    ids = {str(canonical_subject_id(v)) for v in imaging["sub_id"]}
    ids |= set(behaviour_table(project)["subject_key"].astype(str))
    return ids - set(project.excluded)


def export_public_data(project: Project, data_root, id_map_path=None, include_third_party: bool = False) -> dict[str, str]:
    """Write the shareable folder; returns {pseudonym: ''} summary information. Original IDs stay in ``id_map_path``."""
    if project.public:
        raise PipelineError("The export needs the private project (input_root=...), not the shared data.")
    data_root = Path(data_root).resolve()
    id_map_path = Path(id_map_path or project.results_root / "private_id_map.tsv").resolve()
    if data_root in id_map_path.parents:
        raise PipelineError(f"The ID map must stay outside the shared folder: {id_map_path}")
    missing = []
    out = data_root / "derivatives" / "painpred"

    tables = phenotype_tables(project)
    id_map = build_id_map(pd.concat([t[next(c for c in SUBJECT_COLUMNS if c in t.columns)] for t in tables.values()]))
    pd.DataFrame({"original_id": list(id_map), "participant_id": list(id_map.values())}).to_csv(id_map_path, sep="\t", index=False)
    print(f"{len(id_map)} participants; private ID map: {id_map_path}")

    for name, table in tables.items():
        write_table(pseudonymise(table, id_map, name), data_root / "phenotype" / f"{name}.tsv")
    write_table(pd.DataFrame({"participant_id": sorted(id_map.values())}), data_root / "participants.tsv")

    # Figure 2: group mask, sample table (ratings only), per-participant peak values, all 16 analyses
    copy_if_exists(project.group_mask, out / "figure2" / "group_mask.nii.gz", missing)
    sanitized_table(project.subjects, out / "figure2" / "subjects.tsv", id_map, missing,
                    keep=lambda c: c in ("sub_id", *RATING_COLUMNS))
    sanitized_table(project.f2 / "figure_peak_data.tsv", out / "figure2" / "figure_peak_data.tsv", id_map, missing)
    sanitized_table(project.f2 / "summary_all_analyses.tsv", out / "figure2" / "summary_all_analyses.tsv", id_map, missing)
    for contrast in sorted({c.contrast for c in HEAT_CONDITIONS}):
        for rating in MASSUNI_RATINGS:
            source, target = project.analysis_dir(contrast, rating), out / "figure2" / "maps" / f"con{contrast}" / rating
            for key in MAP_FILES:
                copy_if_exists(source / f"{key}.nii.gz", target / f"{key}.nii.gz", missing)
            for name in MAP_TABLES:
                sanitized_table(source / name, target / name, id_map, missing)
    for name in ("local_peaks.csv", "figure_peaks.tsv"):
        sanitized_table(project.f2 / name, out / "figure2" / name, id_map, missing, sep="," if name.endswith(".csv") else "\t")

    # Figure 4: LASSO-PCR results
    for name in LASSO_FILES:
        source, target = project.f4 / name, out / "figure4" / name
        if name.endswith(".tsv"):
            sanitized_table(source, target, id_map, missing)
        else:
            copy_if_exists(source, target, missing)

    # Figure 5, hoeppli mask, group-level maps from the same study
    sanitized_table(project.f5 / "subjects.tsv", out / "figure5" / "subjects.tsv", id_map, missing)
    copy_if_exists(project.hoeppli_mask, out / "shared" / "hoeppli" / "hoeppli_union_mask.nii.gz", missing)
    for source, name in ((project.same_study_48_mask, "48heat_voxelFWE05_pos_mask"), (project.sound_tmap, "sound_tmap"),
                         (project.checkerboard_tmap, "checkerboard_tmap")):
        copy_if_exists(source, out / "group_maps" / f"{name}.nii.gz", missing)

    # Supplement: warmth vs implicit baseline (tables and maps; no settings file, it lists excluded IDs)
    base = project.baseline / "warmth_vs_implicit_baseline"
    target_base = out / "supplementary" / "conventional_implicit_baseline" / "warmth_vs_implicit_baseline"
    for name in ("comparison_summary", "peak_comparisons", "whole_brain_distribution_summary"):
        sanitized_table(base / f"{name}.tsv", target_base / f"{name}.tsv", id_map, missing)
    for folder in sorted(p for p in base.glob("con*_vs_con*/*") if p.is_dir()):
        for image in folder.glob("*.nii.gz"):
            copy_if_exists(image, target_base / folder.relative_to(base) / image.name, missing)

    # Study template used as figure background
    copy_if_exists(project.t1_background, data_root / "external" / "templates" / "T1w_group_template.nii.gz", missing)
    if include_third_party:  # only if the licences allow redistribution
        for source, target in ((project.atlas_nii, "atlas/MNI_Asym_neuromorphometrics.nii.gz"),
                               (project.atlas_xml, "atlas/MNI_Asym_neuromorphometrics.xml")):
            copy_if_exists(source, data_root / "external" / target, missing)
        if project.nps_weights:
            copy_if_exists(project.nps_weights, data_root / "external" / "nps" / "nps_weights_in_study_space.nii.gz", missing)
    write_metadata(data_root)
    check_no_original_ids(data_root)
    if missing:
        shown = "\n  - ".join(missing[:12]) + (f"\n  ... and {len(missing) - 12} more" if len(missing) > 12 else "")
        print(f"{len(missing)} expected file(s) not found and skipped:\n  - {shown}")
    return id_map


def check_no_original_ids(data_root: Path) -> None:
    """Last line of defence: ID columns hold pseudonyms only, and no text file contains paths or original-style IDs."""
    import re

    leak = re.compile(r"sub[-_]\d{2,}|(?<![\w.])(/home/|/Users/|/mnt/|[A-Z]:\\)")
    for path in sorted(Path(data_root).rglob("*")):
        if path.suffix in (".tsv", ".csv", ".json", ".txt", ".xml") and path.is_file():
            hit = leak.search(path.read_text(encoding="utf-8", errors="ignore"))
            if hit:
                raise PipelineError(f"{path}: contains {hit.group(0)!r}. Nothing may be published.")
    id_columns = {*SUBJECT_COLUMNS, "subject_key", "sub_num_int"}
    for path in sorted(Path(data_root).rglob("*")):
        if path.suffix not in (".tsv", ".csv"):
            continue
        table = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", dtype=str)
        for column in id_columns & set(table.columns):
            values = table[column].dropna()
            bad = values[~values.str.startswith("sub-p")]
            if len(bad):
                raise PipelineError(f"{path.name}: column '{column}' still holds original IDs (e.g. {bad.iloc[0]!r}). Nothing may be published.")


def write_metadata(data_root: Path) -> None:
    """BIDS description files and data dictionaries."""
    (data_root / "derivatives/painpred").mkdir(parents=True, exist_ok=True)
    (data_root / "dataset_description.json").write_text(json.dumps({
        "Name": "Pain-evoked neural activity scales with subjective pain across individuals: shared data",
        "BIDSVersion": "1.9.0", "DatasetType": "derivative",
        "License": "CC-BY-4.0",
        "GeneratedBy": [{"Name": "painpred", "Version": "1.0.0"}],
        "Description": "Group-level maps and participant-level summary scores. No individual images are included."},
        indent=2), encoding="utf-8")
    (data_root / "derivatives/painpred/dataset_description.json").write_text(json.dumps({
        "Name": "painpred group-level derivatives", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "painpred", "Version": "1.0.0"}]}, indent=2), encoding="utf-8")
    (data_root / "participants.json").write_text(json.dumps({
        "participant_id": {"Description": "Pseudonym (sub-p<rank>). The rank follows the order of the original IDs."}}, indent=2),
        encoding="utf-8")
    (data_root / ".bidsignore").write_text("derivatives/painpred/figure*/\nderivatives/painpred/shared/\n"
                                           "derivatives/painpred/group_maps/\nderivatives/painpred/supplementary/\n",
                                           encoding="utf-8")


__all__ = ["MissingInputError", "build_id_map", "export_public_data", "pseudonymise"]
