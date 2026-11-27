"""Paths, analysis constants and experimental-condition definitions.

Everything that is a *choice* of the analysis lives here, so that the rest of the
package contains logic only. Paths are derived from two roots:

``input_root``    raw/derived study data (read-only)
``results_root``  everything this code writes (default: ``<input_root>/Results``)

Both can be given explicitly or through the environment variables
``PAINPRED_INPUT_ROOT`` and ``PAINPRED_RESULTS_ROOT``.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .utils import MissingInputError

# --------------------------------------------------------------------------- #
# Sample definition
# --------------------------------------------------------------------------- #
EXCLUDED_SUBJECTS = {
    "45": "Strong movement during scanning (coughing/panic)",
    "109": "Strong movement during scanning (coughing/panic)",
}
EXPECTED_N_IMAGING = 101  # participants with complete imaging + ratings

# --------------------------------------------------------------------------- #
# Experimental conditions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """One heat condition: first-level contrast (vs. 40 °C warmth) and its rating column."""

    contrast: str
    rating: str
    temperature: int
    site: str

    @property
    def label(self) -> str:
        return f"{self.temperature} °C {self.site}"


HEAT_CONDITIONS = (
    Condition("0053", "pain_48_arm", 48, "arm"),
    Condition("0054", "pain_46_arm", 46, "arm"),
    Condition("0055", "pain_48_leg", 48, "leg"),
    Condition("0056", "pain_46_leg", 46, "leg"),
)
HEAT_CONTRASTS = tuple(c.contrast for c in HEAT_CONDITIONS)
RATING_COLUMNS = tuple(c.rating for c in HEAT_CONDITIONS)
PRIMARY = HEAT_CONDITIONS[0]  # 48 °C arm: the training/primary condition everywhere
SOUND_CONTRAST, CHECKERBOARD_CONTRAST = "0048", "0049"

# Order used by the mass-univariate engine (kept for identical summary tables).
MASSUNI_RATINGS = ("pain_48_arm", "pain_48_leg", "pain_46_arm", "pain_46_leg")

# Implicit-baseline contrasts of the supplementary analysis, matched to heat contrasts.
BASELINE_CONTRAST = {"0053": "0010", "0054": "0011", "0055": "0016", "0056": "0017"}

# --------------------------------------------------------------------------- #
# Analysis settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MassUnivariateSettings:
    z_uncorrected: float = 3.1
    cluster_forming_z: float = 3.1
    alpha_voxel_fwe: float = 0.05
    alpha_cluster_fwe: float = 0.05
    n_permutations: int = 10_000
    batch_size: int = 100
    random_seed: int = 42
    mask_threshold_fraction: float = 0.15  # of the 95th percentile of the mean EPI
    mask_reference_percentile: float = 95.0
    cluster_connectivity: int = 3  # 26-neighbourhood


@dataclass(frozen=True)
class LassoPcrSettings:
    outer_folds: int = 5
    outer_repeats: int = 5
    inner_folds: int = 5
    pc_grid: tuple = (5, 10, 20, 30, 40, 50)
    alpha_min_log10: float = -3.0
    alpha_max_log10: float = 1.5
    n_alphas: int = 37
    max_iter: int = 100_000
    tol: float = 1e-5
    rating_bounds: tuple = (0.0, 100.0)
    random_seed: int = 42


@dataclass(frozen=True)
class CombinedModelSettings:  # Figure 6
    iqr_multiplier: float = 3.0
    calibration_targets: tuple = (50, 60, 70, 80)
    expected_paired_n: int | None = 96
    expected_full_n: int | None = 101
    n_repeats: int = 1000
    n_folds: int = 5
    base_seed: int = 20260815
    n_bootstraps: int = 20_000
    n_permutations: int = 20_000
    inference_seed: int = 20260816


@dataclass(frozen=True)
class SupplementSettings:  # Figure S2
    n_permutations: int = 10_000
    batch_size: int = 50
    alpha_fwe: float = 0.05
    random_seed: int = 4201


@dataclass(frozen=True)
class SimulationSettings:  # Figure S1
    n_subjects: int = 100
    true_r: float = 0.50
    response_mean: float = 1.0
    response_sd: float = 1.0
    n_panel_a_repetitions: int = 1_000_000
    panel_a_chunk_size: int = 10_000
    n_whole_brain_repetitions: int = 100
    whole_brain_ratios: tuple = (0.5, 1.0, 1.5, 2.0, 3.0)
    n_true_voxels: int = 5000
    n_null_voxels: int = 95_000
    seed_a: int = 42
    seed_b: int = 44


HOEPPLI_Z_THRESHOLD = 3.1
HOEPPLI_REGISTRATION = "antsRegistrationSyNRepro[s]"
HOEPPLI_SEED = 42
FIRSTLEVEL_MODEL = "Conc_mov24_wm_csf_roi_physio_revisit_hrf_0_none"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
class Project:
    """All input and output locations, derived from two root directories."""

    def __init__(self, input_root: str | Path | None = None, results_root: str | Path | None = None,
                 data_root: str | Path | None = None):
        """Two modes.

        * private: ``input_root`` holds the raw study files (needs the participants' images).
        * public: ``data_root`` is the shared BIDS folder ``data/``; no images are needed, and the
          group-level derivatives are copied into ``results_root`` before use.
        """
        self.public = data_root is not None
        self.excluded = {} if self.public else EXCLUDED_SUBJECTS  # public tables are already filtered
        if self.public:
            self.data_root = Path(data_root).expanduser().resolve()
            if not self.data_root.is_dir():
                raise MissingInputError(f"Data directory does not exist: {self.data_root}")
            self.input_root = self.data_root
            results = results_root or os.environ.get("PAINPRED_RESULTS_ROOT") or self.data_root.parent / "results"
            self.results_root = Path(results).expanduser().resolve()
            self._define_outputs()
            self._define_public_inputs()
            self.stage_public_derivatives()
            return
        input_root = input_root or os.environ.get("PAINPRED_INPUT_ROOT")
        if not input_root:
            raise MissingInputError(
                "No input directory given. Pass `input_root=...` (private mode) or `data_root=...` (public data), "
                "or set PAINPRED_INPUT_ROOT."
            )
        self.input_root = Path(input_root).expanduser().resolve()
        if not self.input_root.is_dir():
            raise MissingInputError(f"Input directory does not exist: {self.input_root}")
        results = results_root or os.environ.get("PAINPRED_RESULTS_ROOT") or self.input_root / "Results"
        self.results_root = Path(results).expanduser().resolve()
        self._define_inputs()
        self._define_outputs()

    def _define_public_inputs(self) -> None:
        """Input locations inside the shared ``data/`` folder."""
        d = self.data_root
        self.mean_ratings = d / "phenotype/ratings_mean.tsv"
        self.trial_ratings = d / "phenotype/ratings_trial.tsv"
        self.calibration = d / "phenotype/calibration.tsv"
        self.nps_scores = d / "phenotype/nps_scores.tsv"
        self.t1_background = d / "external/templates/T1w_group_template.nii.gz"
        atlas = d / "external/atlas/MNI_Asym_neuromorphometrics.nii"  # the export stores it compressed (.nii.gz)
        self.atlas_nii = atlas.with_name(atlas.name + ".gz") if atlas.with_name(atlas.name + ".gz").exists() else atlas
        self.atlas_xml = d / "external/atlas/MNI_Asym_neuromorphometrics.xml"
        self.nps_weight_candidates = (d / "external/nps/nps_weights_in_study_space.nii.gz",)
        self.same_study_48_mask = d / "derivatives/painpred/group_maps/48heat_voxelFWE05_pos_mask.nii.gz"
        self.sound_tmap = d / "derivatives/painpred/group_maps/sound_tmap.nii.gz"
        self.checkerboard_tmap = d / "derivatives/painpred/group_maps/checkerboard_tmap.nii.gz"
        self.cerebellum_mask = d / "external/templates/cerebellum_mask_mni.nii.gz"

    def stage_public_derivatives(self) -> None:
        """Copy the shared group-level derivatives into the results folder; files that changed are replaced."""
        source = self.data_root / "derivatives/painpred"
        if not source.is_dir():
            raise MissingInputError(f"Shared derivatives not found: {source}")
        for path in source.rglob("*"):
            target = self.results_root / path.relative_to(source)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists() or (target.stat().st_size, int(target.stat().st_mtime)) != (path.stat().st_size, int(path.stat().st_mtime)):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    # -- inputs ------------------------------------------------------------- #
    def _define_inputs(self) -> None:
        r = self.input_root
        self.mean_ratings = r / "sub_by_condition_mean_ratings.tsv"
        self.trial_ratings = r / "bids/behav/pain_rating_wide_complete.csv"
        self.calibration = r / "bids/behav/revisitpain_calib_summary.csv"
        self.firstlevel_root = r / "bids/derivatives/spm_firstlevel"
        self.mean_epi = r / "bids/derivatives/nordic_preprocessing/mean_wskull.nii"
        self.t1_background = r / "misc/T1w_desc-brain.nii.gz"
        templates = r / "code/spm_bids_pipeline/templates"
        self.target_t1 = templates / "cb_Template_T1.nii"
        self.target_mask = templates / "brainmask.nii"
        self.hoeppli_t1 = r / "hoeppli/bg_image.nii.gz"
        self.hoeppli_pos = r / "hoeppli/heat_highStim_pos.nii.gz"
        self.hoeppli_neg = r / "hoeppli/heat_highStim_neg.nii.gz"
        self.nps_scores = r / "nps/nps_scores_Conc_mov24_wm_csf_roi_physio_revisit_hrf_0_none_s6.csv"
        nps_dir = r / "bids/derivatives/nordic_preprocessing"
        self.nps_weight_candidates = (
            nps_dir / "nps/nps_weights_in_mean_wskull_space.nii",
            nps_dir / "nps_weights_in_mean_wskull_space.nii",
        )
        self.same_study_48_mask = r / "misc/secondlevel_revisitpain/48heat_voxelFWE05_pos_mask.nii"
        self.atlas_nii = r / "misc/atlas/MNI_Asym_neuromorphometrics.nii"
        self.atlas_xml = r / "misc/atlas/MNI_Asym_neuromorphometrics.xml"
        self.cerebellum_mask = r / "misc/cerebellum_mask_mni.nii.gz"  # optional
        self.sound_tmap = r / "archive/sound.nii"
        self.checkerboard_tmap = r / "archive/checkerboard.nii"

    @property
    def nps_weights(self) -> Path | None:
        return next((p for p in self.nps_weight_candidates if p.exists()), None)

    # -- outputs ------------------------------------------------------------ #
    def _define_outputs(self) -> None:
        o = self.results_root
        self.figures = o / "figures"
        self.qc = o / "qc"
        self.shared = o / "shared" / "hoeppli"
        self.f1, self.f2, self.f3, self.f4, self.f5, self.f6 = (o / f"figure{i}" for i in range(1, 7))
        self.maps = self.f2 / "maps"  # voxelwise results: maps/con0053/pain_48_arm/...
        self.f2_overlap = self.f2 / "overlap"
        self.f5_massuni = self.f5 / "massunivariate"
        self.supplement = o / "supplementary"
        self.simulation = self.supplement / "simulation"
        self.baseline = self.supplement / "conventional_implicit_baseline"
        # Shared intermediate files
        self.group_mask = self.f2 / "group_mask.nii.gz"
        self.subjects = self.f2 / "subjects.tsv"  # the 101-participant imaging sample
        self.hoeppli_mask = self.shared / "hoeppli_union_mask.nii.gz"
        self.lasso_mask = self.f4 / "lasso_mask.nii.gz"
        self.lasso_raw_weights = self.f4 / "lasso_weights_raw.nii.gz"
        self.lasso_scores = self.f4 / "lasso_scores_heat.tsv"
        self.primary_dir = self.maps / f"con{PRIMARY.contrast}" / PRIMARY.rating

    def analysis_dir(self, contrast: str, rating: str) -> Path:
        return self.maps / f"con{contrast}" / rating

    def figure_files(self, stem: str) -> list[Path]:
        return [self.figures / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")]

    def make_output_dirs(self) -> None:
        for d in (self.figures, self.qc, self.shared / "transforms", self.f1, self.f2, self.maps,
                  self.f2_overlap, self.f3, self.f4, self.f5, self.f5_massuni, self.f6,
                  self.simulation, self.baseline, self.qc / "hoeppli"):
            d.mkdir(parents=True, exist_ok=True)

    # -- input validation --------------------------------------------------- #
    def require(self, stage: str) -> None:
        """Raise one error listing *all* missing inputs of a pipeline stage."""
        missing = []
        for name, attr, hint in STAGE_INPUTS[stage]:
            if attr == "nps_weights":
                ok = self.nps_weights is not None
                shown = " or ".join(str(p) for p in self.nps_weight_candidates)
            else:
                path = Path(getattr(self, attr))
                ok, shown = path.exists(), str(path)
            if not ok:
                missing.append(f"  - {name}: {shown}\n      {hint}")
        if missing:
            raise MissingInputError(
                f"Stage '{stage}' cannot run, {len(missing)} required input(s) missing:\n"
                + "\n".join(missing)
            )

    def missing_external_inputs(self) -> list[str]:
        """Every external input (not produced by this pipeline) that is absent, one line each."""
        missing = {}
        for entries in STAGE_INPUTS.values():
            for name, attr, hint in entries:
                if hint != _HINT_INPUT:
                    continue
                if attr == "nps_weights":
                    if self.nps_weights is None:
                        missing[name] = " or ".join(str(p) for p in self.nps_weight_candidates)
                elif not Path(getattr(self, attr)).exists():
                    missing[name] = str(getattr(self, attr))
        return [f"{name}: {path}" for name, path in missing.items()]


_HINT_INPUT = "external input: copy it into the project folder (see README)."
_HINT_F2 = "produced by Figure 2 (run `massuni.run_all` first)."
_HINT_HOEPPLI = "produced by the Hoeppli preprocessing step (`hoeppli.build_mask`)."
_HINT_F4 = "produced by Figure 4 (`signatures.run_lasso_pcr`)."

STAGE_INPUTS = {
    "figure1": [("mean ratings", "mean_ratings", _HINT_INPUT),
                ("trial ratings", "trial_ratings", _HINT_INPUT),
                ("calibration summary", "calibration", _HINT_INPUT)],
    "hoeppli": [("target template", "target_t1", _HINT_INPUT),
                ("Hoeppli anatomy", "hoeppli_t1", _HINT_INPUT),
                ("Hoeppli positive map", "hoeppli_pos", _HINT_INPUT),
                ("Hoeppli negative map", "hoeppli_neg", _HINT_INPUT)],
    "massuni": [("first-level directory", "firstlevel_root", _HINT_INPUT),
                ("mean ratings", "mean_ratings", _HINT_INPUT),
                ("mean EPI", "mean_epi", _HINT_INPUT)],
    "figure2": [("imaging sample", "subjects", _HINT_F2),
                ("primary r-map", "primary_dir", _HINT_F2),
                ("T1 background", "t1_background", _HINT_INPUT)],
    "overlap": [("primary r-map", "primary_dir", _HINT_F2), ("group mask", "group_mask", _HINT_F2),
                ("same-study 48 °C mask", "same_study_48_mask", _HINT_INPUT),
                ("Hoeppli union mask", "hoeppli_mask", _HINT_HOEPPLI)],
    "peaks": [("primary r-map", "primary_dir", _HINT_F2), ("group mask", "group_mask", _HINT_F2),
              ("atlas", "atlas_nii", _HINT_INPUT), ("atlas labels", "atlas_xml", _HINT_INPUT),
              ("same-study 48 °C mask", "same_study_48_mask", _HINT_INPUT),
              ("Hoeppli union mask", "hoeppli_mask", _HINT_HOEPPLI)],
    "atlas": [("atlas", "atlas_nii", _HINT_INPUT), ("atlas labels", "atlas_xml", _HINT_INPUT)],
    "figure3": [("overlap source maps", "maps", _HINT_F2),
                ("NPS scores", "nps_scores", _HINT_INPUT),
                ("T1 background", "t1_background", _HINT_INPUT)],
    "lasso": [("imaging sample", "subjects", _HINT_F2), ("group mask", "group_mask", _HINT_F2),
              ("Hoeppli union mask", "hoeppli_mask", _HINT_HOEPPLI)],
    "figures2": [("mass-univariate maps", "maps", _HINT_F2)],
    "figure4": [("NPS scores", "nps_scores", _HINT_INPUT), ("LASSO weights", "lasso_raw_weights", _HINT_F4)],
    "figure5": [("NPS scores", "nps_scores", _HINT_INPUT), ("LASSO weights", "lasso_raw_weights", _HINT_F4),
                ("sound group T map", "sound_tmap", _HINT_INPUT),
                ("checkerboard group T map", "checkerboard_tmap", _HINT_INPUT)],
    "figure6": [("imaging sample", "subjects", _HINT_F2), ("NPS scores", "nps_scores", _HINT_INPUT),
                ("calibration summary", "calibration", _HINT_INPUT)],
    "supplement": [("group mask", "group_mask", _HINT_F2),
                   ("first-level directory", "firstlevel_root", _HINT_INPUT),
                   ("mean ratings", "mean_ratings", _HINT_INPUT)],
}
