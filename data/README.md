# Shared data

Everything needed to redraw the figures, supplementary figures and tables of the paper. No images of single
participants are included, and participants are identified only by a pseudonym (`sub-p001`, ...).

The folder follows BIDS where it makes sense (`participants.tsv`, `phenotype/`, `derivatives/`).

```
data/
├── dataset_description.json
├── participants.tsv                 participant_id only (n = 101 in the main analyses)
├── phenotype/                       one row per participant (or per participant x calibration level)
│   ├── ratings_mean.tsv             mean pain rating per condition (46/48 °C x arm/leg), 0-100 VAS
│   ├── ratings_trial.tsv            all 12 repetitions per condition
│   ├── calibration.tsv              pre-experiment calibration: temperature estimated for each target VAS
│   └── nps_scores.tsv               NPS expression for the six first-level contrasts
├── derivatives/painpred/            group-level results, same layout as the results/ folder the code writes
│   ├── figure2/                     group mask; 16 voxelwise Spearman analyses (4 contrasts x 4 ratings);
│   │                                peak values per participant; local peak table
│   ├── figure4/                     LASSO-PCR weights, cross-validation tables, signature scores
│   ├── figure5/                     NPS and signature scores for pain and control contrasts
│   ├── group_maps/                  group T maps (sound, checkerboard) and the same-study 48 °C FWE mask
│   ├── shared/hoeppli/              union of the high-heat maps of Hoeppli et al. in study space
│   └── supplementary/               warmth vs implicit-baseline maps and tables
└── external/                        templates and third-party files (see below)
```

## Voxelwise maps

Per analysis (`figure2/maps/con<contrast>/<rating>/`): `rho` (Spearman), `z`, `p_fwe` (max-statistic, 10,000
permutations), `rho_fwe` / `z_fwe` (voxelwise FWE 0.05), `rho_z3p1` / `z_z3p1` (|z| ≥ 3.1, uncorrected),
`rho_cluster_fwe` / `z_cluster_fwe`, plus `clusters.tsv` and `analysis_summary.tsv`. Contrasts: 0053 (48 °C arm),
0054 (46 °C arm), 0055 (48 °C leg), 0056 (46 °C leg).

## Participants

101 participants entered the imaging analyses. Their data are not part of this folder. The pseudonym number is the rank of
the original ID, so the row order of the tables is the same as in the analysis; the key linking pseudonyms to
original IDs is not shared.

## External files

`external/` is only filled with files that may be redistributed. If a file is missing, `external/README.md`
says where to get it and where to put it.

| File | Source |
|---|---|
| `templates/T1w_group_template.nii.gz` | group template of this study, used as figure background |
| `atlas/MNI_Asym_neuromorphometrics.*` | Neuromorphometrics atlas distributed with SPM |
| `nps/nps_weights_in_study_space.nii.gz` | Neurologic Pain Signature (Wager et al. 2013), resampled to the study grid |
| Hoeppli maps | Hoeppli et al., heat stimulus-responsive maps (already combined in `shared/hoeppli`) |
