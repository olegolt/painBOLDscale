<!-- TODO before making the repo public: add source and licence of templates/T1w_group_template.nii.gz -->
# Third-party files

Put the following files here if they are not already present. They are not part of the CC BY release unless the
licence of the original allows it.

* `atlas/MNI_Asym_neuromorphometrics.nii` (or `.nii.gz`) and `.xml`: Neuromorphometrics atlas (ships with SPM12, `tpm/` / `atlas/`).
* `nps/nps_weights_in_study_space.nii.gz` (optional, **not distributed**): NPS weights from the CANlab repository
  (https://github.com/canlab), resampled to the group mask grid (`derivatives/painpred/figure2/group_mask.nii.gz`).
  Without it everything still runs; the NPS brain map in Figure 4 is then left empty and a note says so.
  The NPS scores used in all analyses are part of the shared data.
* `templates/T1w_group_template.nii.gz`: study group template.
