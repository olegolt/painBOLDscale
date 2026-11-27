# Pain-evoked neural activity scales with subjective pain across individuals

Code and data for the paper (add link / DOI). With the shared data you can redraw every figure of the paper and
supplement and rebuild the reported tables on a laptop in a few minutes. The permutation maps and the LASSO-PCR
model are included as results; refitting them needs the participants' images, which are not shared.

```
code/    analysis package (painpred), notebook, tests
data/    shared data in BIDS layout: participant-level scores and group-level maps (see data/README.md)
results/ created by the notebook: figures and tables
```

## Run it

```bash
git clone https://github.com/olegolt/painBOLDscale.git && cd painBOLDscale
conda env create -f environment.yml && conda activate painpred     # or: pip install -r requirements.txt
jupyter notebook code/paper_figures.ipynb
```

The notebook goes through the figures in order. Each step prints what it does and stops with a plain message if a
file is missing or has the wrong columns. The same steps run from the command line:

```bash
python -m painpred --data data --results results --list
python -m painpred --data data --results results --steps figure1 figure2 figure6
```

The final figures are `results/figures/Figure*.{png,pdf,svg}`; tables are under `results/figure*/`.

**Files that are not shared.** The NPS weight map (CANlab) and the Neuromorphometrics atlas are third-party files and
are not part of this repository. Everything runs without them: the NPS scores are shared, the NPS brain map in
Figure 4 stays empty with a note, and the peak and region tables are read from `data/` instead of being rebuilt.
To draw or rebuild them, put the files in `data/external/` (see `data/external/README.md`).

## What needs the raw images

| Step | With shared data |
|---|---|
| Figures 1–6, S1, S2, supplementary figure | redrawn from `data/` |
| `massuni` (16 permutation analyses, ~hours) | skipped; maps are in `data/derivatives/painpred/figure2` |
| `lasso_pcr` (nested CV, ~hours) | skipped; weights and CV tables are in `.../figure4` |
| `hoeppli_mask` (ANTs registration) | skipped; mask is in `.../shared/hoeppli` |

With the raw study files, pass `--input /path/to/study` instead of `--data data` (or set `STUDY_FOLDER` in the
notebook) to rerun these steps. All random procedures are seeded; reruns reproduce the shared maps up to
floating-point noise.

## Where the reported numbers are

| Result | File (under `results/`) |
|---|---|
| Behavioural summaries | `figure1/summary.csv`, `figure1/calibration_summary.csv` |
| Voxelwise analyses (all 16) | `figure2/summary_all_analyses.tsv`, `figure2/maps/*/*/clusters.tsv` |
| Peak table, overlap with reference maps | `figure2/local_peaks.csv`, `figure2/overlap/overlap_summary.tsv` |
| Agreement across ratings, atlas regions | `figure3/overlap_summary.tsv`, `figure3/count4_regions.csv` |
| NPS vs mean pain | `figure3/nps_summary.tsv` |
| LASSO-PCR and NPS cross-validation | `figure4/lasso_cv_metrics.tsv`, `figure4/nps_cv_metrics.tsv`, `figure4/lasso_transfer_stats.tsv` |
| Control contrasts | `figure5/association_stats.tsv`, `figure5/expression_summary.tsv` |
| Combined models | `figure6/oof_metrics.tsv`, `figure6/bootstrap_r2.tsv`, `figure6/cv_repeat_metrics.tsv` |
| Simulation | `supplementary/simulation/panelB_whole_brain_summary.csv` |
| Warmth vs implicit baseline | `supplementary/conventional_implicit_baseline/warmth_vs_implicit_baseline/*.tsv` |

## Tests

```bash
pytest code/tests
```

## For the data owners: creating `data/`

```bash
python code/export_data.py --input /path/to/study --out data --id-map /private/id_map.tsv
python code/check_release.py data code
```

`export_data.py` copies a fixed whitelist of tables and group-level maps, replaces IDs by pseudonyms and strips
image headers. The ID map must stay outside the repository. `check_release.py` looks for leftover paths, original
IDs, 4D images, free text in image headers and files above GitHub's size limit.

## Citation and licence

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`data/LICENSE`); third-party files keep their own licences
(`data/external/README.md`). See `CITATION.cff`.
