"""Code for "Pain evoked neural activity scales with subjective pain across individuals".

Layout
------
config          paths (``Project``), analysis constants, experimental conditions
utils           errors, subject-ID handling, table I/O, small statistics helpers
data            assembling the analysis tables (ratings, NPS scores, LASSO-PCR scores)
imaging         NIfTI helpers (masks, maskers, saving voxel vectors)
massuni         voxelwise Spearman + permutation FWE engine (Figures 2, 3, 5, S2)
hoeppli         registration of the Hoeppli maps into the study template
peaks           local-peak table and overlap statistics (Figure 2)
signatures      NPS and LASSO-PCR cross-validation, full model, scores (Figures 4, 5)
combined        behaviour + NPS prediction analysis (Figure 6)
simulation      reference-variability simulation (Figure S1)
supplementary   implicit-baseline contrasts and paired difference test (Figure S2)
figures         one module per figure
"""

__version__ = "1.0.0"
