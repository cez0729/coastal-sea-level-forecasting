# Main findings used in the project overview

These CSVs contain the values displayed in the manuscript and README figures. They are small, readable reporting tables, not newly computed experiments.

- `expert_comparison.csv`: the original historical expert bank (manuscript Table 1).
- `c4_components.csv`: the matched historical C4 component scores (Table 2).
- `c4_paired_sequence.csv`: reported paired differences and 95% intervals (Sections 6.4 and 6.6).

The C4 reference differs from the original expert bank. Do not subtract the original HS-DT score from the matched C4 score. Component values and reported intervals are rounded to manuscript precision.

The C4 intervals use the reported corrected moving-block analysis. The older 2,000-replicate snapshot formerly under `results/c4/block_bootstrap_168h.csv` is omitted from this release to avoid confusion with the manuscript intervals. The ten-station C4 analysis is post-hoc. The figure caption retains that distinction.
