# Interpreting the results

The release preserves positive and negative controls. It is not a collection of only favorable experiments.

## Separate comparisons

The historical matched C4 reference is HS-DT R²=0.731914. A different historical expert bank gives HS-DT R²=0.7357. These values must not be swapped when calculating the C4 gain.

The two-expert HS-DT combination is compared with individual experts in several periods. The additional ordinary-ensemble control asks a different question: whether cross-supervision averaging exceeds same-supervision averaging. It does not in the ten-station analysis.

The scale-only control keeps the C4 mean unchanged. It isolates the scoring effect of using a dynamic scale rather than one validation-fitted scalar per seed; it is not a separately trained constant-scale model.

## Confidence intervals

The ensemble/scale controls use 5,000 paired circular moving-block resamples. Main block length: 168 origins, with 84/336 sensitivity checks. Caches retain hourly origin order; this release does not independently reconstruct timestamps from raw data. Resampling retains all stations and leads together and shares blocks across pairs and seeds. Intervals condition on the saved five-seed expert bank; overlapping pairs are not independent replicates.

## Evidence boundaries

Historical aligned-forcing results are retrospective. The ordinary-ensemble and scale controls are post-hoc analyses of saved predictions. External C4 was added after the original external evaluation. None becomes an untouched temporal test by renaming the experiment or keeping its weights frozen.

Physical losses and simplified ODE priors are diagnostics, not a complete hydrodynamic solver. Their mixed results do not establish universal gains. Dynamic-scale gains do not imply improved R², and narrower intervals alone do not imply better calibration.
