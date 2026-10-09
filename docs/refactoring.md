# Code organization

This repository keeps the forecasting backbone, HS-DT, aligned C4, physical comparator, required training helpers, data processing, and ensemble/scale controls. Separate hurricane experiments and earlier model-search scripts are stored in the original research workspace.

## Changes

- Replaced numbered file prefixes with descriptive names.
- Grouped code by responsibility rather than experiment order.
- Updated dynamic file-loading paths and repository root discovery.
- Retained state-dict keys and model calculations for checkpoint compatibility.
- Made no-physics, frozen-expert C4 the default; a seed-42 checkpoint override now requires an explicit argument.
- Included the current manuscript PDF. The corresponding Overleaf source is maintained separately.
- Added synthetic model checks and a viewer for saved results.

See `file_mapping.csv` for the source-to-repository file mapping. The original project remains unchanged. File organization and entry-point defaults were updated; saved results were not changed. Full training was not repeated during this reorganization.

VARX helpers were extracted without changing feature construction, the Ridge solver, validation selection, or seven-station output reshape. Some internal variable names remain from earlier experiments to keep this refactor limited to file organization.
