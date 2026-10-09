# 项目文件与目录地图

本文件回答“某个文件为什么存在、应该什么时候使用、是否属于当前主线”。

若需要逐个文件查询，使用同目录的 `FILE_MANIFEST.csv`；本文件负责解释目录和科研角色，CSV 负责精确列出路径、文件类别、大小和更新时间。

## 根目录

| 路径 | 当前作用 | 使用建议 |
|---|---|---|
| `data/` | 原始、处理后和图结构数据 | 研究数据源，保留，不随意改名 |
| `results/` | 主要模型、诊断、HS-DT 和强基线结果 | 优先读取已汇总 CSV，不要把 smoke 结果当正式结果 |
| `数据整理/` | 历史完整脚本和最新实验脚本 | 按脚本编号和本目录实验登记使用 |
| `autodl_bundle/` | 上传 AutoDL 的精简运行包 | 长时间训练使用；不是论文正文 |
| `publication_final/` | 论文、PPT、演讲稿、模板和交付材料 | 对外发送和 Overleaf 使用 |
| `paper_revision/` | 论文草稿、图和早期修订材料 | 继续写作前先和 `publication_final` 对照 |
| `paper_model_code_package_clean/` | 对外发送的清理代码包 | 只需要论文相关代码时使用 |
| `project_overview/` | 早期项目说明 | 有用但部分内容早于 HS-DT，优先以本目录为准 |
| `project_handoff/` | 当前唯一的交接说明 | 新对话先读这里 |
| `archive/` | 历史重复包和旧报告的归档位置 | 不作为当前运行入口 |

## 数据目录

### `data/raw/`

原始下载数据，包括 NOAA CO-OPS 水位和天文潮、ERA5 风压、Copernicus 海流和波浪、局地气象以及 GEBCO 水深。这里的数据是可追溯来源，不应被训练脚本覆盖。

### `data/processed_multiyear_2023_2025/`

当前多年度主输入：

- `water_tide_residual_long.csv`：水位、潮汐和残差；
- `residual_matrix.csv`：站点残差矩阵；
- `tide_matrix.csv`：站点潮汐矩阵；
- `era5_station_hourly.csv`：ERA5风压；
- `surface_currents_station_daily.csv`：表层海流；
- `wave_direction_speed_station_3hourly.csv`：波浪方向和速度/相关波浪变量；
- `noaa_coops_met_station_hourly.csv`：局地气象；
- `gebco_station_depth.csv`：站点水深；
- `station_order.csv`：七个站点顺序；
- `multiyear_data_completeness_overview.csv` 和 `enhanced_forcing_completeness_summary.csv`：完整性和缺失率审计。

### 七个站点

New London、Montauk、Kings Point、The Battery、Sandy Hook、Atlantic City、Cape May，时间范围为2023-01-01至2025-12-31，主水位和 ERA5 表格为小时频率。

## 代码目录

### 数据下载和处理

`数据整理/67_download_noaa_multiyear_raw.py`、`68_download_copernicus_currents_multiyear.py`、`70_download_era5_multiyear.py`、`69_make_noaa_multiyear_residual.py`、`71_make_copernicus_multiyear_station_features.py`、`72_make_era5_multiyear_station_features.py`。

### 四模型归因阶梯

`数据整理/74_multistate_physics_loss_gnn_bigru_v2.py` 是核心多状态 Physics Loss；`76` 做增强强迫和状态消融；`77` 做权重策略；`78` 组织四模型比较。`133_merge_corrected_bigru_ladder.py` 合并修正双向隐藏状态后的五 seed 结果，生成配对统计、论文表格、三联指标图和七站点预测轨迹图。

### 论文级强基线和诊断

`数据整理/101_priority1_publication_experiments.py` 运行 DCRNN、Graph WaveNet 及逐时长、逐站点诊断；`102_priority2_publication_experiments.py` 做锁定 lambda 后的敏感性、缺失率、图权重和失败案例分析；`103` 和 `104` 做强骨干归因与严格匹配物理消融。

### 轨迹生成实验

`87`–`89` 是 Cycle 和趋势一致性；`92` 是单步自回归；它们是补充实验，不应和四模型主阶梯混成一个模型。

### HS-DT-GWN

`数据整理/105_adaptive_multiscale_graph_wavenet.py`、`106_smooth_horizon_gated_gwn.py`、`107_adaptive_residual_gwn_adapter.py` 是候选探索；`108_horizon_specialized_dual_task_gwn.py` 是当前 HS-DT-GWN 实验入口。

### ORC-HS-DT-GWN

`数据整理/131_ode_residual_corrected_hsdt_gwn.py` 在锁定 HS-DT 预测上迁移 ODE-conditioned residual correction，并与 physics、zero-prior 和 persistence-prior correction 做五 seed 配对消融。结果位于 `results/ode_residual_corrected_hsdt_gwn/`，仍属于同一历史 benchmark 上的 post-hoc exploratory extension。

`数据整理/134_confirmatory_hsdt_orc_chronological_refit.py` 对HS-DT和ORC做端到端按时间重训回测：从头训练双专家，冻结融合规则，训练learned-ODE/zero/persistence三个修正器，保存checkpoint、scaler、预测和时间戳，并生成五seed配对统计、逐lead、逐站点和168小时移动块bootstrap。结果位于 `results/confirmatory_hsdt_orc_refit_2025_h2/`；该目录不是smoke，但因2025 H2曾被查看，仍不能标为untouched holdout。

### Physics-conditioned GWN/HS-DT 归因

`数据整理/140_hsdt_expert_physics_conditioned.py`、`142_hsdt_multistate_prior_physics_loss.py`和`143_hsdt_no_prior_finetune_control.py`分别运行专家内causal ODE、ODE上direct physics loss和容量匹配控制；`144_formal_gwn_ode_physics_factorial.py`补充冻结单体GWN的ODE对照；`146_score_multistate_no_prior_control.py`评分已保存的单体GWN no-prior checkpoint。`145_compile_physics_factorial_evidence.py`统一生成跨GNN/GWN/HS-DT配对证据，`147_build_paper_story_assets.py`生成论文故事主表和稳健性图。

### 投稿精简证据分析

`数据整理/132_publication_clean_evidence_analysis.py` 不训练或选择新模型，而是在锁定的五 seed fixed-support Graph WaveNet 预测上补充168小时移动块 bootstrap、逐月份/逐站点稳定性、训练期阈值高低残差分层和总水位重建诊断。结果位于 `results/publication_clean_evidence/`。

### 普通 benchmark 投稿资产

`数据整理/148_build_benchmark_only_submission_assets.py` 只读取已锁定的普通70/15/15五 seed汇总，生成GNN-BiGRU、FS-GWN、HS-DT和ORC的统一层级表，以及direct physics loss、ODE prior和adapter controls的归因图。结果位于 `results/benchmark_only_submission_evidence/`；脚本不重训、不调参，也不读取2025 H2重训结果。

### 物理强迫状态门控扩展

`数据整理/149_physics_regime_switched_hsdt.py` 在普通benchmark上用训练期风应力、气压趋势、流速、波能通量和wave-setup proxy构造因果forcing-intensity index；验证集锁定q75/q90/q95候选中的门槛，强迫状态使用learned-ODE ORC，其余状态使用persistence adapter，并加入反向、时间错位、GNN ODE和GWN direct-physics对照。结果位于 `results/physics_regime_switched_hsdt_benchmark/`，属于机制探索，不覆盖主稿的ORC结论。

## 结果目录

| 结果目录 | 可信用途 |
|---|---|
| `results/priority12_physics_graph_wavenet/` | 五 seed 严格匹配 GWN 多状态/physics 结果 |
| `results/horizon_specialized_dual_task_gwn/` | HS-DT-GWN 五 seed 及逐站点、逐 lead 结果 |
| `results/confirmatory_hsdt_orc_refit_2025_h2/` | HS-DT/ORC从头重训的2025 H2稳健性回测；不是untouched holdout |
| `results/priority1_graph_baselines/` | DCRNN、GWN 等强基线 |
| `results/priority2_evidence_verified/` | 排除旧 lambda 后的去重敏感性证据 |
| `results/ablation/` | 特征、图、状态和模型消融 |
| `results/rolling/` | 严格因果 rolling-origin |
| `results/baselines/` | Persistence、tide-only、BiGRU-only、RF 等简单基线 |
| `results/corrected_bigru_ladder/` | 修正 BiGRU 状态提取后的五 seed 四模型阶梯；正式汇总位于 `merged/` |
| `results/formal_gwn_ode_physics_factorial_2025_h2/` | 冻结单体GWN的causal ODE五seed对照 |
| `results/multistate_no_prior_control_scored_2025_h2/` | 单体GWN no-prior微调checkpoint的统一H2评分 |
| `results/paper_physics_factorial_2025_h2/` | 跨GNN/GWN/HS-DT物理归因矩阵、配对检验和主图 |
| `results/paper_story_2025_h2/` | 完整中文论文故事、统一主表和逐lead/逐站点故事图 |
| `results/benchmark_only_submission_evidence/` | 普通回顾性70/15/15 benchmark的统一模型层级与物理归因投稿资产 |
| `results/physics_regime_switched_hsdt_benchmark/` | 验证锁定的物理强迫状态门控、跨GNN/GWN/HS-DT分层效应和逐lead统计 |
| `results/_smoke_*`、`results/_bench_*` | 快速试跑，不可用于正式论文结论 |

## 论文与交付目录

- `publication_final/overleaf_submission/`：完整 Overleaf 论文和补充 CSV。
- `publication_final/overleaf_sea_level_clean_submission/`：删除后验模型、Cycle、单 seed 递归、test-defined q95 和事件化叙事，并加入已修正五 seed GNN-BiGRU 诊断阶梯后的海平面精简投稿稿；ZIP 位于 `publication_final/overleaf_sea_level_clean_submission.zip`。
- `publication_final/overleaf_sea_level_benchmark_only_submission/`：只使用普通回顾性70/15/15 benchmark的完整稿件，包含GNN-BiGRU、FS-GWN、HS-DT、ORC和匹配物理/适配器对照；不包含2025 H2重训结果，也不声称独立时间holdout。可上传ZIP位于同名`.zip`。
- `publication_final/SEA_LEVEL_PUBLICATION_EXPERIMENT_AUDIT_CN.md`：解释哪些实验保留、删除或降级，以及本轮新增稳健性实验的结论。
- `publication_final/submission_15page_priority12/`：15页短稿，主要是旧版本压缩稿。
- `publication_final/presentation/`：PPT 和演讲材料。
- `publication_final/documents/`：中文解释文档、演讲稿和结果说明。
- `publication_final/templates/Scientific_Reports/`：Scientific Reports 非官方模板，不是论文内容。
- `publication_final/qa/`：渲染后的视觉检查图片。

## 选择原则

如果新对话问“现在应该用什么”：

1. 科学主线看 `publication_final/overleaf_submission/main.tex` 和 `03_EXPERIMENT_REGISTRY_CN.md`。
2. 当前最强结果看 `results/priority12_physics_graph_wavenet/`。
3. HS-DT 扩展看 `results/horizon_specialized_dual_task_gwn/`，但先标记为 exploratory。
4. 代码运行看 `autodl_bundle/sea_level/autodl/AUTODL_RUNBOOK_CN.md`。
5. 不要从根目录散落的旧 zip、旧 PDF 或 `tmp` 目录推断当前结论。

### 物理作用审计

`数据整理/150_physics_effect_audit_benchmark.py` 只读复用普通 benchmark 的五 seed 预测，按训练期逐站点 forcing 阈值输出组件、站点、lead 和 block-bootstrap 物理作用。结果位于 `results/physics_effect_audit_benchmark/`；这是机制审计，不是新模型，也不替换主稿的 HS-DT/ORC 证据。

`数据整理/151_physics_alignment_negative_controls.py` 检查真实 forcing mask、反向 mask 和固定时间错位 mask；`数据整理/152_validation_locked_lagged_physics_gate.py` 只在 validation 选择 forcing lag，再冻结到普通 benchmark 测试。两者均为机制探索，不能替代主模型或独立时间 holdout。

`数据整理/153_build_submission_evidence_pdf.py` 生成投稿前内部证据审计 PDF，输出到 `output/pdf/physics_submission_evidence_audit.pdf`，不覆盖现有 Overleaf 主稿。
