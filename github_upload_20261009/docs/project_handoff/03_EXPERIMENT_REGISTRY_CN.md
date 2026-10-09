# 实验登记表与结论锁定

## 正式主线结果

| 实验 | 入口代码 | 主要结果位置 | 状态 | 可支持的结论 |
|---|---|---|---|---|
| 四模型 GNN-BiGRU 阶梯 | `数据整理/78_final_four_models_enhanced_data.py`、`133_merge_corrected_bigru_ladder.py` | `results/corrected_bigru_ladder/merged/` | 已修正并完成五seed重跑 | Learnable graph改善基础模型；当前ODE-prior无稳定增益；完整Physical配置最好但不是physics-only |
| DCRNN / Graph WaveNet | `数据整理/101_priority1_publication_experiments.py` | `results/priority1_graph_baselines/` | 已完成 | 强时空架构总体超过 GNN-BiGRU |
| 强骨干 physics 匹配实验 | `数据整理/104_priority2_physics_graph_wavenet.py` | `results/priority12_physics_graph_wavenet/` | 已完成 | GWN 上 physics-only 没有可测增益 |
| Physics 状态匹配消融 | `数据整理/76_enhanced_forcing_physics_loss_ablation_v3.py` | `results/ablation/` | 已完成 | 物理项增益小且不普适 |
| Priority-2 权重敏感性 | `数据整理/102_priority2_publication_experiments.py` | `results/priority2_evidence_verified/` | 已完成 | 终点权重、辅助监督和 physics 共同影响远端性能 |
| 严格因果 rolling-origin | `数据整理/96_rolling_origin_validation.py` | `results/rolling/` | 已完成 | physics 收益随时间段改变 |
| 缺失数据和因果预处理 | `数据整理/84_causal_preprocessing_robustness.py` | `results/rolling/` 或对应 outputs | 已完成 | 回顾性插值会影响性能，严格因果结果更低且更现实 |
| 简单基线 | `数据整理/85_simple_baselines_for_paper.py` | `results/baselines/` | 已完成 | 深度时空建模超越简单持续性和潮汐基线，但短时 BiGRU 很有竞争力 |
| 极端事件 | `数据整理/94_event_based_extreme_evaluation.py` | `results/priority1_diagnostics/` | 已完成 | 平均 R2 不等于极端事件预警能力，召回率仍不足 |
| block bootstrap / 配对统计 | `数据整理/95_block_bootstrap_significance.py` | `results/priority1_diagnostics/` | 已完成 | 部分收益可重复，但强骨干 physics 差异接近零 |

## 补充轨迹实验

| 实验 | 入口代码 | 关键结果 | 论文位置建议 |
|---|---|---|---|
| Cycle-plus-trend | `数据整理/87`–`89` | 24h terminal R2 = 0.5453；Cycle 独立增益不稳定 | 补充或 exploratory subsection |
| 单步自回归 | `数据整理/92_autoregressive_rolling_baseline.py` | 严格 held exogenous terminal R2 = -0.0999；oracle forcing = 0.5883 | 说明误差累积和 forcing availability |

## HS-DT-GWN

| 内容 | 当前值 |
|---|---:|
| eta-only GWN sequence R2 | 0.7261 |
| multistate GWN terminal R2 | 0.6153 |
| HS-DT-GWN sequence R2 | 0.7357 |
| HS-DT-GWN terminal R2 | 0.6153 |
| HS-DT-GWN q95 R2 | 0.6292 |
| 相比 eta-only sequence gain | +0.0095，5/5 seed |
| 相比 eta-only terminal gain | +0.0328，4/5 seed |

**锁定说明**：HS-DT 规则是在旧 benchmark 后形成的，当前没有新的完全独立时间留出验证，不能作为无偏的最终投稿模型结论。

## ORC-HS-DT-GWN 后续扩展

| 内容 | 当前值 |
|---|---:|
| ORC-HS-DT sequence R2 | 0.738647 |
| ORC-HS-DT terminal R2 | 0.619144 |
| ORC-HS-DT descriptive q95 R2 | 0.637804 |
| ORC-HS-DT Event CSI | 0.237290 |
| ORC-HS-DT PR-AUC | 0.528804 |
| 相比 HS-DT sequence gain | +0.002985，5/5 seed |
| 相比 HS-DT terminal gain | +0.003876，5/5 seed |
| 相比 HS-DT q95 gain | +0.008607，5/5 seed |

入口代码：`数据整理/131_ode_residual_corrected_hsdt_gwn.py`。结果目录：`results/ode_residual_corrected_hsdt_gwn/`。

**归因边界**：zero-prior 和 persistence-prior adapter 也能带来部分增益；learned ODE 相对 zero-prior 只在 sequence R2 和 PR-AUC 上达到5/5、单侧精确 Wilcoxon `p=0.03125`。Persistence adapter 的 terminal R2 为0.619779，略高于 learned ODE。不能把 ORC 相对 HS-DT 的全部增益归因于 ODE。

**截至131实验时的状态**：这是同一历史 benchmark 上的 post-hoc exploratory extension，不是独立 holdout 结果；当时旧 eta-only GWN没有与锁定HS-DT预测完全匹配的checkpoint。双专家从头训练和完整checkpoint保存缺口已由下面的134实验补齐，但2025 H2曾被查看，因此真正独立确认仍未完成。

## HS-DT / ORC 2025 H2端到端重训回测

入口代码：`数据整理/134_confirmatory_hsdt_orc_chronological_refit.py`；结果目录：`results/confirmatory_hsdt_orc_refit_2025_h2/`。

时间切分为2023–2024训练、2025 H1验证、2025 H2测试。使用严格因果前向填充，scaler、训练图和事件阈值只由训练期构造；每个seed从头训练Eta-only和Multistate Graph WaveNet，冻结HS-DT公式，并训练learned-ODE、zero、persistence三个修正器。适配器训练期间冻结主干始终保持eval，并强制断言主干预测与原Multistate专家逐元素一致。

| 模型 | Sequence R2 | Lead-24 R2 | Lead-24 RMSE (m) | 描述性q95 R2 |
|---|---:|---:|---:|---:|
| Eta-only FS-GWN | 0.647190 +/- 0.009336 | 0.406241 +/- 0.024781 | 0.134606 +/- 0.002781 | 0.524399 +/- 0.022117 |
| Multistate FS-GWN, no physics | 0.643917 +/- 0.010769 | 0.424692 +/- 0.014625 | 0.132512 +/- 0.001684 | 0.505390 +/- 0.035622 |
| HS-DT-GWN | 0.656345 +/- 0.005359 | 0.424692 +/- 0.014625 | 0.132512 +/- 0.001684 | 0.521553 +/- 0.013254 |
| Zero adapter | 0.661566 +/- 0.007556 | 0.424494 +/- 0.014031 | 0.132536 +/- 0.001612 | 0.547988 +/- 0.016392 |
| Persistence adapter | 0.660622 +/- 0.006807 | 0.423593 +/- 0.014320 | 0.132639 +/- 0.001642 | 0.549060 +/- 0.017061 |
| ORC-HS-DT-GWN | 0.661002 +/- 0.007508 | 0.421440 +/- 0.012713 | 0.132888 +/- 0.001454 | 0.561520 +/- 0.020486 |

**配对结论**：HS-DT相对Eta-only的sequence提升+0.009155、相对Multistate提升+0.012428，均5/5 seed、单侧精确Wilcoxon `p=0.03125`；168小时块bootstrap 95% CI分别为`[0.001407, 0.019554]`和`[0.003705, 0.021243]`。HS-DT相对Eta-only的Lead-24提升+0.018451但仅3/5 seed、`p=0.21875`、CI跨0；相对Multistate严格为0。

ORC相对HS-DT的sequence变化+0.004657但仅3/5 seed、`p=0.15625`、95% CI为`[-0.004912, 0.012708]`；Lead-24变化-0.003252，仅2/5 seed。ORC相对zero的sequence变化-0.000565，Lead-24变化-0.003054且0/5 seed获胜；相对persistence的Lead-24变化-0.002152，仅1/5 seed获胜。

**论文状态**：本轮支持把HS-DT作为轨迹融合稳健性实验写入补充材料，但不支持其Lead-24稳定提升，也不支持把ORC作为最终投稿主模型。由于2025 H2在研究中已被查看，本轮证据必须称为end-to-end chronological refit backtest，不能称untouched independent holdout。真正独立验证仍需2026 H1或其他在模型设计时完全不可见的新数据。

## 投稿精简证据补充实验

入口代码：`数据整理/132_publication_clean_evidence_analysis.py`；结果目录：`results/publication_clean_evidence/`。

- Multistate FS-GWN 相对 eta-only FS-GWN 的 lead-24 R2 差值为 `+0.032809`，168小时移动块 bootstrap 95% CI为 `[0.011627, 0.054244]`。
- 该终点提升在7/7个站点平均为正，在5/6个测试月份为正。
- Physics 相对 matched no-physics 的 lead-24 R2 差值为 `-0.000232`，95% CI为 `[-0.000492, 0.000025]`；0/7个站点平均为正。
- 训练期阈值低残差尾部 RMSE 从 eta-only 的 `0.226407 m` 降至 multistate 的 `0.205800 m`；高残差尾部仅从 `0.244775 m` 变化为 `0.244434 m`。
- Multistate 与 eta-only 的差异同时包含辅助状态监督和 lead-24 权重，不能全部归因于辅助状态。

精简 Overleaf 稿位于 `publication_final/overleaf_sea_level_clean_submission/`。该稿包含已修正五 seed BiGRU 诊断阶梯，但不包含 HS-DT/ORC、未重跑的 BiGRU rolling、Cycle、单 seed autoregressive、test-defined q95 或 storm/event 主张；保留无增益的 physics 匹配实验作为中心归因证据。

## 修正后的GNN-BiGRU五seed阶梯

| 模型 | Sequence R2 | Lead-24 R2 | Lead-24 RMSE (m) |
|---|---:|---:|---:|
| Fixed graph | 0.6385 +/- 0.0084 | 0.4925 +/- 0.0208 | 0.1312 +/- 0.0027 |
| Learnable graph | 0.6667 +/- 0.0085 | 0.5229 +/- 0.0177 | 0.1272 +/- 0.0023 |
| ODE-prior | 0.6632 +/- 0.0059 | 0.5189 +/- 0.0179 | 0.1277 +/- 0.0024 |
| Physical-loss完整配置 | 0.6732 +/- 0.0099 | 0.5553 +/- 0.0187 | 0.1228 +/- 0.0026 |

Learnable相对Fixed的sequence提升为+0.0282（5/5 seed，单侧精确Wilcoxon `p=0.03125`）；ODE-prior相对Learnable的lead-24变化为-0.0041；Physical完整配置相对ODE-prior的lead-24提升为+0.0364（5/5 seed，`p=0.03125`）。最后一个差值同时包含多状态目标、终点权重和physics，不能解释为physics-only。

## 关键参数和评估规则

- 输入24小时，直接输出24小时。
- 时间切分按 chronological train/validation/test。
- physics lambda = 0.0002，由 validation 选择并锁定。
- 主指标：sequence residual R2、lead-24 residual R2、RMSE、descriptive q95 residual R2。
- q95 子集不能称作部署阈值。
- total water level 指标必须说明它包含已知或对齐的 tide prediction。

## 结果读取纪律

1. 优先读取 `mean_std.csv`、`paired_tests.csv`、`per_horizon.csv` 和论文 supplementary CSV。
2. `partial.csv`、`training_log.csv` 只能用于诊断训练过程，不能替代汇总结果。
3. 旧的 `0.0003` lambda worker 结果必须排除。
4. smoke 和 benchmark screen 结果只用于发现候选结构，不能写入主表。
5. 如果新实验改变了 BiGRU hidden-state 提取、数据切分或输入特征，必须重新标记为新实验，不得覆盖旧结果。

## 投稿前剩余项目

1. BiGRU 双向最终隐藏状态提取和四模型五 seed 重跑已于2026-07-28完成；旧 BiGRU 数值继续禁用。
2. HS-DT 的2025 H2端到端严格时间重训、逐 lead、逐站点、事件和计算成本已完成；由于2025 H2曾在开发中被查看，只能称 chronological refit backtest，真正独立验证仍需冻结的2026数据。
3. 正文已统一使用 fixed-support Graph WaveNet variant（FS-GWN），不得改写为完整 canonical adaptive Graph WaveNet。

## 2026-07-31 HS-DT 双专家物理信息注入探索

入口代码：`数据整理/140_hsdt_expert_physics_conditioned.py`；统计与图：`数据整理/141_analyze_hsdt_expert_physics.py`；结果：`results/hsdt_expert_physics_conditioned/`。

该实验把只依赖预测起点前状态和 last-input forcing 的 causal ODE prior 通过逐 horizon 门控注入 eta-only 与 multistate 两个 FS-GWN 专家，并联合微调专家与先验分支。协议仍为2023--2024训练、2025 H1选择、2025 H2回测，五个seed为42、123、2024、2025和3407。

| 模型 | Sequence R2 | Lead-24 R2 | 描述性 q95 R2 | Event PR-AUC |
|---|---:|---:|---:|---:|
| HS-DT baseline | 0.656345 +/- 0.005359 | 0.424692 +/- 0.014625 | 0.521553 +/- 0.013254 | 0.352328 +/- 0.017452 |
| Multistate expert + causal prior | 0.663952 +/- 0.002368 | 0.427200 +/- 0.010596 | 0.543662 +/- 0.015391 | 0.371265 +/- 0.023916 |
| Both experts + causal prior | 0.669496 +/- 0.007805 | 0.427200 +/- 0.010596 | 0.543001 +/- 0.030798 | 0.371265 +/- 0.023916 |

双专家增强相对HS-DT的sequence提升为`+0.013151`，5/5 seed、单侧精确Wilcoxon `p=0.03125`，168小时移动块bootstrap 95% CI为`[0.006626, 0.018921]`。Lead-24仅提高`+0.002508`，4/5 seed、`p=0.21875`、CI跨零，不得声称稳定终点提升。描述性q95提高`+0.021448`（4/5，`p=0.0625`），PR-AUC提高`+0.018936`（5/5，`p=0.03125`）。

对照显示，post-fusion ODE只改善Lead-24而降低sequence与q95；专家内注入同时改善轨迹和尾部指标，支持“物理信息的注入位置重要”。但本实验在2025 H2已被查看后形成，属于post-hoc chronological backtest，不是独立holdout；联合微调和门控也增加了容量，不能把全部增益归因于physics。完整边界见`results/hsdt_expert_physics_conditioned/EXPERIMENT_REPORT_CN.md`。

No-prior capacity control显示完全相同轮数的双专家继续微调可达到 sequence R2 `0.667285 +/- 0.001172`、Lead-24 `0.435667 +/- 0.018478`；相对该控制，双专家 prior 仅 sequence `+0.002211`（4/5，`p=0.3125`），Lead-24 `-0.008467`（2/5，`p=0.84375`）。因此双专家 prior 相对冻结HS-DT的`+0.013151`不能全部归因于physics。

`数据整理/142_hsdt_multistate_prior_physics_loss.py`激活了可微`lambda=0.0002` physical residual loss。五 seed相对prior-only的净增量为sequence `+0.0000018`、Lead-24 `+0.000027`，无显著额外贡献。当前最稳健归因是强GWN骨干和专家微调最主要，直接physics loss在该设置下近似中性。

## 2026-07-31 单体 GWN 冻结 ODE 与跨骨干证据矩阵

入口代码：`数据整理/144_formal_gwn_ode_physics_factorial.py`；统一汇总代码：`数据整理/145_compile_physics_factorial_evidence.py`。结果分别位于 `results/formal_gwn_ode_physics_factorial_2025_h2/` 和 `results/paper_physics_factorial_2025_h2/`。

单体 multistate FS-GWN 从严格时间重训 checkpoint 初始化并完全冻结，只训练 causal ODE prior 与逐 horizon gate。五 seed、2025 H2回测结果为：

| 配置 | Sequence R2 | Lead-24 R2 | 描述性 q95 R2 | Event PR-AUC |
|---|---:|---:|---:|---:|
| Multistate FS-GWN baseline | 0.643917 +/- 0.010769 | 0.424692 +/- 0.014625 | 0.505390 +/- 0.035622 | 0.352328 +/- 0.017452 |
| Frozen GWN + causal ODE prior | 0.647409 +/- 0.007302 | 0.419725 +/- 0.007937 | 0.496590 +/- 0.021853 | 0.359729 +/- 0.010025 |

冻结 ODE 相对纯 GWN 的 sequence 差值为`+0.003492`（3/5，单侧精确 Wilcoxon `p=0.3125`），Lead-24为`-0.004967`，q95为`-0.008799`；Event PR-AUC为`+0.007400`（4/5，`p=0.09375`）。该对照排除了 GWN 微调容量，但没有证明稳定 physics-only 增益。结合 joint ODE、no-prior fine-tuning 和 direct physics loss，当前结论仍是：联合模型相对冻结基线的收益主要由微调容量和注入结构共同产生，physics 特异贡献尚未建立。

同时对已保存的单体 GWN no-prior 微调 checkpoint 做了 H2 评分：no-prior 的 Sequence R2 为`0.655563 +/- 0.012238`、Lead-24 为`0.435667 +/- 0.018478`、q95 为`0.544395 +/- 0.044651`。joint ODE 相对该容量控制的差值为 sequence `+0.004440`（4/5，`p=0.21875`）、Lead-24 `-0.008467`、q95 `+0.005584`、PR-AUC `+0.003688`，均未达到稳定显著。因此单体 GWN 的 q95 优势只能称为描述性候选结果，不能归因于 physics-only。

## 2026-07-29投稿前严格证据闭环

入口代码：`数据整理/135_confirmatory_physical_bigru_chronological_refit.py`、`数据整理/136_build_submission_strict_evidence.py`。

时间协议：2023--2024训练、2025 H1验证、2025 H2回测；严格因果前向填充，scaler、训练图和事件阈值仅由训练期构造。五个配对seed为42、123、2024、2025和3407。GNN-BiGRU的无physics与physics配置使用完全相同的多状态骨架、初始化、批次、优化器、辅助状态权重和lead-24权重，唯一预测损失差异为`lambda=0`与`lambda=0.0002`。

| 严格协议模型 | Sequence R2 | Lead-24 R2 | Lead-24 RMSE (m) |
|---|---:|---:|---:|
| GNN-BiGRU multistate, no physics | 0.597648 +/- 0.008207 | 0.362912 +/- 0.019505 | 0.139442 +/- 0.002117 |
| GNN-BiGRU multistate + physics | 0.593628 +/- 0.009438 | 0.355689 +/- 0.016971 | 0.140233 +/- 0.001839 |
| Eta-only FS-GWN | 0.647190 +/- 0.009336 | 0.406241 +/- 0.024781 | 0.134606 +/- 0.002781 |
| Multistate FS-GWN, no physics | 0.643917 +/- 0.010769 | 0.424692 +/- 0.014625 | 0.132512 +/- 0.001684 |
| HS-DT-GWN | 0.656345 +/- 0.005359 | 0.424692 +/- 0.014625 | 0.132512 +/- 0.001684 |

锁定结论：

- Multistate FS-GWN相对无physics GNN-BiGRU的sequence和lead-24提升分别为`+0.046269`和`+0.061780`，均为5/5 seed、单侧精确Wilcoxon `p=0.03125`；168小时块bootstrap 95% CI分别为`[0.029680, 0.069018]`和`[0.020015, 0.110108]`。
- GNN-BiGRU中physics相对无physics的sequence和lead-24变化分别为`-0.004020`和`-0.007223`，均仅2/5 seed改善；对应95% CI为`[-0.011137, 0.000798]`和`[-0.015920, -0.000383]`。因此该physics项在第二骨架上也没有增益，并在本次锁定回测中轻微降低终点性能。
- HS-DT相对eta-only和multistate FS-GWN的sequence提升分别为`+0.009155`和`+0.012428`，均5/5 seed；统一bootstrap 95% CI分别为`[0.001163, 0.019919]`和`[0.003451, 0.021085]`。
- HS-DT的lead-24与multistate专家严格相同；相对eta-only仅3/5 seed改善，CI `[-0.011054, 0.056274]`跨零，不得声称稳定终点提升。
- 统一投稿证据位于`results/submission_strict_combined_evidence/`；最终Overleaf目录位于`publication_final/overleaf_sea_level_final_submission/`。

## 2026-07-31 普通回顾性 benchmark-only 投稿版

入口代码：`数据整理/148_build_benchmark_only_submission_assets.py`；证据目录：`results/benchmark_only_submission_evidence/`；Overleaf目录：`publication_final/overleaf_sea_level_benchmark_only_submission/`。

该版本仅使用项目原始 chronological 70%/15%/15% train/validation/test benchmark，统一列出修正后的GNN-BiGRU阶梯、FS-GWN强基线、HS-DT双专家和ORC残差适配器。它有意不引用2023--2024训练、2025 H1验证、2025 H2测试的重训结果。原有严格证据目录保留且不覆盖，但不进入这个稿件或其补充文件。

| 模型 | Sequence R2 | Lead-24 R2 | 描述性 q95 R2 |
|---|---:|---:|---:|
| Physical-loss GNN-BiGRU完整配置 | 0.673187 | 0.555286 | 0.517410 |
| Eta-only FS-GWN | 0.726117 | 0.582458 | 0.627116 |
| Multistate FS-GWN，无physics | 0.725023 | 0.615267 | 0.617901 |
| HS-DT-GWN | 0.735662 | 0.615267 | 0.629197 |
| ORC-HS-DT-GWN | 0.738647 | 0.619144 | 0.637804 |

该普通benchmark支持描述性层级`dual expert > FS-GWN > GNN-BiGRU`。但HS-DT的Lead-24按定义直接取multistate专家，因此只能称“保持0.615267”，不能称独立提升。ORC相对HS-DT的Sequence、Lead-24和q95差值分别为`+0.002985`、`+0.003876`和`+0.008607`；相对zero adapter仅为`+0.001255`、`+0.001209`，而persistence adapter的Lead-24为`0.619779`，略高于ORC。

物理归因必须保留负结果：GNN ODE-prior相对learnable GNN的Sequence/Lead-24变化为`-0.003566/-0.004064`；FS-GWN direct physics loss相对matched no-physics为`-0.000119/-0.000232`。因此可发表叙述是“物理贡献依赖注入机制，ORC是最强的physics-conditioned候选，但其全增益不是physics-only”，不能写成“physics普遍提升”或“ORC全部增益来自ODE”。

证据等级仍为retrospective aligned-forcing benchmark，不是独立未来时段、外部站点或issue-time operational validation。稿件的`EVIDENCE_STATUS.md`列出允许与禁止的声明。smoke、partial、旧`lambda=0.0003`和旧PDF继续禁止引用。

## 2026-07-31 物理强迫状态门控扩展（普通benchmark）

入口代码：`数据整理/149_physics_regime_switched_hsdt.py`；结果目录：`results/physics_regime_switched_hsdt_benchmark/`。

该实验没有重新训练GWN，也没有修改既有结果。它把训练期风应力、气压趋势、流速、波能通量和wave-setup proxy做站点内robust标准化，得到因果forcing-intensity index；只在验证集从预先固定的q75、q90、q95候选中选择门槛。普通测试期中，高强迫状态采用learned-ODE ORC，低强迫状态采用persistence adapter；反向门控和168小时错位门控用于负对照。

验证集选择了训练期 `q95`，高强迫样本约占 `8.0%`。五seed普通benchmark结果如下：

| 模型 | Sequence R2 | Lead-24 R2 | 描述性q95 R2 | Event PR-AUC |
|---|---:|---:|---:|---:|
| HS-DT-GWN | 0.735662 | 0.615267 | 0.629197 | 0.524586 |
| HS-DT + persistence adapter | 0.738107 | 0.619779 | 0.633625 | 0.525554 |
| ORC-HS-DT-GWN | 0.738647 | 0.619144 | 0.637804 | 0.528804 |
| Physics-Regime-Switched HS-DT | 0.738151 | **0.620088** | 0.635187 | 0.527472 |

PRS相对原HS-DT的Sequence/Lead-24变化为 `+0.002489/+0.004820`，均5/5 seed；相对persistence adapter仅 `+0.000043/+0.000308`，Lead-24为3/5 seed、精确Wilcoxon `p=0.15625`。因此它可以称为“终点专门化候选”，不能称为稳定统一提升模型。

## 高强迫区间的跨骨干物理归因

下面所有阈值均由训练期q95定义，覆盖测试起点约8.0%；正值表示降低MSE。

| 对比 | Sequence MSE变化 | Lead-24 MSE变化 | Lead-24正向seed数 |
|---|---:|---:|---:|
| GNN ODE-prior - learnable GNN | -0.00027695 | **+0.00083073** | 3/5 |
| GWN physics-loss - no-physics | -0.00002677 | -0.00004527 | 0/5 |
| ORC - HS-DT | **+0.00027078** | **+0.00082681** | 5/5 |
| ORC - zero adapter | +0.00007721 | +0.00015042 | 4/5 |
| ORC - persistence adapter | +0.00001828 | +0.00013014 | 3/5 |

逐lead文件 `physics_regime_per_lead.csv` 和图 `physics_regime_per_lead_effect.png` 显示：GNN ODE-prior的收益主要出现在远期且波动较大；GWN直接physical loss接近中性或负；ORC在多数lead保持正向，且高强迫时的Lead-24收益最稳定。

当前最详细、最可辩护的结论是：**物理信息的作用不是统一地改变所有预测，而是帮助识别何时应使用动力修正；这种作用在强风/强流/高波能状态的远期预测最明显。** 但ORC相对persistence的差值仍小，PRS尚不能替代主稿ORC，也不能把适配器全部增益称为physics-only。

主要文件：`EXPERIMENT_REPORT_CN.md`、`validation_threshold_selection.csv`、`physics_regime_effects.csv`、`physics_regime_effect_summary.csv`、`physics_regime_per_lead.csv`、`physics_regime_switched_model_comparison.png`、`physics_regime_effect_map.png`和`physics_regime_per_lead_effect.png`。

## 2026-07-31 物理作用详细审计（普通 benchmark）

入口：`数据整理/150_physics_effect_audit_benchmark.py`；结果：`results/physics_effect_audit_benchmark/`。

该脚本不重新训练、不用测试标签选阈值，只对已有五 seed 预测按训练期逐站点 q75/q90/q95 forcing 阈值进行分层，并报告组件、站点、lead 和 24-origin block bootstrap。q95 综合强迫区间中，ORC 相对 HS-DT 的 Lead-24 MSE 平均减少 `0.000270782`（约 `1.058%` 的高强迫基线误差，5/5 seed 为正）；直接 GWN physics loss 相对 no-physics 的变化为 `-0.00002677`（0/5 seed 为正）；GNN ODE prior 的条件收益不稳定。该结果支持“物理信息主要用于识别强迫状态下何时启用动力修正”，不支持 physics loss 普遍提升或把适配器全部增益归因于 physics。

## 2026-08-01 物理门控对齐性与滞后搜索

入口：`数据整理/151_physics_alignment_negative_controls.py`、`数据整理/152_validation_locked_lagged_physics_gate.py`；结果分别位于 `results/physics_alignment_negative_controls/` 和 `results/lagged_physics_gate_benchmark/`。

固定 q95 mask 的对齐性负对照发现，1--6 小时错位 mask 在部分指标上同样有效，说明当前 forcing index 尚不能证明唯一的物理对齐关系。验证集限定 q75/q90/q95 与 0/6/12/24/72/168 h 候选后选择 q95、12 h lag；测试集相对 persistence 的 Lead-24 MSE reduction 仅 `0.00001891`，因此该方法不升级为主模型，只作为失败/中性搜索记录和后续改进依据。

## 2026-08-01 投稿前证据审计 PDF

`数据整理/153_build_submission_evidence_pdf.py` 生成 `output/pdf/physics_submission_evidence_audit.pdf`。PDF 汇总普通 benchmark 主排序、物理条件效应、负对照、滞后门控和投稿前最低补强项。它明确不承诺接收概率，不把普通 benchmark 写成独立未来 holdout，也不覆盖现有 Overleaf 主稿。

## 2026-08-10 2026 H1 与 July 独立时间验证

数据准备与冻结评估入口：`数据整理/158_prepare_frozen_2026_holdout.py`、`159_evaluate_frozen_2026_holdout.py`、`160_safety_gated_dual_head_gwn.py`、`161_evaluate_frozen_safety_gated_dual_head_gwn.py`、`162_prepare_frozen_2026_july_holdout.py`、`163_formal_hsdt_independent_validation.py`。主要结果位于 `results/frozen_2026_h1_physics_reliability_gwn/`、`results/frozen_2026_july_safety_gated_dual_head_gwn/` 和 `results/formal_hsdt_independent_validation_2026/`。

2026 H1 首次冻结验证显示，Physics-Reliability 候选相对 matched no-physics ridge 的 sequence、Lead-24 和 q95 R2 分别变化 `-0.011852`、`-0.014383` 和 `-0.082614`，均为 0/5 seed 改善；Event PR-AUC 提高 `+0.016125`，5/5 seed，单侧精确 Wilcoxon `p=0.03125`。因此 H1 不支持 physics 连续修正，只支持事件排序辅助信号。

H1 随后仅作为开发集构建共享 station x 4-hour-group safety gate，并在查看 July 目标前冻结 gate、评估入口、数据准备入口和确认标准。2026 July 一次性验证中，safety-gated 双头相对 matched no-physics 的 sequence、Lead-24 和 q95 R2 分别变化 `-0.002450`、`-0.011948` 和 `-0.022406`；sequence 0/5 seed 改善，168 小时 bootstrap CI 为 `[-0.005299, 0.002675]`。独立 physics 事件头的 PR-AUC 仍提高 `+0.000522`，5/5 seed，`p=0.03125`，但绝对增益很小。该候选正式失败，不得在同一 July 数据上重调后再称独立验证，也不得升级为连续水位主模型。

## 2026-08-11 复杂候选筛选与 residual early-stopping 修正

入口代码：`数据整理/172_uncertainty_supervised_reliability_moe.py`、`173_terminal_physics_adapter_screen.py`、`174_horizon_residual_boosted_gwn_screen.py`、`175_joint_terminal_head_residual_gwn.py` 和 `176_validation_safe_horizon_mask.py`。总览位于 `results/FINAL_MODEL_EXPLORATION_SUMMARY_CN.md`。

Reliability router 的最好 AUROC 约为 `0.552`；terminal physics adapter、joint terminal head 和验证期小样本 residual booster 均没有超过 horizon-only VARX/GWN mixture。旧 full-train residual GWN 曾在测试集得到 sequence R2 `0.80045`，Lead-24 protected 版本曾得到 `0.80055/0.68625/0.69914`，但这些值现在不可引用。

代码审计发现旧 `174` 的 residual head 虽然零初始化，却没有把训练前的 epoch 0 anchor 纳入 early-stopping checkpoint 候选。旧逻辑会在所有训练 epoch 都让验证目标变差时仍强制保留非零修正。修正后最佳 epoch 为 `0`，residual GWN 和 Lead-24 protected 版本的 sequence/Lead-24/q95 R2 均与 horizon anchor 完全相同：`0.798939/0.686250/0.693825`；预测逐元素最大绝对差为 `0.0`。

验证锁定的逐 lead 安全掩码只选择第2和第7小时，测试 sequence R2 为 `0.798909`，低于 anchor。因此该路线停止，不扩展五 seed，不进入论文主表。旧的正向测试值属于 checkpoint 选择缺陷下的偶然结果，不是 target leakage，也不能作为复杂模型优于强基线的证据。修正版结果位于 `results/horizon_protected_residual_gwn_earlystop_fixed_20260811/`。

## 2026-08-11 CF-MOR-GWN P0 残差可预测性代理诊断

入口代码：`数据整理/177_cf_mor_gwn_p0_residual_diagnostic.py`、`178_cf_mor_gwn_p0_structure_audit.py`；结果目录：`results/cf_mor_gwn_p0_residual_diagnostic_20260811/`。

按照 `CF-MOR-GWN_主模型设计与实验路线说明.docx` 的最低成本阶段，先运行 purged blocked cross-fitted VARX proxy，而不是直接运行预计超过14小时CPU的三个正式FS-GWN OOF fold。训练期分为5个时间块，完整输入/目标窗口与held block相交的样本均从训练补集剔除；anchor alpha只由训练期OOF MSE选择。Residual Ridge和零初始化MLP只学习OOF anchor error；验证前半选择模型和bounded-correction kappa，后半确认；未生成测试预测、未计算测试指标。

最佳anchor alpha为`100`，最终选择`no correction`、`kappa=0`。Calibration residual R2为`-0.054037`，Confirmation为`-0.093494`；MLP最佳epoch为`0`。结构审计显示平均8小时残差ACF为`0.332867`，但24小时ACF降至`0.150920`，且0%的station-horizon单元在5个block中保持完全一致的偏差方向。局部相关存在但没有转化为跨时段可迁移的Ridge/MLP修正信号。

状态：`STOP_BEFORE_FS_GWN_OOF`。这只说明P0 proxy未通过，不等于完整CF-MOR-GWN已经被训练或永久否定；但按照预先停止规则，当前不启动正式FS-GWN OOF，也不把CF-MOR-GWN升级为论文主模型。完整边界见`P0_EXPERIMENT_REPORT_CN.md`。

固定 HS-DT-GWN 相对 Multistate FS-GWN 的 sequence residual R2 在两个新时间段均稳定提高：

| 时间段 | Multistate FS-GWN | HS-DT-GWN | 平均提升 | seed 胜率 | 168h bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| 2026 H1 | 0.523230 | 0.538477 | +0.015247 | 5/5 | [0.008283, 0.021647] |
| 2026 July | 0.391619 | 0.409345 | +0.017725 | 5/5 | [0.009072, 0.028388] |

HS-DT 的 Lead-24 按定义完全采用 Multistate 专家，因此两个时间段的 Lead-24 与 Multistate 严格相同，不能声称额外终点提升。July q95 R2 的绝对值很低且 HS-DT 相对 Multistate 不稳定，极端预测仍是明确短板。锁定结论为：**HS-DT-GWN 已获得作为 sequence 主模型的双新时间段支持；Multistate GWN 保留为强基线和 Lead-24 专家；Physics-Reliability 只保留为事件排序辅助头，physics 连续修正未获独立验证支持。**

最小可移交包：`final_models_python_package_20260810.zip`；ZIP SHA-256 为 `89F48E9C0D49654C229F96C29741A7E06CE540B5C7A94D7525ECECD669E1A1C8`。包内不含 H1/July 验证目标、历史结果、smoke/partial 结果或旧 PDF。

## 2026-08-11 逐时效专门化与专家互补审计

入口：`数据整理/179_horizon_specialization_complementarity.py`；结果：`results/horizon_specialization_complementarity_20260811/`。

该脚本复用既有 retrospective 与严格 2025 H2 五 seed 预测，不重新训练。严格 2025 H2 中，Multistate 专家在 11 个 lead 优于 Eta-only，Eta-only 在 13 个 lead 优于 Multistate；专家误差平均相关系数为 `0.932068`，seed 范围为 `0.876304--0.962878`。HS-DT 的 sequence R2 相对 Eta-only 和 Multistate 分别平均提高 `+0.009155` 和 `+0.012428`，均为 5/5 seed 获胜。状态：`HSI_COMPLEMENTARITY_SUPPORTED`。该结果支持与监督方式相关的时效专门化和双专家互补，不支持严格因果的“supervision-induced”表述，也不支持 HS-DT 的 Lead-24 额外提升，因为 Lead-24 按定义直接使用 Multistate 专家。

## 2026-08-11 VARX 严格时间协议验证

入口：`数据整理/180_strict_varx_chronological_validation.py`；结果：`results/strict_varx_chronological_validation_2025_h2/`。

VARX-Ridge 使用严格因果预处理，训练期为 2023--2024、验证期为 2025 H1、测试期为 2025 H2；Ridge alpha 只由验证集选择，未来 residual 不作为输入。严格 2025 H2 的 Sequence/Lead-24/描述性 q95 R2 为 `0.714563/0.429092/0.642917`；普通 retrospective 对应为 `0.789727/0.644221/0.718639`。Sequence 从 retrospective 到 strict 的相对衰减为 `9.52%`，低于 Eta-only、Multistate 和 HS-DT。2025 H2 已在项目开发中被查看，因此证据等级是 chronological refit backtest，不是 untouched holdout。

## 2026-08-11 VARX--深度模型鲁棒性配对审计

入口：`数据整理/181_strict_varx_deep_robustness_audit.py`；结果：`results/strict_varx_deep_robustness_audit_2025_h2/`。

VARX 在 24 个 lead 的五 seed 平均曲线上均高于 Eta-only GWN、Multistate GWN 和 HS-DT。其 Sequence R2 平均优势分别为 `+0.067373/+0.070646/+0.058218`，q95 优势为 `+0.118519/+0.137528/+0.121364`，三组比较均 5/5 seed 获胜；Sequence 的 168 小时移动块 bootstrap 区间对三个模型均在 5/5 seed 完全高于零。Lead-24 相对 Eta-only、Multistate 和 HS-DT 分别为 4/5、3/5、3/5 seed 获胜，且后两组没有 seed 的区间完全高于零。结论是 VARX 具有更强的整体时间稳定性，但不是 Lead-24 的稳定全面统治；论文需要采用 linear--deep complementary robustness，而不能继续声称深度模型在所有协议总体最强。

## 2026-08-11 跨架构 physics utility matrix

入口：`数据整理/182_architecture_conditional_physics_utility.py`；结果：`results/architecture_conditional_physics_utility_20260811/`。

该审计不重新训练，将严格 2025 H2 的 matched physics increment、capacity-confounded candidate 和 capacity-only control 明确分层，并把普通 retrospective 高强迫分析放在独立表格与图面板中。严格协议共有 16 个匹配的“物理增量 × 指标”组合，正向且单侧精确 Wilcoxon `p<=0.05` 的为 0 个。GNN-BiGRU direct physics loss 的 Sequence 变化为 `-0.004020`；GWN joint ODE 相对同容量 no-prior control 为 `+0.004440`，HS-DT dual ODE 相对同容量 control 为 `+0.002211`，二者均不显著且 Lead-24 为负。普通 benchmark 的 q95 combined forcing 中只有 `ORC - HS-DT` 为 5/5 seed 正向，但该对比包含 adapter 容量混杂。锁定结论：physics utility 依赖 backbone、supervision、injection 和 metric，不存在已确认的普遍连续水位精度增益。

## 2026-08-11 冻结 VARX 的 2026 H1/July 诊断

入口：`数据整理/183_frozen_varx_2026_diagnostic.py`；结果：`results/frozen_varx_2026_diagnostic_20260811/`。

该实验只使用 2023--2024 训练、2025 H1 选择的 VARX-Ridge，锁定 `alpha=100`，未在 2025 H2 或 2026 target 上重拟合。H1 与 July 数据扩展中的训练/验证设计矩阵逐元素一致，VARX 与五 seed 深度预测的时间戳和真实目标也逐元素对齐。由于 2026 H1 和 July 在该诊断设计前已经被查看，结果等级强制标记为 `diagnostic_only_already_viewed_not_independent_confirmation`。

2026 H1 中 VARX Sequence/Lead-24 R2 为 `0.431512/-0.179686`，低于 HS-DT 的 `0.538477/0.290481`；相对 HS-DT 的 Sequence 差值为 `-0.106965`，0/5 seed 获胜。2026 July 中 VARX 为 `0.365286/-0.116782`，低于 HS-DT 的 `0.409345/0.420625`；Sequence 差值为 `-0.044059`，0/5 seed 获胜。逐 lead 图显示 VARX 在早期仍强，但约第 9--13 小时后被深度模型反超，并在两个时期的 Lead-24 变为负值。该结果推翻“VARX 普遍时间更稳定”的宽泛表述，支持 period- and horizon-dependent model hierarchy。

## 2026-08-11 主线机制图与新版论文

入口：`数据整理/184_build_mainline_mechanism_assets.py`；结果：`results/paper_mainline_assets_20260811/`。脚本生成三时期逐 lead 模型层级图、station × horizon supervision specialization 热图和精简 physics utility 主图。

新版双栏主稿位于 `publication_final/overleaf_submission_mainline_20260811/`，标题为 `Supervision-Induced Expert Complementarity and Temporal Model-Hierarchy Reversals in Coastal Sea-Level Residual Forecasting`。主稿严格区分 retrospective、2025 H2 chronological refit 和 2026 evidence tier；将 HS-DT 定位为深度 Sequence 主模型，将 VARX 结论收缩为时段/时效依赖，将 physics 定位为条件性边际效用，并将失败复杂候选整理为 complexity-boundary 负控。Tectonic/BibTeX 编译成功，最终 PDF 为 7 页双栏，逐页视觉检查未发现裁切、重叠或表格越界。

## 2026-08-12 论文逻辑评估整改、证据账本与冻结清单

入口：`数据整理/185_logic_revision_evidence_audit.py`；结果：`results/manuscript_logic_revision_20260811/`。同时修订了 `数据整理/179_horizon_specialization_complementarity.py` 与 `数据整理/184_build_mainline_mechanism_assets.py` 的主线图措辞和指标范围。

本次整改依据 `当前论文逻辑评估与修改建议_20260811.docx`，完成以下可立即解决的事项：

- 标题从 `Supervision-Induced` 改为更稳妥的 `Supervision-Associated`，因为两专家虽然共享输入、时间切分、隐藏骨干超参数和优化设置，但输出头维度与训练目标不同，不能解释为严格单因素因果处理。
- 在 Methods 中明确 HS-DT 规则是在 Tier-A benchmark 后形成；Tier-A 分数为 post-hoc，2025 H2 为已查看的 refit backtest，2026 deep 为 frozen evaluation，2026 VARX 仍是 diagnostic-only。
- Event PR-AUC 从主线正文与物理图移出；主图仅保留 Sequence、Lead-24 和描述性 q95 连续指标。
- 五 seed 精确 Wilcoxon 降为补充性描述；主线使用 effect size、5/5 或 4/5 方向一致性和 168 h moving-block bootstrap。
- Related Work 新增 multi-horizon forecasting、forecast combination/MoE、out-of-time evaluation 和 theory-guided/scientific ML 的 10 篇核验 DOI 文献。

审计脚本生成 6 条 machine-readable claim ledger。关键锁定结果为：HS-DT 相对 Multistate 的 Sequence 增量在 2025 H2、2026 H1 和 2026 July 分别为 `+0.012428/+0.015247/+0.017725`，对应 168 h 区间均高于零；HS-DT Lead-24 仍与 Multistate 完全相同。Matched physics 中最大 Sequence 增量为 `+0.004440`，四个 Lead-24 匹配对照仅 1 个为微小正值，不能声称 physics 普遍提分或解释 `0.6153`。

脚本按 `alpha=100` 重新拟合并冻结 2023--2024 VARX 系数，与既有严格 2025 H2 VARX 预测的逐元素最大绝对差为 `0.0`。Prospective confirmation manifest 共登记 15 个唯一 SHA-256：10 个 Eta/Multistate GWN checkpoint、4 个关键脚本和 1 个冻结 VARX 系数文件。该 manifest 只表示下一次真正未见验证已具备冻结条件，不表示外部验证已经完成。

当前修订稿位于 `publication_final/overleaf_submission_logic_revision_20260811/`，标题为 `Supervision-Associated Expert Complementarity and Temporal Model-Hierarchy Changes in Coastal Sea-Level Residual Forecasting`。Tectonic/BibTeX 编译成功，最终 PDF 为 8 页 A4 双栏；所有 8 页已视觉检查，未发现裁切、重叠、缺图或参考文献中夹入结果浮动体。旧 `publication_final/overleaf_submission_mainline_20260811/` 保留为历史版本，不再作为当前入口。

最终可上传 ZIP：`publication_final/overleaf_submission_logic_revision_20260812_final.zip`。PDF SHA-256：`D8734731D7E3C10520F1B2A2CA9285D228D52F982D0245538EE7FFE559F9C963`；ZIP SHA-256：`F7D8B3827667F6E56987D4428F0C00F9814C5D4AFCD22DE9843492976A1F2771`。

## 2026-08-12 Delaware Bay--River 外部地区确认（当前最新证据）

入口：`数据整理/186_download_prepare_delaware_bay_external_confirmation.py`、`187_run_delaware_bay_external_confirmation.py`、`188_analyze_delaware_bay_external_confirmation.py`。冻结协议：`configs/delaware_bay_external_confirmation_20260812.json`及其freeze manifest。结果：`results/delaware_bay_external_confirmation_20260812/`。论文：`publication_final/overleaf_submission_external_confirmation_20260812/`。

10/10个非重叠站通过事前覆盖率门槛，目标水位、潮汐和残差在三个时间段均为100%覆盖。正式模型在修复局地气象派生量的二次前向保持后从新数据重新训练；修复前输出已隔离，禁止作为证据。五seed独立运行均值为：Eta-only Sequence/Lead-24 R2=`0.663913/0.521657`，Multistate=`0.675723/0.574273`，HS-DT=`0.680256/0.574273`；确定性VARX=`0.765687/0.576291`。

冻结判定为2/3通过：

- supervision structure：失败。Eta仅赢2/24 leads和0/10站；Multistate赢22/24 leads和10/10站。
- HS-DT complementarity：通过。相对Eta为`+0.016343`、5/5 seed、168h CI `[0.009489, 0.024196]`；相对Multistate为`+0.004533`、4/5 seed、CI `[-0.001438, 0.010116]`。后者区间跨0，必须降格表述。
- VARX/deep逐lead层级：通过。五seed预测均值曲线上Lead 1--8的HS-DT减VARX平均为`-0.118445`，Lead-24为`+0.013336`；但仅2/5单seed在Lead-24超过VARX。

论文可引用结论：第二地区支持低复杂度双专家的正向Sequence互补点估计和时效依赖的线性--深度层级变化，但不支持原区域13比11的具体监督专门化模式。VARX是外部总体Sequence最强模型，HS-DT不是全面冠军。证据等级为 **prospectively locked spatial confirmation**；因使用与原地区相同的2025 H2测试日历，且只做本地哈希锁定，不能称later-period temporal holdout、公开preregistration或第三方independent validation。

## 2026-08-17 C4 概率动态双专家五种子确认与因果填补敏感性

入口：`additional_experiments/three_person_tiera_extensions_20260816/person3_probabilistic/run_probabilistic.py`、`analyze_c4_confirmation.py`、`run_causal_sensitivity.py` 和 `analyze_causal_sensitivity.py`。结果位于 `additional_experiments/three_person_tiera_extensions_20260816/runs/person3_probabilistic_formal_screening/` 与 `runs/person3_probabilistic_causal_sensitivity/`；完整报告为 `additional_experiments/three_person_tiera_extensions_20260816/C4_FIVE_SEED_CAUSAL_CONFIRMATION_REPORT_CN.md`。

C4（`DG_JOINT_PHYS_SPLINE_PROB`）包含 Eta-only/Multistate 两个 GWN 专家、动态 gate、joint training、四状态可学习 ODE 物理残差、spline 激活和 heteroscedastic Gaussian 输出。五个注册 seed 均从头训练，checkpoint 只按 validation Gaussian NLL 选择，sigma scaling 只用 validation。

五种子 C4 与 HS-DT 的 Sequence R2 分别为 `0.735307 +/- 0.009281` 和 `0.735662 +/- 0.003759`，平均差 `-0.000355`、2/5 seed 获胜；Lead-24 为 `0.597428` 对 `0.615267`，平均差 `-0.017839`、1/5 seed 获胜。训练期按站点 q95 R2 平均差为 `+0.012900`、3/5 seed 获胜，但五种子平均预测的168h区间为 `[-0.006826, 0.042617]`。Lead-24 的168h区间为 `[-0.027920, -0.003048]`，完全低于0。

严格因果前向填补 seed 42 重训使 Sequence/Lead-24/训练期按站点 q95 R2 相对历史离线填补分别下降 `-0.055266/-0.160329/-0.134591`；三项168h区间均完全低于0。该结果不是目标泄漏证明，但说明 C4 对 offline interpolation 高度敏感，不能直接解释为 operational forecast 性能。

锁定状态：`C4_NOT_CONFIRMED_AS_MAIN_MODEL`。seed 42 的 `0.7495/0.6333/0.7012` 只能作为单 seed screening，不得单独作为最终结果。C4 可保留为概率/尾部探索扩展；当前深度 Sequence 主模型仍为 HS-DT-GWN。
# 2026-08-18 C4 与正式 HS-DT 专家对齐审计和 seed 42 筛选

入口：`additional_experiments/c4_hsdt_aligned_20260818/run_aligned_c4.py`；干净重算：`reevaluate_clean.py`；正式结果：`clean_evaluation/`；完整报告：`AUDIT_AND_RESULTS_CN.md`。

代码审计确认，原始 C4 `DG_JOINT_PHYS_SPLINE_PROB` 的两个 GWN 专家从随机初始化联合训练，`formal_results_root=None` 且变体 `frozen=False`，因此没有加载用户正式 HS-DT 的 Eta-only/Multistate checkpoint。原始 C4只能称为独立动态双专家概率 GWN，不能称为 HS-DT 升级。

对齐版加载并冻结用户正式 seed 42 Eta-only 和 Multistate checkpoint，冻结专家始终保持 eval，初始化保留 lead 1--23 等权、lead 24 使用 Multistate 的 HS-DT 规则，只训练动态 gate、零初始化有界修正和 Gaussian 概率头。专家重算预测与正式保存预测最大差为 `2.38e-7/7.15e-7`，目标差为0。

seed 42 的固定 HS-DT Sequence/Lead-24/q95 R2 为 `0.735398/0.628132/0.632365`；对齐无 physics C4 为 `0.744165/0.632280/0.666397`，增量为 `+0.008766/+0.004147/+0.034032`。匹配 physics 版为 `0.744144/0.632215/0.666330`，相对无 physics 净变化 `-0.000020/-0.000064/-0.000068`。因此当前正向筛选支持动态适配容量，不支持 physics-only 增益；仍需其余四个注册 seed 确认，不能升级为最终主模型。

第一次尝试的 `runs/` 因冻结专家 BatchNorm running statistics 漂移而无效；`runs_fixed_bn/` 的 checkpoint 有效，但其最初保存的原始 Multistate 字段记录错误，唯一可引用指标是从相同 checkpoint 修正重算的 `clean_evaluation/clean_summary.csv`。
# 2026-08-18 三人实验分工与精简交付包

交付目录：`delivery_20260818/`。我的部分运行 C4-HSDT-Frozen no-physics 五种子主模型确认；同学 A 运行 Local/N0 空间信息消融；同学 B 运行同一正式专家上的 C4-HSDT-Frozen physics matched ablation。三者共享同一 Tier-A 数据和运行时，测试指标由主研究者统一汇总。

为避免旧包中重复数据、预测缓存和 checkpoint 导致体积过大，当前交付改为一份 `common_resources_20260818.zip`（约13.94 MiB）加三个代码包（各约0.01 MiB）。共享包不含历史结果、smoke/partial、PDF 或旧 checkpoint；B 和主模型先从头按锁定协议生成正式专家。校验和见 `delivery_20260818/archives/SHA256SUMS.txt`。
