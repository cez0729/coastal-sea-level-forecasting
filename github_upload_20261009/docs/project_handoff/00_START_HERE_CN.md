# 海平面项目交接入口

更新时间：2026-08-17

这套文档是新对话理解项目的唯一推荐入口。先读本文件，再按顺序阅读 `01_RESEARCH_STORY_CN.md`、`02_FILE_MAP_CN.md`、`03_EXPERIMENT_REGISTRY_CN.md` 和 `04_DATA_PREPROCESSING_CN.md`。

## 一句话说明项目

本项目研究美国东北部七个沿海站点未来24小时的非潮汐海平面残差预测，比较图时空网络、ODE启发特征、多状态监督和物理残差正则化分别带来了什么作用。

## 2026-08-12 逻辑评估整改后的当前投稿主线

新加入的同协议 VARX、2026 时段诊断和 Delaware Bay--River 第二地区确认共同改变了“总体冠军”叙述。当前主稿不再声称 GWN、HS-DT 或 VARX 在所有时期总体最强，而采用以下锁定结论：

1. Graph WaveNet 是当前最强的深度图时空骨干；HS-DT-GWN 是获得多时段支持的深度 Sequence 主模型。
2. Eta-only 与 Multistate 专家在原区域呈现非单调的优势分工；严格 2025 H2 为13 vs 11个lead，平均误差相关为`0.932`。该具体模式未迁移到外部区域，后者为Multistate赢22/24 leads和10/10站。
3. VARX 在 retrospective 和严格 2025 H2 总体领先，但冻结规则在 2026 H1/July 的事后诊断中低于 HS-DT，并在远期 lead 明显退化；正确结论是模型层级依赖 period 和 horizon，而不是线性模型普遍更鲁棒。
4. Physics 的主论据改为 matched effect size 与方向一致性：最大匹配 Sequence 增量为 `+0.004440`，GWN/HS-DT ODE 对照的 Lead-24 均为负；五 seed 精确检验只作为描述性证据。
5. 当前主稿位于 `publication_final/overleaf_submission_external_confirmation_20260812/`，标题为 `Supervision-Associated Expert Complementarity and Temporal Model-Hierarchy Changes in Coastal Sea-Level Residual Forecasting`。2026 的 HS-DT 专家比较是冻结评估；后来加入的原区域 VARX 对比只允许称 diagnostic-only。
6. 十站第二地区在目标下载和神经评分前完成本地协议锁定，三项复现标准通过2项：专家互补和逐lead线性--深度排序变化通过，具体supervision specialization失败。它是prospectively locked spatial confirmation，不是后续时间holdout或公开预注册。
7. Event PR-AUC 已移出主线；描述性 q95 也不是 operational storm-surge event。下一步仍需公开预注册的后续时间段或水文动力差异更大的地区。
8. 2026-08-17 完成 C4 概率动态双专家五种子确认：Sequence 均值 `0.735307`，略低于 HS-DT 的 `0.735662`；Lead-24 低 `0.017839`，168h区间完全低于0。严格因果填补 seed 42 又显著下降，因此 C4 不升级为最终主模型，只保留为概率/尾部探索扩展。

## Retrospective 历史核心结论

1. **Graph WaveNet 是总体最强的架构基线**：sequence residual R2 = 0.7261，24小时终点 residual R2 = 0.5825，描述性 q95 residual R2 = 0.6271。
2. **Multistate Graph WaveNet（不加 physics）是24小时终点最高模型**：terminal residual R2 = 0.6153。
3. **在完全相同的 Multistate Graph WaveNet 上加入 physics 后没有带来增益**：terminal R2 变化为 -0.000232 +/- 0.000189，仅1/5个 seed 获胜。
4. **修正双向隐藏状态后的五 seed GNN-BiGRU 阶梯中，完整 Physical-loss 配置优于基础固定图模型**：sequence R2 从0.6385升至0.6732，24小时终点从0.4925升至0.5553；但这部分差距同时包含可学习图、多状态辅助监督、终点加权和物理项，不能全部归因于 physics。
5. 匹配的 GNN-BiGRU 消融显示 physics-only 的增益较小且依赖配置：状态消融约 +0.0042，预设 Priority-2 权重下约 +0.0094，只有3/5个 seed 提升。
6. **Cycle 项没有被独立证明有效**；Cycle-plus-trend 24小时 terminal R2 = 0.5453，但与 trend-only 几乎相同。
7. **直接多输出优于严格自回归滚动**；在没有未来外部强迫的递归实验中，24小时 terminal R2 = -0.0999，反映误差累积和未来 forcing 不可得问题。

## HS-DT-GWN 的当前状态

HS-DT-GWN（Horizon-Specialized Dual-Task Graph WaveNet）起源于后验探索，现在已经作为固定、低自由度的深度专家组合写入主论文；它是深度Sequence主模型，但不是所有模型和所有时效上的全局冠军。

它使用两个 Graph WaveNet 专家：

- eta-only GWN：只预测 residual eta，轨迹和 q95 较强；
- multistate GWN：同时预测 eta、流速状态和波浪状态，24小时终点较强。

当前规则是：预测第1–23小时使用两个专家平均，第24小时使用 multistate 专家。五个 seed 的 HS-DT-GWN 结果为 sequence R2 = 0.7357、terminal R2 = 0.6153、q95 R2 = 0.6292。

历史benchmark结果有价值，但规则是在分析该benchmark后提出，因此该分数仍有后验设计风险。冻结公式后的2025 H2端到端重训、2026深度模型评估和十站第二地区空间确认均支持Sequence正向互补；外部相对Multistate的增量只有`+0.004533`且168h区间跨0，因此应称可复现的低复杂度互补点估计，而不是确定的大幅优势。真正更强的确认仍需公开预注册的later-period holdout。

## ORC-HS-DT-GWN 后续探索

2026-07-27新增 **ODE Residual-Corrected HS-DT-GWN（ORC-HS-DT-GWN）**：保留锁定 HS-DT 预测，并迁移 learned-ODE residual adapter 的修正量。五 seed 的 sequence R2 = 0.738647、terminal R2 = 0.619144、描述性 q95 R2 = 0.637804、Event CSI = 0.237290、PR-AUC = 0.528804；相对 HS-DT 的主要指标均5/5 seed改善。

该模型仍是同一历史 benchmark 上的 post-hoc exploratory extension。Zero/persistence adapter也能贡献部分增益，且 persistence terminal R2 = 0.619779，略高于 learned ODE；不能把全部提升归因于ODE。完整登记见 `03_EXPERIMENT_REGISTRY_CN.md`，实验报告见 `results/ode_residual_corrected_hsdt_gwn/EXPERIMENT_REPORT_CN.md`。

## 2026-07-31 双专家物理 prior 探索

`results/hsdt_expert_physics_conditioned/` 将 causal ODE prior 通过 horizon gate 注入两个 HS-DT GWN 专家。相对冻结 HS-DT，双专家 prior 的 sequence R2 为 `0.669496 +/- 0.007805`，提高 `+0.013151`；但相同轮数的 no-prior fine-tuning control 已达到 sequence `0.667285`、Lead-24 `0.435667`，prior 模型没有超过该控制的终点表现。因此它只能称为“prior + fine-tuning capacity”的 exploratory extension，不能声称 physics-only 提升。

`results/hsdt_multistate_prior_physics_loss_active/` 激活可微 `lambda=0.0002` physical residual loss 后，相对 prior-only 五 seed 净增量接近零（sequence `+0.0000018`、Lead-24 `+0.000027`）。直接 physics loss 在强 GWN 设置下仍未显示独立贡献。

`results/formal_gwn_ode_physics_factorial_2025_h2/` 新增单体 multistate GWN 的冻结 ODE 五 seed 对照：只训练 causal ODE prior 和 horizon gate，不微调 GWN。相对纯 GWN，sequence R2 `+0.003492`（3/5，`p=0.3125`）、Lead-24 `-0.004967`、q95 `-0.008799`、Event PR-AUC `+0.007400`（4/5，`p=0.09375`）。因此冻结 ODE 只有事件排序的弱趋势，不能称为稳定精度提升。跨 GNN/GWN/HS-DT 的统一证据矩阵、配对统计和主图位于 `results/paper_physics_factorial_2025_h2/`。

单体 GWN joint ODE 的 q95 R2 为 `0.549979`，高于同协议 no-prior 微调 control 的 `0.544395`，但配对差值仅 `+0.005584`、未显著；因此它是“physics-conditioned extreme-forecast candidate”，不是已确认的 physics-only 最强模型。

## 2025 H2端到端重训回测

2026-07-28已冻结HS-DT公式和ORC修正比例，使用严格因果预处理完成五seed端到端按时间重训：训练期为2023–2024，验证期为2025 H1，测试期为2025 H2。两个Graph WaveNet专家均从头训练，ORC同时与匹配refiner/gate的zero和persistence控制比较。

- HS-DT sequence R2 = 0.656345 +/- 0.005359；相对Eta-only提升+0.009155、相对Multistate提升+0.012428，均5/5 seed，单侧精确Wilcoxon p=0.03125；168小时块bootstrap 95% CI分别为[0.001407, 0.019554]和[0.003705, 0.021243]。
- HS-DT Lead-24 R2 = 0.424692 +/- 0.014625；按定义与Multistate完全相同，相对Eta-only仅3/5 seed改善，95% CI跨0，不能声称终点稳定提升。
- ORC sequence R2 = 0.661002 +/- 0.007508，但相对HS-DT仅3/5 seed、p=0.15625、95% CI为[-0.004912, 0.012708]；Lead-24平均反而下降0.003252。
- ORC在Lead-24上0/5 seed优于zero控制，1/5 seed优于persistence控制，不能把它作为确认的HS-DT升级或把收益归因于ODE。

该实验消除了旧checkpoint和旧预测复用问题，可以支持“HS-DT轨迹融合在新时间切分重训中稳定改善”的稳健性结论；但研究过程中已经查看过2025 H2，因此证据等级仍是 **end-to-end chronological refit backtest**，不是untouched independent holdout。完整结果位于 `results/confirmatory_hsdt_orc_refit_2025_h2/`。

## 当前论文定位

最稳妥的论文定位是：

> A rigorous attribution study of physics regularization, multistate supervision, and graph temporal architectures for coastal sea-level residual forecasting.

不要把论文定位成“Physics Loss 打败所有模型”，也不要把图残差称为完整浅水方程求解器。论文的价值在于把架构、辅助状态、损失设计和物理项拆开，报告了正面和负面结果，并说明物理收益依赖 backbone 和 loss design。

## 当前任务定义

- 历史窗口：过去24小时。
- 预测窗口：未来24小时。
- 输出方式：direct multi-output，一次输出全部24个未来小时。
- 主预测目标：非潮汐残差 eta = observed water level - astronomical tide。
- 总水位重建：predicted total level = predicted residual + tide prediction。
- 数据切分：按时间顺序约 70%/15%/15% train/validation/test。
- 主研究区：美国东北部七个站点，2023–2025小时数据。
- 训练时 physics weight：lambda = 0.0002，必须由 validation 锁定，不能根据 test 重新选择。

## 推荐阅读顺序

1. `project_handoff/01_RESEARCH_STORY_CN.md`：从问题到结论的故事线。
2. `project_handoff/02_FILE_MAP_CN.md`：每个主要目录和文件的作用。
3. `project_handoff/03_EXPERIMENT_REGISTRY_CN.md`：代码、结果、结论和可否引用。
4. `project_handoff/04_DATA_PREPROCESSING_CN.md`：数据来源、变量和因果性边界。
5. `project_handoff/05_NEW_CONVERSATION_PROMPT_CN.md`：复制到新对话的最短说明。
6. `project_handoff/06_CLEANUP_LOG_CN.md`：为什么一些旧路径已经归档或删除。
7. `project_handoff/FILE_MANIFEST.csv`：研究文件逐文件索引；排除了本机环境和缓存。
8. `publication_final/overleaf_submission_external_confirmation_20260812/main.pdf`：当前完整论文版本。
9. `results/horizon_specialized_dual_task_gwn/NEW_MODEL_EXPERIMENT_REPORT_CN.md`：HS-DT-GWN 的独立实验记录。

## 不能混用的结果

- 旧的未修正四模型 GNN-BiGRU 结果不能继续作为正式证据；修正后的五 seed 阶梯应单独作为诊断表，与 Graph WaveNet 强骨干主表区分。
- `lambda=0.0003` 的旧 Priority-2 结果不能替代锁定的 `lambda=0.0002`。
- test-defined q95 只可称为 descriptive q95，不可称为业务事件阈值。
- oracle future-forcing 自回归结果只能作为 forcing availability diagnostic。
- HS-DT的Tier-A历史分数仍是post-hoc；后续冻结评估和第二地区空间确认可正式引用，但不能升级为全面超过VARX或已经完成later-period temporal confirmation。

## 继续工作前必须注意

2026-07-28已将 BiGRU 表征修正为 `torch.cat([h_n[-2], h_n[-1]], dim=-1)` 并完成五 seed 四模型重跑。正式结果位于 `results/corrected_bigru_ladder/merged/`，已加入海平面精简稿。Graph WaveNet 结果不受这一具体 BiGRU 修正影响。

## 2026-08-12 Delaware Bay--River 第二地区前瞻锁定空间确认

当前投稿入口更新为 `publication_final/overleaf_submission_external_confirmation_20260812/`。第二地区包含10个与原7站完全不重叠的 Delaware Bay--Delaware River 站点，沿用相同34类输入与24小时直接预测任务。协议在目标下载和神经测试指标产生前本地哈希锁定：2023--2024训练、2025 H1验证、2025 H2测试；模型为VARX、Eta-only FS-GWN、Multistate FS-GWN和固定HS-DT；五个seed为42、123、2024、2025、3407。

外部主结果为：VARX Sequence R2=`0.765687`；Eta-only=`0.663913 +/- 0.011455`；Multistate=`0.675723 +/- 0.006147`；HS-DT=`0.680256 +/- 0.005841`。HS-DT相对Eta-only提高`+0.016343`（5/5 seed，168h CI `[0.009489, 0.024196]`），相对Multistate提高`+0.004533`（4/5 seed，CI `[-0.001438, 0.010116]`）。后一个区间跨0，因此只能称正点估计，不能称高精度确认。HS-DT Lead-24按定义与Multistate完全相同。

三项事前判据中2项通过：低自由度双专家互补通过；VARX与深度模型的逐lead排序变化通过；原区域的supervision specialization不通过。外部区域中Eta仅赢2/24 leads、0/10站，Multistate赢22/24 leads和10/10站。论文必须同时报告这项失败，结论应是“互补和时效层级具有一定迁移性，但具体监督分工依赖地区”。VARX仍是外部Sequence总体最强模型；五seed预测均值曲线仅在Lead-24上由HS-DT比VARX高`0.013336`，且只有2/5单seed超过VARX，不能写成稳定全面反超。

该实验可称 **prospectively locked spatial confirmation**，不能称公开预注册、第三方独立复核或later-period temporal holdout。两个地区使用同一2025 H2日历测试期，下一项真正关键补强仍是公开预注册的后续时间段或水文动力差异更大的第三地区。外部实验正式结果位于 `results/delaware_bay_external_confirmation_20260812/`；预处理泄露修复前输出已隔离到 `results/archive_invalid_external_doublefill_20260812/`，严禁引用。
