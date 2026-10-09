# 研究故事线与最终解释

## 第一幕：先预测真正困难的部分

潮位图可以较好地描述天文潮，但实际水位还会受到风、气压、海流和波浪影响。于是项目没有直接把可预测的潮汐周期当成全部任务，而是先计算：

\[
\eta(t) = z_{observed}(t) - z_{tide}(t),
\]

其中 eta 是非潮汐水位残差。模型真正学习的是残差，最后再加回未来潮汐得到总水位。

## 第二幕：把沿海站点组织成图

七个站点不是七条互不相关的时间序列。站点之间存在空间距离、共同天气系统和海流传播关系，因此把站点作为图节点，把距离、训练期残差相关性或可学习图权重作为空间信息，再用时间编码器处理过去24小时。

## 第三幕：逐步拆分模型贡献

最早的四模型阶梯不是为了声称最终冠军，而是为了回答“每一项到底做了什么”：

| 阶梯 | 作用 |
|---|---|
| Fixed-graph GNN-BiGRU | 基础固定空间图和时间编码器 |
| Learnable-graph GNN-BiGRU | 测试空间关系是否应由数据学习 |
| ODE-prior GNN-BiGRU | 加入 persistence、图残差和局地趋势等动态先验特征 |
| Physical-loss GNN-BiGRU | 预测 eta、流速和波浪状态，并加入图离散动力残差 |

ODE-prior 不是数值 ODE 求解器；它是把过去数据构成的动态特征交给网络。Physical-loss 也不是完整水动力 PDE 求解器；它是一个软正则项。

## 第四幕：强基线改变了排名

加入 DCRNN 和 Graph WaveNet 后，研究发现架构容量和时间建模能力比早期 GNN-BiGRU 更强：

| 模型 | Sequence residual R2 | 24h terminal residual R2 | q95 residual R2 |
|---|---:|---:|---:|
| Fixed GNN-BiGRU（修正后） | 0.6385 | 0.4925 | 0.4344 |
| Learnable graph GNN-BiGRU（修正后） | 0.6667 | 0.5229 | 0.4886 |
| ODE-prior GNN-BiGRU（修正后） | 0.6632 | 0.5189 | 0.4944 |
| Physical-loss GNN-BiGRU（修正后） | 0.6732 | 0.5553 | 0.5174 |
| DCRNN | 0.6963 | 0.5541 | 0.5533 |
| Eta-only Graph WaveNet | **0.7261** | 0.5825 | **0.6271** |
| Multistate GWN, no physics | 0.7250 | **0.6153** | 0.6179 |
| Multistate GWN + physics | 0.7249 | 0.6150 | 0.6174 |

这一步很重要：论文不能隐藏强基线，也不能把早期 physical-loss GNN-BiGRU 的提升写成总体最优。

## 第五幕：把 physics 的真实贡献单独拿出来

修正后的完整 Physical-loss 配置相对于固定图 GNN-BiGRU 的 terminal R2 提升约0.0628，5/5个 seed 提升；但这不是 physics-only 的公平差值，因为同时改变了图、辅助状态和损失设计。ODE-prior相对Learnable graph反而变化约-0.0041，说明这组三个动态先验通道没有得到稳定支持。

匹配实验给出更可靠的归因：

- 相同多状态 GNN-BiGRU 的状态消融：physics-only 约 +0.0042 terminal R2；
- 预设终点和辅助任务权重下：physics 平均 +0.0094，只有3/5 seed 提升；
- 相同 Multistate Graph WaveNet：physics 变化 -0.000232 +/- 0.000189，仅1/5 seed 获胜。

结论不是“物理没有价值”，而是：物理正则的价值是小幅、条件依赖的远端稳定化作用，并且会被更强时空架构和多状态监督掩盖。

## 第六幕：预测方式也会改变结果

Cycle-plus-trend 仍是 direct multi-output 模型，训练时约束未来轨迹的趋势和潜在一致性；它不是自回归。结果为24小时 terminal R2 = 0.5453，但 component ablation 显示：

- no cycle/no trend：0.5437；
- trend only：0.5438；
- cycle only：0.5372；
- cycle + trend：0.5453。

因此可以说轨迹一致性思路可行，但不能说 Cycle 项已经有稳定独立贡献。

严格单步递归会把上一步预测再送回输入窗口。未来外部变量不可得时，24小时 terminal R2 = -0.0999；提供真实未来外部变量时为0.5883，但后者不是可部署结果。这个对照支持主实验采用一次输出未来24小时。

## 第七幕：HS-DT-GWN 是后续扩展

HS-DT-GWN 发现 eta-only 和 multistate GWN 在不同预测目标上具有互补性：前者擅长整体轨迹，后者擅长24小时终点。当前等权规则在第1–23小时平均两个专家，第24小时使用 multistate 专家，五 seed 结果为：

- sequence R2：0.7357；
- terminal R2：0.6153；
- q95 R2：0.6292。

但是它是基于已有 benchmark 提出的后验规则，尚未完成独立 chronological holdout。因此当前文章中应把它写成 exploratory extension，或者冻结规则后补验证再作为主模型。

## 第八幕：冻结后重训支持轨迹融合，但不支持ORC稳健升级

冻结HS-DT公式和ORC修正比例后，项目用2023–2024训练、2025 H1验证、2025 H2测试，从头训练了五seed双专家和三个修正器控制。HS-DT的sequence R2为0.6563，相对Eta-only和Multistate分别提升0.0092和0.0124，均5/5 seed改善，168小时块bootstrap区间完全大于0。这支持HS-DT作为“轨迹融合”稳健性实验。

但是HS-DT的第24小时按定义直接取Multistate预测，Lead-24 R2为0.4247；相对Eta-only只在3/5 seed改善，区间跨0。因此原先“Multistate终点专家可稳定保住终点优势”的解释没有在新切分上完全复现。

ORC的sequence R2为0.6610，但相对HS-DT仅3/5 seed改善、区间跨0，Lead-24平均下降0.0033；它在Lead-24上也没有超过zero和persistence控制。ORC不能进入当前投稿主模型，只能保留为探索性尾部/事件修正候选。

这一轮仍不是完全独立holdout，因为2025 H2在研究过程中已经被查看。真正确认性结论仍需2026 H1或模型设计时完全不可见的新数据。

## 最终一句话

本研究最可信的结论是：**多状态监督和强时空架构带来的收益比简化物理残差更稳定，物理正则可以在部分 GNN-BiGRU 设置下改善远期预测，但其贡献依赖模型骨干、损失权重和可用外部强迫，不能被解释为普遍的精度提升。**
