# 数据、变量与因果性说明

## 数据时间和空间范围

- 区域：美国东北部沿海。
- 站点：New London、Montauk、Kings Point、The Battery、Sandy Hook、Atlantic City、Cape May。
- 水位和 ERA5：2023-01-01 至 2025-12-31，小时级。
- 表层海流：主要为日级产品，再对齐到小时任务。
- 波浪：主要为3小时级产品，再对齐到小时任务。

## 数据来源和变量

| 变量组 | 具体内容 | 主要来源 |
|---|---|---|
| 水位 | 观测总水位、天文潮、非潮汐残差 | NOAA CO-OPS |
| 大气 | 10m风 u/v、风速、气压、气压趋势/异常 | ERA5 |
| 海流 | 表层 u/v、流速、流向 | Copernicus Marine |
| 海浪 | 有义波高、周期、方向、Stokes 漂移、波能、能量通量、setup proxy | Copernicus Marine / NDBC增强数据 |
| 局地气象 | 局地风、阵风、气压、气温、水温 | NOAA CO-OPS / NDBC |
| 静态信息 | 水深、逆水深、站点坐标、距离 | GEBCO 和站点元数据 |
| 图结构 | 固定距离图、训练期残差相关图、可学习图 | 由训练数据和站点信息构造 |

数据下载和处理脚本、网址及文件清单见根目录 `数据来源与下载网址.md` 和 `data/processed_multiyear_2023_2025/` 下的 inventory 文件。

## 目标构造

\[
\eta_t = z_t^{observed} - z_t^{tide}.
\]

模型预测未来 residual eta；应用层总水位为：

\[
\hat z_{t+h}^{total} = \hat\eta_{t+h} + z_{t+h}^{tide}.
\]

主论文应优先报告 residual 指标，因为 total-water-level R2 会受到容易预测的潮汐部分显著影响。

## 预处理边界

主 benchmark 是 retrospective aligned-forcing benchmark：部分海流、波浪和局地变量来自对齐后的历史产品，某些缺失值处理使用了回顾性插值或 edge filling。因此它回答的是“在已对齐历史强迫可用时，模型能恢复多少未来残差”，不是完整实时业务测试。

严格 causal 版本采用：

- 只用训练期拟合 scaler；
- 训练期构造残差相关图；
- 只使用预测时点以前可获得的信息进行缺失填补；
- rolling-origin 时间折叠；
- 不把未来 residual target 送入输入。

严格因果实验的结果通常低于主 benchmark，且 physics 的收益随时间段改变。这不是失败，而是对回顾性结果的边界校正。

## 缺失率

主处理后水位、ERA5、海流和波浪表格完整性较高；局地 NOAA 气象变量缺失明显，其中局地风速/阵风覆盖率约42.9%，缺失率约57.1%。因此论文必须说明缺失处理，并避免宣称当前模型已经具备实时部署条件。

## 数据泄露判断

当前主模型没有把未来 residual target 直接作为输入，输入窗口也严格位于预测起点之前；scaler 和训练图原则上从训练期构造。因此不存在最直接的 target leakage。

但回顾性外部强迫和双向/回顾性插值可能比 issue-time 数据更有利，这属于 preprocessing/forcing availability 风险。正式论文中应称主结果为 retrospective benchmark，并把 strict-causal rolling-origin 结果作为稳健性检查。

## 未来数据扩展

若加入2026数据或新的时间折，必须：

1. 保存原始下载文件和下载时间；
2. 更新 `multiyear_data_inventory`；
3. 不改变已有 test 结果文件；
4. 先冻结 HS-DT 规则，再只用新时间段验证；
5. 单独建立新结果目录和实验配置 JSON。
