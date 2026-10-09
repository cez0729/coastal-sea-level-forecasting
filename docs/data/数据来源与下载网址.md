# 数据来源与下载网址

本文档整理当前项目中各类数据的来源网站、对应本地文件以及后续可下载同类数据的入口。

## 1. NOAA 实测水位数据

- 数据内容：小时级实测水位 `Water Level`、测量不确定性 `Sigma`、质量标记 `I/L`
- 本地位置：`data/raw/2025小时级实测水位/`
- 新增多年原始数据位置：`data/raw/NOAA_hourly_water_level_2023_2025/`
- 新增多年覆盖范围：2023-01-01 00:00 至 2025-12-31 23:00，7 个站点每站 26,304 条小时记录
- 新增多年处理后数据位置：`data/processed_multiyear_2023_2025/water_level_matrix.csv`
- 来源机构：NOAA Center for Operational Oceanographic Products and Services (CO-OPS)
- 下载入口：[NOAA CO-OPS Data API](https://api.tidesandcurrents.noaa.gov/api/prod/)
- 站点查询：[NOAA Tides & Currents Station Selection](https://tidesandcurrents.noaa.gov/stations.html)

## 2. NOAA 天文潮预测数据

- 数据内容：小时级天文潮预测 `Prediction`
- 本地位置：`data/raw/2025 年小时级天文潮/`
- 新增多年原始数据位置：`data/raw/NOAA_hourly_tide_predictions_2023_2025/`
- 新增多年覆盖范围：2023-01-01 00:00 至 2025-12-31 23:00，7 个站点每站 26,304 条小时记录
- 新增完整性索引：`data/raw/NOAA_multiyear_2023_2025_inventory.csv`
- 新增多年 residual 数据位置：`data/processed_multiyear_2023_2025/water_tide_residual_long.csv`、`data/processed_multiyear_2023_2025/residual_matrix.csv`
- 来源机构：NOAA CO-OPS
- 下载入口：[NOAA CO-OPS Data API](https://api.tidesandcurrents.noaa.gov/api/prod/)
- 潮汐预测页面示例：[NOAA Tide Predictions](https://tidesandcurrents.noaa.gov/noaatidepredictions.html)

## 3. NOAA 站点元数据

- 数据内容：站点名称、经纬度、州、时区、潮汐类型、基准面、调和常数链接等
- 本地位置：`data/raw/站台元数据/`
- 来源机构：NOAA CO-OPS Metadata API
- 下载入口：[NOAA CO-OPS Metadata API](https://api.tidesandcurrents.noaa.gov/mdapi/prod/)
- 示例格式：`https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}.json`

## 4. ERA5 气象强迫数据

- 数据内容：10 m 风速 U/V 分量 `u10/v10`、海平面气压 `msl`
- 本地位置：`data/raw/ERA5气象强迫数据/`
- 来源机构：ECMWF / Copernicus Climate Data Store
- 下载入口：[ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

## 5. Copernicus Marine 海浪数据

- 数据内容：显著波高 `VHM0`、峰值周期 `VTPK`、平均波周期 `VTM10`、平均波向 `VMDR`
- 本地位置：`data/raw/海浪数据.nc`
- 来源机构：Copernicus Marine Service
- 产品 ID：`GLOBAL_MULTIYEAR_WAV_001_032`
- 下载入口：[Global Ocean Waves Reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_WAV_001_032/description)

## 6. Copernicus Marine 海流数据

- 数据内容：东西向海水速度 `uo`、南北向海水速度 `vo`
- 当前状态：已下载并处理为站点级日尺度海流数据
- 原始本地位置：`data/raw/copernicus_currents/`
- 处理后本地位置：`data/processed/surface_currents_station_daily.csv`
- 来源机构：Copernicus Marine Service
- 产品 ID：`GLOBAL_MULTIYEAR_PHY_001_030`
- 数据集 ID：`cmems_mod_glo_phy_my_0.083deg_P1D-m`
- 已下载变量：`uo`、`vo`
- 派生变量：`current_speed_mps`、`current_direction_deg_toward`
- 下载入口：[Global Ocean Physics Reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)

## 6.1 后续多状态物理模型建议补充的数据

为了支持 `η + u + v + W` 多状态浅水方程启发模型，建议将下列强迫数据扩展到与 NOAA 多年水位一致的 2023-2025 时间范围：

- ERA5 气象强迫：`u10`、`v10`、`msl`。入口：[ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
- Copernicus Marine 海浪：`VHM0`、`VTPK`、`VTM10`、`VMDR`。入口：[Global Ocean Waves Reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_WAV_001_032/description)
- Copernicus Marine 海流：`uo`、`vo`。入口：[Global Ocean Physics Reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
- 可选增强变量：海温、盐度、海表高度、混合层深度。入口同上：[Global Ocean Physics Reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
- 可选河口入流变量：USGS 河流流量 `discharge`。入口：[USGS Water Data](https://waterdata.usgs.gov/nwis)

## 7. GEBCO 水深与海底地形数据

- 数据内容：海底高程 `elevation`，项目中进一步换算为水深 `depth`
- 本地位置：`data/raw/GEBCO_gebco_unzip/`
- 来源机构：GEBCO / Nippon Foundation-GEBCO Seabed 2030 Project
- 下载入口：[GEBCO Gridded Bathymetry Data](https://www.gebco.net/data-products/gridded-bathymetry-data)

## 8. 台风与热带气旋数据

- 数据内容：风暴编号、时间、位置、最大风速、中心气压、风圈半径、移动速度、移动方向等
- 本地位置：`data/raw/台风数据.csv`
- 来源机构：NOAA National Centers for Environmental Information (NCEI)
- 数据集：International Best Track Archive for Climate Stewardship (IBTrACS)
- 下载入口：[NOAA NCEI IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)

## 9. 项目中当前使用的 NOAA 站点

- `8461490` New London, CT
- `8510560` Montauk, NY
- `8516945` Kings Point, NY
- `8518750` The Battery, NY
- `8531680` Sandy Hook, NJ
- `8534720` Atlantic City, NJ
- `8536110` Cape May, NJ

这些站点可以通过 NOAA 站点页面查询，例如：

- [NOAA station page example: The Battery, NY](https://tidesandcurrents.noaa.gov/stationhome.html?id=8518750)
- [NOAA station page example: Montauk, NY](https://tidesandcurrents.noaa.gov/stationhome.html?id=8510560)
