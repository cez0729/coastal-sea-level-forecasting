# -*- coding: utf-8 -*-

import datetime
import shutil
import zipfile
from pathlib import Path


# ============================================================
# 只打包 PPT 中涉及到的核心实验代码
# ============================================================

PPT_RELATED_CODE_FILES = [
    # 任务模式对比：direct vs multi-output
    "52_GNN_BiGRU_real_NOAA_task_mode.py",

    # 特征消融
    "53_GNN_BiGRU_feature_ablation.py",

    # 图结构消融
    "54_GNN_BiGRU_graph_ablation.py",

    # 消融后选出的最终 Pure GNN 配置
    "55_GNN_BiGRU_final_selected_config.py",

    # 可学习图融合 GNN-BiGRU
    "56_GNN_BiGRU_learnable_graph_fusion_v2.py",

    # 物理 ODE baseline
    "58_physics_multi_ode_baselines.py",

    # Physics-guided GNN：物理预测 + GNN 修正
    "59_physics_guided_learnable_graph_gnn_bigru.py",

    # 总结果汇总与图表生成
    "60_final_model_comparison_summary.py",

    # 第一版 physics-informed loss
    # 这篇 PPT 中用于和 optimized physics loss 对比
    "61_physics_informed_loss_gnn_bigru.py",

    # 优化后的 physics-informed loss
    "62_optimized_physics_loss_gnn_bigru.py",
]


# ============================================================
# 只打包 PPT 中涉及到的 outputs 子目录
# 如果某个文件夹不存在，脚本会记录 missing，但不会中断
# ============================================================

PPT_RELATED_OUTPUT_DIRS = [
    # direct vs multi-output 任务模式实验
    "task_mode_ablation",
    "real_NOAA_task_mode",

    # 特征消融
    "feature_ablation",

    # 图结构消融
    "graph_ablation",

    # 最终 Pure GNN 最优配置
    "final_selected_config",
    "GNN_BiGRU_final_selected_config",

    # 可学习图融合
    "learnable_graph_fusion",
    "learnable_graph_fusion_v2",

    # 物理 ODE baseline
    "physics_multi_ode_baselines",

    # Physics-guided GNN
    "physics_guided_learnable_graph_gnn_bigru",

    # 第一版 physics loss
    "physics_informed_loss_gnn_bigru",

    # 优化版 physics loss
    "optimized_physics_informed_loss_gnn_bigru",

    # 最终总汇总
    "final_model_comparison",
]


# ============================================================
# 只额外收集这些 PPT 常用结果文件类型
# 不包括 .pt / .npz / .npy，避免 zip 太大
# ============================================================

RESULT_EXTENSIONS = {
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".txt",
    ".json",
}


def log(message=""):
    print(message, flush=True)


def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_outputs_root(code_dir: Path) -> Path:
    """
    自动寻找 outputs 文件夹。
    优先：
    1. 当前代码目录下的 outputs
    2. 代码目录上一级的 outputs
    """
    candidates = [
        code_dir / "outputs",
        code_dir.parent / "outputs",
    ]

    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "没有找到 outputs 文件夹。已检查以下路径：\n" + checked
    )


def copy_file(src: Path, dst: Path, copied: list, missing: list):
    if src.exists() and src.is_file():
        mkdir(dst.parent)
        shutil.copy2(src, dst)
        copied.append(str(src))
        return True

    missing.append(str(src))
    return False


def copy_dir_selected_files(src_dir: Path, dst_dir: Path, copied: list, missing: list):
    """
    复制某个 outputs 子目录下的结果文件。
    只复制 csv/png/pdf/txt/json 等轻量结果文件。
    不复制模型权重和大数组文件。
    """
    if not src_dir.exists() or not src_dir.is_dir():
        missing.append(str(src_dir))
        return

    mkdir(dst_dir)

    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue

        if src.suffix.lower() not in RESULT_EXTENSIONS:
            continue

        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        copy_file(src, dst, copied, missing)


def collect_all_selected_result_files(outputs_root: Path, dst_dir: Path, copied: list, missing: list):
    """
    额外把 outputs 下所有 csv/png/pdf/txt/json 结果集中复制一份，
    方便别人快速查看，不需要到各个子文件夹里找。
    """
    mkdir(dst_dir)

    for src in outputs_root.rglob("*"):
        if not src.is_file():
            continue

        if src.suffix.lower() not in RESULT_EXTENSIONS:
            continue

        try:
            rel = src.relative_to(outputs_root)
            safe_name = "__".join(rel.parts)
        except Exception:
            safe_name = src.name

        copy_file(src, dst_dir / safe_name, copied, missing)


def zip_folder(src_folder: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in src_folder.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(src_folder.parent)
                zf.write(file, arcname)


def get_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size

    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def write_manifest(
    package_folder: Path,
    code_dir: Path,
    outputs_root: Path,
    zip_path: Path,
    copied: list,
    missing: list,
):
    manifest = package_folder / "MANIFEST.txt"

    lines = []
    lines.append("PPT Related Code and Outputs Package")
    lines.append("=" * 80)
    lines.append(f"Created time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Package purpose:")
    lines.append("  This zip contains only the code files and output results related to the final PPT.")
    lines.append("  It does not include raw data, PPT files, Word documents, virtual environments, or old zip packages.")
    lines.append("")
    lines.append(f"Code directory: {code_dir}")
    lines.append(f"Outputs root: {outputs_root}")
    lines.append(f"Zip file: {zip_path}")
    lines.append("")
    lines.append("Included code files:")
    for name in PPT_RELATED_CODE_FILES:
        lines.append(f"  - {name}")
    lines.append("")
    lines.append("Included output folders:")
    for name in PPT_RELATED_OUTPUT_DIRS:
        lines.append(f"  - {name}")
    lines.append("")
    lines.append("Copied items:")
    for item in copied:
        lines.append(f"  OK: {item}")
    lines.append("")
    lines.append("Missing items:")
    for item in missing:
        lines.append(f"  MISSING: {item}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  Missing optional output folders may be normal if that experiment used a different folder name.")
    lines.append("  Check the outputs/selected_result_files folder for a flat collection of key result files.")
    lines.append("")

    manifest.write_text("\n".join(lines), encoding="utf-8")


def write_short_readme(package_folder: Path):
    readme = package_folder / "README.txt"

    content = """This package contains only the code files and output results used in the final PPT.

Included:
1. PPT-related Python scripts
2. PPT-related outputs folders
3. A flat selected_result_files folder containing csv/png/pdf/txt/json results
4. MANIFEST.txt

Not included:
1. Raw data
2. PPT files
3. Word documents
4. Virtual environment
5. Old zip packages
6. Model checkpoint files such as .pt
7. Prediction arrays such as .npz or .npy

Main conclusion:
- 1h: learnable graph GNN-BiGRU performs best.
- 12h and 24h: physics-guided GNN-BiGRU performs best.
- Optimized physics-informed loss is more stable than the first physics loss version, but it is still not better than the physics-guided correction model.
"""
    readme.write_text(content, encoding="utf-8")


def main():
    log("=" * 80)
    log("开始打包：最终 PPT 涉及到的代码 + outputs 结果")
    log("=" * 80)

    code_dir = Path(__file__).resolve().parent
    outputs_root = find_outputs_root(code_dir)

    export_dir = code_dir / "package_exports"
    mkdir(export_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"PPT_Related_Code_Outputs_{timestamp}"

    package_folder = export_dir / package_name
    zip_path = export_dir / f"{package_name}.zip"

    if package_folder.exists():
        shutil.rmtree(package_folder)

    mkdir(package_folder)

    copied = []
    missing = []

    code_out = package_folder / "code"
    outputs_out = package_folder / "outputs"
    selected_out = outputs_out / "selected_result_files"

    mkdir(code_out)
    mkdir(outputs_out)
    mkdir(selected_out)

    log(f"代码目录：{code_dir}")
    log(f"outputs 目录：{outputs_root}")
    log(f"打包临时文件夹：{package_folder}")
    log(f"最终 zip 文件：{zip_path}")
    log("-" * 80)

    log("正在复制 PPT 涉及的 Python 代码...")
    for filename in PPT_RELATED_CODE_FILES:
        copy_file(code_dir / filename, code_out / filename, copied, missing)

    log("-" * 80)

    log("正在复制 PPT 涉及的 outputs 子目录...")
    for dirname in PPT_RELATED_OUTPUT_DIRS:
        src_dir = outputs_root / dirname
        dst_dir = outputs_out / dirname
        copy_dir_selected_files(src_dir, dst_dir, copied, missing)

    log("-" * 80)

    log("正在集中收集 outputs 下所有轻量结果文件...")
    collect_all_selected_result_files(outputs_root, selected_out, copied, missing)

    log("-" * 80)

    log("正在写入 README 和 MANIFEST...")
    write_short_readme(package_folder)
    write_manifest(
        package_folder=package_folder,
        code_dir=code_dir,
        outputs_root=outputs_root,
        zip_path=zip_path,
        copied=copied,
        missing=missing,
    )

    log("正在生成 zip 压缩包...")
    zip_folder(package_folder, zip_path)

    folder_size = get_size(package_folder)
    zip_size = get_size(zip_path)

    log("=" * 80)
    log("打包完成！")
    log("=" * 80)
    log(f"打包文件夹位置：{package_folder}")
    log(f"zip 压缩包位置：{zip_path}")
    log(f"打包文件夹大小：{format_size(folder_size)}")
    log(f"zip 压缩包大小：{format_size(zip_size)}")
    log("")
    log("你要发给别人的文件是：")
    log(str(zip_path))
    log("=" * 80)


if __name__ == "__main__":
    main()