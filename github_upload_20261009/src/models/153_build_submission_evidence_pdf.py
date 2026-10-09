from __future__ import annotations

"""Build a compact Chinese submission-evidence PDF without changing the main paper."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "physics_submission_evidence_audit.pdf"
AUDIT = ROOT / "results" / "physics_effect_audit_benchmark"
ALIGN = ROOT / "results" / "physics_alignment_negative_controls"
LAG = ROOT / "results" / "lagged_physics_gate_benchmark"


def register_fonts() -> tuple[str, str]:
    regular = r"C:\Windows\Fonts\Deng.ttf"
    bold = r"C:\Windows\Fonts\Dengb.ttf"
    pdfmetrics.registerFont(TTFont("NotoSC", regular))
    pdfmetrics.registerFont(TTFont("NotoSCBold", bold))
    return "NotoSC", "NotoSCBold"


def table(data, widths, header=True, font="NotoSC"):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style.extend([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#25465A")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "NotoSCBold")])
    t.setStyle(TableStyle(style))
    return t


def build() -> None:
    regular, bold = register_fonts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle", parent=styles["Title"], fontName=bold, fontSize=21, leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#193B4D"), spaceAfter=16))
    styles.add(ParagraphStyle(name="CNH1", parent=styles["Heading1"], fontName=bold, fontSize=15, leading=20, textColor=colors.HexColor("#193B4D"), spaceBefore=8, spaceAfter=8))
    styles.add(ParagraphStyle(name="CNH2", parent=styles["Heading2"], fontName=bold, fontSize=11.5, leading=15, textColor=colors.HexColor("#2F7D6D"), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="CNBody", parent=styles["BodyText"], fontName=regular, fontSize=9.5, leading=15, alignment=TA_LEFT, spaceAfter=7))
    styles.add(ParagraphStyle(name="CNSmall", parent=styles["BodyText"], fontName=regular, fontSize=8, leading=11, textColor=colors.HexColor("#4A5560")))
    story = []
    story.append(Spacer(1, 2.1 * cm))
    story.append(Paragraph("海平面预测项目\n投稿前证据与物理机制审计", styles["CNTitle"]))
    story.append(Paragraph("普通 70/15/15 benchmark | 五 seed | 24 小时预测 | 2026-08-01", styles["CNBody"]))
    story.append(Spacer(1, 0.7 * cm))
    story.append(Paragraph("用途说明", styles["CNH1"]))
    story.append(Paragraph("本报告用于投稿前内部决策，汇总物理损失、ODE prior、ORC 修正、forcing-regime 门控和滞后门控的可复现实验。它不覆盖现有 Overleaf 主稿，不把普通 benchmark 写成独立未来 holdout，也不承诺任何固定接收概率。", styles["CNBody"]))
    story.append(Paragraph("核心判断：物理信息的作用是条件性的。直接 physics loss 在强 GWN 上没有稳定增益；ODE-conditioned residual correction 在高强迫状态的远期预测中最有一致性，但总体提升仍属于小幅、机制性证据。", styles["CNBody"]))
    story.append(PageBreak())

    story.append(Paragraph("1. 当前主结果", styles["CNH1"]))
    story.append(Paragraph("普通 benchmark 的主排序保持为 HS-DT-GWN > FS-GWN > GNN-BiGRU。ORC 是当前最强的物理相关候选，但不应把它的全部提升归因于 ODE，因为适配器容量和残差校正结构也会贡献增益。", styles["CNBody"]))
    main_data = [
        ["模型", "Sequence R2", "Lead-24 R2", "q95 R2", "定位"],
        ["FS-GWN eta-only", "0.7261", "0.5825", "0.6271", "强 baseline"],
        ["FS-GWN multistate", "0.7250", "0.6153", "0.6179", "强 baseline"],
        ["HS-DT-GWN", "0.7357", "0.6153", "0.6292", "双专家主模型"],
        ["ORC-HS-DT-GWN", "0.7386", "0.6191", "0.6378", "物理条件修正候选"],
        ["PRS-HS-DT", "0.7382", "0.6201", "0.6352", "终点专门化探索"],
    ]
    story.append(table(main_data, [4.0 * cm, 2.3 * cm, 2.3 * cm, 2.0 * cm, 5.0 * cm]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("不应写入主结论的结果", styles["CNH2"]))
    story.append(Paragraph("GWN direct physics loss 相对 matched no-physics 的全样本 Lead-24 R2 变化约为 -0.000232；GNN ODE prior 的总体收益也不稳定。因此文章应把它们作为负对照和机制边界，而不是强行包装成普遍提升。", styles["CNBody"]))
    story.append(PageBreak())

    story.append(Paragraph("2. 物理作用的细分证据", styles["CNH1"]))
    story.append(Paragraph("审计使用训练期逐站点 q75/q90/q95 forcing 阈值，测试阶段只进行分层统计。q95 综合强迫区间约覆盖测试起点的 8%。", styles["CNBody"]))
    q95 = pd.read_csv(AUDIT / "component_regime_effect_summary.csv")
    q95 = q95[q95["train_quantile"] == 0.95]
    rows = [["比较", "高强迫 Lead-24 MSE reduction", "相对高强迫误差", "正向 seed"]]
    wanted = ["orc_minus_hsdt", "orc_minus_persistence", "gwn_physics_loss_minus_no_physics", "gnn_ode_prior_minus_learnable"]
    labels = {"orc_minus_hsdt": "ORC - HS-DT", "orc_minus_persistence": "ORC - persistence", "gwn_physics_loss_minus_no_physics": "GWN physics loss - no physics", "gnn_ode_prior_minus_learnable": "GNN ODE prior - learnable"}
    for name in wanted:
        r = q95[(q95.component == "combined") & (q95.comparison == name)].iloc[0]
        rows.append([labels[name], f"{r.MSE_reduction_mean:.8f}", f"{r.relative_MSE_reduction_pct_mean:.3f}%", f"{int(r.seed_wins)}/5"])
    story.append(table(rows, [6.0 * cm, 4.0 * cm, 3.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 8))
    story.append(Image(str(AUDIT / "component_effect_q95.png"), width=16.8 * cm, height=8.0 * cm))
    story.append(Paragraph("图 1. q95 高强迫状态下不同物理比较的 Lead-24 MSE reduction。正值表示候选模型降低误差。", styles["CNSmall"]))
    story.append(PageBreak())

    story.append(Paragraph("3. 对齐性负对照与滞后门控", styles["CNH1"]))
    story.append(Paragraph("为了检验收益是否来自物理状态与修正时机的匹配，分别测试了真实 forcing mask、反向 mask 和预先固定的时间错位 mask。结果显示，1-6 小时错位控制有时同样有效，说明当前 forcing index 尚不能证明唯一的物理对齐关系。", styles["CNBody"]))
    align = pd.read_csv(ALIGN / "alignment_negative_control_summary.csv")
    rows = [["Mask", "Sequence reduction", "Lead-24 reduction", "正向 Lead-24 seed"]]
    for _, r in align.iterrows():
        rows.append([str(r["mask"]), f"{r.sequence_MSE_reduction_mean:.8f}", f"{r.lead24_MSE_reduction_mean:.8f}", f"{int(r.positive_lead24_seeds)}/5"])
    story.append(table(rows, [4.0 * cm, 4.0 * cm, 4.0 * cm, 4.0 * cm]))
    story.append(Spacer(1, 8))
    story.append(Image(str(ALIGN / "alignment_negative_controls.png"), width=16.8 * cm, height=7.7 * cm))
    story.append(Paragraph("图 2. 对齐、反向与错位 mask 的负对照。该结果不支持把当前门控描述成已经验证的唯一因果 forcing 识别器。", styles["CNSmall"]))
    story.append(PageBreak())

    story.append(Paragraph("4. 验证集锁定的滞后 forcing 实验", styles["CNH1"]))
    story.append(Paragraph("候选滞后 {0, 6, 12, 24, 72, 168} 小时和 q75/q90/q95 在 validation 上预先限定并选择，最终选择 q95、12 小时滞后。测试集相对 persistence 的 Lead-24 MSE reduction 为 0.00001891，幅度很小；因此它是机制探索，不应替代 ORC。" , styles["CNBody"]))
    lag = pd.read_csv(LAG / "lagged_gate_test_summary.csv")
    rows = [["模型", "Sequence MSE", "Lead-24 MSE"]]
    for _, r in lag.iterrows():
        rows.append([str(r["model"]), f"{r.seq_MSE_mean:.8f}", f"{r.lead24_MSE_mean:.8f}"])
    story.append(table(rows, [7.0 * cm, 4.5 * cm, 4.5 * cm]))
    story.append(Spacer(1, 8))
    story.append(Image(str(LAG / "lag_validation_selection.png"), width=16.8 * cm, height=7.7 * cm))
    story.append(Paragraph("图 3. 滞后候选只在 validation 选择，测试结果不用于反向调参。", styles["CNSmall"]))
    story.append(PageBreak())

    story.append(Paragraph("5. 投稿判断与最低补强项", styles["CNH1"]))
    story.append(Paragraph("目前可以形成一篇有竞争力的机器学习海平面预测论文，但不能依据现有结果声称 60% 的 SCI 三区接收概率。接收率受期刊、审稿人、选题竞争、语言和新颖性判断影响，无法由模型 R2 直接换算。", styles["CNBody"]))
    story.append(Paragraph("推荐故事线", styles["CNH2"]))
    story.append(Paragraph("(1) GNN-BiGRU 物理损失作为早期探索，说明简单 physics loss 并不一定有效；(2) 固定支持图 Graph WaveNet 提供强 baseline；(3) HS-DT 用 eta-only 与 multistate 专家分工，改善轨迹和终点稳定性；(4) ODE-conditioned residual correction 将物理信息放到预测后端，在高强迫远期区间得到最一致收益；(5) 负对照和滞后搜索限定物理结论边界。", styles["CNBody"]))
    story.append(Paragraph("投稿前建议", styles["CNH2"]))
    checklist = [
        ["项目", "状态"],
        ["五 seed、统一 70/15/15 benchmark、强 baseline 对照", "已完成"],
        ["direct physics loss、ODE prior、adapter controls", "已完成并保留负结果"],
        ["高强迫/逐 lead/逐站点物理效应", "已完成"],
        ["forcing 对齐性负对照与滞后搜索", "已完成，结果为探索性"],
        ["完全不可见的新年份或外部站点验证", "仍建议补充，当前未作为主稿证据"],
        ["将 ORC 全部增益归因于 physics-only", "禁止"],
    ]
    story.append(table(checklist, [9.5 * cm, 7.0 * cm]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("目标期刊可优先核对 Natural Hazards 的官方 scope（明确包含 oceanographic hazards 与 storm surges）：https://link.springer.com/journal/11069。具体分区和接收难度应以投稿年份的 JCR/中科院分区及近期文章为准，不在本报告中虚构。", styles["CNSmall"]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor("#66737D"))
        canvas.drawString(2 * cm, 1.1 * cm, "Sea-level forecasting | submission evidence audit")
        canvas.drawRightString(19 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm, title="Submission Evidence and Physics Mechanism Audit")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
