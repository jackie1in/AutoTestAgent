#!/usr/bin/env python3
"""Generate Graph Agent design document PPT."""

import logging
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import nsmap
from pptx.oxml import parse_xml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Theme ──────────────────────────────────────────────────────────
PRIMARY   = RGBColor(0x1A, 0x36, 0x5D)   # Deep navy
ACCENT    = RGBColor(0x00, 0x96, 0xC7)   # Cyan accent
DARK      = RGBColor(0x1F, 0x29, 0x33)   # Near black
LIGHT     = RGBColor(0xF5, 0xF7, 0xFA)   # Off-white
GRAY      = RGBColor(0x6B, 0x72, 0x80)   # Text gray
GREEN     = RGBColor(0x10, 0xB9, 0x81)   # Success green
ORANGE    = RGBColor(0xF5, 0x9E, 0x0B)   # Warning orange

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_bg(slide, color=LIGHT):
    """Fill slide background."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, shape_type, left, top, width, height, fill=None, line=None):
    s = slide.shapes.add_shape(shape_type, left, top, width, height)
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    return s


def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=DARK, align=PP_ALIGN.LEFT, font_name="Microsoft YaHei"):
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = align
    return tx


def add_title_slide(slide, title, subtitle=""):
    add_bg(slide, PRIMARY)
    # Accent bar
    add_shape(slide, MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.8), Inches(13.333), Inches(0.08), fill=ACCENT)
    add_textbox(slide, Inches(0.8), Inches(2.0), Inches(11.5), Inches(1.2),
                title, font_size=44, bold=True, color=LIGHT, align=PP_ALIGN.LEFT)
    if subtitle:
        add_textbox(slide, Inches(0.8), Inches(3.1), Inches(11.5), Inches(0.8),
                    subtitle, font_size=20, color=LIGHT, align=PP_ALIGN.LEFT)
    add_textbox(slide, Inches(0.8), Inches(6.5), Inches(11.5), Inches(0.5),
                "技术团队  |  2026/04/22", font_size=14, color=RGBColor(0x99, 0xAA, 0xBB), align=PP_ALIGN.LEFT)


def add_section_slide(slide, number, title):
    add_bg(slide, PRIMARY)
    add_textbox(slide, Inches(0.8), Inches(2.6), Inches(11.5), Inches(0.8),
                number, font_size=72, bold=True, color=ACCENT, align=PP_ALIGN.LEFT)
    add_textbox(slide, Inches(0.8), Inches(3.5), Inches(11.5), Inches(1.0),
                title, font_size=36, bold=True, color=LIGHT, align=PP_ALIGN.LEFT)


def add_content_slide(slide, title, bullets, note=""):
    add_bg(slide, LIGHT)
    # Top accent bar
    add_shape(slide, MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.06), fill=ACCENT)
    # Title
    add_textbox(slide, Inches(0.6), Inches(0.25), Inches(12), Inches(0.7),
                title, font_size=28, bold=True, color=PRIMARY, align=PP_ALIGN.LEFT)
    # Content
    tx = slide.shapes.add_textbox(Inches(0.6), Inches(1.1), Inches(12), Inches(5.8))
    tf = tx.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = b
        p.font.size = Pt(16)
        p.font.color.rgb = DARK
        p.font.name = "Microsoft YaHei"
        p.space_before = Pt(12)
        p.space_after = Pt(4)
        p.level = 0
    if note:
        add_textbox(slide, Inches(0.6), Inches(6.8), Inches(12), Inches(0.4),
                    note, font_size=12, color=GRAY, align=PP_ALIGN.LEFT)


def add_table_slide(slide, title, headers, rows):
    add_bg(slide, LIGHT)
    add_shape(slide, MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.06), fill=ACCENT)
    add_textbox(slide, Inches(0.6), Inches(0.25), Inches(12), Inches(0.7),
                title, font_size=28, bold=True, color=PRIMARY, align=PP_ALIGN.LEFT)

    cols = len(headers)
    rows_cnt = len(rows) + 1
    table = slide.shapes.add_table(rows_cnt, cols, Inches(0.6), Inches(1.1), Inches(12), Inches(0.6 * rows_cnt)).table

    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.font.color.rgb = LIGHT
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.name = "Microsoft YaHei"
        p.alignment = PP_ALIGN.CENTER

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(val)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(13)
            p.font.color.rgb = DARK
            p.font.name = "Microsoft YaHei"
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            if r_idx % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xED, 0xF2, 0xF7)


def add_diagram_slide(slide, title, lines):
    add_bg(slide, LIGHT)
    add_shape(slide, MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.06), fill=ACCENT)
    add_textbox(slide, Inches(0.6), Inches(0.25), Inches(12), Inches(0.7),
                title, font_size=28, bold=True, color=PRIMARY, align=PP_ALIGN.LEFT)

    y = 1.1
    box_w = Inches(11.5)
    box_h = Inches(0.55)
    for line in lines:
        if line.startswith("  "):
            # indented sub-line
            add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(y), box_w - Inches(0.4), box_h,
                      fill=LIGHT, line=ACCENT)
            add_textbox(slide, Inches(1.2), Inches(y + 0.08), box_w - Inches(0.8), Inches(0.4),
                        line.strip(), font_size=14, color=DARK)
        else:
            add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(y), box_w, box_h,
                      fill=RGBColor(0xE8, 0xF4, 0xFA), line=ACCENT)
            add_textbox(slide, Inches(0.8), Inches(y + 0.08), box_w - Inches(0.4), Inches(0.4),
                        line, font_size=15, bold=True, color=PRIMARY)
        y += 0.65


# ═══════════════════════════════════════════════════════════════════
# Build slides
# ═══════════════════════════════════════════════════════════════════

# ── P1 封面 ───────────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_title_slide(slide, "Graph Agent 自动测绘系统",
                "基于 LLM + 知识图谱的 Web 应用自动化探索与建模方案")

# ── P2 背景与痛点 ────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_table_slide(slide, "背景与痛点：为什么需要自动测绘？",
    ["痛点", "现状", "影响"],
    [
        ["应用规模大", "管理后台动辄数百页面、数千交互点", "人工梳理耗时数周"],
        ["变更频繁", "前端迭代快，UI 结构经常变化", "测试用例快速失效"],
        ["知识流失", "业务逻辑分散在代码和文档中", "新人上手成本高"],
        ["回归困难", "不清楚改了哪里会影响哪里", "回归范围靠猜测"],
    ])

# ── P3 总体架构 ───────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_diagram_slide(slide, "总体架构：四层一体化设计",
    ["应用层 — CLI / API / Web 控制台",
     "  提供命令行、REST API、可视化界面多种使用方式",
     "协调层 — 会话管理 · 任务调度 · 断点续传",
     "  智能分配探索任务，支持中断恢复与进度持久化",
     "智能引擎 — LLM 意图推断 · DOM 分析 · 策略决策",
     "  三重识别保障：LLM 语义理解 + CSS 选择器 + JS 启发式扫描",
     "执行层 — 浏览器自动化 (Playwright)",
     "  真实浏览器操作，支持 SPA 路由感知与多 iframe 场景",
     "知识层 — Neo4j 图数据库 · 状态-转移图谱",
     "  结构化存储应用状态、交互路径、业务意图与验证点"])

# ── P4 核心流程：发现阶段 ─────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_table_slide(slide, "核心流程 ①：发现阶段 — 像人一样“看”页面",
    ["能力", "实现方式", "价值"],
    [
        ["导航菜单识别", "自动提取侧边栏/顶部菜单，支持 Ant Design / Element UI 等框架", "建立应用骨架地图"],
        ["功能区域发现", "识别表单、表格、Tab、弹窗、分页器等 12 类功能区域", "精准定位交互范围"],
        ["SPA 路由感知", "区分单页应用 vs 多页应用，自动捕获路由变化", "适配现代前端架构"],
    ])

# ── P5 核心流程：探索阶段 ─────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_diagram_slide(slide, "核心流程 ②：探索阶段 — ReAct 智能探索循环",
    ["观察 (Observe) — 获取当前页面 DOM 快照、URL、标题、元素索引",
     "思考 (Think) — LLM 分析未探索元素，制定下一步行动计划",
     "行动 (Act) — 执行 click / input / select / scroll 等浏览器操作",
     "记录 (Record) — 检测到页面变化时，自动记录 State → Transition → Intent",
     "",
     "探索策略：深度优先 · 全元素覆盖 · 安全边界 · 智能回退"])

# ── P6 核心流程：智能增强 ─────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_table_slide(slide, "核心流程 ③：智能增强 — 让探索“更聪明”",
    ["能力", "说明"],
    [
        ["业务意图推断", "每次交互后自动识别业务语义，如 user.login.submit、order.list.filter"],
        ["CAPTCHA 自动识别", "检测验证码 → 截图 → LLM Vision 识别 → 自动填充"],
        ["自动登录", "识别登录表单，自动填入环境变量配置的测试账号密码"],
        ["Waypoint 记忆", "基于历史探索知识避免重复路径，优先发现新区域"],
        ["覆盖率驱动调度", "优先探索未覆盖区域，陈旧区域（>7 天）自动重探"],
    ])

# ── P7 数据存储设计 ───────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_diagram_slide(slide, "数据存储：Neo4j 知识图谱 — 活的业务地图",
    ["App（被测应用）— 顶层聚合节点，关联所有 Session 与 State",
     "  Session（探索会话）— 一次完整的测绘任务，记录时间、策略、统计",
     "  State（页面状态）— URL + DOM 指纹唯一标识一个可达页面",
     "    Zone（功能区域）— 绑定到 State 的 12 类功能区域",
     "  Transition（交互转移）— 用户操作路径，带置信度评分与验证次数",
     "    Intent（业务意图）— 可理解的业务行为标签，REALIZES 关系关联",
     "    Checkpoint（验证点）— 每个转移的前后置校验规则"])

# ── P8 关键特性 ───────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_table_slide(slide, "系统关键特性",
    ["特性", "描述"],
    [
        ["断点续传", "每 10 分钟自动保存进度，中断后可从任务断点恢复"],
        ["事务安全", "单次探索结果批量原子写入 Neo4j，失败自动回滚"],
        ["并发探索", "多个 Zone 并行探索，默认并发 3，大站点可扩至 6"],
        ["增量更新", "同一 Transition 多次验证后置信度递增，动态内容宽限期保护"],
        ["冲突解决", "相同起终点的多条路径，优先保留更短 Selector，自动降级冗余路径"],
        ["时间预算控制", "会话级总时间预算（默认 6 小时），保障资源可控"],
    ])

# ── P9 预期效果 ───────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_table_slide(slide, "预期效果：量化收益对比",
    ["指标", "人工方式", "自动测绘", "提升"],
    [
        ["应用梳理周期", "2-3 周", "2-4 小时", "数十倍加速"],
        ["页面覆盖率", "约 60%（易遗漏弹窗/深层路径）", "> 95%", "大幅提升"],
        ["用例维护成本", "前端变更后全面返工", "自动检测变化、增量更新", "持续降本"],
        ["业务知识沉淀", "文档分散、易过期", "结构化图谱、可追溯", "资产化"],
    ])

# ── P10 实施路线 ──────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_table_slide(slide, "实施路线：分阶段落地",
    ["阶段", "时间", "目标"],
    [
        ["Phase 1", "2 周", "单页面探索闭环跑通，基础图谱入库"],
        ["Phase 2", "4 周", "多页面会话协调 + 任务调度 + 覆盖率分析"],
        ["Phase 3", "6 周", "深度探索（表单组合、弹窗全遍历）+ 断点续传"],
        ["Phase 4", "8 周", "集成测试用例生成、回归路径推荐、可视化大盘"],
    ])

# ── P11 总结 ──────────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, PRIMARY)
add_textbox(slide, Inches(0.8), Inches(1.5), Inches(11.5), Inches(0.8),
            "总结", font_size=20, bold=True, color=ACCENT, align=PP_ALIGN.LEFT)
add_textbox(slide, Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.2),
            "Graph Agent 让机器像测试工程师一样浏览应用、理解业务、记录路径，",
            font_size=24, color=LIGHT, align=PP_ALIGN.LEFT)
add_textbox(slide, Inches(0.8), Inches(2.9), Inches(11.5), Inches(0.8),
            "最终构建一张实时更新的业务知识图谱。",
            font_size=24, bold=True, color=LIGHT, align=PP_ALIGN.LEFT)

# 核心价值卡片
shapes = [
    ("降本", "大幅减少人工梳理\n和用例维护成本", GREEN),
    ("提效", "快速获得应用\n完整交互地图", ACCENT),
    ("保质", "覆盖率驱动\n减少遗漏路径", ORANGE),
    ("沉淀", "业务知识从人脑\n转移到图谱", RGBColor(0x8B, 0x5C, 0xF6)),
]
for i, (title, desc, color) in enumerate(shapes):
    x = Inches(0.8 + i * 3.0)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(4.0), Inches(2.6), Inches(2.0), fill=color)
    add_textbox(slide, x + Inches(0.15), Inches(4.15), Inches(2.3), Inches(0.5),
                title, font_size=22, bold=True, color=LIGHT, align=PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.15), Inches(4.7), Inches(2.3), Inches(1.2),
                desc, font_size=14, color=LIGHT, align=PP_ALIGN.CENTER)

# ── Save ───────────────────────────────────────────────────────────
output_path = "/Users/linhai/AutoTestAgent/GraphAgent_设计汇报_20260422.pptx"
prs.save(output_path)
logger.info("PPT saved to: %s", output_path)
