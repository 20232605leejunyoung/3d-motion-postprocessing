"""
report/build_report.py
---------------------------------------------------------------
과제 템플릿(Assignment_3D_motiondata.pptx)을 채워 최종 보고서 PPT를 생성한다.
p1_3 ~ p1_5 를 먼저 실행해 data/output 아래 그림과 지표가 있어야 한다.

Usage:
    python report/build_report.py <template.pptx> [output.pptx]
"""
import os
import sys
import json
import copy

import numpy as np
import pandas as pd
from PIL import Image
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "data", "output", "figures")
OUT = os.path.join(ROOT, "data", "output")
NAME = "tail_sunmi_dance"

BLUE = RGBColor(0x00, 0x6C, 0xB8)
TEAL = RGBColor(0x49, 0xA0, 0x9A)
RED = RGBColor(0xE4, 0x57, 0x2E)
TXT = RGBColor(0x33, 0x33, 0x33)
MUTED = RGBColor(0x6B, 0x72, 0x80)
LINE = RGBColor(0xD5, 0xDB, 0xE3)
TINT = RGBColor(0xF3, 0xF7, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "맑은 고딕"


# ------------------------------------------------------------------ helpers
def remove(shape):
    el = shape._element
    el.getparent().remove(el)


def set_text_keep_format(shape, text):
    """첫 run 서식을 유지하고 텍스트만 교체."""
    p = shape.text_frame.paragraphs[0]
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in shape.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def _word_wrap_rules(paragraph):
    """템플릿 마스터가 latinLnBrk/eaLnBrk=1 (단어 중간 줄바꿈 허용)이라
    '3|0 fps', '위팔|이' 처럼 끊기므로, 직접 만든 문단은 띄어쓰기 단위로 줄바꿈."""
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("latinLnBrk", "0")
    pPr.set("eaLnBrk", "0")


def _add_runs(paragraph, text, apply_font):
    """'\n' 이 들어간 텍스트는 줄바꿈(<a:br/>)으로 나눠 run 을 만든다."""
    parts = text.split("\n")
    for k, part in enumerate(parts):
        if k > 0:
            paragraph.add_line_break()
        r = paragraph.add_run()
        r.text = part
        apply_font(r.font)
        _korean_lang(r)


def _korean_lang(run):
    """run 언어를 ko-KR 로 지정 — PowerPoint 가 한글을 단어(어절) 단위로 줄바꿈하도록."""
    rPr = run._r.get_or_add_rPr()
    rPr.set("lang", "ko-KR")
    rPr.set("altLang", "en-US")


def _bullet(paragraph, char="•", indent_in=0.16, color="006CB8"):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent_in))))
    pPr.set("indent", str(-int(Inches(indent_in))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buFont", "a:buClr"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    buClr = etree.SubElement(pPr, qn("a:buClr"))
    srgb = etree.SubElement(buClr, qn("a:srgbClr"))
    srgb.set("val", color)
    buFont = etree.SubElement(pPr, qn("a:buFont"))
    buFont.set("typeface", "Arial")
    buChar = etree.SubElement(pPr, qn("a:buChar"))
    buChar.set("char", char)


def add_text(slide, x, y, w, h, paras, size=11, color=TXT, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, margin=0.0, space_after=0, line_spacing=None, name=None):
    """paras: list of paragraphs; each paragraph is str or list of (text, {opts}) runs.
    paragraph-level opts via dict {"runs": [...], "bullet": True, "align": ...}"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    m = Inches(margin)
    tf.margin_left = tf.margin_right = m
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        popts = {}
        if isinstance(para, dict):
            popts = para
            runs = para["runs"]
        elif isinstance(para, str):
            runs = [(para, {})]
        else:
            runs = para
        p.alignment = popts.get("align", align)
        if space_after or popts.get("space_after"):
            p.space_after = Pt(popts.get("space_after", space_after))
        if line_spacing:
            p.line_spacing = line_spacing
        _word_wrap_rules(p)
        if popts.get("bullet"):
            _bullet(p, color=popts.get("bullet_color", "006CB8"))
        for text, o in runs:
            def fmt(f, o=o):
                f.size = Pt(o.get("size", popts.get("size", size)))
                f.bold = o.get("bold", bold)
                f.italic = o.get("italic", False)
                f.color.rgb = o.get("color", popts.get("color", color))
                f.name = o.get("font", FONT)
            _add_runs(p, text, fmt)
    return tb


def add_box(slide, x, y, w, h, fill=WHITE, line=LINE, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            line_w=0.75, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def box_text(shape, paras, size=10, color=TXT, bold=False, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, margin=0.05):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        _word_wrap_rules(p)
        runs = [(para, {})] if isinstance(para, str) else para
        for text, o in runs:
            def fmt(f, o=o):
                f.size = Pt(o.get("size", size))
                f.bold = o.get("bold", bold)
                f.color.rgb = o.get("color", color)
                f.name = FONT
            _add_runs(p, text, fmt)


def add_badge(slide, x, y, d, text, fill=BLUE, size=10):
    s = add_box(slide, x, y, d, d, fill=fill, line=None, shape=MSO_SHAPE.OVAL)
    box_text(s, [text], size=size, color=WHITE, bold=True, margin=0)
    return s


def add_picture_fit(slide, path, x, y, max_w, max_h, center=True, name=None):
    im = Image.open(path)
    ar = im.size[1] / im.size[0]
    w = max_w
    h = w * ar
    if h > max_h:
        h = max_h
        w = h / ar
    px = x + (max_w - w) / 2 if center else x
    pic = slide.shapes.add_picture(path, Inches(px), Inches(y), Inches(w), Inches(h))
    if name:
        pic.name = name
    return pic, (px, y, w, h)


def add_arrow(slide, x, y, w=0.22, h=0.22, color=MUTED):
    s = add_box(slide, x, y, w, h, fill=color, line=None, shape=MSO_SHAPE.RIGHT_ARROW)
    return s


def card(slide, x, y, w, h, num, title, num_fill=BLUE, title_color=BLUE, name=None):
    add_box(slide, x, y, w, h, name=name)
    if num:
        add_badge(slide, x + 0.12, y + 0.11, 0.27, num, fill=num_fill, size=10)
        add_text(slide, x + 0.46, y + 0.08, w - 0.55, 0.32, [title], size=12, bold=True,
                 color=title_color, anchor=MSO_ANCHOR.MIDDLE)
    else:
        add_text(slide, x + 0.14, y + 0.08, w - 0.25, 0.32, [title], size=12, bold=True,
                 color=title_color, anchor=MSO_ANCHOR.MIDDLE)


def style_chart_axes(chart, size=9):
    from pptx.enum.chart import XL_TICK_LABEL_POSITION  # noqa: F401
    ca, va = chart.category_axis, chart.value_axis
    for ax in (ca, va):
        ax.tick_labels.font.size = Pt(size)
        ax.tick_labels.font.color.rgb = TXT
        ax.tick_labels.font.name = FONT
        ax.format.line.color.rgb = LINE
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE9, 0xEE)
    va.major_gridlines.format.line.width = Pt(0.5)
    ca.has_major_gridlines = False


def _tick_skip_one(axis):
    """카테고리 라벨을 하나도 건너뛰지 않도록 <c:tickLblSkip val=1/> 지정."""
    ax = axis._element
    for el in ax.findall(qn("c:tickLblSkip")):
        ax.remove(el)
    skip = etree.SubElement(ax, qn("c:tickLblSkip"))
    skip.set("val", "1")
    # schema order: tickLblSkip must come after c:lblOffset / before c:tickMarkSkip, c:noMultiLvlLbl
    nm = ax.find(qn("c:noMultiLvlLbl"))
    if nm is not None:
        nm.addprevious(skip)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ------------------------------------------------------------------ data
def load_data():
    metrics = pd.read_csv(os.path.join(OUT, f"{NAME}_metrics.csv"), index_col=0)
    with open(os.path.join(OUT, f"{NAME}_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    vis = np.load(os.path.join(ROOT, "data", "processed", f"{NAME}_vis.npy"))
    groups = [("얼굴", range(0, 11)), ("어깨·팔꿈치", range(11, 15)), ("손목·손", range(15, 23)),
              ("골반·무릎", range(23, 27)), ("발목", [27, 28]), ("뒤꿈치", [29, 30]), ("발끝", [31, 32])]
    low = [(g, float((vis[:, list(idx)] < 0.5).mean() * 100)) for g, idx in groups]
    return metrics, info, low


# ------------------------------------------------------------------ slides
def slide1(s):
    # 부제 + 작성자
    add_text(s, 0.74, 4.42, 9.8, 0.42,
             ["선미 ‘꼬리(Tail)’ 댄스 커버 영상 · MediaPipe BlazePose 3D 모션 후처리"],
             size=16, color=RGBColor(0xD6, 0xE9, 0xF8), name="Subtitle")
    add_text(s, 0.74, 4.92, 6.0, 0.36, ["이준영 · 20232605"], size=14,
             color=RGBColor(0xD6, 0xE9, 0xF8), name="Author")
    notes(s, "Assignment 02 — 기말 프로젝트용 댄스 영상에서 3D 모션을 추출하고, 문제점을 찾아 "
             "제공된 후처리와 추가 후처리를 적용해 비교한 결과입니다.")


def slide2(s, m, info, low):
    for sh in list(s.shapes):
        if sh.name in ("TextBox 33", "TextBox 2", "TextBox 3"):
            remove(sh)
    raw = m.loc["Raw"]

    # ---- Left: Data acquisition ----
    add_text(s, 0.48, 1.38, 4.3, 0.34, ["① 데이터 획득 (Data Acquisition)"], size=14, bold=True, color=BLUE)
    add_text(s, 0.48, 1.76, 4.3, 1.2, [
        {"runs": [("영상  ", {"bold": True}), ("선미 ‘꼬리’ 댄스 커버 (YouTube Shorts) · 1080×1920 · 30 fps · 545 프레임", {})], "bullet": True, "space_after": 2},
        {"runs": [("모델  ", {"bold": True}), ("MediaPipe BlazePose (complexity 2) · world landmark 33관절, 단위 m", {})], "bullet": True, "space_after": 2},
        {"runs": [("코드  ", {"bold": True}), ("제공 코드 p1_create_raw_keypoints.py 의 추출 함수를 그대로 사용 → 545/545 프레임 검출", {})], "bullet": True},
    ], size=10.5)

    cy, ch = 3.05, 4.1
    card(s, 0.48, cy, 4.3, ch, "1", "지면 기준 없음 · 골반 원점 고정")
    add_picture_fit(s, os.path.join(FIG, "fig_a_acquisition.png"), 0.6, cy + 0.46, 4.06, ch - 0.98)
    add_text(s, 0.62, cy + ch - 0.5, 4.05, 0.44, [
        [("world landmark는 골반이 항상 원점 → Unity 바닥(y=0) 아래로 발이 평균 ", {}),
         (f"{-raw['lowest_foot_mean']:.2f} m", {"bold": True, "color": RED}), (" 묻힘 (100% 프레임)", {})]],
        size=10, color=TXT)

    # ---- Right: problem cards ----
    add_text(s, 5.0, 1.38, 7.85, 0.34, ["② 문제점 분석 (Problem Identification)"], size=14, bold=True, color=BLUE)
    gx, gy, gw, gh, gap = 5.0, 1.8, 3.825, 2.6, 0.18
    pos = [(gx, gy), (gx + gw + 0.2, gy), (gx, gy + gh + gap - 0.03), (gx + gw + 0.2, gy + gh + gap - 0.03)]

    # 2) spike
    x, y = pos[0]
    card(s, x, y, gw, gh, "2", "떨림(jitter) & 순간 튐(spike)")
    add_picture_fit(s, os.path.join(FIG, "fig_b_spike.png"), x + 0.1, y + 0.45, gw - 0.2, 1.65)
    add_text(s, x + 0.14, y + gh - 0.5, gw - 0.26, 0.44, [
        [("손목 속도 최대 ", {}), (f"21.4 m/s", {"bold": True, "color": RED}),
         (", 평균 관절 가속도 ", {}), (f"{raw['mean_acc']:.1f} m/s²", {"bold": True, "color": RED}),
         (" → 사람이 낼 수 없는 움직임", {})]], size=10)

    # 3) bone length
    x, y = pos[1]
    card(s, x, y, gw, gh, "3", "뼈 길이 불일치 (깊이 오차)")
    add_picture_fit(s, os.path.join(FIG, "fig_c_bone.png"), x + 0.1, y + 0.45, gw - 0.2, 1.65)
    add_text(s, x + 0.14, y + gh - 0.5, gw - 0.26, 0.44, [
        [("팔이 카메라를 향하면 위팔이 0.02 m까지 짧아짐 (CV 23%), 좌우 허벅지 ", {}),
         ("0.42 vs 0.35 m", {"bold": True, "color": RED})]], size=10)

    # 4) low visibility — native chart
    x, y = pos[2]
    card(s, x, y, gw, gh, "4", "가려짐 → 저신뢰 관절")
    cd = CategoryChartData()
    cd.categories = [g for g, _ in low]
    cd.add_series("visibility < 0.5 (%)", [round(v, 1) for _, v in low])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x + 0.1), Inches(y + 0.42),
                            Inches(gw - 0.2), Inches(1.72), cd)
    gf.name = "Chart Visibility"
    ch_ = gf.chart
    ch_.has_title = False
    ch_.has_legend = False
    ch_.font.size = Pt(9)
    ch_.font.name = FONT
    plot = ch_.plots[0]
    plot.gap_width = 45
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = '0"%"'
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(8.5)
    dl.font.color.rgb = TXT
    ser = plot.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = RGBColor(0xB8, 0xC4, 0xD2)
    for i, (_, v) in enumerate(low):
        if v >= 10:
            pt = ser.points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = RED
    style_chart_axes(ch_, size=8.5)
    ch_.category_axis.reverse_order = True
    _tick_skip_one(ch_.category_axis)
    ch_.value_axis.maximum_scale = 25
    ch_.value_axis.has_major_gridlines = False
    ch_.value_axis.visible = False
    add_text(s, x + 0.14, y + gh - 0.5, gw - 0.26, 0.44, [
        [("뒤꿈치·손 관절은 ", {}), ("13~20% 프레임", {"bold": True, "color": RED}),
         ("에서 visibility < 0.5 → 위치를 믿기 어려움", {})]], size=10)

    # 5) summary stats
    x, y = pos[3]
    card(s, x, y, gw, gh, None, "Raw 데이터 품질 요약", title_color=TXT)
    stats = [(f"{raw['below_ground_pct']:.0f}%", "바닥 아래 프레임"),
             (f"{raw['mean_acc']:.1f}", "평균 가속도 (m/s²)"),
             (f"{raw['max_speed']:.1f}", "최대 관절 속도 (m/s)"),
             (f"{raw['bone_cv']:.1f}%", "뼈 길이 변동 CV")]
    sw, sh_ = (gw - 0.3) / 2, 0.98
    for i, (val, lab) in enumerate(stats):
        sx = x + 0.12 + (i % 2) * (sw + 0.06)
        sy = y + 0.5 + (i // 2) * (sh_ + 0.06)
        b = add_box(s, sx, sy, sw, sh_, fill=TINT, line=None, radius=0.08)
        box_text(b, [[(val, {"size": 24, "bold": True, "color": RED})],
                     [(lab, {"size": 10, "color": MUTED})]], anchor=MSO_ANCHOR.MIDDLE)

    notes(s, "왼쪽: 제공 코드(p1)를 그대로 사용해 545프레임 전부에서 BlazePose 33관절 3D 좌표를 추출했습니다. "
             "문제 1) world landmark는 골반이 원점에 고정되어 있어 Unity에 그대로 넣으면 발이 바닥 아래 0.83m에 위치합니다. "
             "문제 2) 손목 속도가 최대 21 m/s까지 튀는 등 프레임 간 떨림이 심합니다. "
             "문제 3) 팔이 카메라를 향할 때 깊이 추정 오류로 위팔 길이가 0.02~0.33m로 변하고 좌우 허벅지 길이도 다릅니다. "
             "문제 4) 뒤꿈치·손처럼 가려지기 쉬운 관절은 visibility가 낮습니다.")


def slide3(s, m, info):
    for sh in list(s.shapes):
        if sh.name in ("TextBox 33", "TextBox 3"):
            remove(sh)

    # Lane A — provided
    ly, lh = 1.42, 0.92
    lab = add_box(s, 0.48, ly, 1.62, lh, fill=BLUE, line=None, radius=0.08)
    box_text(lab, [[("A. 제공 방법", {"bold": True, "size": 12, "color": WHITE})],
                   [("p1_2_post_processing.py", {"size": 8.5, "color": RGBColor(0xD6, 0xE9, 0xF8)})]])
    stepsA = [
        [("load_raw_keypoints()", {"bold": True}), ],
        [("프레임마다 발끝(31·32)", {"bold": True})],
        [("y < 0 관절 → clamp(0)", {"bold": True})],
        [("export_unity_csv()", {"bold": True})],
    ]
    subsA = ["npz → (T, 33, 3) 시퀀스", "최저점을 지면 y=0 으로 이동", "땅 아래로 내려간 값을 0으로", "→ *_target.csv (Unity)"]
    x0, xr = 2.3, 12.85
    n, ag = 4, 0.4
    bw = (xr - x0 - (n - 1) * ag) / n
    for i in range(n):
        bx = x0 + i * (bw + ag)
        b = add_box(s, bx, ly, bw, lh, fill=WHITE, line=BLUE, radius=0.08, line_w=1.0)
        box_text(b, [stepsA[i], [(subsA[i], {"size": 10, "color": TXT})]], size=10.5, color=BLUE)
        if i < n - 1:
            add_arrow(s, bx + bw + (ag - 0.22) / 2, ly + lh / 2 - 0.11, color=BLUE)

    # Lane B — bonus
    ly2, lh2 = 2.52, 1.2
    lab = add_box(s, 0.48, ly2, 1.62, lh2, fill=TEAL, line=None, radius=0.08)
    box_text(lab, [[("B. 추가 후처리", {"bold": True, "size": 12, "color": WHITE})],
                   [("(Bonus)", {"bold": True, "size": 11, "color": WHITE})],
                   [("utils/motion_filters.py", {"size": 8.5, "color": RGBColor(0xDF, 0xF2, 0xF0)})]])
    stepsB = [
        ("① 결측 프레임 복원", "타임라인 유지\n선형 보간"),
        ("② 이상치 제거", "Hampel 필터 +\n7 m/s 속도 제한"),
        ("③ 저신뢰 관절", "visibility < 0.3\n→ 보간 (≤10f)"),
        ("④ 시간 스무딩", "Butterworth 6 Hz\nzero-phase"),
        ("⑤ 뼈 길이 고정", "중앙값·좌우 대칭\n방향 4 Hz 평활"),
        ("⑥ 지면 정렬", "발 6관절 최저점=0\nclamp 불필요"),
    ]
    n, ag = 6, 0.26
    bw = (xr - x0 - (n - 1) * ag) / n
    for i, (t, sub) in enumerate(stepsB):
        bx = x0 + i * (bw + ag)
        b = add_box(s, bx, ly2, bw, lh2, fill=RGBColor(0xEE, 0xF7, 0xF6), line=TEAL, radius=0.08, line_w=1.0)
        paras = [[(t, {"bold": True, "size": 10.5, "color": RGBColor(0x2E, 0x7D, 0x77)})]]
        for line in sub.split("\n"):
            paras.append([(line, {"size": 9.5, "color": TXT})])
        box_text(b, paras, margin=0.03)
        if i < n - 1:
            add_arrow(s, bx + bw + (ag - 0.16) / 2, ly2 + lh2 / 2 - 0.08, w=0.16, h=0.16, color=TEAL)
    log = info["refine_log"]
    add_text(s, x0, ly2 + lh2 + 0.05, xr - x0, 0.28, [[
        ("처리량  ", {"bold": True, "color": TEAL}),
        (f"Hampel {log['hampel_joint_frames']} · 속도 제한 {log['speed_limit_joint_frames']} · "
         f"저신뢰 {log['low_vis_joint_frames']} 관절-프레임 보간 (전체 17,985)   →   출력 ", {}),
        ("*_refined.csv", {"bold": True}), (" (제공 CSV와 같은 형식 → Unity 로더 그대로 사용)", {})]],
        size=9.5, color=MUTED)

    # Bottom-left: provided effect
    by, bh = 4.18, 2.97
    card(s, 0.48, by, 7.95, bh, "!", "제공 방법(A)을 적용해 보니: 지면 문제는 해결, 새로운 부작용 2가지",
         num_fill=RED, title_color=TXT)
    add_picture_fit(s, os.path.join(FIG, "fig_e_provided_effect.png"), 0.58, by + 0.44, 7.75, 1.98)
    clamp = info["provided_clamp"]
    add_text(s, 0.62, by + bh - 0.5, 7.7, 0.46, [
        [("① 발끝 떨림이 몸 전체 높이로 옮겨가 골반 상하 가속도 ", {}),
         (f"{m.loc['Raw', 'root_acc_y']:.2f} → {m.loc['Provided', 'root_acc_y']:.2f} m/s²", {"bold": True, "color": RED})],
        [(f"② 발끝만 기준으로 삼아 뒤꿈치가 {clamp['clamped_frame_pct']:.1f}% 프레임에서 최대 ", {}),
         (f"{clamp['max_clamp_depth_m'] * 100:.1f} cm", {"bold": True, "color": RED}), (" 바닥에 눌려 잘림", {})]],
        size=10)

    # Bottom-right: cutoff rationale
    rx, rw = 8.6, 4.25
    card(s, rx, by, rw, bh, None, "스무딩 강도 선택 근거", title_color=TXT)
    add_picture_fit(s, os.path.join(FIG, "fig_g_cutoff.png"), rx + 0.1, by + 0.44, rw - 0.2, 1.86)
    add_text(s, rx + 0.14, by + bh - 0.67, rw - 0.26, 0.62, [
        [("차단 주파수를 낮출수록 떨림↓ 원본과 차이↑. 인체 동작 분석 표준인 ", {}),
         ("6 Hz", {"bold": True, "color": BLUE}), (" 에서 원본과 1.1 cm 차이로 떨림 65% 감소", {})]],
        size=10)

    notes(s, "A는 수업에서 제공된 p1_2 방법으로, 프레임마다 양 발끝 중 낮은 점을 y=0으로 옮기고 0 아래는 잘라냅니다. "
             "바닥 관통은 해결되지만, 발끝 좌표의 떨림이 몸 전체의 상하 떨림이 되고 뒤꿈치가 바닥에 눌리는 부작용이 있습니다. "
             "B는 추가로 구현한 6단계 후처리입니다. 이상치를 먼저 제거하고(Hampel, 속도 7 m/s 제한, 저신뢰 관절 보간), "
             "6 Hz zero-phase Butterworth로 떨림을 줄인 뒤, 뼈 길이를 중앙값으로 고정하고 좌우를 대칭으로 맞췄습니다. "
             "마지막으로 발목·뒤꿈치·발끝 6관절 중 최저점을 지면에 맞춰 clamp 없이 바닥 관통을 없앴습니다.")


def slide4(s, m, info):
    for sh in list(s.shapes):
        if sh.name in ("TextBox 33", "TextBox 3"):
            remove(sh)
        if sh.name == "TextBox 6":
            set_text_keep_format(sh, "Results & Evaluation — 정량 비교")
    R, P, F = m.loc["Raw"], m.loc["Provided"], m.loc["Refined (Bonus)"]

    def pct(a, b):
        return f"{(b - a) / a * 100:+.0f}%" if a > 1e-9 else "—"

    rows = [
        ("평균 관절 가속도 (떨림, m/s²)", f"{R.mean_acc:.1f}", f"{P.mean_acc:.1f}", f"{F.mean_acc:.1f}", pct(R.mean_acc, F.mean_acc)),
        ("손목 가속도 (m/s²)", f"{R.wrist_acc:.1f}", f"{P.wrist_acc:.1f}", f"{F.wrist_acc:.1f}", pct(R.wrist_acc, F.wrist_acc)),
        ("최대 관절 속도 (m/s)", f"{R.max_speed:.1f}", f"{P.max_speed:.1f}", f"{F.max_speed:.1f}", pct(R.max_speed, F.max_speed)),
        ("7 m/s 초과 튐 (%)", f"{R.spike_pct:.2f}", f"{P.spike_pct:.2f}", f"{F.spike_pct:.2f}", pct(R.spike_pct, F.spike_pct)),
        ("뼈 길이 변동 CV (%)", f"{R.bone_cv:.1f}", f"{P.bone_cv:.1f}", f"{F.bone_cv:.1f}", "0으로"),
        ("좌우 뼈 길이 차이 (%)", f"{R.lr_asym:.1f}", f"{P.lr_asym:.1f}", f"{F.lr_asym:.1f}", "0으로"),
        ("바닥 아래 프레임 (%)", f"{R.below_ground_pct:.0f}", f"{P.below_ground_pct:.0f}", f"{F.below_ground_pct:.0f}", "0으로"),
        ("골반 상하 가속도 (m/s²)", f"{R.root_acc_y:.2f}", f"{P.root_acc_y:.2f}", f"{F.root_acc_y:.2f}", f"{pct(P.root_acc_y, F.root_acc_y)} *"),
        ("원본 대비 자세 차이 (cm)", "—", f"{P.mpjpe_vs_raw_cm:.1f}", f"{F.mpjpe_vs_raw_cm:.1f}", ""),
    ]
    tx, ty, tw = 0.48, 1.42, 6.95
    rh = 0.34
    cols = [2.75, 0.98, 1.05, 1.07, 1.1]
    gf = s.shapes.add_table(len(rows) + 1, 5, Inches(tx), Inches(ty), Inches(tw), Inches(rh * (len(rows) + 1)))
    gf.name = "Table Metrics"
    tbl = gf.table
    tbl.first_row = True
    tbl.horz_banding = False
    for i, cw in enumerate(cols):
        tbl.columns[i].width = Inches(cw)
    head = ["지표", "Raw", "Provided", "Refined", "개선율"]
    head_fill = [RGBColor(0x4A, 0x55, 0x63), RED, BLUE, TEAL, RGBColor(0x4A, 0x55, 0x63)]
    for r in range(len(rows) + 1):
        tbl.rows[r].height = Inches(rh)
        for c in range(5):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            _word_wrap_rules(p)
            run = p.add_run()
            if r == 0:
                run.text = head[c]
                run.font.bold = True
                run.font.color.rgb = WHITE
                run.font.size = Pt(10.5)
                cell.fill.solid()
                cell.fill.fore_color.rgb = head_fill[c]
            else:
                val = rows[r - 1][c]
                run.text = val
                run.font.size = Pt(10.5)
                run.font.color.rgb = TXT
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r % 2 else RGBColor(0xF6, 0xF8, 0xFA)
                if c == 3:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0x2E, 0x7D, 0x77)
                if c == 4 and val:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0x2E, 0x7D, 0x77)
            run.font.name = FONT
            _korean_lang(run)
    add_text(s, tx, ty + rh * (len(rows) + 1) + 0.04, tw, 0.24,
             ["* 골반 상하 가속도는 Provided 대비. 떨림·속도는 골반 기준 상대 좌표로 계산 (17,985 관절-프레임)"],
             size=8.5, color=MUTED)

    # interpretation
    iy = 5.3
    card(s, tx, iy, tw, 7.15 - iy, None, "해석", title_color=TXT)
    add_text(s, tx + 0.14, iy + 0.42, tw - 0.28, 7.15 - iy - 0.48, [
        {"runs": [("제공 방법", {"bold": True, "color": BLUE}),
                  ("은 몸 전체를 위아래로 옮길 뿐 자세는 그대로 → 떨림·뼈 길이 지표가 Raw와 같고, 골반 상하 떨림이 새로 생김", {})],
         "bullet": True, "space_after": 3},
        {"runs": [("떨림 감소는 대부분 ", {}), ("이상치 제거 + 스무딩", {"bold": True, "color": TEAL}),
                  (" (30.6 → 12.4), 뼈 길이 고정·방향 평활로 10.7까지 추가 감소", {})],
         "bullet": True, "bullet_color": "49A09A", "space_after": 3},
        {"runs": [("원본과의 차이 4.1 cm 중 스무딩 몫은 1.1 cm — 나머지는 ", {}),
                  ("접힌 팔 길이를 복원", {"bold": True}), ("한 결과 (의도된 변화)", {})],
         "bullet": True, "bullet_color": "49A09A"},
    ], size=10.5)

    # right: trajectory
    rx, rw = 7.65, 5.2
    add_text(s, rx, 1.38, rw, 0.3, [[("오른손목 궤적 비교 ", {"bold": True, "color": TXT}),
                                     ("(12–17.5 s, 골반 기준)", {"color": MUTED, "size": 10})]], size=12)
    add_picture_fit(s, os.path.join(FIG, "fig_f_trajectory.png"), rx, 1.7, rw, 3.3)

    # right-bottom: stage ablation (native chart)
    ay = 5.15
    add_text(s, rx, ay, rw, 0.3, [[("단계별 떨림 감소 ", {"bold": True, "color": TXT}),
                                   ("(평균 관절 가속도, m/s²)", {"color": MUTED, "size": 10})]], size=12)
    cats = ["Raw", "Provided", "②이상치", "④스무딩", "⑤뼈 길이", "⑥최종"]
    vals = [R.mean_acc, P.mean_acc, m.loc["stage:outlier_removed", "mean_acc"],
            m.loc["stage:smoothed", "mean_acc"], m.loc["stage:bone_fixed", "mean_acc"], F.mean_acc]
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("평균 관절 가속도", [round(float(v), 1) for v in vals])
    gfc = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(rx), Inches(ay + 0.3), Inches(rw),
                             Inches(7.15 - ay - 0.3), cd)
    gfc.name = "Chart Ablation"
    c = gfc.chart
    c.has_title = False
    c.has_legend = False
    c.font.size = Pt(9)
    c.font.name = FONT
    plot = c.plots[0]
    plot.gap_width = 55
    plot.has_data_labels = True
    plot.data_labels.number_format = "0.0"
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    plot.data_labels.font.size = Pt(9)
    plot.data_labels.font.bold = True
    plot.data_labels.font.color.rgb = TXT
    ser = plot.series[0]
    colors = [RED, BLUE, TEAL, TEAL, TEAL, TEAL]
    for i, col in enumerate(colors):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = col
    style_chart_axes(c, size=9)
    c.value_axis.maximum_scale = 36
    c.value_axis.has_major_gridlines = False
    c.value_axis.visible = False

    notes(s, "표는 Raw, 제공 방법, 추가 후처리 결과의 품질 지표입니다. 제공 방법은 바닥 관통만 100%에서 0%로 해결했고 "
             "자세 관련 지표는 Raw와 같습니다. 추가 후처리는 평균 떨림을 65%, 손목 떨림을 69% 줄였고, 7 m/s를 넘는 튐을 거의 없앴으며, "
             "뼈 길이 변동과 좌우 차이를 0으로 만들었습니다. 오른쪽 그래프에서 Raw의 손목 속도 튐(최대 21 m/s)이 사라지면서도 "
             "동작의 큰 흐름은 유지되는 것을 볼 수 있습니다.")


def slide5(s, m, info):
    for sh in list(s.shapes):
        if sh.name in ("TextBox 33", "TextBox 3"):
            remove(sh)
        if sh.name == "TextBox 6":
            set_text_keep_format(sh, "Results & Evaluation — 시각화 비교")
    F = m.loc["Refined (Bonus)"]
    R = m.loc["Raw"]

    lx, lw = 0.48, 8.15
    _, (px, py, pw, ph) = add_picture_fit(s, os.path.join(FIG, "strip_487.png"), lx, 1.42, lw, 4.6,
                                          center=False, name="Comparison Frame")
    add_box(s, px, py, pw, ph, fill=None, line=LINE, shape=MSO_SHAPE.RECTANGLE)
    add_text(s, lx, py + ph + 0.06, lw, 0.3, [[
        ("frame 487 (16.2 s) · 오른팔을 머리 뒤로 올린 동작  ", {"bold": True}),
        ("— 같은 좌표·스케일로 그린 4분할 동기화 영상의 한 장면 (회색 선 = 오른손목 최근 15프레임 궤적)", {"color": MUTED})]],
        size=10)
    # bonus visualization tools
    ty = py + ph + 0.45
    b = add_box(s, lx, ty, lw, 7.15 - ty, fill=TINT, line=None, radius=0.08)
    add_text(s, lx + 0.15, ty + 0.06, lw - 0.3, 7.15 - ty - 0.1, [
        [("Bonus 시각화  ", {"bold": True, "color": BLUE}),
         ("p1_5_render_comparison_video.py", {"bold": True}), (" 영상·Raw·Provided·Refined 4분할 MP4   ", {}),
         ("p1_6_viser_compare.py", {"bold": True}), (" viser 3D 좌우 동시 재생   ", {}),
         ("*_refined.csv", {"bold": True}), (" Unity BlazePoseDataFile 로더 호환", {})]],
        size=10, anchor=MSO_ANCHOR.MIDDLE)

    # right column
    rx, rw = 8.83, 4.02
    oy, oh = 1.42, 3.1
    card(s, rx, oy, rw, oh, None, "관찰 (frame 487)", title_color=TXT)
    items = [
        (RED, "Raw", "골반 원점 고정 → 하체가 바닥 아래 0.84 m, 오른 위팔이 0.056 m로 접힘"),
        (BLUE, "Provided", "바닥 위로 올라왔지만 자세는 Raw 그대로 — 팔 접힘·떨림 유지"),
        (TEAL, "Refined", "위팔 0.237 m 복원, 손목 궤적이 매끄럽고 두 발이 바닥에 안착"),
    ]
    iy = oy + 0.5
    for col, name, desc in items:
        add_box(s, rx + 0.16, iy + 0.07, 0.14, 0.14, fill=col, line=None, shape=MSO_SHAPE.OVAL)
        add_text(s, rx + 0.4, iy, rw - 0.55, 0.82, [
            [(name, {"bold": True, "color": col, "size": 11})],
            [(desc, {"size": 10})]], size=10)
        iy += 0.86

    cy = oy + oh + 0.15
    card(s, rx, cy, rw, 7.15 - cy, None, "결론 및 한계", title_color=TXT)
    add_text(s, rx + 0.14, cy + 0.42, rw - 0.26, 7.15 - cy - 0.48, [
        {"runs": [("결론  ", {"bold": True, "color": BLUE}),
                  (f"제공 방법은 지면 문제만 해결. 추가 후처리로 떨림 {abs((F.mean_acc - R.mean_acc) / R.mean_acc * 100):.0f}%↓, "
                   f"튐 99%↓, 뼈 길이 CV 0%, 관통 0%", {})], "bullet": True, "space_after": 3},
        {"runs": [("한계  ", {"bold": True, "color": RED}),
                  ("전역 이동(root translation) 미복원 · 점프 표현 불가 · 허리를 숙일 때 깊이 오류(frame 391)는 길이 고정만으론 해결 불가", {})],
         "bullet": True, "bullet_color": "E4572E", "space_after": 3},
        {"runs": [("향후  ", {"bold": True, "color": TEAL}),
                  ("2D 골반 궤적으로 root 이동 복원, 발 접촉 IK", {})], "bullet": True, "bullet_color": "49A09A"},
    ], size=10.5)

    notes(s, "영상과 세 가지 모션 데이터를 같은 좌표계에서 동시에 재생한 비교 영상입니다. "
             "Raw는 하체가 바닥 아래에 있고, 오른팔이 카메라를 향하면서 위팔이 0.056m로 접혀 있습니다. "
             "제공 방법은 바닥 위로 올라왔지만 자세 자체는 같아 팔이 여전히 접혀 있습니다. "
             "추가 후처리 결과는 위팔 길이가 복원되고 손목 궤적이 부드럽습니다. "
             "한계로는 전역 이동이 없고, 프레임별 지면 정렬이라 점프를 표현할 수 없으며, 허리를 깊게 숙이는 동작처럼 "
             "깊이 방향 자체가 틀린 경우는 길이 고정만으로 고칠 수 없습니다.")


def main():
    tmpl = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "report", "Assignment02_3D_Motion_PostProcessing.pptx")
    m, info, low = load_data()
    prs = Presentation(tmpl)
    sl = prs.slides
    slide1(sl[0])
    slide2(sl[1], m, info, low)
    slide3(sl[2], m, info)
    slide4(sl[3], m, info)
    slide5(sl[4], m, info)
    prs.save(out)
    print("Saved →", out)


if __name__ == "__main__":
    main()
