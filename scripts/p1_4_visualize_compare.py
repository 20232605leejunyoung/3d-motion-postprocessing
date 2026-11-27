"""
p1_4_visualize_compare.py
---------------------------------------------------------------
Step 2 & 4. Problem Identification / Results & Analysis 시각화

p1_3 실행 후 data/processed/*.npy 를 읽어 보고서용 그림을 만든다.
출력: data/output/figures/*.png

  fig_a_acquisition.png      영상 프레임 + Raw 스켈레톤 (Unity 바닥 y=0 기준 → 발이 바닥 아래)
  fig_b_spike.png            오른손목 속도 — 7 m/s 초과 튐(spike)
  fig_c_bone.png             뼈 길이 변화 — 원근 단축/좌우 비대칭
  fig_d_glitch.png           튐 발생 3프레임 겹쳐 그리기 (손 순간이동)
  fig_e_provided_effect.png  제공 방법의 부작용 — 골반 높이 떨림 / 뒤꿈치 clamp
  fig_f_trajectory.png       Before / After 손목 궤적 + 속도
  fig_g_cutoff.png           Butterworth 차단 주파수 선택 근거 (떨림 vs 원본과의 차이)

Usage:
    python scripts/p1_4_visualize_compare.py [name]
"""
import os
import sys
import json

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.motion_filters import (  # noqa: E402
    root_relative, butterworth_smooth, compute_metrics, mpjpe, load_raw_with_meta,
    refine_motion, FOOT_JOINTS,
)
from utils.viser_test import POSE_CONNECTIONS  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "data", "output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FPS = 30.0
C_RAW, C_PROV, C_REF = "#E4572E", "#006CB8", "#49A09A"
C_TXT, C_GRID, C_MUTED = "#333333", "#E3E6EA", "#8A9099"

# ---- Korean font -------------------------------------------------
for fam in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(fam == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": "#B8BDC4", "axes.labelcolor": C_TXT, "xtick.color": C_TXT,
    "ytick.color": C_TXT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})

LEFT_SIDE = {11, 13, 15, 23, 25, 27, 29, 31}


def draw_skeleton(ax, pts, color, lw=2.0, alpha=1.0, point=True):
    for a, b in POSE_CONNECTIONS:
        ax.plot(pts[[a, b], 0], pts[[a, b], 1], color=color, lw=lw, alpha=alpha,
                solid_capstyle="round")
    # head: nose + shoulder midpoint
    neck = (pts[11] + pts[12]) / 2
    ax.plot([neck[0], pts[0, 0]], [neck[1], pts[0, 1]], color=color, lw=lw, alpha=alpha)
    if point:
        ax.scatter(pts[:33, 0], pts[:33, 1], s=6, color=color, alpha=alpha, zorder=3)


def speed(seq):
    return np.linalg.norm(np.diff(root_relative(seq), axis=0), axis=2) * FPS


def bone_len(seq, a, b):
    return np.linalg.norm(seq[:, a] - seq[:, b], axis=1)


def video_frame(name, idx):
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "raw_videos", f"{name}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, im = cap.read()
    cap.release()
    return im[:, :, ::-1] if ok else None


# ==================================================================
def fig_a_acquisition(name, raw, frame=150):
    im = video_frame(name, frame)
    fig = plt.figure(figsize=(4.3, 3.6))
    ax0 = fig.add_axes([0.0, 0.02, 0.36, 0.94])
    ax0.imshow(im[260:1660, 80:1000])
    ax0.set_axis_off()
    ax0.set_title(f"입력 영상 (frame {frame})", fontsize=9, color=C_TXT)

    ax = fig.add_axes([0.47, 0.1, 0.5, 0.86])
    ax.axhspan(-1.1, 0, color="#F6D5CC", alpha=0.6, lw=0)
    ax.axhline(0, color="#7A4A3E", lw=1.4)
    ax.text(-0.68, 0.035, "Unity 바닥 (y=0)", fontsize=8, color="#7A4A3E", ha="left")
    ax.text(-0.68, -1.03, "바닥 아래", fontsize=8, color="#B4442A", ha="left")
    p = raw[frame]
    draw_skeleton(ax, p, C_RAW)
    low = p[list(FOOT_JOINTS), 1].min()
    ax.annotate(f"발 y = {low:.2f} m", xy=(p[31, 0], low), xytext=(-0.68, -0.92),
                fontsize=8.5, color="#B4442A", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#B4442A", lw=1))
    mh = (p[23] + p[24]) / 2
    ax.scatter([mh[0]], [mh[1]], s=40, facecolor="white", edgecolor=C_TXT, zorder=5)
    ax.annotate("골반 = 원점 고정", xy=(mh[0] - 0.02, mh[1] - 0.02), xytext=(-0.68, -0.25),
                fontsize=8.5, color=C_TXT, arrowprops=dict(arrowstyle="->", color=C_TXT, lw=1))
    ax.set_xlim(-0.7, 0.4)
    ax.set_ylim(-1.1, 0.75)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Raw 3D (정면 투영)", fontsize=9, color=C_TXT)
    fig.savefig(os.path.join(FIG_DIR, "fig_a_acquisition.png"))
    plt.close(fig)


def fig_b_spike(raw):
    sp = speed(raw)
    t = np.arange(1, raw.shape[0]) / FPS
    j = 16
    s = sp[:, j]
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.plot(t, s, color=C_RAW, lw=0.9)
    ax.axhline(7, color=C_TXT, ls="--", lw=0.9)
    ax.text(0.15, 7.7, "한계 7 m/s", ha="left", fontsize=7.5, color=C_TXT)
    over = s > 7
    ax.scatter(t[over], s[over], s=10, color="#9E2A12", zorder=3)
    k = int(np.argmax(s))
    ax.annotate(f"최대 {s[k]:.1f} m/s", xy=(t[k], s[k]), xytext=(t[k] - 6.5, s[k] - 1.5),
                fontsize=8, color="#9E2A12", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#9E2A12", lw=0.9))
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, max(20, s.max() * 1.1))
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("오른손목 속도 (m/s)")
    fig.savefig(os.path.join(FIG_DIR, "fig_b_spike.png"))
    plt.close(fig)


def fig_c_bone(raw):
    t = np.arange(raw.shape[0]) / FPS
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    items = [((12, 14), "오른쪽 위팔", C_RAW), ((23, 25), "왼쪽 허벅지", "#7B5EA7"),
             ((24, 26), "오른쪽 허벅지", "#D49A2A")]
    for (a, b), lab, c in items:
        L = bone_len(raw, a, b)
        ax.plot(t, L, color=c, lw=0.9, label=f"{lab} (CV {L.std() / L.mean() * 100:.0f}%)")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("뼈 길이 (m)")
    ax.legend(fontsize=7, loc="lower left", ncol=3, handlelength=1.2, columnspacing=0.8,
              bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    fig.savefig(os.path.join(FIG_DIR, "fig_c_bone.png"))
    plt.close(fig)


def fig_d_glitch(raw, refined):
    sp = speed(raw)[:, 16]
    k = int(np.argmax(sp)) + 1           # frame index where jump lands
    frames = [k - 1, k, k + 1]
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.3))
    for ax, seq, col, title in [(axes[0], raw, C_RAW, "Raw"), (axes[1], refined, C_REF, "Refined")]:
        base = root_relative(seq)
        for i, f in enumerate(frames):
            draw_skeleton(ax, base[f], col, lw=1.6, alpha=[0.3, 1.0, 0.55][i], point=False)
        w = base[frames, 16]
        ax.plot(w[:, 0], w[:, 1], "-", color="#222", lw=0.9, zorder=4)
        ax.scatter(w[:, 0], w[:, 1], s=14, color="#222", zorder=5)
        for f, (x, y) in zip(frames, w[:, :2]):
            ax.text(x - 0.03, y + 0.02, str(f), fontsize=6.5, color="#222", ha="right")
        c = base[frames, 11:17].reshape(-1, 3)
        cx, cy = c[:, 0].mean(), c[:, 1].mean()
        ax.set_aspect("equal")
        ax.set_xlim(cx - 0.42, cx + 0.42)
        ax.set_ylim(cy - 0.38, cy + 0.38)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sname in ["left", "bottom"]:
            ax.spines[sname].set_visible(False)
        ax.set_title(f"{title}  (frame {frames[0]}–{frames[-1]})", fontsize=8.5, color=C_TXT)
    fig.savefig(os.path.join(FIG_DIR, "fig_d_glitch.png"))
    plt.close(fig)
    return frames


def fig_e_provided_effect(raw, target, refined):
    t = np.arange(raw.shape[0]) / FPS
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.1), gridspec_kw={"wspace": 0.22})
    ax = axes[0]
    for seq, c, lab in [(raw, C_RAW, "Raw"), (target, C_PROV, "Provided"), (refined, C_REF, "Refined")]:
        hip = ((seq[:, 23] + seq[:, 24]) / 2)[:, 1]
        ax.plot(t, hip - np.median(hip), color=c, lw=0.9 if c != C_PROV else 0.8, label=lab)
    ax.set_xlim(8, 14)
    ax.set_ylim(-0.1, 0.13)
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("골반 높이 변화 (m)")
    ax.set_title("① 프레임별 지면 보정 → 발끝 떨림이 몸 전체로 전파", fontsize=9, color=C_TXT, loc="left")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.03))

    ax = axes[1]
    seq = raw.copy()
    ground = np.minimum(seq[:, 31, 1], seq[:, 32, 1])
    heel = seq[:, 29, 1] - ground
    ax.fill_between(t, heel, 0, where=heel < 0, color="#E98A70", lw=0)
    ax.plot(t, heel, color=C_TXT, lw=0.8)
    ax.axhline(0, color="#7A4A3E", lw=1)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.05, 0.12)
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("왼쪽 뒤꿈치 높이 (m)")
    pct = (heel < 0).mean() * 100
    k = int(np.argmin(heel))
    ax.annotate(f"clamp(0)로 잘림\n(최대 {-heel[k] * 100:.1f} cm)", xy=(t[k], heel[k]),
                xytext=(9.6, -0.043), fontsize=7.5, color="#B4442A",
                arrowprops=dict(arrowstyle="->", color="#B4442A", lw=0.9))
    ax.set_title(f"② 발끝만 기준 → 뒤꿈치가 바닥에 눌림 ({pct:.1f}% 프레임)", fontsize=9, color=C_TXT, loc="left")
    fig.savefig(os.path.join(FIG_DIR, "fig_e_provided_effect.png"))
    plt.close(fig)


def fig_f_trajectory(raw, target, refined, span=(12.0, 17.5)):
    t = np.arange(raw.shape[0]) / FPS
    a, b = int(span[0] * FPS), int(span[1] * FPS)
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 3.3), sharex=True, gridspec_kw={"hspace": 0.12})
    ax = axes[0]
    for seq, c, lab, lw in [(raw, C_RAW, "Raw = Provided (자세 변화 없음)", 1.0), (refined, C_REF, "Refined (Bonus)", 1.8)]:
        y = root_relative(seq)[:, 16, 1]
        ax.plot(t[a:b], y[a:b], color=c, lw=lw, label=lab)
    ax.set_ylabel("오른손목 높이 (m)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.16))
    ax = axes[1]
    for seq, c, lw in [(raw, C_RAW, 1.0), (refined, C_REF, 1.8)]:
        s = speed(seq)[:, 16]
        ax.plot(t[1:][a:b], s[a:b], color=c, lw=lw)
    ax.axhline(7, color=C_TXT, ls="--", lw=0.8)
    ax.set_ylabel("속도 (m/s)")
    ax.set_xlabel("시간 (s)")
    ax.set_xlim(span)
    fig.savefig(os.path.join(FIG_DIR, "fig_f_trajectory.png"))
    plt.close(fig)


def fig_g_cutoff(name, raw):
    frames, seq, vis = load_raw_with_meta(os.path.join(ROOT, "data", "raw_keypoints", f"{name}.npz"))
    cutoffs = [2, 3, 4, 6, 8, 10, 12]
    jit, dev = [], []
    for c in cutoffs:
        _, r, _ = refine_motion(frames, seq, vis, cutoff=c)
        jit.append(compute_metrics(r)["mean_acc"])
        dev.append(mpjpe(butterworth_smooth(seq, cutoff=c), raw) * 100)
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.plot(cutoffs, jit, "o-", color=C_REF, lw=1.5, ms=4, label="떨림 (m/s²)")
    ax.set_xlabel("Butterworth 차단 주파수 (Hz)")
    ax.set_ylabel("떨림 (m/s²)", color=C_REF)
    ax2 = ax.twinx()
    ax2.plot(cutoffs, dev, "s--", color=C_MUTED, lw=1.2, ms=3.5, label="원본과의 차이 (cm)")
    ax2.set_ylabel("원본과의 차이 (cm)", color=C_MUTED)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.axvline(6, color=C_PROV, lw=1, alpha=0.6)
    ax.text(6.25, max(jit) * 0.92, "선택: 6 Hz", fontsize=8, color=C_PROV)
    fig.savefig(os.path.join(FIG_DIR, "fig_g_cutoff.png"))
    plt.close(fig)
    return dict(zip(cutoffs, zip(jit, dev)))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "tail_sunmi_dance"
    raw = np.load(os.path.join(PROC, f"{name}_raw.npy"))
    target = np.load(os.path.join(PROC, f"{name}_target.npy"))
    refined = np.load(os.path.join(PROC, f"{name}_refined.npy"))

    fig_a_acquisition(name, raw)
    fig_b_spike(raw)
    fig_c_bone(raw)
    frames = fig_d_glitch(raw, refined)
    fig_e_provided_effect(raw, target, refined)
    fig_f_trajectory(raw, target, refined)
    cut = fig_g_cutoff(name, raw)

    info = {"glitch_frames": frames, "cutoff_sweep": {str(k): v for k, v in cut.items()}}
    with open(os.path.join(FIG_DIR, "figure_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print("Figures →", FIG_DIR)


if __name__ == "__main__":
    main()
