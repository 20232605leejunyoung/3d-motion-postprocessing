"""
p1_5_render_comparison_video.py
---------------------------------------------------------------
[Bonus] Improved visualization

원본 영상과 3가지 모션 데이터를 한 화면에서 동기화 재생하는 비교 영상(MP4) 생성.

    | Video | RAW | PROVIDED (p1_2) | REFINED (Bonus) |

  - 모든 패널이 같은 월드 좌표/스케일을 사용 → Raw 는 Unity 바닥(y=0) 아래로 묻혀 보임
  - 오른손목 최근 15프레임 궤적(trail)을 그려 떨림을 눈으로 비교
  - 뼈 길이 / 바닥 관통 여부를 패널 하단에 실시간 표시

출력: data/output/{name}_comparison.mp4, data/output/figures/strip_{frame}.png

Usage:
    python scripts/p1_5_render_comparison_video.py [name]
"""
import os
import sys

import cv2
import numpy as np
import imageio.v2 as imageio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import POSE_CONNECTIONS  # noqa: E402
from utils.motion_filters import FOOT_JOINTS  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "data", "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

H = 1080
VID_W = 609                     # ≈1080 * 9/16 (전체 폭 1920 맞춤)
PANEL_W = 437
PX_PER_M = 360.0
FLOOR_ROW = 760                 # y=0 이 그려질 화면 높이(px)
TRAIL = 15

# BGR
COLORS = {
    "RAW": (46, 87, 228),        # #E4572E
    "PROVIDED": (184, 108, 0),   # #006CB8
    "REFINED": (154, 160, 73),   # #49A09A
}
LEFT_SIDE = {11, 13, 15, 23, 25, 27, 29, 31}


def to_px(p, cx):
    x = int(round(cx + p[0] * PX_PER_M))
    y = int(round(FLOOR_ROW - p[1] * PX_PER_M))
    return x, y


def draw_panel(seq, t, label, color):
    img = np.full((H, PANEL_W, 3), 250, np.uint8)
    # underground
    img[FLOOR_ROW:] = (232, 236, 246)
    # floor grid (perspective-free simple lines)
    for k in range(-3, 4):
        x = int(PANEL_W / 2 + k * 0.25 * PX_PER_M)
        cv2.line(img, (x, FLOOR_ROW), (x, FLOOR_ROW + 8), (190, 190, 190), 1)
    cv2.line(img, (0, FLOOR_ROW), (PANEL_W, FLOOR_ROW), (62, 74, 122), 2)
    cv2.putText(img, "floor y=0", (8, FLOOR_ROW - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (62, 74, 122), 1, cv2.LINE_AA)

    cx = PANEL_W / 2
    pts = seq[t]
    light = tuple(int(c + (255 - c) * 0.45) for c in color)
    # trail of right wrist
    a = max(0, t - TRAIL)
    tr = [to_px(seq[k, 16], cx) for k in range(a, t + 1)]
    for i in range(1, len(tr)):
        alpha = i / len(tr)
        col = tuple(int(255 - (255 - c) * alpha) for c in (40, 40, 40))
        cv2.line(img, tr[i - 1], tr[i], col, 2, cv2.LINE_AA)
    # skeleton
    for i, j in POSE_CONNECTIONS:
        c = light if (i in LEFT_SIDE and j in LEFT_SIDE) else color
        cv2.line(img, to_px(pts[i], cx), to_px(pts[j], cx), c, 6, cv2.LINE_AA)
    neck = (pts[11] + pts[12]) / 2
    cv2.line(img, to_px(neck, cx), to_px(pts[0], cx), color, 6, cv2.LINE_AA)
    for k in range(0, 11):
        cv2.circle(img, to_px(pts[k], cx), 4, color, -1, cv2.LINE_AA)
    for k in range(11, 33):
        cv2.circle(img, to_px(pts[k], cx), 5, (60, 60, 60), -1, cv2.LINE_AA)

    # header
    cv2.rectangle(img, (0, 0), (PANEL_W, 64), color, -1)
    cv2.putText(img, label, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    # stats (상단 — 스켈레톤과 겹치지 않도록)
    low = pts[list(FOOT_JOINTS), 1].min()
    arm = np.linalg.norm(pts[12] - pts[14])
    status = "BELOW FLOOR" if low < -0.01 else "on floor"
    st_col = (40, 40, 190) if low < -0.01 else (50, 50, 50)
    cv2.putText(img, f"lowest foot: {low:+.2f} m ({status})", (14, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, st_col, 1, cv2.LINE_AA)
    cv2.putText(img, f"R upper arm: {arm:.3f} m", (14, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    return img


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "tail_sunmi_dance"
    seqs = {
        "RAW": np.load(os.path.join(PROC, f"{name}_raw.npy")),
        "PROVIDED": np.load(os.path.join(PROC, f"{name}_target.npy")),
        "REFINED": np.load(os.path.join(PROC, f"{name}_refined.npy")),
    }
    labels = {"RAW": "RAW", "PROVIDED": "PROVIDED (p1_2)", "REFINED": "REFINED (Bonus)"}
    T = seqs["RAW"].shape[0]

    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "raw_videos", f"{name}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_path = os.path.join(OUT, f"{name}_comparison.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7,
                                macro_block_size=8, pixelformat="yuv420p")
    strip_frames = {150, 391, 487}

    for t in range(T):
        ok, frame = cap.read()
        if not ok:
            break
        vid = cv2.resize(frame, (VID_W, H))
        cv2.putText(vid, f"frame {t:03d}  ({t / fps:5.2f}s)", (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panels = [draw_panel(seqs[k], t, labels[k], COLORS[k]) for k in ["RAW", "PROVIDED", "REFINED"]]
        canvas = np.hstack([vid] + panels)
        for x in np.cumsum([VID_W] + [PANEL_W] * 2):
            cv2.line(canvas, (int(x), 0), (int(x), H), (210, 210, 210), 2)
        writer.append_data(canvas[:, :, ::-1])
        if t in strip_frames:
            cv2.imwrite(os.path.join(FIG, f"strip_{t:03d}.png"), canvas)
    writer.close()
    cap.release()
    print("Saved →", out_path)


if __name__ == "__main__":
    main()
