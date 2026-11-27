"""
p1_6_viser_compare.py
---------------------------------------------------------------
[Bonus] viser 3D 듀얼 뷰 — 수업 제공 utils/viser_test.PoseViser.play_two_sequences 사용.

    왼쪽 : Provided (p1_2 결과, *_target.npy)   또는 --raw 옵션 시 Raw
    오른쪽: Refined  (Bonus 결과, *_refined.npy)

Usage:
    python scripts/p1_6_viser_compare.py                 # provided vs refined
    python scripts/p1_6_viser_compare.py --raw           # raw vs refined
    python scripts/p1_6_viser_compare.py --name my_video
브라우저에서 터미널에 출력되는 주소(기본 http://localhost:8080)를 연다. Ctrl+C 로 종료.
"""
import os
import sys
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tail_sunmi_dance")
    ap.add_argument("--raw", action="store_true", help="왼쪽에 Provided 대신 Raw 표시")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--offset", type=float, default=1.2, help="두 스켈레톤 사이 간격 (m)")
    args = ap.parse_args()

    left = np.load(os.path.join(PROC, f"{args.name}_{'raw' if args.raw else 'target'}.npy"))
    right = np.load(os.path.join(PROC, f"{args.name}_refined.npy"))
    # viser 는 z 가 화면 안쪽 → ex1_viser.py 와 동일하게 z 부호 반전
    left, right = left.copy(), right.copy()
    left[:, :, 2] *= -1
    right[:, :, 2] *= -1

    vis = PoseViser(fps=args.fps)
    vis.play_two_sequences(left, right, offset=args.offset)


if __name__ == "__main__":
    main()
