"""
p1_3_post_processing_pipeline.py
---------------------------------------------------------------
Step 3. Post-processing

  (A) Provided method  — p1_2_post_processing.py 의 함수를 그대로 호출
        load_raw_keypoints → create_height_corrected_target → export_unity_csv
        출력: data/processed/{name}_raw.npy, _target.npy, _target.csv
        (p1_2 의 main() 과 동일한 결과. main() 은 마지막에 viser 무한 재생으로
         멈추기 때문에 함수만 가져와 사용)

  (B) Bonus method     — utils/motion_filters.refine_motion
        결측 프레임 복원 → Hampel + 속도 제한 이상치 제거 → 저신뢰 관절 보간
        → Butterworth 6Hz → 뼈 길이 고정/좌우 대칭 → 발 관절 6개 기준 지면 정렬
        출력: data/processed/{name}_refined.npy, _refined.csv (Unity 호환)

  (C) 품질 지표 비교 → data/output/{name}_metrics.csv

Usage:
    python scripts/p1_3_post_processing_pipeline.py
"""
import os
import sys
import json
from glob import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.p1_2_post_processing import (  # noqa: E402  (provided code)
    load_raw_keypoints,
    create_height_corrected_target,
    export_unity_csv,
    RAW_KEYPOINTS_DIR,
    OUT_DIR,
)
from utils.motion_filters import (  # noqa: E402  (bonus code)
    load_raw_with_meta,
    refine_motion,
    compute_metrics,
    mpjpe,
)

REPORT_DIR = os.path.join(ROOT, "data", "output")
os.makedirs(REPORT_DIR, exist_ok=True)

FPS = 30.0
CUTOFF_HZ = 6.0


def clamp_stats(raw_seq):
    """제공 방법의 clamp(0) 가 실제로 얼마나 개입했는지 측정."""
    seq = raw_seq.copy()
    ground = np.minimum(seq[:, 31, 1], seq[:, 32, 1])
    seq[:, :, 1] -= ground[:, None]
    below = seq[:, :, 1] < 0
    return {
        "clamped_joint_frames": int(below.sum()),
        "clamped_frame_pct": float(below.any(axis=1).mean() * 100),
        "clamped_joints": sorted(set(np.where(below)[1].tolist())),
        "max_clamp_depth_m": float(max(0.0, -seq[:, :, 1].min())),
    }


def main():
    paths = sorted(glob(os.path.join(RAW_KEYPOINTS_DIR, "*.npz")))
    print(f"Found {len(paths)} raw keypoint files")

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")
        print(f"\n=== {base} ===")

        # ---------------- (A) Provided ----------------
        raw_seq = load_raw_keypoints(path)
        np.save(os.path.join(OUT_DIR, f"{base}_raw.npy"), raw_seq)
        target_seq = create_height_corrected_target(raw_seq)
        np.save(os.path.join(OUT_DIR, f"{base}_target.npy"), target_seq)
        export_unity_csv(target_seq, os.path.join(OUT_DIR, f"{base}_target.csv"))

        # ---------------- (B) Bonus -------------------
        frames, seq, vis = load_raw_with_meta(path)
        _, refined, log, stages = refine_motion(frames, seq, vis, fps=FPS, cutoff=CUTOFF_HZ,
                                                return_stages=True)
        np.save(os.path.join(OUT_DIR, f"{base}_refined.npy"), refined)
        np.save(os.path.join(OUT_DIR, f"{base}_vis.npy"), vis)
        for k, v in stages.items():
            np.save(os.path.join(OUT_DIR, f"{base}_stage_{k}.npy"), v)
        export_unity_csv(refined, os.path.join(OUT_DIR, f"{base}_refined.csv"))
        print("Refine log:", log)

        # ---------------- (C) Metrics -----------------
        rows = []
        for name, s in [("Raw", raw_seq), ("Provided", target_seq), ("Refined (Bonus)", refined)]:
            m = compute_metrics(s, fps=FPS)
            m["mpjpe_vs_raw_cm"] = mpjpe(s, raw_seq) * 100
            m["method"] = name
            rows.append(m)
        # 단계별 기여도 (ablation)
        for name in ["outlier_removed", "smoothed", "bone_fixed", "grounded"]:
            m = compute_metrics(stages[name], fps=FPS)
            m["mpjpe_vs_raw_cm"] = mpjpe(stages[name], raw_seq) * 100
            m["method"] = f"stage:{name}"
            rows.append(m)
        df = pd.DataFrame(rows).set_index("method")
        out_csv = os.path.join(REPORT_DIR, f"{base}_metrics.csv")
        df.to_csv(out_csv, float_format="%.4f", encoding="utf-8-sig")

        info = {"refine_log": log, "provided_clamp": clamp_stats(raw_seq),
                "frames": int(raw_seq.shape[0]), "fps": FPS, "cutoff_hz": CUTOFF_HZ}
        with open(os.path.join(REPORT_DIR, f"{base}_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        pd.set_option("display.width", 200)
        print(df.round(3).to_string())
        print(json.dumps(info, indent=2, ensure_ascii=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
