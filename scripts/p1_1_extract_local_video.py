"""
p1_1_extract_local_video.py
---------------------------------------------------------------
Step 1. Data Acquisition

수업에서 제공된 p1_create_raw_keypoints.py 의 extract_3d_keypoints()를
그대로 재사용하여, 이미 내려받은 로컬 영상(data/raw_videos/*.mp4)에서
BlazePose 3D world landmark (33 joints) 를 추출한다.

  - 원본 코드는 YouTube URL → pytubefix 다운로드 → 추출 순서이지만,
    기말 프로젝트 영상은 이미 로컬에 있으므로 다운로드 단계만 건너뛴다.
  - 추출 로직(MediaPipe Pose, model_complexity=2, smooth_landmarks=True,
    y축 반전, npz 저장 형식)은 제공 코드와 100% 동일하다.

Usage:
    python scripts/p1_1_extract_local_video.py                # raw_videos 폴더 전체
    python scripts/p1_1_extract_local_video.py my_video.mp4   # 특정 파일만
"""
import os
import sys
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 제공 코드의 추출 함수를 그대로 사용
from scripts.p1_create_raw_keypoints import (  # noqa: E402
    extract_3d_keypoints,
    RAW_VIDEO_DIR,
    RAW_KEYPOINT_DIR,
)


def main():
    if len(sys.argv) > 1:
        videos = [os.path.join(RAW_VIDEO_DIR, v) if not os.path.isabs(v) else v
                  for v in sys.argv[1:]]
    else:
        videos = sorted(glob(os.path.join(RAW_VIDEO_DIR, "*.mp4")))

    print(f"Found {len(videos)} video(s)")
    for video_path in videos:
        name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"▶ Extracting: {name}")
        extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, name)


if __name__ == "__main__":
    main()
