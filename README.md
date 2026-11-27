# Assignment 02 — 3D Motion Data Post-Processing

기말 프로젝트 영상(선미 ‘꼬리(Tail)’ 댄스 커버, YouTube Shorts)에서 MediaPipe BlazePose로
3D 모션을 추출하고, 문제점을 분석한 뒤 **제공된 후처리(p1_2)** 와 **추가 후처리(Bonus)** 를
적용해 전후를 비교한다.

- 보고서: [`report/Assignment02_3D_Motion_PostProcessing.pptx`](report/Assignment02_3D_Motion_PostProcessing.pptx)
- 비교 영상: [`data/output/tail_sunmi_dance_comparison.mp4`](data/output/tail_sunmi_dance_comparison.mp4) (영상 | Raw | Provided | Refined)
- 원본 영상: <https://www.youtube.com/shorts/SQJdJ1jctk8> — 저작권 문제로 저장소에는 넣지 않았다.
  직접 실행하려면 영상을 받아 `data/raw_videos/tail_sunmi_dance.mp4` 로 저장한다.
  추출된 키포인트(`data/raw_keypoints/tail_sunmi_dance.npz`)가 포함되어 있어 후처리(`p1_3`)와 viser 비교(`p1_6`)는
  영상 없이 실행되고, 영상 프레임을 쓰는 `p1_4`·`p1_5`만 원본 영상이 필요하다.

## 파이프라인

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 1. Data Acquisition | `scripts/p1_1_extract_local_video.py` | 제공 코드 `p1_create_raw_keypoints.extract_3d_keypoints()`를 그대로 import해 로컬 영상에서 33관절 world landmark 추출 → `data/raw_keypoints/*.npz` |
| 3A. Provided post-processing | `scripts/p1_3_post_processing_pipeline.py` | 제공 코드 `p1_2_post_processing`의 `load_raw_keypoints → create_height_corrected_target → export_unity_csv` 호출 → `*_target.npy/csv` |
| 3B. Bonus post-processing | `utils/motion_filters.py` | 결측 프레임 복원 → Hampel + 7 m/s 속도 제한 이상치 제거 → 저신뢰(vis<0.3) 관절 보간 → Butterworth 6 Hz zero-phase → 뼈 길이 고정·좌우 대칭(방향 4 Hz 평활, 손가락은 손목에 강체 부착) → 발 6관절 기준 지면 정렬 → `*_refined.npy/csv` |
| 2 & 4. 시각화·평가 | `scripts/p1_4_visualize_compare.py` | 문제점/전후 비교 그래프 → `data/output/figures/` |
| Bonus 시각화 | `scripts/p1_5_render_comparison_video.py` | 4분할 동기화 비교 MP4 |
| Bonus 시각화 | `scripts/p1_6_viser_compare.py` | viser 3D 듀얼 뷰 (제공 `PoseViser.play_two_sequences` 사용) |
| 보고서 | `report/build_report.py` | 과제 템플릿에 결과를 채워 PPT 생성 |

`*_refined.csv`는 제공 CSV와 같은 형식(`frame, landmark, x, y, z, visibility`)이라 Unity
`BlazePoseDataFile` 로더에서 그대로 불러올 수 있다.

## 결과 요약 (545 프레임, 30 fps)

| 지표 | Raw | Provided | Refined (Bonus) |
|---|---|---|---|
| 평균 관절 가속도 (떨림, m/s²) | 30.6 | 30.6 | **10.7** (−65%) |
| 손목 가속도 (m/s²) | 58.8 | 58.8 | **18.2** (−69%) |
| 최대 관절 속도 (m/s) | 23.1 | 23.1 | **7.0** |
| 7 m/s 초과 튐 (% 관절-프레임) | 1.06 | 1.06 | **0.01** |
| 뼈 길이 변동 CV (%) | 11.3 | 11.3 | **0.0** |
| 좌우 뼈 길이 차이 (%) | 13.5 | 13.5 | **0.0** |
| 바닥 아래 프레임 (%) | 100 | 0 | **0** |
| 골반 상하 가속도 (m/s²) | 0.05 | 5.50 | **2.12** |

- 제공 방법은 바닥 관통만 해결하고 자세는 그대로이며, 발끝 떨림이 몸 전체 상하 떨림으로 전파되고
  왼쪽 뒤꿈치가 6.8% 프레임에서 최대 3.1 cm clamp 된다.
- 한계: 전역 이동(root translation) 미복원, 프레임별 지면 정렬이라 점프 표현 불가, 허리를 숙일 때
  깊이 방향 오류는 뼈 길이 고정만으로 해결되지 않음.

## 실행 방법

```bash
conda env create -f environment.yml     # 또는: pip install -r requirements.txt (Python 3.10)
conda activate ai_ml_final

# 영상을 data/raw_videos/ 에 넣은 뒤
python scripts/p1_1_extract_local_video.py
python scripts/p1_3_post_processing_pipeline.py
python scripts/p1_4_visualize_compare.py
python scripts/p1_5_render_comparison_video.py
python scripts/p1_6_viser_compare.py            # http://localhost:8080
python report/build_report.py <Assignment_3D_motiondata.pptx 템플릿 경로>
```

환경: Python 3.10, mediapipe 0.10.21 (수업 `environment.yml` 기준).
