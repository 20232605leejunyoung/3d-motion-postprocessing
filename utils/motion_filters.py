"""
utils/motion_filters.py
---------------------------------------------------------------
BlazePose 3D motion (T, 33, 3) 후처리 + 품질 평가 유틸리티.

[Bonus] 제공된 후처리(p1_2: 프레임별 지면 보정 + clamp) 외에 추가로 구현한 방법:
  1) fill_missing_frames      : 검출 실패 프레임 복원 (타임라인 유지 + 선형 보간)
  2) hampel_filter            : 순간 튐(spike) 이상치 검출 → NaN
     speed_limit_filter       : 7 m/s 초과 이동(물리적으로 불가능) → NaN
  3) mask_low_visibility      : visibility 낮은 관절 → NaN (짧은 구간만)
  4) interpolate_nans         : NaN 구간 선형 보간
  5) butterworth_smooth       : 4차 Butterworth 저역통과 (zero-phase, filtfilt)
  6) normalize_bone_lengths   : 뼈 길이 고정 + 좌우 대칭화 (방향은 유지)
  7) align_to_ground          : 발 관절 6개 기준 지면 정렬 (clamp 없음)

좌표계: p1 과 동일 (y-up, 단위 m, 골반 중심 원점의 MediaPipe world landmark)
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt

# ------------------------------------------------------------------
# BlazePose 관절 정의
# ------------------------------------------------------------------
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
FOOT_JOINTS = (27, 28, 29, 30, 31, 32)          # ankle, heel, foot_index (L/R)
HEAD_JOINTS = tuple(range(0, 11))

# (parent, child) — 몸통에서 바깥쪽 순서 (위상 정렬)
LIMB_BONES = [
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),   # left arm + hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),   # right arm + hand
    (23, 25), (25, 27), (27, 29), (27, 31),             # left leg + foot
    (24, 26), (26, 28), (28, 30), (28, 32),             # right leg + foot
]
# 좌우 대칭 뼈 쌍
SYMMETRIC_PAIRS = [
    ((11, 13), (12, 14)), ((13, 15), (14, 16)),
    ((15, 17), (16, 18)), ((15, 19), (16, 20)), ((15, 21), (16, 22)),
    ((23, 25), (24, 26)), ((25, 27), (26, 28)),
    ((27, 29), (28, 30)), ((27, 31), (28, 32)),
]
# 평가에 쓰는 주요 팔다리 뼈 8개
MAIN_BONES = {
    "L upper arm": (11, 13), "L forearm": (13, 15),
    "R upper arm": (12, 14), "R forearm": (14, 16),
    "L thigh": (23, 25), "L shank": (25, 27),
    "R thigh": (24, 26), "R shank": (26, 28),
}


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def load_raw_with_meta(npz_path):
    """p1 의 npz 를 읽어 (frames, seq(T,33,3), vis(T,33)) 반환.
    제공 코드의 load_raw_keypoints 와 달리 frame 번호/visibility 를 보존한다."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame", "landmark", "x", "y", "z", "visibility"]).astype(float)
    frames, seq, vis = [], [], []
    for f, g in df.groupby("frame"):
        g = g.sort_values("landmark")
        frames.append(int(f))
        seq.append(g[["x", "y", "z"]].values)
        vis.append(g["visibility"].values)
    return np.array(frames), np.array(seq), np.array(vis)


# ------------------------------------------------------------------
# 1) Missing frames
# ------------------------------------------------------------------
def fill_missing_frames(frames, seq, vis):
    """검출 실패로 빠진 프레임을 타임라인에 다시 넣고 선형 보간.
    (제공 코드는 빠진 프레임을 그냥 건너뛰어 동작이 빨라지는 문제가 있음)"""
    full = np.arange(frames.min(), frames.max() + 1)
    missing = np.setdiff1d(full, frames)
    if len(missing) == 0:
        return full, seq.copy(), vis.copy(), missing
    T, J, C = len(full), seq.shape[1], seq.shape[2]
    out = np.empty((T, J, C))
    for j in range(J):
        for c in range(C):
            out[:, j, c] = np.interp(full, frames, seq[:, j, c])
    vis_out = np.zeros((T, J))
    vis_out[np.isin(full, frames)] = vis
    return full, out, vis_out, missing


# ------------------------------------------------------------------
# 2) Spike removal — Hampel filter
# ------------------------------------------------------------------
def hampel_filter(seq, half_window=3, n_sigma=3.0, min_mad=0.005):
    """각 관절 좌표의 이동 중앙값에서 n_sigma*MAD 이상 벗어난 값 → NaN.
    min_mad(m): 정지 구간에서 미세 잡음까지 이상치로 잡지 않도록 하는 하한."""
    T = seq.shape[0]
    w = 2 * half_window + 1
    pad = np.pad(seq, ((half_window, half_window), (0, 0), (0, 0)), mode="edge")
    win = sliding_window_view(pad, w, axis=0)            # (T,33,3,w)
    med = np.median(win, axis=-1)
    mad = 1.4826 * np.median(np.abs(win - med[..., None]), axis=-1)
    mad = np.maximum(mad, min_mad)
    outlier = np.abs(seq - med) > n_sigma * mad
    # 한 좌표라도 튀면 그 관절 전체(x,y,z)를 이상치로 처리
    joint_outlier = outlier.any(axis=2)
    out = seq.copy()
    out[joint_outlier] = np.nan
    assert out.shape[0] == T
    return out, joint_outlier


def root_relative(seq):
    """골반 중심(23,24 중점)을 원점으로 옮긴 좌표 — 전역 이동과 분리된 '자세'만 본다."""
    return seq - ((seq[:, L_HIP] + seq[:, R_HIP]) / 2)[:, None]


def speed_limit_filter(seq, fallback, fps=30.0, vmax=7.0, max_gap=10, iters=3):
    """물리적으로 불가능한 속도(vmax m/s 초과)로 이동한 관절-프레임 → NaN → 보간.
    빠른 팔 동작 구간에서는 MAD 가 커져 Hampel 이 놓치는 튐을 잡기 위한 보완책."""
    x = seq.copy()
    flagged = np.zeros(seq.shape[:2], dtype=bool)
    for _ in range(iters):
        sp = np.linalg.norm(np.diff(root_relative(x), axis=0), axis=2) * fps
        bad = np.zeros(seq.shape[:2], dtype=bool)
        bad[1:] = sp > vmax
        if not bad.any():
            break
        flagged |= bad
        x[bad] = np.nan
        x = interpolate_nans(x, fallback, max_gap=max_gap)
    return x, flagged


# ------------------------------------------------------------------
# 3) Low-visibility masking
# ------------------------------------------------------------------
def mask_low_visibility(seq, vis, threshold=0.3):
    out = seq.copy()
    low = vis < threshold
    out[low] = np.nan
    return out, low


# ------------------------------------------------------------------
# 4) NaN interpolation (짧은 구간만)
# ------------------------------------------------------------------
def interpolate_nans(seq, fallback, max_gap=10):
    """NaN 구간을 선형 보간. max_gap 프레임보다 긴 구간은 보간이 부정확하므로
    fallback(원본) 값으로 되돌린다."""
    out = seq.copy()
    T = seq.shape[0]
    t = np.arange(T)
    for j in range(seq.shape[1]):
        nan = np.isnan(out[:, j, 0])
        if not nan.any():
            continue
        if nan.all():
            out[:, j] = fallback[:, j]
            continue
        for c in range(3):
            out[nan, j, c] = np.interp(t[nan], t[~nan], out[~nan, j, c])
        # 긴 구간은 원본 복원
        idx = np.flatnonzero(nan)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for r in runs:
            if len(r) > max_gap:
                out[r, j] = fallback[r, j]
    return out


# ------------------------------------------------------------------
# 5) Temporal smoothing
# ------------------------------------------------------------------
def butterworth_smooth(seq, fps=30.0, cutoff=6.0, order=4):
    """Zero-phase Butterworth low-pass. 인체 동작 분석에서 널리 쓰는 6Hz 기준.
    filtfilt 로 앞뒤 두 번 걸러 위상 지연(동작이 늦게 따라오는 현상)이 없다."""
    b, a = butter(order, cutoff / (fps / 2.0), btype="low")
    return filtfilt(b, a, seq, axis=0, padtype="odd", padlen=min(3 * max(len(a), len(b)), seq.shape[0] - 1))


def moving_average(seq, window=5):
    """Unity BlazePoseDataFile.cs 에 들어있는 5프레임 이동평균과 같은 방식 (비교용)."""
    k = window // 2
    pad = np.pad(seq, ((k, k), (0, 0), (0, 0)), mode="edge")
    return sliding_window_view(pad, window, axis=0).mean(axis=-1)


# ------------------------------------------------------------------
# 6) Bone-length normalization
# ------------------------------------------------------------------
def _unit(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def _bone_len(seq, a, b):
    return np.linalg.norm(seq[:, a] - seq[:, b], axis=1)


HAND_JOINTS = {15: (17, 19, 21), 16: (18, 20, 22)}     # wrist → pinky/index/thumb


def normalize_bone_lengths(seq, symmetric=True, dir_cutoff=4.0, fps=30.0):
    """모든 프레임에서 뼈 길이를 시퀀스 중앙값으로 고정(+좌우 평균으로 대칭화).
    각 뼈의 '방향'만 입력에서 가져오고 길이는 고정해 부모→자식 순서로 다시 쌓는다.

    - 팔이 카메라를 향할 때(원근 단축) 뼈가 짧아지며 방향이 불안정해지므로,
      방향 벡터를 dir_cutoff Hz 로 한 번 더 저역통과한 뒤 사용한다.
    - 손가락(17~22)은 추정이 부정확해 길이 고정 대신 손목에 강체로 붙여 이동."""
    src = seq
    out = seq.copy()

    # --- target lengths ---
    target = {bone: np.median(_bone_len(src, *bone)) for bone in LIMB_BONES}
    hip_w = np.median(_bone_len(src, L_HIP, R_HIP))
    sh_w = np.median(_bone_len(src, L_SHOULDER, R_SHOULDER))
    mid_hip_src = (src[:, L_HIP] + src[:, R_HIP]) / 2
    mid_sh_src = (src[:, L_SHOULDER] + src[:, R_SHOULDER]) / 2
    spine = np.median(np.linalg.norm(mid_sh_src - mid_hip_src, axis=1))
    if symmetric:
        for lb, rb in SYMMETRIC_PAIRS:
            m = (target[lb] + target[rb]) / 2
            target[lb] = target[rb] = m

    # --- torso (root = mid-hip, 그대로 유지) ---
    hip_dir = _unit(src[:, L_HIP] - src[:, R_HIP])
    out[:, L_HIP] = mid_hip_src + hip_dir * hip_w / 2
    out[:, R_HIP] = mid_hip_src - hip_dir * hip_w / 2
    mid_sh_new = mid_hip_src + _unit(mid_sh_src - mid_hip_src) * spine
    sh_dir = _unit(src[:, L_SHOULDER] - src[:, R_SHOULDER])
    out[:, L_SHOULDER] = mid_sh_new + sh_dir * sh_w / 2
    out[:, R_SHOULDER] = mid_sh_new - sh_dir * sh_w / 2
    # 머리는 어깨 중심과 같이 이동
    shift = (mid_sh_new - mid_sh_src)[:, None, :]
    out[:, list(HEAD_JOINTS)] = src[:, list(HEAD_JOINTS)] + shift

    # --- limbs (부모 → 자식) ---
    finger_bones = {(w, f) for w, fs in HAND_JOINTS.items() for f in fs}
    for p, c in LIMB_BONES:
        if (p, c) in finger_bones:
            continue
        direction = _unit(src[:, c] - src[:, p])
        if dir_cutoff:
            direction = _unit(butterworth_smooth(direction[:, None, :], fps=fps, cutoff=dir_cutoff)[:, 0])
        out[:, c] = out[:, p] + direction * target[(p, c)]

    # --- hands: 손목에 강체로 부착 ---
    for w, fingers in HAND_JOINTS.items():
        for f in fingers:
            out[:, f] = out[:, w] + (src[:, f] - src[:, w])
    return out


# ------------------------------------------------------------------
# 7) Ground alignment
# ------------------------------------------------------------------
def align_to_ground(seq, foot_joints=FOOT_JOINTS):
    """발목·뒤꿈치·발끝 6개 관절 중 가장 낮은 점을 y=0 에 맞춘다.
    제공 방법은 발끝(31,32)만 보고 나머지를 clamp(0) 하므로 뒤꿈치가 바닥에 눌려
    발 모양이 찌그러지는데, 여기서는 모든 발 관절을 기준으로 삼아 clamp 가 필요 없다."""
    out = seq.copy()
    ground = out[:, list(foot_joints), 1].min(axis=1)
    out[:, :, 1] -= ground[:, None]
    return out, ground


# ------------------------------------------------------------------
# Full bonus pipeline
# ------------------------------------------------------------------
def refine_motion(frames, seq, vis, fps=30.0, cutoff=6.0, vis_threshold=0.3,
                  hampel_half_window=3, hampel_sigma=3.0, vmax=7.0, max_gap=10,
                  symmetric=True, return_stages=False):
    """Bonus 후처리 전체 파이프라인.
    결측 프레임 복원 → Hampel → 속도 제한 → 저신뢰 관절 보간 → Butterworth
    → 뼈 길이 고정/대칭화 → 지면 정렬"""
    log, stages = {}, {}
    full, x, v, missing = fill_missing_frames(frames, seq, vis)
    log["missing_frames"] = int(len(missing))
    base = x.copy()

    # (1) 이상치 제거
    x, spikes = hampel_filter(x, hampel_half_window, hampel_sigma)
    log["hampel_joint_frames"] = int(spikes.sum())
    x = interpolate_nans(x, fallback=base, max_gap=max_gap)
    x, fast = speed_limit_filter(x, base, fps=fps, vmax=vmax, max_gap=max_gap)
    log["speed_limit_joint_frames"] = int(fast.sum())
    x_low, low = mask_low_visibility(x, v, vis_threshold)
    log["low_vis_joint_frames"] = int(low.sum())
    x = interpolate_nans(x_low, fallback=base, max_gap=max_gap)
    stages["outlier_removed"] = x.copy()

    # (2) 시간축 스무딩
    x = butterworth_smooth(x, fps=fps, cutoff=cutoff)
    stages["smoothed"] = x.copy()

    # (3) 뼈 길이 일관성
    x = normalize_bone_lengths(x, symmetric=symmetric)
    stages["bone_fixed"] = x.copy()

    # (4) 지면 정렬
    x, ground = align_to_ground(x)
    stages["grounded"] = x.copy()
    if return_stages:
        return full, x, log, stages
    return full, x, log


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(seq, fps=30.0, speed_limit=7.0):
    """seq: (T,33,3)  →  dict of quality metrics.
    떨림/속도는 골반 기준 상대 좌표로 계산 (지면 정렬로 인한 몸 전체 상하 이동은
    root_acc_y 로 따로 평가)."""
    rel = root_relative(seq)
    vel = np.diff(rel, axis=0) * fps
    acc = np.diff(rel, n=2, axis=0) * fps * fps
    speed = np.linalg.norm(vel, axis=2)
    accm = np.linalg.norm(acc, axis=2)

    bone_cv = []
    for (a, b) in MAIN_BONES.values():
        L = _bone_len(seq, a, b)
        bone_cv.append(L.std() / L.mean() * 100)
    asym = []
    for lb, rb in SYMMETRIC_PAIRS[:2] + SYMMETRIC_PAIRS[5:7]:
        l, r = _bone_len(seq, *lb).mean(), _bone_len(seq, *rb).mean()
        asym.append(abs(l - r) / ((l + r) / 2) * 100)

    foot_y = seq[:, list(FOOT_JOINTS), 1]
    lowest = foot_y.min(axis=1)
    mid_hip = (seq[:, L_HIP] + seq[:, R_HIP]) / 2
    root_acc_y = np.abs(np.diff(mid_hip[:, 1], n=2)) * fps * fps

    return {
        "mean_acc": float(accm.mean()),                       # m/s^2  (jitter)
        "wrist_acc": float(accm[:, [15, 16]].mean()),
        "spike_pct": float((speed > speed_limit).mean() * 100),  # % joint-frames
        "max_speed": float(speed.max()),
        "bone_cv": float(np.mean(bone_cv)),                    # %
        "lr_asym": float(np.mean(asym)),                       # %
        "lowest_foot_mean": float(lowest.mean()),              # m (0 = on ground)
        "lowest_foot_std": float(lowest.std()),
        "below_ground_pct": float((foot_y < -0.01).any(axis=1).mean() * 100),
        "root_acc_y": float(root_acc_y.mean()),
    }


def mpjpe(a, b):
    """두 시퀀스의 평균 관절 위치 차이 (m). 지면 정렬로 생긴 수직 이동은 제외하기 위해
    골반 중심을 원점으로 맞춘 뒤 비교."""
    ra = a - ((a[:, L_HIP] + a[:, R_HIP]) / 2)[:, None]
    rb = b - ((b[:, L_HIP] + b[:, R_HIP]) / 2)[:, None]
    return float(np.linalg.norm(ra - rb, axis=2).mean())
