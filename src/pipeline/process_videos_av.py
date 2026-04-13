#!/usr/bin/env python3
"""
Batch-process dataset videos with a fine-tuned EmoNet checkpoint (same path as real-time webcam).

For each video:
  - Read frames (OpenCV).
  - Face crop + resize 256×256 (default: **FaceAlignment** + 30% pad, same as AFEW-VA training;
    optional **SFD** + 20px for webcam-style demos), then convert to the chosen color space.
  - Split time into consecutive segments of --segment_frames (default 20).
  - Per segment: mean valence/arousal over frames with a successful face detection.
  - Per video/trial: combine segment (V,A) with ``--trial-aggregation`` (default: mean over
    segments; optional ``median``, ``last``, or learned ``gru`` via ``--aggregator-checkpoint``).

At ~30 FPS, a 60 s clip has ~1800 frames -> 90 segments of 20 frames (matches the sketch).

Subjective ratings: if your study used DEAP-style SAM ratings on 1–9 (neutral 5), use
`--subjective_scale deap_1_9_to_neg1_1` so comparisons use (rating - 5) / 4 in [-1, 1],
aligned with EmoNet valence/arousal. Raw values are still stored on each video record.

When enough trials are paired, ``comparison`` reports CCC, PCC (``pcc_*`` / ``pearson_*``),
RMSE, SAGR (``emonet.metrics`` definitions), plus MAE for convenience.

Requires the `emonet` package and dependencies from the full repo (face_alignment, etc.).
Set PYTHONPATH to the repo root that contains the `emonet/` folder, or run from that root.

Physio-Code dataset roots: ``.../Data/<participant_id>/P1/`` (full cohort; often **two** duplicate
``.mp4`` names for one session) or ``.../Data_ECG_GSR/<participant_id>/P1/`` (subset). Both use
``Video/*.mp4``, ``Ratings/Participant<id>.csv`` (``videoName``, ``arousal``, ``valence``,
``startTime``, ``endTime``), and ``CameraEvents/dbtest_data.csv``.

If there is **one** long session recording per participant (not one file per stimulus), use
``--timed_trials``: trial segments are cut with wall-clock times from the ratings CSV, aligned
to ``VIDEO_START`` in CameraEvents (same machine local time as the GUI timestamps).

Example::

  python process_videos_av.py --physio_session /media/mabl-main/Data/Physio-code/Data_ECG_GSR/15/P1 \\
    --timed_trials --checkpoint .../best_model.pth --color_space lab \\
    --subjective_scale deap_1_9_to_neg1_1 -o p15_trials.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import face_alignment
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from skimage import color as skcolor

# Resolve imports: repo root must contain `emonet/`
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _import_emonet():
    try:
        from emonet.models import EmoNet

        return EmoNet
    except ImportError:
        pass
    alt = os.environ.get("EMONET_REPO")
    if alt:
        p = Path(alt).resolve()
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        from emonet.models import EmoNet

        return EmoNet
    print(
        "Could not import emonet. Clone the full emonet_feature_extraction repo and either:\n"
        "  export PYTHONPATH=/path/to/emonet_feature_extraction\n"
        "  or export EMONET_REPO=/path/to/emonet_feature_extraction",
        file=sys.stderr,
    )
    raise SystemExit(1)


EmoNet = _import_emonet()


def preprocess_face(face_rgb_uint8: np.ndarray, color_space: str) -> np.ndarray:
    """Face crop 256×256 RGB uint8 -> float32 H×W×3 in [0,1] in target space (matches training / webcam)."""
    cs = color_space.lower()
    if cs == "rgb":
        return face_rgb_uint8.astype(np.float32) / 255.0

    if cs == "hsv":
        face_hsv = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2HSV)
        face_hsv = face_hsv.astype(np.float32)
        face_hsv[:, :, 0] = face_hsv[:, :, 0] * (255.0 / 179.0)
        return face_hsv / 255.0

    if cs == "ycbcr":
        ycc = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2YCrCb)
        return ycc.astype(np.float32) / 255.0

    if cs == "xyz":
        xyz = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2XYZ)
        return xyz.astype(np.float32) / 255.0

    if cs == "lab":
        rgb = face_rgb_uint8.astype(np.float32) / 255.0
        lab_np = skcolor.rgb2lab(rgb)
        out = lab_np.astype(np.float32)
        out[:, :, 0] = out[:, :, 0] / 100.0
        out[:, :, 1] = (out[:, :, 1] + 128.0) / 255.0
        out[:, :, 2] = (out[:, :, 2] + 128.0) / 255.0
        return out

    raise ValueError(f"Unknown color space: {color_space}")


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file():
        print(
            f"Checkpoint not found: {path}\n"
            "Pass the real path to your trained weights, e.g. "
            ".../output_lab/best_model.pth (must match --color_space).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        raw = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(str(path), map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def load_video_frames(video_path: Path) -> Tuple[List[np.ndarray], Optional[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    fps_out = fps if fps and fps > 1e-3 else None
    return frames, fps_out


def load_video_frames_range(
    video_path: Path, start_frame: int, end_frame: int
) -> Tuple[List[np.ndarray], Optional[float]]:
    """Load frames [start_frame, end_frame) without reading the whole file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame, int(end_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    frames: List[np.ndarray] = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    fps_out = fps if fps and fps > 1e-3 else None
    return frames, fps_out


def parse_camera_csv_timestamp(ts: str) -> datetime:
    ts = ts.strip()
    return datetime.fromisoformat(ts)


def parse_ratings_gui_time(s: str) -> datetime:
    """Parse GUI export like 'Mon Apr 21 2025 16:32:04 GMT-0400 (Eastern...)' as naive local time."""
    s = s.strip()
    if " GMT" in s:
        s = s.split(" GMT")[0].strip()
    return datetime.strptime(s, "%a %b %d %Y %H:%M:%S")


def _camera_event_matches_session_mp4(payload: str, session_mp4_name: str) -> bool:
    """CameraEvents may list ``20250421T163059Z.mp4`` while Video/ has ``15_20250421T163059Z.mp4``."""
    if "VIDEO_START" not in payload:
        return False
    parts = [p.strip() for p in payload.split(",")]
    if len(parts) < 2:
        return session_mp4_name in payload
    listed = parts[1]
    return (
        listed == session_mp4_name
        or session_mp4_name.endswith(listed)
        or listed in session_mp4_name
    )


def listed_video_name_from_camera_events(camera_events_csv: Path) -> Optional[str]:
    """Filename from VIDEO_START (e.g. ``20250421T163059Z.mp4``)."""
    with open(camera_events_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or "VIDEO_START" not in row[1]:
                continue
            parts = [p.strip() for p in row[1].split(",")]
            if len(parts) >= 2 and parts[1].lower().endswith(".mp4"):
                return parts[1]
    return None


def _disk_mp4_matches_listed(disk_name: str, listed: str) -> bool:
    return (
        disk_name == listed
        or disk_name.endswith(listed)
        or listed in disk_name
    )


def pick_session_mp4(video_dir: Path, cam_csv: Path) -> Path:
    """
    Choose which session recording to use. ``.../Data/<id>/P1/Video`` may contain two exports
    of the same take (e.g. ``20250421T163059Z.mp4`` and ``15_20250421T163059Z.mp4``); we use
    ``CameraEvents`` VIDEO_START and prefer an exact filename match, otherwise the largest file.
    """
    mp4s = sorted(video_dir.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 files in {video_dir}")

    listed: Optional[str] = None
    if cam_csv.is_file():
        listed = listed_video_name_from_camera_events(cam_csv)

    if listed:
        matches = [p for p in mp4s if _disk_mp4_matches_listed(p.name, listed)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            exact = [p for p in matches if p.name == listed]
            if len(exact) == 1:
                return exact[0]
            return max(matches, key=lambda p: p.stat().st_size)

    if len(mp4s) == 1:
        return mp4s[0]

    raise ValueError(
        f"Multiple .mp4 files in {video_dir} and no VIDEO_START in {cam_csv} to disambiguate: "
        f"{[p.name for p in mp4s]}"
    )


def video_start_time_from_camera_events(
    camera_events_csv: Path, session_mp4_name: str
) -> datetime:
    with open(camera_events_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            if _camera_event_matches_session_mp4(row[1], session_mp4_name):
                return parse_camera_csv_timestamp(row[0])
    raise FileNotFoundError(
        f"No VIDEO_START line matching {session_mp4_name} in {camera_events_csv}"
    )


def load_physio_timed_trials(
    ratings_csv: Path,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    with open(ratings_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vn = (row.get("videoName") or "").strip()
            if not vn:
                continue
            try:
                av = float(row["arousal"])
                vv = float(row["valence"])
            except (KeyError, ValueError):
                continue
            try:
                t0 = parse_ratings_gui_time(row["startTime"])
                t1 = parse_ratings_gui_time(row["endTime"])
            except (KeyError, ValueError):
                continue
            rows_out.append(
                {
                    "videoName": vn,
                    "arousal_subj": av,
                    "valence_subj": vv,
                    "start": t0,
                    "end": t1,
                }
            )
    return rows_out


def _fa_landmark_type():
    try:
        return face_alignment.LandmarksType.TWO_D
    except AttributeError:
        return face_alignment.LandmarksType._2D


def build_face_aligner(device: torch.device) -> "face_alignment.FaceAlignment":
    dev = "cuda" if device.type == "cuda" else "cpu"
    return face_alignment.FaceAlignment(
        _fa_landmark_type(), flip_input=False, device=dev
    )


def crop_face_fan(
    frame_rgb: np.ndarray,
    fa: "face_alignment.FaceAlignment",
    image_size: int,
    padding: float = 0.3,
) -> Optional[np.ndarray]:
    """Same geometry as ``AFEWVA._detect_and_crop_face`` / ``train_emonet_lab``."""
    try:
        landmarks = fa.get_landmarks(frame_rgb)
        if landmarks is None or len(landmarks) == 0:
            return None
        lm = landmarks[0]
        x_min, y_min = np.min(lm, axis=0).astype(int)
        x_max, y_max = np.max(lm, axis=0).astype(int)
        w_box = int(x_max - x_min)
        h_box = int(y_max - y_min)
        x_min = max(0, int(x_min - padding * w_box))
        y_min = max(0, int(y_min - padding * h_box))
        x_max = min(frame_rgb.shape[1], int(x_max + padding * w_box))
        y_max = min(frame_rgb.shape[0], int(y_max + padding * h_box))
        crop = frame_rgb[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (image_size, image_size))
    except Exception:
        return None


def crop_face_sfd(
    frame_rgb: np.ndarray,
    sfd: SFDDetector,
    image_size: int,
    pad_px: int = 20,
) -> Optional[np.ndarray]:
    with torch.no_grad():
        faces = sfd.detect_from_image(frame_rgb[:, :, ::-1])
    if len(faces) == 0:
        return None
    x1, y1, x2, y2 = [int(x) for x in faces[0][:4]]
    y1c, y2c = max(0, y1 - pad_px), min(frame_rgb.shape[0], y2 + pad_px)
    x1c, x2c = max(0, x1 - pad_px), min(frame_rgb.shape[1], x2 + pad_px)
    crop = frame_rgb[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (image_size, image_size))


def frame_valence_arousal(
    net: torch.nn.Module,
    frame_rgb: np.ndarray,
    color_space: str,
    image_size: int,
    device: torch.device,
    face_detector: str,
    fa: Optional["face_alignment.FaceAlignment"],
    sfd: Optional[SFDDetector],
    clamp_output: bool = True,
) -> Optional[Tuple[float, float]]:
    if face_detector == "fan":
        assert fa is not None
        resized = crop_face_fan(frame_rgb, fa, image_size)
    else:
        assert sfd is not None
        resized = crop_face_sfd(frame_rgb, sfd, image_size)
    if resized is None:
        return None
    proc = preprocess_face(resized, color_space)
    tensor = torch.from_numpy(proc).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(tensor)
    if isinstance(out, dict):
        v_t = out["valence"].reshape(-1)[0]
        a_t = out["arousal"].reshape(-1)[0]
    else:
        pred = out.reshape(out.shape[0], -1)
        v_t, a_t = pred[0, -2], pred[0, -1]
    if clamp_output:
        v = float(v_t.clamp(-1.0, 1.0).cpu().item())
        a = float(a_t.clamp(-1.0, 1.0).cpu().item())
    else:
        v = float(v_t.cpu().item())
        a = float(a_t.cpu().item())
    return v, a


def _process_rgb_frames(
    frames: List[np.ndarray],
    fps_meta: Optional[float],
    net: torch.nn.Module,
    color_space: str,
    segment_frames: int,
    image_size: int,
    device: torch.device,
    face_detector: str,
    fa: Optional["face_alignment.FaceAlignment"],
    sfd: Optional[SFDDetector],
    global_frame_offset: int = 0,
    clamp_output: bool = True,
    trial_aggregation: str = "mean",
    agg_module: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    from segment_aggregator import trial_va_from_segments

    n = len(frames)
    segment_va: List[Dict[str, Any]] = []
    seg_idx = 0
    for start in range(0, n, segment_frames):
        chunk = frames[start : start + segment_frames]
        vs: List[float] = []
        aa: List[float] = []
        for frame in chunk:
            va = frame_valence_arousal(
                net,
                frame,
                color_space,
                image_size,
                device,
                face_detector,
                fa,
                sfd,
                clamp_output=clamp_output,
            )
            if va is not None:
                vs.append(va[0])
                aa.append(va[1])
        g0 = global_frame_offset + start
        g1 = global_frame_offset + min(start + segment_frames, n)
        if vs:
            segment_va.append(
                {
                    "segment_index": seg_idx,
                    "frame_start": g0,
                    "frame_end": g1,
                    "n_frames_with_face": len(vs),
                    "valence_mean": float(np.mean(vs)),
                    "arousal_mean": float(np.mean(aa)),
                }
            )
        else:
            segment_va.append(
                {
                    "segment_index": seg_idx,
                    "frame_start": g0,
                    "frame_end": g1,
                    "n_frames_with_face": 0,
                    "valence_mean": None,
                    "arousal_mean": None,
                }
            )
        seg_idx += 1

    valid = [s for s in segment_va if s["valence_mean"] is not None]
    v_a, a_a = trial_va_from_segments(
        valid,
        trial_aggregation,
        device,
        agg_module,
        clamp_output,
    )

    return {
        "n_frames": n,
        "fps_reported": fps_meta,
        "n_segments": len(segment_va),
        "segments": segment_va,
        "video_mean_valence": v_a,
        "video_mean_arousal": a_a,
        "trial_aggregation": trial_aggregation,
    }


def process_video(
    video_path: Path,
    net: torch.nn.Module,
    color_space: str,
    segment_frames: int,
    image_size: int,
    device: torch.device,
    face_detector: str,
    fa: Optional["face_alignment.FaceAlignment"],
    sfd: Optional[SFDDetector],
    clamp_output: bool = True,
    trial_aggregation: str = "mean",
    agg_module: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    frames, fps_meta = load_video_frames(video_path)
    inner = _process_rgb_frames(
        frames,
        fps_meta,
        net,
        color_space,
        segment_frames,
        image_size,
        device,
        face_detector,
        fa,
        sfd,
        global_frame_offset=0,
        clamp_output=clamp_output,
        trial_aggregation=trial_aggregation,
        agg_module=agg_module,
    )
    inner["video_path"] = str(video_path)
    return inner


def process_video_frame_range(
    video_path: Path,
    net: torch.nn.Module,
    color_space: str,
    segment_frames: int,
    image_size: int,
    device: torch.device,
    start_frame: int,
    end_frame: int,
    face_detector: str,
    fa: Optional["face_alignment.FaceAlignment"],
    sfd: Optional[SFDDetector],
    clamp_output: bool = True,
    trial_aggregation: str = "mean",
    agg_module: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    frames, fps_meta = load_video_frames_range(video_path, start_frame, end_frame)
    inner = _process_rgb_frames(
        frames,
        fps_meta,
        net,
        color_space,
        segment_frames,
        image_size,
        device,
        face_detector,
        fa,
        sfd,
        global_frame_offset=start_frame,
        clamp_output=clamp_output,
        trial_aggregation=trial_aggregation,
        agg_module=agg_module,
    )
    inner["video_path"] = str(video_path)
    inner["frame_range"] = [start_frame, end_frame]
    return inner


def load_subjective_csv(path: Path) -> Dict[str, Tuple[float, float]]:
    """Map video key -> (valence_subj, arousal_subj).

    Supports generic column names and Physio-Code ``Participant<id>.csv`` (``videoName``,
    ``valence``, ``arousal``). Multiple lookup keys are stored per row so ``.mp4`` basenames
    match ratings whether ``videoName`` is a full filename or a stem.
    """
    out: Dict[str, Tuple[float, float]] = {}

    def pick(row: Dict[str, str], *names: str) -> Optional[str]:
        lower = {k.lower().strip(): k for k in row}
        for n in names:
            if n.lower() in lower:
                return row[lower[n.lower()]]
        return None

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out

        for row in reader:
            key = pick(
                row,
                "videoname",
                "video",
                "file",
                "path",
                "id",
                "video_id",
            )
            if not key:
                continue
            vs = pick(row, "valence_subj", "valence", "v_p", "v_sub", "subjective_valence")
            ar = pick(row, "arousal_subj", "arousal", "a_p", "a_sub", "subjective_arousal")
            if vs is None or ar is None:
                continue
            val, aro = float(vs), float(ar)
            out[key.strip()] = (val, aro)
            out[Path(key).name] = (val, aro)
            out[Path(key).stem] = (val, aro)
    return out


def resolve_physio_session(session: Path) -> Tuple[Path, Path, str]:
    """
    Resolve Physio-Code session paths: ``.../Data/<participant_id>/P1`` with ``Video/`` and
    ``Ratings/Participant<participant_id>.csv``.

    ``session`` may be ``.../Data/<id>`` (if ``P1`` exists inside) or ``.../Data/<id>/P1``.
    """
    root = session.resolve()
    if root.name == "P1":
        p1 = root
    elif (root / "P1").is_dir():
        p1 = root / "P1"
    else:
        raise ValueError(
            f"Expected a path like .../Data/<participant_id>/P1 or .../Data/<participant_id> "
            f"containing P1/; got {session}"
        )

    participant_id = p1.parent.name
    video_dir = p1 / "Video"
    ratings_csv = p1 / "Ratings" / f"Participant{participant_id}.csv"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Physio-Code: missing Video directory: {video_dir}")
    if not ratings_csv.is_file():
        raise FileNotFoundError(f"Physio-Code: missing ratings CSV: {ratings_csv}")
    return video_dir, ratings_csv, participant_id


def session_to_p1(session: Path) -> Path:
    r = session.resolve()
    if r.name == "P1":
        return r
    if (r / "P1").is_dir():
        return r / "P1"
    raise ValueError(f"Not a Physio session path with P1/: {session}")


def run_physio_timed_trials(
    p1_dir: Path,
    ratings_csv: Path,
    participant_id: str,
    args: argparse.Namespace,
    device: torch.device,
    net: torch.nn.Module,
    face_detector: str,
    fa: Optional["face_alignment.FaceAlignment"],
    sfd: Optional[SFDDetector],
    agg_module: Optional[torch.nn.Module] = None,
) -> None:
    """One session take (possibly multiple .mp4 duplicates) + CameraEvents + Ratings times."""
    video_dir = p1_dir / "Video"
    cam_csv = p1_dir / "CameraEvents" / "dbtest_data.csv"
    if not cam_csv.is_file():
        print(f"Missing CameraEvents CSV: {cam_csv}", file=sys.stderr)
        raise SystemExit(1)

    try:
        session_mp4 = pick_session_mp4(video_dir, cam_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"--timed_trials: {e}", file=sys.stderr)
        raise SystemExit(1)

    all_mp4 = sorted(video_dir.glob("*.mp4"))
    if len(all_mp4) > 1:
        print(
            f"Using session file {session_mp4.name} (VIDEO_START / duplicate disambiguation); "
            f"also in folder: {[p.name for p in all_mp4 if p != session_mp4]}",
        )

    t_video0 = video_start_time_from_camera_events(cam_csv, session_mp4.name)
    trials = load_physio_timed_trials(ratings_csv)
    if not trials:
        print(f"No timed trial rows in {ratings_csv}", file=sys.stderr)
        raise SystemExit(1)

    cap = cv2.VideoCapture(str(session_mp4))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps < 1e-3:
        print("Could not read FPS from session MP4; time→frame mapping may be wrong.", file=sys.stderr)
        fps = 25.0

    print(
        f"Timed trials: participant {participant_id}, {session_mp4.name}, "
        f"clock0={t_video0}, fps={fps:.3f}, {len(trials)} rating rows"
    )

    trial_results: List[Dict[str, Any]] = []
    pred_v: List[float] = []
    pred_a: List[float] = []
    subj_v: List[float] = []
    subj_a: List[float] = []
    n_ratings_end_past_video = 0

    for tr in trials:
        sec_s = (tr["start"] - t_video0).total_seconds()
        sec_e = (tr["end"] - t_video0).total_seconds()
        f0 = int(sec_s * fps)
        f1 = int(sec_e * fps)
        if f1 > n_frames_total:
            n_ratings_end_past_video += 1
        f0 = max(0, min(f0, max(0, n_frames_total - 1)))
        f1 = max(0, min(f1, n_frames_total))
        if f1 <= f0:
            err_rec: Dict[str, Any] = {
                "videoName": tr["videoName"],
                "error": "bad_or_empty_frame_span",
                "frame_start": f0,
                "frame_end": f1,
                "subjective_valence": tr["valence_subj"],
                "subjective_arousal": tr["arousal_subj"],
                "subjective_matched": True,
            }
            if getattr(args, "subjective_scale", None) == "deap_1_9_to_neg1_1":
                vc, ac = scale_subjective_deap_to_model(tr["valence_subj"], tr["arousal_subj"])
                err_rec["subjective_valence_scaled"] = vc
                err_rec["subjective_arousal_scaled"] = ac
            trial_results.append(err_rec)
            continue

        print(f"  {tr['videoName'][:60]} ... frames [{f0}, {f1})")
        clamp_out = not getattr(args, "no_clamp_output", False)
        r = process_video_frame_range(
            session_mp4,
            net,
            args.color_space,
            args.segment_frames,
            args.image_size,
            device,
            f0,
            f1,
            face_detector,
            fa,
            sfd,
            clamp_output=clamp_out,
            trial_aggregation=args.trial_aggregation,
            agg_module=agg_module,
        )
        r["videoName"] = tr["videoName"]
        r["subjective_valence"] = tr["valence_subj"]
        r["subjective_arousal"] = tr["arousal_subj"]
        r["subjective_matched"] = True
        if args.subjective_scale == "deap_1_9_to_neg1_1":
            vc, ac = scale_subjective_deap_to_model(tr["valence_subj"], tr["arousal_subj"])
            r["subjective_valence_scaled"] = vc
            r["subjective_arousal_scaled"] = ac
            sv_use, sa_use = vc, ac
        else:
            sv_use, sa_use = tr["valence_subj"], tr["arousal_subj"]
        if not (np.isnan(r["video_mean_valence"]) or np.isnan(r["video_mean_arousal"])):
            pred_v.append(r["video_mean_valence"])
            pred_a.append(r["video_mean_arousal"])
            subj_v.append(sv_use)
            subj_a.append(sa_use)
        trial_results.append(r)

    if n_ratings_end_past_video > 0:
        print(
            f"WARNING: {n_ratings_end_past_video} trial(s) have rating end times past the last frame "
            f"({n_frames_total}); spans are clamped (session shorter than subjective log). "
            f"Run: python3 physio_trial_qc.py <this_json>",
            file=sys.stderr,
        )

    summary: Dict[str, Any] = {
        "timed_trials": True,
        "physio_participant_id": participant_id,
        "physio_data_note": "Aligned to CameraEvents VIDEO_START and Ratings startTime/endTime (naive local).",
        "session_mp4": str(session_mp4.resolve()),
        "video_clock_zero": t_video0.isoformat(),
        "fps_reported_session": fps,
        "n_frames_session": n_frames_total,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "color_space": args.color_space,
        "segment_frames": args.segment_frames,
        "subjective_scale": args.subjective_scale,
        "subjective_csv": str(ratings_csv.resolve()),
        "camera_events_csv": str(cam_csv.resolve()),
        "n_trials_csv": len(trials),
        "face_detector": face_detector,
        "clamp_output": not getattr(args, "no_clamp_output", False),
        "trial_aggregation": getattr(args, "trial_aggregation", "mean"),
        "aggregator_checkpoint": str(Path(args.aggregator_checkpoint).resolve())
        if getattr(args, "aggregator_checkpoint", None)
        else None,
        "trials": trial_results,
    }
    if n_ratings_end_past_video > 0:
        summary["session_coverage"] = {
            "n_trials_rating_end_after_last_frame": n_ratings_end_past_video,
            "note": "Ratings map past n_frames_session; trials clamped. See physio_trial_qc.py SESSION_* flags.",
        }

    if args.physio_session:
        summary["physio_session"] = str(Path(args.physio_session).resolve())

    if len(pred_v) >= 2:
        pv, pa = np.array(pred_v), np.array(pred_a)
        sv, sa = np.array(subj_v), np.array(subj_a)
        pcc_v, pcc_a = pearson_r(pv, sv), pearson_r(pa, sa)
        sagr_v, sagr_a = sagr_metric(sv, pv), sagr_metric(sa, pa)
        summary["comparison"] = {
            "n_paired": len(pred_v),
            "metrics_note": "CCC, PCC (pcc_* / pearson_*), RMSE, SAGR match AFEW-VA paper conventions; subjective scaled per --subjective_scale.",
            "mae_valence": mae(pv, sv),
            "mae_arousal": mae(pa, sa),
            "rmse_valence": rmse_metric(sv, pv),
            "rmse_arousal": rmse_metric(sa, pa),
            "pearson_valence": pcc_v,
            "pearson_arousal": pcc_a,
            "pcc_valence": pcc_v,
            "pcc_arousal": pcc_a,
            "ccc_valence": concordance_ccc(pv, sv),
            "ccc_arousal": concordance_ccc(pa, sa),
            "sagr_valence": sagr_v,
            "sagr_arousal": sagr_a,
            "sagr_overall": float((sagr_v + sagr_a) / 2.0),
        }
    else:
        summary["comparison"] = {
            "note": "Fewer than 2 trials with valid model outputs; expand spans or check face detection."
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def rmse_metric(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Root mean squared error (same definition as ``emonet.metrics.RMSE``)."""
    return float(np.sqrt(np.mean((ground_truth - predictions) ** 2)))


def sagr_metric(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Sign agreement rate (same definition as ``emonet.metrics.SAGR``)."""
    return float(np.mean(np.sign(ground_truth) == np.sign(predictions)))


def concordance_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin’s CCC (common for continuous affect)."""
    if len(x) < 2:
        return float("nan")
    xm, ym = float(np.mean(x)), float(np.mean(y))
    xv, yv = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - xm) * (y - ym)))
    denom = xv + yv + (xm - ym) ** 2
    if denom < 1e-12:
        return float("nan")
    return 2.0 * cov / denom


def scale_subjective_deap_to_model(v: float, a: float) -> Tuple[float, float]:
    """Map DEAP/SAM 1..9 to [-1, 1] (neutral 5 -> 0)."""
    return (v - 5.0) / 4.0, (a - 5.0) / 4.0


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="EmoNet batch video A/V (segmented) for comparison to subjective ratings"
    )
    parser.add_argument(
        "--physio_session",
        type=str,
        default=None,
        help=(
            "Physio-Code session folder: .../Data/<participant_id>/P1 (or parent containing P1/). "
            "Uses <session>/P1/Video for clips and <session>/P1/Ratings/Participant<id>.csv for "
            "subjective arousal/valence unless --subjective_csv is set explicitly."
        ),
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing video files (required if --physio_session not set)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth for the chosen color space",
    )
    parser.add_argument(
        "--color_space",
        type=str,
        default="rgb",
        choices=["rgb", "hsv", "ycbcr", "xyz", "lab"],
        help="Must match how the checkpoint was trained",
    )
    parser.add_argument(
        "--segment_frames",
        type=int,
        default=20,
        help="Frames per segment (e.g. 20 at ~30 FPS -> ~90 segments/minute)",
    )
    parser.add_argument(
        "--subjective_csv",
        type=str,
        default=None,
        help="Optional CSV with subjective valence/arousal per video for comparison",
    )
    parser.add_argument(
        "-o",
        "--output_json",
        type=str,
        default="video_av_results.json",
        help="Write full results here",
    )
    parser.add_argument(
        "--timed_trials",
        action="store_true",
        help=(
            "Physio-Code session with one continuous face MP4: segment by Ratings startTime/endTime "
            "using CameraEvents VIDEO_START as t=0. Requires --physio_session."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include videos in subdirectories of --video_dir",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".mp4,.avi,.mov,.mkv",
        help="Comma-separated list of file extensions to include",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--subjective_scale",
        type=str,
        default="none",
        choices=["none", "deap_1_9_to_neg1_1"],
        help="Scale subjective CSV ratings to [-1,1] for metrics (DEAP/SAM 1–9 in CVPR-style study)",
    )
    parser.add_argument(
        "--face_detector",
        type=str,
        default="fan",
        choices=["fan", "sfd"],
        help=(
            "fan=FaceAlignment landmarks + 30%% padding (matches AFEW-VA / train_emonet_*). "
            "sfd=SFD box + 20px pad (old webcam-style; often mismatches training)."
        ),
    )
    parser.add_argument(
        "--no_clamp_output",
        action="store_true",
        help="Do not clamp V/A to [-1,1]; use for debugging saturated predictions.",
    )
    parser.add_argument(
        "--trial-aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "last", "gru"],
        help=(
            "Combine per-segment valence/arousal into one trial/video summary. "
            "'gru' uses GRUSegmentAggregator weights from --aggregator-checkpoint."
        ),
    )
    parser.add_argument(
        "--aggregator-checkpoint",
        type=str,
        default=None,
        help="Path to .pth state_dict for GRUSegmentAggregator (required if --trial-aggregation gru).",
    )
    parser.add_argument(
        "--aggregator-hidden",
        type=int,
        default=32,
        help="GRU hidden size (must match trained aggregator checkpoint).",
    )
    parser.add_argument(
        "--aggregator-layers",
        type=int,
        default=1,
        help="GRU num_layers (must match trained aggregator checkpoint).",
    )
    args = parser.parse_args()

    physio_participant_id: Optional[str] = None
    if args.physio_session:
        vdir, default_ratings, physio_participant_id = resolve_physio_session(
            Path(args.physio_session)
        )
        args.video_dir = str(vdir)
        if args.subjective_csv is None:
            args.subjective_csv = str(default_ratings)
    elif not args.video_dir:
        print(
            "Provide --physio_session (Physio-Code layout) or --video_dir.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    print(f"Device: {device}, color_space={args.color_space}, checkpoint={args.checkpoint}")
    net = EmoNet(n_expression=8).to(device)
    load_checkpoint(net, Path(args.checkpoint).resolve(), device)
    net.eval()

    agg_module: Optional[torch.nn.Module] = None
    if args.trial_aggregation == "gru":
        if not args.aggregator_checkpoint:
            print("--aggregator-checkpoint is required when --trial-aggregation gru", file=sys.stderr)
            raise SystemExit(2)
        from segment_aggregator import load_gru_aggregator

        agg_module = load_gru_aggregator(
            Path(args.aggregator_checkpoint).resolve(),
            device,
            hidden_size=int(args.aggregator_hidden),
            num_layers=int(args.aggregator_layers),
        )
        print(f"Segment aggregator (GRU): {args.aggregator_checkpoint}")

    fa: Optional[Any] = None
    sfd_det: Optional[SFDDetector] = None
    if args.face_detector == "fan":
        print("Face crop: FaceAlignment + 30% pad (matches AFEW-VA training).")
        fa = build_face_aligner(device)
    else:
        print("Face crop: SFD + 20px pad (legacy; may not match AFEW-VA training).")
        sfd_det = SFDDetector(device=str(device))

    if args.timed_trials:
        if not args.physio_session:
            print("--timed_trials requires --physio_session", file=sys.stderr)
            raise SystemExit(2)
        if not args.subjective_csv:
            print("--timed_trials needs a ratings CSV (set --physio_session to auto-fill).", file=sys.stderr)
            raise SystemExit(2)
        p1 = session_to_p1(Path(args.physio_session))
        run_physio_timed_trials(
            p1,
            Path(args.subjective_csv),
            p1.parent.name,
            args,
            device,
            net,
            args.face_detector,
            fa,
            sfd_det,
            agg_module=agg_module,
        )
        return

    video_dir = Path(args.video_dir).resolve()
    exts = {e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower() for e in args.extensions.split(",")}

    videos: List[Path] = []
    if args.recursive:
        for p in sorted(video_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                videos.append(p)
    else:
        for p in sorted(video_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                videos.append(p)

    if not videos:
        print(f"No videos found under {video_dir}", file=sys.stderr)
        raise SystemExit(1)

    subjective: Dict[str, Tuple[float, float]] = {}
    if args.subjective_csv:
        subjective = load_subjective_csv(Path(args.subjective_csv))
        print(f"Loaded {len(subjective)} subjective rows from CSV")

    results: List[Dict[str, Any]] = []
    pred_v: List[float] = []
    pred_a: List[float] = []
    subj_v: List[float] = []
    subj_a: List[float] = []

    for vp in videos:
        print(f"Processing {vp.name} ...")
        r = process_video(
            vp,
            net,
            args.color_space,
            args.segment_frames,
            args.image_size,
            device,
            args.face_detector,
            fa,
            sfd_det,
            clamp_output=not args.no_clamp_output,
            trial_aggregation=args.trial_aggregation,
            agg_module=agg_module,
        )
        results.append(r)

        key = vp.name
        if key not in subjective and vp.stem in subjective:
            key = vp.stem
        r["subjective_matched"] = bool(subjective) and key in subjective
        if key in subjective:
            vp_sub, ap_sub = subjective[key]
            r["subjective_valence"] = vp_sub
            r["subjective_arousal"] = ap_sub
            if args.subjective_scale == "deap_1_9_to_neg1_1":
                vc, ac = scale_subjective_deap_to_model(vp_sub, ap_sub)
                r["subjective_valence_scaled"] = vc
                r["subjective_arousal_scaled"] = ac
                subj_use_v, subj_use_a = vc, ac
            else:
                subj_use_v, subj_use_a = vp_sub, ap_sub
            if not (np.isnan(r["video_mean_valence"]) or np.isnan(r["video_mean_arousal"])):
                pred_v.append(r["video_mean_valence"])
                pred_a.append(r["video_mean_arousal"])
                subj_v.append(subj_use_v)
                subj_a.append(subj_use_a)

    summary: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "color_space": args.color_space,
        "segment_frames": args.segment_frames,
        "subjective_scale": args.subjective_scale,
        "face_detector": args.face_detector,
        "clamp_output": not args.no_clamp_output,
        "trial_aggregation": args.trial_aggregation,
        "aggregator_checkpoint": str(Path(args.aggregator_checkpoint).resolve())
        if args.aggregator_checkpoint
        else None,
        "n_videos": len(results),
        "videos": results,
    }
    if args.physio_session:
        summary["physio_session"] = str(Path(args.physio_session).resolve())
        summary["physio_participant_id"] = physio_participant_id
    if args.subjective_csv:
        summary["subjective_csv"] = str(Path(args.subjective_csv).resolve())

    unmatched = [r["video_path"] for r in results if not r.get("subjective_matched", False)]
    if subjective:
        summary["pairing"] = {
            "n_videos_processed": len(results),
            "n_with_subjective_row": len(results) - len(unmatched),
            "n_unmatched": len(unmatched),
            "unmatched_video_paths": unmatched,
        }

    if len(pred_v) >= 2:
        pv, pa = np.array(pred_v), np.array(pred_a)
        sv, sa = np.array(subj_v), np.array(subj_a)
        pcc_v, pcc_a = pearson_r(pv, sv), pearson_r(pa, sa)
        sagr_v, sagr_a = sagr_metric(sv, pv), sagr_metric(sa, pa)
        summary["comparison"] = {
            "n_paired": len(pred_v),
            "metrics_note": "CCC, PCC (pcc_* / pearson_*), RMSE, SAGR match AFEW-VA paper conventions; subjective scaled per --subjective_scale.",
            "mae_valence": mae(pv, sv),
            "mae_arousal": mae(pa, sa),
            "rmse_valence": rmse_metric(sv, pv),
            "rmse_arousal": rmse_metric(sa, pa),
            "pearson_valence": pcc_v,
            "pearson_arousal": pcc_a,
            "pcc_valence": pcc_v,
            "pcc_arousal": pcc_a,
            "ccc_valence": concordance_ccc(pv, sv),
            "ccc_arousal": concordance_ccc(pa, sa),
            "sagr_valence": sagr_v,
            "sagr_arousal": sagr_a,
            "sagr_overall": float((sagr_v + sagr_a) / 2.0),
        }
    elif subjective:
        summary["comparison"] = {
            "note": "Not enough paired clips with both model output and subjective row (need ≥2). Check pairing.unmatched_video_paths and CSV videoName vs .mp4 basenames."
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run_cli()
