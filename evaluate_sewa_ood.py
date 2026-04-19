#!/usr/bin/env python3
"""
Out-of-distribution evaluation on SEWA-style frame folders (e.g. HRI-AVC/SEWA).

Walks ``<root>/<split>/<clip_id>/`` with ``*.png`` and ``Arousal_Valence.json`` (splits 01/02).
Runs the same face crop + color space + EmoNet forward as ``process_videos_av.py``, then reports
frame-level agreement with JSON ``valence`` / ``arousal`` (CCC, PCC, RMSE, MAE, SAGR).

Split 04 typically has frames but no labels on-disk; it is skipped unless you add JSON.

Example::

  python3 evaluate_sewa_ood.py \\
    --sewa-root /media/mabl-main/Data-Karthik/HRI-AVC/SEWA \\
    --checkpoint /path/to/best_model_lab.pth --color_space lab \\
    -o sewa_ood_lab.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from face_alignment.detection.sfd.sfd_detector import SFDDetector

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def parse_label_scale(spec: str) -> float:
    """
    Parse SEWA label scale specification.

    - ``auto``: use 1000.0 (expected for labels in [-1000, 1000])
    - numeric string: explicit divisor (e.g., ``1`` or ``1000``)
    """
    s = str(spec).strip().lower()
    if s == "auto":
        return 1000.0
    try:
        v = float(s)
    except ValueError as e:
        raise ValueError(f"Invalid --label-scale value {spec!r}; use 'auto' or numeric divisor.") from e
    if v <= 0:
        raise ValueError(f"--label-scale must be > 0, got {v}")
    return v


def load_sewa_annotations(
    json_path: Path, label_scale_divisor: float, clip_labels: bool
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Optional[np.ndarray]]]:
    """Return per-frame (valence, arousal) and optional landmark array (N,2) for cropping."""
    with open(json_path) as f:
        raw = json.load(f)
    frames = raw.get("frames") or {}
    out: Dict[int, Tuple[float, float]] = {}
    lms: Dict[int, Optional[np.ndarray]] = {}
    for k, rec in frames.items():
        if not isinstance(rec, dict):
            continue
        try:
            idx = int(str(k))
        except ValueError:
            continue
        v = float(rec["valence"]) / label_scale_divisor
        a = float(rec["arousal"]) / label_scale_divisor
        if clip_labels:
            v = float(np.clip(v, -1.0, 1.0))
            a = float(np.clip(a, -1.0, 1.0))
        out[idx] = (v, a)
        lm_raw = rec.get("landmarks")
        lm_arr: Optional[np.ndarray]
        if isinstance(lm_raw, list) and len(lm_raw) >= 2:
            arr = np.asarray(lm_raw, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 2 and np.isfinite(arr).all():
                lm_arr = arr
            else:
                lm_arr = None
        else:
            lm_arr = None
        lms[idx] = lm_arr
    return out, lms


def load_sewa_frame_labels(
    json_path: Path, label_scale_divisor: float, clip_labels: bool
) -> Dict[int, Tuple[float, float]]:
    labels, _ = load_sewa_annotations(json_path, label_scale_divisor, clip_labels)
    return labels


def list_png_frames(clip_dir: Path) -> List[Path]:
    paths = [p for p in clip_dir.iterdir() if p.suffix.lower() == ".png" and p.is_file()]
    paths.sort(key=lambda p: int(p.stem))
    return paths


def crop_face_from_landmarks(
    frame_rgb: np.ndarray,
    landmarks: np.ndarray,
    image_size: int,
    padding: float = 0.3,
) -> Optional[np.ndarray]:
    """BBox from landmarks + relative pad; same style as FAN crop in process_videos_av."""
    if landmarks.shape[0] < 2:
        return None
    try:
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        w_box = float(x_max - x_min)
        h_box = float(y_max - y_min)
        if w_box < 1.0 or h_box < 1.0:
            return None
        x_min_i = max(0, int(x_min - padding * w_box))
        y_min_i = max(0, int(y_min - padding * h_box))
        x_max_i = min(frame_rgb.shape[1], int(x_max + padding * w_box))
        y_max_i = min(frame_rgb.shape[0], int(y_max + padding * h_box))
        crop = frame_rgb[y_min_i:y_max_i, x_min_i:x_max_i]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (image_size, image_size))
    except Exception:
        return None


def moving_average_1d(x: List[float], k: int) -> List[float]:
    if k <= 1 or len(x) < k:
        return list(x)
    arr = np.asarray(x, dtype=np.float64)
    ker = np.ones(k, dtype=np.float64) / float(k)
    out = np.convolve(arr, ker, mode="same")
    return [float(v) for v in out]


def infer_va_on_face_rgb(
    pva: Any,
    net: torch.nn.Module,
    face_rgb_uint8: np.ndarray,
    color_space: str,
    image_size: int,
    device: torch.device,
    clamp_output: bool,
) -> Optional[Tuple[float, float]]:
    """Run EmoNet on an RGB face crop (uint8), already W=H=image_size preferred."""
    if face_rgb_uint8 is None or face_rgb_uint8.size == 0:
        return None
    if (
        face_rgb_uint8.shape[0] != image_size
        or face_rgb_uint8.shape[1] != image_size
    ):
        face_rgb_uint8 = cv2.resize(face_rgb_uint8, (image_size, image_size))
    proc = pva.preprocess_face(face_rgb_uint8, color_space)
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


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; returns NaN if undefined (no variance), without numpy corrcoef divide warnings."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else float("nan")


def eval_one_clip(
    pva: Any,
    clip_dir: Path,
    labels: Dict[int, Tuple[float, float]],
    landmarks: Optional[Dict[int, Optional[np.ndarray]]],
    crop_source: str,
    landmark_padding: float,
    smooth_k: int,
    net: torch.nn.Module,
    color_space: str,
    image_size: int,
    device: torch.device,
    face_detector: str,
    fa: Optional[Any],
    sfd: Optional[SFDDetector],
    clamp_output: bool,
    max_frames: Optional[int],
) -> Tuple[Dict[str, Any], List[float], List[float], List[float], List[float]]:
    paths = list_png_frames(clip_dir)
    if max_frames is not None:
        paths = paths[: max_frames]

    pred_v: List[float] = []
    pred_a: List[float] = []
    gt_v: List[float] = []
    gt_a: List[float] = []
    n_skip_no_label = 0
    n_skip_no_face = 0
    n_skip_no_landmarks = 0
    n_imread_fail = 0

    for p in paths:
        idx = int(p.stem)
        if idx not in labels:
            n_skip_no_label += 1
            continue
        gv, ga = labels[idx]
        img = cv2.imread(str(p))
        if img is None:
            n_imread_fail += 1
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        va: Optional[Tuple[float, float]] = None
        if crop_source == "landmarks":
            lm = landmarks.get(idx) if landmarks is not None else None
            if lm is None:
                n_skip_no_landmarks += 1
                continue
            face = crop_face_from_landmarks(rgb, lm, image_size, landmark_padding)
            if face is None:
                n_skip_no_landmarks += 1
                continue
            va = infer_va_on_face_rgb(
                pva, net, face, color_space, image_size, device, clamp_output
            )
        else:
            va = pva.frame_valence_arousal(
                net,
                rgb,
                color_space,
                image_size,
                device,
                face_detector,
                fa,
                sfd,
                clamp_output=clamp_output,
            )
        if va is None:
            n_skip_no_face += 1
            continue
        pv, pa = va
        pred_v.append(pv)
        pred_a.append(pa)
        gt_v.append(gv)
        gt_a.append(ga)

    if smooth_k > 1 and len(pred_v) >= smooth_k:
        pred_v = moving_average_1d(pred_v, smooth_k)
        pred_a = moving_average_1d(pred_a, smooth_k)

    pva_np = np.asarray(pred_v, dtype=np.float64)
    paa_np = np.asarray(pred_a, dtype=np.float64)
    gva_np = np.asarray(gt_v, dtype=np.float64)
    gaa_np = np.asarray(gt_a, dtype=np.float64)

    clip_metrics: Dict[str, Any] = {
        "n_frames_png": len(list_png_frames(clip_dir)),
        "n_evaluated_paths": len(paths),
        "n_paired": int(len(pred_v)),
        "n_skip_no_label": n_skip_no_label,
        "n_skip_no_face": n_skip_no_face,
        "n_skip_no_landmarks": n_skip_no_landmarks,
        "n_imread_fail": n_imread_fail,
        "crop_source": crop_source,
        "smooth_k": int(smooth_k),
    }
    if len(pred_v) >= 2:
        clip_metrics.update(
            {
                "ccc_valence": pva.concordance_ccc(pva_np, gva_np),
                "ccc_arousal": pva.concordance_ccc(paa_np, gaa_np),
                "pearson_valence": safe_pearson(pva_np, gva_np),
                "pearson_arousal": safe_pearson(paa_np, gaa_np),
                "rmse_valence": pva.rmse_metric(gva_np, pva_np),
                "rmse_arousal": pva.rmse_metric(gaa_np, paa_np),
                "mae_valence": pva.mae(pva_np, gva_np),
                "mae_arousal": pva.mae(paa_np, gaa_np),
                "sagr_valence": pva.sagr_metric(gva_np, pva_np),
                "sagr_arousal": pva.sagr_metric(gaa_np, paa_np),
            }
        )
    else:
        clip_metrics["note"] = "Fewer than 2 paired frames; CCC/PCC undefined."

    return clip_metrics, pred_v, pred_a, gt_v, gt_a


def aggregate_micro(
    pva: Any,
    all_pv: List[float],
    all_pa: List[float],
    all_gv: List[float],
    all_ga: List[float],
) -> Dict[str, float]:
    pv = np.asarray(all_pv, dtype=np.float64)
    pa = np.asarray(all_pa, dtype=np.float64)
    gv = np.asarray(all_gv, dtype=np.float64)
    ga = np.asarray(all_ga, dtype=np.float64)
    if len(pv) < 2:
        return {}
    return {
        "ccc_valence": pva.concordance_ccc(pv, gv),
        "ccc_arousal": pva.concordance_ccc(pa, ga),
        "pearson_valence": safe_pearson(pv, gv),
        "pearson_arousal": safe_pearson(pa, ga),
        "rmse_valence": pva.rmse_metric(gv, pv),
        "rmse_arousal": pva.rmse_metric(ga, pa),
        "mae_valence": pva.mae(pv, gv),
        "mae_arousal": pva.mae(pa, ga),
        "sagr_valence": pva.sagr_metric(gv, pv),
        "sagr_arousal": pva.sagr_metric(ga, pa),
        "n_paired_frames": float(len(pv)),
    }


def mean_clip_metrics(
    per_clip: List[Dict[str, Any]], keys: List[str]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        xs = [float(c[k]) for c in per_clip if k in c and np.isfinite(float(c[k]))]
        if xs:
            out[f"mean_clip_{k}"] = float(np.mean(xs))
            out[f"se_clip_{k}"] = (
                float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
            )
    out["n_clips_averaged"] = float(len(per_clip))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SEWA / HRI-AVC OOD frame-level V/A evaluation (same inference as process_videos_av)"
    )
    ap.add_argument(
        "--sewa-root",
        type=Path,
        default=Path("/media/mabl-main/Data-Karthik/HRI-AVC/SEWA"),
        help="Parent folder containing split subdirs (01, 02, ...)",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="01,02",
        help="Comma-separated split folder names to include",
    )
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--color_space",
        type=str,
        default="rgb",
        choices=["rgb", "hsv", "ycbcr", "xyz", "lab"],
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument(
        "--face_detector",
        type=str,
        default="fan",
        choices=["fan", "sfd"],
        help="Used when --crop-source is fan or sfd (ignored for landmarks crop).",
    )
    ap.add_argument(
        "--crop-source",
        type=str,
        default="fan",
        choices=["fan", "sfd", "landmarks"],
        help=(
            "fan/sfd: full-frame detection as in process_videos_av. "
            "landmarks: crop from SEWA JSON landmarks (bbox + padding)."
        ),
    )
    ap.add_argument(
        "--landmark-padding",
        type=float,
        default=0.3,
        help="Relative pad around landmark bbox when --crop-source landmarks.",
    )
    ap.add_argument(
        "--smooth-k",
        type=int,
        default=1,
        help="Odd/even moving average length on predictions within each clip after pairing (1=off; needs enough frames per clip).",
    )
    ap.add_argument(
        "--no_clamp_output",
        action="store_true",
        help="Match process_videos_av --no_clamp_output",
    )
    ap.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Process at most this many clips per split (debug)",
    )
    ap.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=None,
        help="Only first N frames per clip (debug)",
    )
    ap.add_argument(
        "--per-clip-json",
        action="store_true",
        help="Include full per-clip metric dicts in output (large)",
    )
    ap.add_argument(
        "--label-scale",
        type=str,
        default="auto",
        help=(
            "Divisor applied to SEWA JSON labels before metrics. "
            "'auto' uses 1000 (for labels in about [-1000,1000]); "
            "use '1' if labels are already in [-1,1]."
        ),
    )
    ap.add_argument(
        "--no-clip-labels",
        action="store_true",
        help="Do not clip scaled ground-truth labels to [-1,1].",
    )
    ap.add_argument("-o", "--output", type=Path, default=Path("sewa_ood_results.json"))
    args = ap.parse_args()

    out_stem_l = args.output.stem.lower()
    if "landmark" in out_stem_l and args.crop_source != "landmarks":
        print(
            "Warning: output filename looks like a landmark run, but "
            f"--crop-source is {args.crop_source!r}. "
            "Use --crop-source landmarks if you intended JSON-landmark crops.",
            file=sys.stderr,
            flush=True,
        )
    if "smooth" in out_stem_l and int(args.smooth_k) <= 1:
        print(
            "Warning: output filename looks like a smoothed run, but "
            f"--smooth-k is {args.smooth_k} (off). "
            "Pass e.g. --smooth-k 5 if you intended within-clip prediction smoothing.",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"evaluate_sewa_ood: crop_source={args.crop_source} smooth_k={args.smooth_k} "
        f"label_scale={args.label_scale!r} -> {args.output}",
        flush=True,
    )

    import process_videos_av as pva

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    net = pva.EmoNet(n_expression=8).to(device)
    pva.load_checkpoint(net, Path(args.checkpoint).resolve(), device)
    net.eval()

    fa = None
    sfd_det = None
    if args.crop_source == "fan":
        fa = pva.build_face_aligner(device)
    elif args.crop_source == "sfd":
        sfd_det = SFDDetector(device=str(device))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    root = args.sewa_root.resolve()
    label_scale_divisor = parse_label_scale(args.label_scale)
    clip_labels = not args.no_clip_labels

    # ``frame_valence_arousal`` picks FAN vs SFD from this string; it must match
    # which detector object we built. Default --face-detector fan + --crop-source
    # sfd would otherwise pass fa=None while requesting FAN (assert / wrong).
    if args.crop_source == "fan":
        face_detector_run = "fan"
    elif args.crop_source == "sfd":
        face_detector_run = "sfd"
    else:
        face_detector_run = args.face_detector  # landmarks path ignores it

    summary: Dict[str, Any] = {
        "protocol": {
            "description": "Frame-level OOD: EmoNet vs Arousal_Valence.json; paired only when crop succeeds.",
            "sewa_root": str(root),
            "splits": splits,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "color_space": args.color_space,
            "face_detector": face_detector_run,
            "crop_source": args.crop_source,
            "landmark_padding": float(args.landmark_padding),
            "smooth_k": int(args.smooth_k),
            "clamp_output": not args.no_clamp_output,
            "label_scale_divisor": label_scale_divisor,
            "clip_scaled_labels_to_neg1_1": clip_labels,
            "aggregation_note": "Both micro_pooled_frames (frame-weighted) and macro_mean_over_clips (clip-weighted) are reported.",
        },
        "by_split": {},
    }

    metric_keys = [
        "ccc_valence",
        "ccc_arousal",
        "pearson_valence",
        "pearson_arousal",
        "rmse_valence",
        "rmse_arousal",
    ]

    pooled_pv: List[float] = []
    pooled_pa: List[float] = []
    pooled_gv: List[float] = []
    pooled_ga: List[float] = []

    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            summary["by_split"][split] = {"error": f"missing directory {split_dir}"}
            continue

        clip_dirs = sorted(
            [p for p in split_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        if args.max_clips is not None:
            clip_dirs = clip_dirs[: args.max_clips]

        per_clip: List[Dict[str, Any]] = []
        all_pv: List[float] = []
        all_pa: List[float] = []
        all_gv: List[float] = []
        all_ga: List[float] = []
        n_skipped_no_json = 0
        raw_label_v: List[float] = []
        raw_label_a: List[float] = []

        for cd in clip_dirs:
            jpath = cd / "Arousal_Valence.json"
            if not jpath.is_file():
                n_skipped_no_json += 1
                continue
            labels, lms = load_sewa_annotations(jpath, label_scale_divisor, clip_labels)
            if labels:
                raw_label_v.extend([x[0] for x in labels.values()])
                raw_label_a.extend([x[1] for x in labels.values()])
            lm_dict = lms if args.crop_source == "landmarks" else None
            cm, pv_l, pa_l, gv_l, ga_l = eval_one_clip(
                pva,
                cd,
                labels,
                lm_dict,
                args.crop_source,
                float(args.landmark_padding),
                int(args.smooth_k),
                net,
                args.color_space,
                args.image_size,
                device,
                face_detector_run,
                fa,
                sfd_det,
                clamp_output=not args.no_clamp_output,
                max_frames=args.max_frames_per_clip,
            )
            cm["clip_id"] = cd.name
            cm["split"] = split
            if args.per_clip_json:
                per_clip.append(cm)
            else:
                per_clip.append({k: cm[k] for k in ["clip_id", "split", "n_paired", *metric_keys] if k in cm})

            all_pv.extend(pv_l)
            all_pa.extend(pa_l)
            all_gv.extend(gv_l)
            all_ga.extend(ga_l)

        pooled_pv.extend(all_pv)
        pooled_pa.extend(all_pa)
        pooled_gv.extend(all_gv)
        pooled_ga.extend(all_ga)

        micro = aggregate_micro(pva, all_pv, all_pa, all_gv, all_ga)
        macro = mean_clip_metrics(
            [c for c in per_clip if "ccc_valence" in c],
            metric_keys,
        )

        summary["by_split"][split] = {
            "n_clip_dirs": len(clip_dirs),
            "n_with_json": len(clip_dirs) - n_skipped_no_json,
            "n_skipped_no_json": n_skipped_no_json,
            "scaled_label_range": {
                "valence_min": float(np.min(raw_label_v)) if raw_label_v else None,
                "valence_max": float(np.max(raw_label_v)) if raw_label_v else None,
                "arousal_min": float(np.min(raw_label_a)) if raw_label_a else None,
                "arousal_max": float(np.max(raw_label_a)) if raw_label_a else None,
            },
            "micro_pooled_frames": micro,
            "macro_mean_over_clips": macro,
            "per_clip": per_clip,
        }

    summary["micro_pooled_all_splits_frames"] = aggregate_micro(
        pva, pooled_pv, pooled_pa, pooled_gv, pooled_ga
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
