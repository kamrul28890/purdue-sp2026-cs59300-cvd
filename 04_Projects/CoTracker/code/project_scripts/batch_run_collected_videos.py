#!/usr/bin/env python
"""Run CoTracker on the collected project video set and save reproducible outputs."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
VIDEO_DIRS = [
    ROOT / "project_data" / "raw_videos" / "official_assets",
    ROOT / "project_data" / "raw_videos" / "internet_archive",
]
VIDEO_EXTENSIONS = {".mp4", ".gif", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class ClipMetadata:
    original_frames: int
    clip_start_frame: int
    clip_frames: int
    fps: float
    resolution: str


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=f"batch_collected_{today}")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["online", "offline"],
        choices=["online", "offline"],
        help="Run the online model, the offline model, or both.",
    )
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=180)
    parser.add_argument(
        "--clip_start_fraction",
        type=float,
        default=0.10,
        help="For long videos, start this fraction into the video before taking max_frames.",
    )
    parser.add_argument(
        "--online_checkpoint",
        default=str(ROOT / "checkpoints" / "scaled_online.pth"),
    )
    parser.add_argument(
        "--offline_checkpoint",
        default=str(ROOT / "checkpoints" / "scaled_offline.pth"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output videos if they already exist.",
    )
    return parser.parse_args()


def collect_video_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in VIDEO_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(path)
    return paths


def read_clip(path: Path, max_frames: int, clip_start_fraction: float) -> tuple[np.ndarray, ClipMetadata]:
    if path.suffix.lower() == ".gif":
        frames = imageio.mimread(path)
        frames_np = [np.asarray(frame)[..., :3] for frame in frames]
        clip_frames = frames_np[:max_frames]
        if not clip_frames:
            raise RuntimeError("No frames decoded from GIF.")
        height, width = clip_frames[0].shape[:2]
        metadata = ClipMetadata(
            original_frames=len(frames_np),
            clip_start_frame=0,
            clip_frames=len(clip_frames),
            fps=10.0,
            resolution=f"{width}x{height}",
        )
        return np.stack(clip_frames), metadata

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("VideoCapture could not open the input video.")

    original_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_start_frame = 0
    if original_frames > max_frames:
        clip_start_frame = min(
            original_frames - max_frames,
            max(0, int(original_frames * clip_start_fraction)),
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)

    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

    if not frames:
        raise RuntimeError("No frames decoded from the input video.")

    metadata = ClipMetadata(
        original_frames=original_frames,
        clip_start_frame=clip_start_frame,
        clip_frames=len(frames),
        fps=fps if fps > 0 else 10.0,
        resolution=f"{width}x{height}",
    )
    return np.stack(frames), metadata


def build_predictor(mode: str, checkpoint: str, device: str) -> CoTrackerPredictor:
    if mode == "online":
        predictor = CoTrackerPredictor(
            checkpoint=checkpoint,
            offline=False,
            window_len=16,
        )
    else:
        predictor = CoTrackerPredictor(
            checkpoint=checkpoint,
            offline=True,
            window_len=60,
        )
    predictor = predictor.to(device)
    predictor.eval()
    return predictor


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = DEFAULT_DEVICE
    run_name = args.run_name
    video_paths = collect_video_paths()

    if not video_paths:
        raise RuntimeError("No collected videos were found under project_data/raw_videos/.")

    output_root = ROOT / "project_results" / "videos" / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []

    checkpoint_by_mode = {
        "online": args.online_checkpoint,
        "offline": args.offline_checkpoint,
    }
    predictors = {
        mode: build_predictor(mode=mode, checkpoint=checkpoint_by_mode[mode], device=device)
        for mode in args.modes
    }

    for path in video_paths:
        frames, metadata = read_clip(
            path=path,
            max_frames=args.max_frames,
            clip_start_fraction=args.clip_start_fraction,
        )
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)

        for mode in args.modes:
            mode_dir = output_root / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            output_stem = f"{path.stem}_{mode}"
            output_path = mode_dir / f"{output_stem}.mp4"

            row = {
                "run_name": run_name,
                "video_id": path.stem,
                "source_group": path.parent.name,
                "mode": mode,
                "input_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "output_path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
                "device": device,
                "grid_size": str(args.grid_size),
                "clip_start_frame": str(metadata.clip_start_frame),
                "clip_frames": str(metadata.clip_frames),
                "original_frames": str(metadata.original_frames),
                "fps": f"{metadata.fps:.3f}",
                "resolution": metadata.resolution,
                "runtime_seconds": "",
                "status": "",
                "error": "",
            }

            if output_path.exists() and not args.overwrite:
                row["status"] = "skipped_existing_output"
                run_rows.append(row)
                failure_rows.append(
                    {
                        "video_id": path.stem,
                        "mode": mode,
                        "camera_motion": "",
                        "occlusion": "",
                        "blur": "",
                        "low_texture": "",
                        "scale_change": "",
                        "lighting_change": "",
                        "overall_quality": "",
                        "summary": "",
                        "failure_notes": (
                            f"Manual review pending. Existing output: {row['output_path']} "
                            f"(clip {metadata.clip_start_frame}:{metadata.clip_start_frame + metadata.clip_frames})."
                        ),
                    }
                )
                continue

            start_time = time.perf_counter()
            try:
                with torch.inference_mode():
                    pred_tracks, pred_visibility = predictors[mode](
                        video_tensor,
                        grid_size=args.grid_size,
                    )

                visualizer = Visualizer(
                    save_dir=str(mode_dir),
                    pad_value=120,
                    linewidth=3,
                    fps=max(1, int(round(metadata.fps))),
                )
                visualizer.visualize(
                    video_tensor,
                    pred_tracks,
                    pred_visibility,
                    query_frame=0,
                    filename=output_stem,
                )

                row["status"] = "completed"
                row["runtime_seconds"] = f"{time.perf_counter() - start_time:.3f}"
            except Exception as exc:
                row["status"] = "failed"
                row["runtime_seconds"] = f"{time.perf_counter() - start_time:.3f}"
                row["error"] = str(exc)
            finally:
                run_rows.append(row)
                failure_rows.append(
                    {
                        "video_id": path.stem,
                        "mode": mode,
                        "camera_motion": "",
                        "occlusion": "",
                        "blur": "",
                        "low_texture": "",
                        "scale_change": "",
                        "lighting_change": "",
                        "overall_quality": "",
                        "summary": "",
                        "failure_notes": (
                            f"Manual review pending. Output: {row['output_path']} "
                            f"(clip {metadata.clip_start_frame}:{metadata.clip_start_frame + metadata.clip_frames}, "
                            f"status={row['status']})."
                        ),
                    }
                )
                if device == "cuda":
                    torch.cuda.empty_cache()

        del video_tensor
        if device == "cuda":
            torch.cuda.empty_cache()

    run_summary_path = ROOT / "project_results" / "tables" / f"{run_name}_summary.csv"
    failure_seed_path = ROOT / "project_results" / "tables" / f"{run_name}_failure_analysis.csv"

    write_csv(
        run_summary_path,
        run_rows,
        [
            "run_name",
            "video_id",
            "source_group",
            "mode",
            "input_path",
            "output_path",
            "device",
            "grid_size",
            "clip_start_frame",
            "clip_frames",
            "original_frames",
            "fps",
            "resolution",
            "runtime_seconds",
            "status",
            "error",
        ],
    )
    write_csv(
        failure_seed_path,
        failure_rows,
        [
            "video_id",
            "mode",
            "camera_motion",
            "occlusion",
            "blur",
            "low_texture",
            "scale_change",
            "lighting_change",
            "overall_quality",
            "summary",
            "failure_notes",
        ],
    )

    print(f"Run summary written to: {run_summary_path.relative_to(ROOT)}")
    print(f"Failure analysis seed written to: {failure_seed_path.relative_to(ROOT)}")
    print(f"Output videos saved under: {output_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
