"""Video metadata and deterministic key-frame extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata for an input video."""

    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


@dataclass
class FrameInfo:
    """A saved analysis frame and its source location."""

    index: int
    source_index: int
    timestamp_s: float
    path: Path
    width: int
    height: int
    camera: str = "default"
    action: list[float] | None = None
    action_delta: list[float] | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def read_video_info(video: Path) -> VideoInfo:
    """Read metadata without loading the complete video into memory."""
    try:
        reader = imageio.get_reader(str(video))
        meta = reader.get_meta_data()
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = int(meta.get("nframes", 0) or 0)
        reader.close()
    except Exception as exc:
        raise RuntimeError(f"could not open video {video}: {exc}") from exc

    fps = float(meta.get("fps") or 0.0)
    if fps <= 0:
        raise RuntimeError(f"video {video} has no usable FPS metadata")
    if frame_count <= 0:
        raise RuntimeError(f"video {video} has no readable frames")
    size = meta.get("size")
    if not size or len(size) != 2:
        raise RuntimeError(f"video {video} has no usable size metadata")
    width, height = int(size[0]), int(size[1])
    return VideoInfo(fps, frame_count, width, height, frame_count / fps)


def _resize_frame(frame: np.ndarray, max_edge: int) -> np.ndarray:
    """Resize an RGB frame while preserving its aspect ratio."""
    image = Image.fromarray(np.asarray(frame).astype(np.uint8))
    longest = max(image.size)
    if longest <= max_edge:
        return np.asarray(image)
    scale = max_edge / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS))


def extract_keyframes(
    video: Path,
    output_dir: Path,
    *,
    frame_count: int | None = None,
    sample_stride: int = 10,
    max_image_edge: int = 1280,
) -> tuple[VideoInfo, list[FrameInfo]]:
    """Sample frames at a stride, optionally capped by ``frame_count``."""
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least 1")
    if frame_count is not None and frame_count < 1:
        raise ValueError("frame_count must be at least 1 when provided")
    if max_image_edge < 64:
        raise ValueError("max_image_edge must be at least 64")
    info = read_video_info(video)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_indices = list(range(0, info.frame_count, sample_stride))
    if not source_indices or source_indices[-1] != info.frame_count - 1:
        source_indices.append(info.frame_count - 1)
    if frame_count is not None and len(source_indices) > frame_count:
        positions = np.linspace(0, len(source_indices) - 1, frame_count, dtype=int)
        source_indices = [source_indices[int(pos)] for pos in positions]
    source_indices = list(dict.fromkeys(source_indices))

    reader = imageio.get_reader(str(video))
    result: list[FrameInfo] = []
    try:
        for saved_index, source_index in enumerate(source_indices):
            frame = reader.get_data(source_index)
            frame = _resize_frame(frame, max_image_edge)
            path = output_dir / f"frame_{saved_index:04d}.jpg"
            imageio.imwrite(path, frame, quality=90)
            result.append(
                FrameInfo(
                    index=saved_index,
                    source_index=source_index,
                    timestamp_s=source_index / info.fps,
                    path=path,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                )
            )
    except Exception as exc:
        raise RuntimeError(f"could not extract frames from {video}: {exc}") from exc
    finally:
        reader.close()
    return info, result


def video_info_dict(info: VideoInfo) -> dict[str, object]:
    """Return JSON-compatible video metadata."""
    return asdict(info)
