"""Dataset catalog manager for indexed test videos, frames, and consensus masks.

Enables the researcher to select any untouched test benchmark frame (e.g. video 21 - frame 040)
or browse videos with ground truth comparisons.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from PIL import Image


class DatasetCatalog:
    """Indexes video frames, ground truth masks, and split assignments."""

    def __init__(self, workspace_root: Optional[Path] = None):
        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parent.parent.parent

        self.root = workspace_root
        self.data_dir = self.root / "video-frame" / "video-frame"
        self.frames_dir = self.data_dir / "frames"
        self.consensus_dir = self.data_dir / "consensus_masks"
        self.split_csv_path = self.root / "video_split_assignment.csv"

        self.split_df: Optional[pd.DataFrame] = None
        if self.split_csv_path.exists():
            try:
                self.split_df = pd.read_csv(self.split_csv_path)
            except Exception as e:
                print(f"[DatasetCatalog] Warning: Failed to read split csv: {e}")

    def list_videos(self) -> List[Dict[str, Any]]:
        """Returns indexed videos with split info and frame counts."""
        if not self.frames_dir.exists():
            return []

        videos = []
        for vid_dir in sorted(self.frames_dir.iterdir()):
            if not vid_dir.is_dir():
                continue

            vid_name = vid_dir.name
            frame_count = len(list(vid_dir.glob("*.jpg")))
            has_masks = (self.consensus_dir / vid_name).exists()

            split = "unknown"
            if self.split_df is not None and "video" in self.split_df.columns:
                match = self.split_df[self.split_df["video"] == vid_name]
                if not match.empty:
                    split = match.iloc[0]["split"]

            videos.append({
                "video": vid_name,
                "split": split,
                "frame_count": frame_count,
                "has_consensus_masks": has_masks,
                "is_test_benchmark": (split == "test")
            })

        # Sort so test benchmark videos appear first
        videos.sort(key=lambda x: (0 if x["is_test_benchmark"] else 1, x["video"]))
        return videos

    def list_frames_for_video(self, video_name: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Returns ordered frame stems for a selected video."""
        vid_dir = self.frames_dir / video_name
        if not vid_dir.exists():
            return []

        frames = []
        for img_path in sorted(vid_dir.glob("*.jpg"))[:limit]:
            stem = img_path.stem
            mask_path = self.consensus_dir / video_name / f"{stem}.png"
            frames.append({
                "video": video_name,
                "stem": stem,
                "filename": img_path.name,
                "has_mask": mask_path.exists()
            })
        return frames

    def get_frame_paths(self, video_name: str, stem: str) -> Dict[str, Optional[Path]]:
        """Resolves filesystem paths for a frame image and its consensus mask."""
        img_path = self.frames_dir / video_name / f"{stem}.jpg"
        mask_path = self.consensus_dir / video_name / f"{stem}.png"

        return {
            "image": img_path if img_path.exists() else None,
            "mask": mask_path if mask_path.exists() else None
        }

    def load_frame(self, video_name: str, stem: str) -> Optional[Image.Image]:
        """Loads PIL image for the requested frame."""
        paths = self.get_frame_paths(video_name, stem)
        if paths["image"] and paths["image"].exists():
            return Image.open(paths["image"])
        return None

    def load_mask(self, video_name: str, stem: str) -> Optional[Image.Image]:
        """Loads PIL consensus mask if present."""
        paths = self.get_frame_paths(video_name, stem)
        if paths["mask"] and paths["mask"].exists():
            return Image.open(paths["mask"])
        return None
