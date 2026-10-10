"""Artifact indexing, provenance and display-only image operations.

No notebook execution, model loading, or measurements from rendered figures.
All paths sent to clients are repository-relative, and file access is allowlisted.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODELS = [
    {"id": "unetpp", "name": "U-Net++", "encoder": "ResNet-34", "notebook": "unetpp-res34.ipynb", "color": "#2265ab"},
    {"id": "deeplabv3plus", "name": "DeepLabV3+", "encoder": "ResNet-50", "notebook": "deeplabv3plus.ipynb", "color": "#328579"},
    {"id": "segformer_b1", "name": "SegFormer-B1", "encoder": "MiT-B1", "notebook": "segformer_b1.ipynb", "color": "#7967ad"},
    {"id": "fpn", "name": "FPN", "encoder": "EfficientNet-B2", "notebook": "fpn1.ipynb", "color": "#ad733e"},
]
METRICS = {
    "dice": {"label": "Dice", "description": "Overlap between prediction and reference; higher is better."},
    "iou": {"label": "IoU", "description": "Intersection divided by union of wet pixels; higher is better."},
    "precision": {"label": "Precision", "description": "Fraction of predicted wet pixels that are correct; penalizes false positives."},
    "recall": {"label": "Recall", "description": "Fraction of reference wet pixels recovered; penalizes false negatives."},
    "accuracy": {"label": "Pixel accuracy", "description": "Fraction of all pixels classified correctly, including dry background."},
    "war_mae": {"label": "WAR MAE", "description": "Mean absolute error of published WAR ratios; lower is better (ratio units)."},
    "war_rmse": {"label": "WAR RMSE", "description": "Root mean squared WAR error; emphasizes larger errors (ratio units)."},
    "war_r2": {"label": "WAR R²", "description": "Fit of evaluated WAR estimates. This is not a segmentation metric or evidence of sorptivity prediction."},
}
WAR_NOTE = "Published WAR uses wet pixels / all image pixels: the notebooks call compute_war without a specimen ROI. Specimen-normalized WAR requires a validated full-specimen ROI, which is unavailable. Values below preserve the published definition."
XAI_NOTE = "XAI ROI is a bounding envelope of the union of a video's consensus wet masks, padded by 8 pixels and extended to the image bottom. It is an approximate ROI, not a validated full-specimen annotation; dry specimen regions may be excluded. Attribution fractions retain that limitation."


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def png_bytes(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def annotation_votes(paths):
    """Match notebook foreground rule: any nonzero RGB channel, >=2/4 votes."""
    if len(paths) != 4 or not all(p.is_file() for p in paths):
        raise ValueError("All four annotator masks are required for agreement.")
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        arrays.append(array.max(axis=-1) > 0)
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("Annotator mask dimensions differ; agreement is unavailable.")
    return np.stack(arrays).sum(axis=0)


class Catalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.data = "video-frame/video-frame"
        self.warnings = []
        self.files = {}
        self.by_path = {}
        roots = [self.root / d for d in ("evaluation", "outputs", "xai_gradcam_outputs", "video-frame")]
        paths = [p for d in roots if d.exists() for p in d.rglob("*") if p.is_file()]
        paths += list(self.root.glob("*.ipynb")) + [self.root / "README.md", self.root / "video_split_assignment.csv"]
        for path in paths:
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".csv", ".json", ".pdf", ".ipynb", ".md"}:
                continue
            if not path.is_file() or not path.resolve().is_relative_to(self.root):
                continue
            relative = path.relative_to(self.root).as_posix()
            key = hashlib.sha256(relative.encode()).hexdigest()[:20]
            record = {"id": key, "path": relative, "name": path.name, "type": path.suffix[1:].upper(), "bytes": path.stat().st_size, "url": f"/api/files/{key}"}
            self.files[key] = record
            self.by_path[relative] = record
        self.splits = self.read_csv("video_split_assignment.csv", ("video", "split", "frames"))
        self.videos = []
        for row in self.splits:
            if not re.fullmatch(r"video \d+", row["video"]) or row["split"] not in {"train", "val", "test"} or number(row["frames"]) is None:
                self.warnings.append("Invalid row in video_split_assignment.csv; skipped.")
                continue
            frames = sorted(p.stem for p in (self.root / self.data / "frames" / row["video"]).glob("*.jpg"))
            self.videos.append({"video": row["video"], "split": row["split"], "declared_frames": int(float(row["frames"])), "frames": frames})
            if len(frames) != int(float(row["frames"])):
                self.warnings.append(f"{row['video']}: manifest frame count differs from available images.")
        self.videos.sort(key=lambda v: int(v["video"].split()[-1]) if v["video"].split()[-1].isdigit() else 999)
        self.frame_keys = {(v["video"], s) for v in self.videos for s in v["frames"]}
        expected = {(v["video"], s) for v in self.videos if v["split"] == "test" for s in v["frames"]}
        self.benchmarks = []
        self.predictions = {}
        self.coverage = {}
        for model in MODELS:
            mid = model["id"]
            path = f"evaluation/{mid}/metrics_test.csv"
            for row in self.read_csv(path, ("mode", *METRICS)):
                if row["mode"] not in {"raw", "physics_enforced"}:
                    self.warnings.append(f"Unknown mode in {path}; skipped.")
                    continue
                metrics = {k: number(row.get(k)) for k in METRICS}
                if any(v is None for v in metrics.values()):
                    self.warnings.append(f"Missing or non-finite metric in {path}; displayed as unavailable.")
                self.benchmarks.append({"model": mid, "mode": row["mode"], **metrics, "source": path})
            pp = f"evaluation/{mid}/test_frame_predictions.csv"
            rows = self.read_csv(pp, ("video", "stem", "war_gt", "war_pred_raw", "war_pred_phys"))
            self.predictions[mid] = {(r["video"], r["stem"]): {k: number(r.get(k)) for k in ("war_gt", "war_pred_raw", "war_pred_phys")} for r in rows}
            self.coverage[mid] = {"frames": len(rows), "matches_test": set(self.predictions[mid]) == expected and len(rows) == len(expected), "source": pp}
            if not self.coverage[mid]["matches_test"]:
                self.warnings.append(f"{model['name']}: evaluated frame IDs differ from the current test manifest.")
        self.xai_rows = self.read_csv("xai_gradcam_outputs/tables/xai_spatial_attribution_diagnostics.csv", ("video", "stem", "dice", "roi_cam_fraction"))
        self.xai = {(r["video"], r["stem"]): {k: number(v) if k not in {"video", "stem", "image_id"} else v for k, v in r.items()} for r in self.xai_rows}
        self.cases = self.read_csv("xai_gradcam_outputs/tables/representative_cases.csv", ("video", "stem", "category"))
        self.xai_summary = self.read_json("xai_gradcam_outputs/tables/xai_summary.json")
        self.checkpoints = [p.relative_to(self.root).as_posix() for pattern in ("*.pth", "*.pt", "*.ckpt") for p in self.root.rglob(pattern) if p.resolve().is_relative_to(self.root) and not any(part.startswith('.') for part in p.relative_to(self.root).parts)]
        self.audit()

    def read_csv(self, relative, required=()):
        try:
            with (self.root / relative).open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, strict=True)
                if not set(required).issubset(reader.fieldnames or []):
                    raise ValueError("required columns missing")
                rows = list(reader)
                if any(None in r or any(r.get(k) is None for k in required) for r in rows):
                    raise ValueError("malformed rows")
                return rows
        except (OSError, ValueError, csv.Error, UnicodeError) as exc:
            self.warnings.append(f"{relative}: unavailable ({type(exc).__name__}).")
            return []

    def read_json(self, relative):
        try:
            with (self.root / relative).open(encoding="utf-8") as stream:
                result = json.load(stream, parse_float=number, parse_constant=lambda _: None)
            if not isinstance(result, dict):
                raise ValueError("expected object")
            return result
        except (OSError, ValueError, UnicodeError) as exc:
            self.warnings.append(f"{relative}: unavailable ({type(exc).__name__}).")
            return {}

    def audit(self):
        self.warnings.extend([
            "README dataset counts (24 videos; 17/4/3 split) conflict with the root manifest (28 videos; 19/6/3). The app uses the manifest and checks evaluated frame IDs against it. Seed 42 is documented in the notebooks, not stored in the CSV.",
            "README benchmark numbers differ from the evaluation CSVs. Model-specific metrics_test.csv files are the displayed source of truth.",
            "Checkpoint hashes and run IDs are absent. Matching frame IDs verifies cohort alignment, but cannot prove checkpoint identity between CSVs and figures.",
            "The XAI summary reports a separate U-Net++ evaluation (Dice 0.964798…), while the benchmark CSV reports 0.965182…. These runs are displayed separately.",
            "Physics enforcement modifies masks (connected components, morphology and column filling). However, the notebooks reuse raw segmentation overlap metrics in both summary rows; only WAR metrics are evaluated separately.",
        ])
        nested = self.read_csv(f"{self.data}/video_split_assignment.csv", ("video", "split", "frames"))
        if nested and sorted((r['video'], r['split'], r['frames']) for r in nested) != sorted((r['video'], r['split'], r['frames']) for r in self.splits):
            self.warnings.append("The nested dataset split manifest differs from the root manifest.")
        self.seg_figures = {}
        figure_rows = self.read_csv("outputs/segformer_visualizations/segformer_war_comparison.csv", ("video", "stem", "filename", "war_gt", "war_pred_raw", "war_pred_phys"))
        differing = 0
        for row in figure_rows:
            key = (row["video"], row["stem"])
            self.seg_figures[key] = row
            old = self.predictions["segformer_b1"].get(key)
            if old and any(number(row[k]) is not None and old[k] is not None and abs(number(row[k]) - old[k]) > 1e-8 for k in old):
                differing += 1
        if differing:
            self.warnings.append(f"SegFormer-B1 figure-export WAR differs from benchmark predictions for {differing} frames. Explorer figure values use segformer_war_comparison.csv; benchmark values remain separate.")

    def file(self, relative):
        return self.by_path.get(relative)

    def path_for_id(self, key):
        if key not in self.files:
            raise KeyError("Artifact unavailable")
        path = (self.root / self.files[key]["path"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise KeyError("Artifact unavailable")
        return path

    def metadata(self):
        counts = {s: {"videos": sum(v["split"] == s for v in self.videos), "frames": sum(v["declared_frames"] for v in self.videos if v["split"] == s)} for s in ("train", "val", "test")}
        return {"models": MODELS, "metrics": METRICS, "splits": counts, "videos": self.videos, "benchmarks": self.benchmarks, "coverage": self.coverage, "warnings": self.warnings, "war_note": WAR_NOTE, "xai_note": XAI_NOTE, "xai_summary": self.xai_summary, "cases": self.cases, "checkpoints": self.checkpoints, "live_inference": False, "vlm": False,
                "artifacts": [f for f in self.files.values() if not f["path"].startswith("video-frame/")]}

    def validate_frame(self, video, stem):
        if (video, stem) not in self.frame_keys:
            raise KeyError("Frame unavailable in the current manifest")

    def visual(self, model, video, stem):
        """Verified figure-to-frame mapping. Never infer arbitrary grid rows."""
        if model == "segformer_b1":
            row = self.seg_figures.get((video, stem))
            if row:
                return self.file("outputs/segformer_visualizations/" + row["filename"]), 0, "single", "physics_enforced"
        if model == "unetpp":
            rows = {("video 24", "ezgif-frame-001"): 0, ("video 1", "ezgif-frame-046"): 1, ("video 21", "ezgif-frame-040"): 2}
            if (video, stem) in rows:
                return self.file("evaluation/unetpp/unetpp_visual_predictions.png"), rows[(video, stem)], "unet", "raw"
        if (video, stem) == ("video 21", "ezgif-frame-040") and model in {"fpn", "deeplabv3plus"}:
            name = "fpn" if model == "fpn" else "deeplabv3p"
            return self.file(f"evaluation/{model}/{name}_visual_predictions.png"), 0, "single", "physics_enforced"
        return None, 0, None, None

    def frame(self, video, stem):
        self.validate_frame(video, stem)
        result = {"video": video, "stem": stem, "split": next(v["split"] for v in self.videos if v["video"] == video),
                  "original": self.file(f"{self.data}/frames/{video}/{stem}.jpg"), "consensus": self.file(f"{self.data}/consensus_masks/{video}/{stem}.png"),
                  "annotators": [self.file(f"{self.data}/annotations/annotator {n}/{video}/{stem}.png") for n in range(1, 5)], "models": []}
        for model in MODELS:
            mid = model["id"]
            figure, _, _, error_mode = self.visual(mid, video, stem)
            values = self.predictions[mid].get((video, stem))
            source = f"evaluation/{mid}/test_frame_predictions.csv"
            if mid == "segformer_b1" and figure:
                row = self.seg_figures[(video, stem)]
                values = {k: number(row[k]) for k in ("war_gt", "war_pred_raw", "war_pred_phys")}
                source = "outputs/segformer_visualizations/segformer_war_comparison.csv"
            result["models"].append({**model, "figure": figure, "values": values, "source": source, "error_mode": error_mode})
        prefix = f"xai_panel_{video}_{stem}.png"
        result["xai_figures"] = [f for f in self.files.values() if f["path"].startswith("xai_gradcam_outputs/") and f["name"] == prefix]
        result["xai_metrics"] = self.xai.get((video, stem))
        return result

    @lru_cache(maxsize=128)
    def panel(self, model, video, stem, view, mode):
        self.validate_frame(video, stem)
        if mode not in {"raw", "physics_enforced"} or view not in {"prediction", "error", "input", "reference"}:
            raise ValueError("Unknown display mode")
        figure, row, template, error_mode = self.visual(model, video, stem)
        if not figure or (view == "error" and error_mode != mode):
            raise KeyError("No saved panel for this frame and mode")
        col = {"input": 0, "reference": 1, "prediction": 2 if mode == "raw" else 3, "error": 4}[view]
        with Image.open(self.path_for_id(figure["id"])) as image:
            # Pixel bounds inspected against the committed figure layouts. These
            # crops are for visualization only, never used as prediction arrays.
            if template == "single":
                if image.size != (5235, 1062):
                    raise ValueError("Figure layout changed; use the original download")
                x, y, size = [30, 1101, 2172, 3243, 4314][col], 141, 891
            else:
                if image.size != (5400, 3150):
                    raise ValueError("Figure layout changed; use the original download")
                x, y, size = [99, 1168, 2237, 3306, 4376][col], [109, 1144, 2179][row], 926
            return png_bytes(image.crop((x, y, x + size, y + size)).convert("RGB"))

    @lru_cache(maxsize=32)
    def agreement(self, video, stem):
        self.validate_frame(video, stem)
        records = [self.file(f"{self.data}/annotations/annotator {n}/{video}/{stem}.png") for n in range(1, 5)]
        if not all(records):
            raise ValueError("All four annotator masks are required for agreement.")
        paths = [self.path_for_id(record["id"]) for record in records]
        votes = annotation_votes(paths)
        colors = np.array([[24, 39, 59], [247, 187, 83], [247, 187, 83], [247, 187, 83], [49, 156, 148]], dtype=np.uint8)
        return png_bytes(Image.fromarray(colors[votes]))

    def comparison(self, video, stem, mode):
        self.validate_frame(video, stem)
        if mode not in {"raw", "physics_enforced"}:
            raise ValueError("Unknown display mode")
        frame = self.frame(video, stem)
        canvas = Image.new("RGB", (1440, 980), "#f3f6f9")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 16), f"{video} / {stem} | {mode} | Saved research figures", fill="#132c47")
        tiles = [("Input frame (native aspect)", frame["original"]), ("Consensus reference (native aspect)", frame["consensus"])]
        for i, (title, artifact) in enumerate(tiles):
            draw.text((24 + i * 480, 50), title, fill="#132c47")
            if artifact:
                with Image.open(self.path_for_id(artifact["id"])) as image:
                    image = image.convert("RGB"); image.thumbnail((430, 380))
                    canvas.paste(image, (24 + i * 480, 82))
        draw.text((980, 80), "Display crops are precomputed.\nMissing panels are unavailable.\nWAR uses full-image denominator.\nCheckpoint identity unverified.", fill="#132c47", spacing=10)
        for i, model in enumerate(MODELS):
            x, y = 24 + i * 354, 500
            draw.text((x, y), f"{model['name']} / {model['encoder']}", fill="#132c47")
            try:
                image = Image.open(io.BytesIO(self.panel(model["id"], video, stem, "prediction", mode)))
                image.thumbnail((330, 330)); canvas.paste(image, (x, y + 28))
            except (KeyError, ValueError, OSError):
                draw.text((x, y + 100), "Prediction image unavailable", fill="#657286")
            values = frame["models"][i]["values"]
            if values:
                draw.text((x, 878), f"Published WAR: {values.get('war_pred_raw' if mode == 'raw' else 'war_pred_phys')}", fill="#132c47")
        draw.text((24, 940), "Source files and run limitations are available in the application. No model inference performed.", fill="#657286")
        return png_bytes(canvas)
