"""Asynchronous HTTP Application Server for the Sorptivity XAI Workbench.

Built on Python aiohttp.web. Handles dataset indexing, live U-Net++ segmentation,
WAR calculation, and Grad-CAM++ saliency map generation.
"""

import os
import io
import json
from pathlib import Path
from typing import Optional
from PIL import Image
from aiohttp import web

from .model import get_model, DEVICE
from .dataset_catalog import DatasetCatalog
from .pipeline import SorptivityPipeline

pipeline_instance: Optional[SorptivityPipeline] = None
catalog_instance: Optional[DatasetCatalog] = None

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def get_pipeline() -> SorptivityPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = SorptivityPipeline()
    return pipeline_instance


def get_catalog() -> DatasetCatalog:
    global catalog_instance
    if catalog_instance is None:
        catalog_instance = DatasetCatalog()
    return catalog_instance


# -----------------------------------------------------------------------------
# API Handlers
# -----------------------------------------------------------------------------

async def handle_status(request: web.Request) -> web.Response:
    """Returns runtime telemetry, model metadata, and GPU hardware status."""
    pipe = get_pipeline()
    return web.json_response({
        "status": "online",
        "device": str(DEVICE),
        "cuda_available": (DEVICE.type == "cuda"),
        "model_metadata": pipe.metadata,
        "static_dir_exists": STATIC_DIR.exists()
    })


async def handle_list_videos(request: web.Request) -> web.Response:
    """Lists indexed videos and split classifications."""
    catalog = get_catalog()
    videos = catalog.list_videos()
    return web.json_response({"videos": videos, "total": len(videos)})


async def handle_list_frames(request: web.Request) -> web.Response:
    """Lists frames for a specific video."""
    video = request.query.get("video")
    if not video:
        return web.json_response({"error": "Missing 'video' query parameter"}, status=400)

    catalog = get_catalog()
    frames = catalog.list_frames_for_video(video)
    return web.json_response({"video": video, "frames": frames, "total": len(frames)})


async def handle_analyze_benchmark(request: web.Request) -> web.Response:
    """Runs pipeline on an indexed dataset frame (with automatic Ground Truth mask)."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload"}, status=400)

    video = data.get("video")
    stem = data.get("stem")
    if not video or not stem:
        return web.json_response({"error": "Both 'video' and 'stem' parameters are required"}, status=400)

    catalog = get_catalog()
    pil_img = catalog.load_frame(video, stem)
    if pil_img is None:
        return web.json_response({"error": f"Frame not found: '{video}' - '{stem}'"}, status=404)

    gt_mask_pil = catalog.load_mask(video, stem)

    pipe = get_pipeline()
    target_layer = data.get("target_layer", "decoder_x_0_4")
    threshold = float(data.get("threshold", 0.50))
    apply_physics = bool(data.get("apply_physics", True))
    bottom_margin = int(data.get("bottom_margin", 32))
    colormap = data.get("colormap", "jet")
    heatmap_alpha = float(data.get("heatmap_alpha", 0.55))
    include_simulated_aug = bool(data.get("include_simulated_augmentation", False))

    result = pipe.run_analysis(
        pil_image=pil_img,
        gt_mask_pil=gt_mask_pil,
        target_layer_name=target_layer,
        pred_threshold=threshold,
        apply_physics=apply_physics,
        bottom_margin=bottom_margin,
        colormap=colormap,
        heatmap_alpha=heatmap_alpha,
        include_simulated_augmentation=include_simulated_aug,
    )

    result["specimen_info"] = {
        "source": "dataset_benchmark",
        "video": video,
        "stem": stem,
        "has_ground_truth": (gt_mask_pil is not None)
    }
    return web.json_response(result)


async def handle_analyze_upload(request: web.Request) -> web.Response:
    """Handles uploaded custom images and optional ground truth masks via multipart/form-data."""
    reader = await request.multipart()

    image_bytes = None
    mask_bytes = None
    target_layer = "decoder_x_0_4"
    threshold = 0.50
    apply_physics = True
    bottom_margin = 32
    colormap = "jet"
    heatmap_alpha = 0.55
    include_simulated_aug = False

    while True:
        part = await reader.next()
        if part is None:
            break

        if part.name == "image":
            image_bytes = await part.read()
        elif part.name == "gt_mask":
            mask_bytes = await part.read()
        elif part.name == "target_layer":
            target_layer = (await part.read()).decode("utf-8")
        elif part.name == "threshold":
            try:
                threshold = float((await part.read()).decode("utf-8"))
            except ValueError:
                pass
        elif part.name == "apply_physics":
            val = (await part.read()).decode("utf-8").lower()
            apply_physics = (val in ("1", "true", "yes"))
        elif part.name == "bottom_margin":
            try:
                bottom_margin = int((await part.read()).decode("utf-8"))
            except ValueError:
                pass
        elif part.name == "colormap":
            colormap = (await part.read()).decode("utf-8")
        elif part.name == "heatmap_alpha":
            try:
                heatmap_alpha = float((await part.read()).decode("utf-8"))
            except ValueError:
                pass
        elif part.name == "include_simulated_augmentation":
            val = (await part.read()).decode("utf-8").lower()
            include_simulated_aug = (val in ("1", "true", "yes"))

    if not image_bytes:
        return web.json_response({"error": "No 'image' file provided in upload"}, status=400)

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return web.json_response({"error": f"Failed to parse uploaded image: {str(e)}"}, status=400)

    gt_mask_pil = None
    if mask_bytes and len(mask_bytes) > 0:
        try:
            gt_mask_pil = Image.open(io.BytesIO(mask_bytes))
        except Exception:
            pass

    pipe = get_pipeline()
    result = pipe.run_analysis(
        pil_image=pil_img,
        gt_mask_pil=gt_mask_pil,
        target_layer_name=target_layer,
        pred_threshold=threshold,
        apply_physics=apply_physics,
        bottom_margin=bottom_margin,
        colormap=colormap,
        heatmap_alpha=heatmap_alpha,
        include_simulated_augmentation=include_simulated_aug,
    )

    result["specimen_info"] = {
        "source": "custom_upload",
        "has_ground_truth": (gt_mask_pil is not None)
    }
    return web.json_response(result)


# -----------------------------------------------------------------------------
# Static App Handler
# -----------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.FileResponse:
    index_file = STATIC_DIR / "index.html"
    return web.FileResponse(index_file)


def create_app() -> web.Application:
    """Builds and returns the configured web application."""
    app = web.Application(client_max_size=32 * 1024 * 1024)

    # API Routes
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/catalog/videos", handle_list_videos)
    app.router.add_get("/api/catalog/frames", handle_list_frames)
    app.router.add_post("/api/analyze/benchmark", handle_analyze_benchmark)
    app.router.add_post("/api/analyze/upload", handle_analyze_upload)

    # Index
    app.router.add_get("/", handle_index)

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.router.add_static("/static/", path=str(STATIC_DIR), name="static")

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sorptivity XAI Workbench Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    print("=" * 70)
    print("      CEMENTITIOUS SORPTIVITY XAI WORKBENCH SERVER")
    print("=" * 70)
    print(f"Host: http://{args.host}:{args.port}")
    print(f"Device: {DEVICE}")

    app = create_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
