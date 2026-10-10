# Cementitious Research Explorer

A local, read-only frontend for the actual repository artifacts. No training,
notebook execution, model inference or remote API calls occur when the app opens.

## Launch

Python 3.10+ is required. No Node.js, GPU, PyTorch or web framework is needed to
run the application. Its only Python dependencies are NumPy and Pillow.

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-frontend.txt
.venv\Scripts\python -m research_app.server
```

On macOS/Linux, use `.venv/bin/python` in place of `.venv\Scripts\python`.
Open **http://127.0.0.1:8765**. Stop the server with Ctrl+C.

Windows shortcut:

```powershell
.\start-frontend.ps1
```

The launcher looks for a configured `RESEARCH_PYTHON`, `.venv`, a normal Python
installation, then the optional Codex desktop bundled runtime. It checks for
NumPy and Pillow and does not install packages automatically. If PowerShell
execution policy prevents running the script, use the Python command above.

```powershell
.\start-frontend.ps1 -Port 8766 -Python C:\path\to\python.exe
```

Configuration:

| Setting | Default | Purpose |
| --- | --- | --- |
| `--root` / `RESEARCH_ROOT` | Repository root | Read research artifacts from another checkout |
| `--port` | `8765` | Local HTTP port |
| `--host` | `127.0.0.1` | Listen on loopback; intended for local use |
| `RESEARCH_HTTP_LOG=1` | Disabled | Enable request logs |
| `RESEARCH_PYTHON` | Auto-detected by launcher | Python executable for the Windows shortcut |

Restart the server after changing scientific artifacts to rebuild the index and
clear display caches. The server is a local research viewer, not a production
internet deployment. No credentials are needed.

## Features and their sources

| Section | Implementation |
| --- | --- |
| Research overview | Manifest-derived counts, CSV-derived headline benchmarks, workflow, notebook downloads, provenance notes |
| Segmentation explorer | Synchronized video/frame, raw/physics masks, mode-correct error maps, CSV WAR, full-size image inspection, comparison PNG download |
| Model benchmark | All 8 requested metrics, selectable metric bars, raw/physics rows, per-model sources and cohort checks |
| Explainability laboratory | Saved six-panel Grad-CAM++ figures, frame diagnostics, representative case categories, layer comparison, boundary/failure figures, correlations |
| Dataset & annotations | All 28 videos / 1,250 frames, original RGB, four original masks, stored consensus, live disagreement visualization |
| Research outputs | Search/type filters, PNG preview, CSV/JSON/PDF/notebook downloads |
| VLM analysis | Explicit planned/unavailable module and evidence requirements; no simulated response |

**Precomputed:** all segmentation predictions, metrics, WAR values, probability
visualizations, Grad-CAM++, correlations and research figures.

**Live display computation:** four-annotator agreement (any nonzero color channel
is foreground, matching the notebook), safe figure crops, comparison image
assembly, absolute WAR differences from CSV values, search/filtering, and cohort
checks. This is not live model inference. Stored consensus masks are displayed
directly; the original artifacts are never overwritten.

**Unavailable:** trained checkpoints, standalone prediction/probability arrays,
live inference, live Grad-CAM++, VLM backend, validated specimen ROI measurements,
and validated sorptivity regression. The Segmentation explorer includes an Upload image button for JPEG, PNG and
WebP previews (20 MB / 40 million pixels maximum). Images stay in browser memory,
can be replaced or removed, and clear on refresh. No model results or dataset
metrics are attached to an upload. Adding checkpoint files alone does not enable inference.

## Artifact mapping and provenance audit

The actual dataset is under `video-frame/video-frame/`, with `frames/`,
`annotations/annotator 1` through `annotator 4`, and `consensus_masks/`.
There are 1,250 RGB images, 5,000 annotator masks and 1,250 stored consensus masks.
The README's alternate dataset paths are not used.

The root `video_split_assignment.csv` specifies:

| Split | Videos | Frames |
| --- | ---: | ---: |
| Train | 19 | 863 |
| Validation | 6 | 169 |
| Test | 3 | 218 |

The test videos are 1, 21 and 24. Both committed split manifests agree. All four
model prediction CSVs contain exactly the 218 current test-frame IDs. Seed 42 is
documented in notebooks, not in the CSV. README counts of 24 videos, 17 training
videos / 1,234 frames and 4 validation videos / 290 frames conflict with these
files. The interface uses the root manifest and exposes the discrepancy.

Benchmark values come from `evaluation/<model>/metrics_test.csv`, not README
numbers. The combined four-architecture benchmark CSV is downloadable. The
top-level `evaluation/metrics_test.csv` is not silently merged with the explicitly
named ResNet-34 subdirectory results: the U-Net++ notebook output path is generic
and multiple U-Net++ variants exist.

### Prediction figure coverage

| Architecture | Saved, frame-resolved benchmark figures |
| --- | --- |
| FPN — EfficientNet-B2 | video 21 / ezgif-frame-040 |
| DeepLabV3+ — ResNet-50 | video 21 / ezgif-frame-040 |
| SegFormer-B1 — MiT-B1 | All 218 test frames |
| U-Net++ — ResNet-34 | video 24 / 001, video 1 / 046, video 21 / 040; additional selected XAI cases in the laboratory |

The default is video 21 / frame 040 so all four benchmark images are visible.
Other frames retain each model's available CSV WAR and show unavailable image
states. Mapping of the three U-Net++ figure rows and the single FPN/DeepLabV3+
figures was verified against the image titles. It is deliberately not inferred
from generic notebook sample indices: committed figures differ from some current
notebook plotting code.

Prediction panels are display crops from the saved PNGs, not recovered binary
arrays. Known pixel bounds and expected dimensions are checked. Changed layouts
fail explicitly and the source figure remains downloadable. No metrics are
calculated from rendered crops.

Error map convention is **green TP, red FP, blue FN, black TN**. U-Net++'s saved
benchmark error map uses **raw** predictions. FPN, DeepLabV3+ and SegFormer-B1 saved
error maps use **physics-enforced** predictions. Switching to the other mode shows
an unavailable error-map state instead of relabelling a saved image.

SegFormer-B1's `outputs/segformer_visualizations/segformer_war_comparison.csv`
differs from its benchmark prediction CSV on **22 frames**. Explorer WAR beside
those images uses the figure-export CSV. The benchmark dashboard retains the
model evaluation CSV. Checkpoint hashes and immutable run IDs are absent, so
matching cohorts does not prove checkpoint identity. This limitation is visible.

### WAR and physics enforcement

The intended specimen WAR is wet pixels inside specimen ROI / specimen ROI
pixels. The committed evaluation notebooks call `compute_war(mask)` without an
ROI, invoking the **whole-image denominator**. All displayed published WAR values
retain this definition. The app never invents a specimen ROI or recomputes a
purported specimen-normalized ratio.

Physics enforcement changes binary masks via post-processing. However, notebook
summary rows reuse the raw segmentation overlap metrics for both modes. Only WAR
metrics are separately evaluated for physics-enforced masks. The interface
explains this and does not claim independently measured post-processed Dice/IoU.

### XAI

`xai_gradcam_outputs/tables/xai_summary.json` reports Dice **0.9647983657**,
while the U-Net++ benchmark CSV reports **0.9651819425**. The interface keeps these
evaluation sources separate rather than combining their summaries. Attribution
diagnostics cover all 218 test frames; six-panel figures cover selected cases.

The primary layer is `decoder.blocks["x_0_4"]`. Stored figures include probability
and CAM color scales, overlay, raw prediction and ground truth. Original figures
are shown intact and can be enlarged/downloaded; numerical probability arrays
are not available.

XAI ROI is a padded bounding envelope from the union of each test video's wet
consensus masks, extended down to the image bottom. It is an approximation, not a
validated full-specimen annotation and may omit dry specimen regions. Background
and ROI attribution percentages inherit this limitation. Grad-CAM++ is post-hoc
attribution, not uncertainty or causal proof. Correlations do not establish
causality. `build_xai_notebook.py` is mentioned in the README but absent.

## Architecture and future integration

- `research_app/catalog.py`: one model/metric registry, CSV/JSON readers, artifact
  allowlist, frame/figure mapping, provenance checks, cached display operations.
- `research_app/server.py`: Python standard-library threaded HTTP server; read-only
  JSON/image/file routes. Files are served only by indexed opaque IDs; arbitrary
  paths and traversal are rejected. No training modules are imported.
- `research_app/static/`: buildless HTML, CSS and modular JavaScript, responsive
  navigation, state, accessible controls and image dialog.
- `tests/`: backend/data/API tests, JS state tests and optional browser tests.

Future inference should be a separate backend adapter with verified architecture,
checkpoint hash, training preprocessing and validated ROI metadata. Inspection
found 448×448 image resizing, ImageNet normalization and sigmoid threshold 0.5 in
the notebooks, but a complete adapter must verify these against the supplied
checkpoint. Use evaluation/no-gradient mode and cache loaded models. Grad-CAM++
requires its own gradient-enabled explanation path. Do not enable either feature
with untrained weights or by running training cells.

Future VLM calls belong on a credential-protected backend. No key input or external
call is included. The planned UI describes required evidence and the distinction
between observations, hypotheses and computed measurements.

## Verification

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
node --test tests/ui.test.mjs
```

The Python suite covers real metric/cohort mappings, figure availability, correct
error modes, colored-annotation voting, consensus agreement, malformed/missing
files, non-finite values, allowlisted access, HTTP modules and comparison downloads.
JS tests cover navigation, synchronized selection, modes, unavailable values and
HTML escaping.

Optional browser checks, with the server running:

```powershell
npm install --no-save playwright
npx playwright install chromium
node tests/browser.cjs
```

For installed Microsoft Edge, use `$env:BROWSER_CHANNEL='msedge'` instead of
installing Chromium. `PLAYWRIGHT_MODULE` may point to an existing Playwright
package directory. `RESEARCH_URL` overrides the test URL; `SCREENSHOT_DIR` saves
screenshots. Tests exercise all seven sections, selection/mode changes, empty
states, downloads, image dialog, XAI case selection, annotations, artifact search
and desktop/tablet/mobile layouts.
