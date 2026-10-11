/**
 * Sorptivity XAI Workbench Application Controller
 * Handles UI interactions, split-curtain comparison slider,
 * parameters, and data telemetry rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const el = {
    // Status
    devicePill: document.getElementById("devicePill"),
    modelPill: document.getElementById("modelPill"),
    latencyPill: document.getElementById("latencyPill"),
    statusDot: document.getElementById("statusDot"),

    // Modes & Inputs
    sourceModeBenchmark: document.getElementById("sourceModeBenchmark"),
    sourceModeUpload: document.getElementById("sourceModeUpload"),
    benchmarkSection: document.getElementById("benchmarkSection"),
    uploadSection: document.getElementById("uploadSection"),

    videoSelect: document.getElementById("videoSelect"),
    frameSelect: document.getElementById("frameSelect"),
    benchmarkChips: document.getElementById("benchmarkChips"),

    fileInput: document.getElementById("fileInput"),
    dropzone: document.getElementById("dropzone"),
    dropzoneText: document.getElementById("dropzoneText"),
    maskFileInput: document.getElementById("maskFileInput"),
    maskDropzoneText: document.getElementById("maskDropzoneText"),

    // Parameters
    targetLayerSelect: document.getElementById("targetLayerSelect"),
    thresholdSlider: document.getElementById("thresholdSlider"),
    thresholdVal: document.getElementById("thresholdVal"),
    applyPhysicsCheck: document.getElementById("applyPhysicsCheck"),
    bottomMarginInput: document.getElementById("bottomMarginInput"),
    colormapSelect: document.getElementById("colormapSelect"),
    alphaSlider: document.getElementById("alphaSlider"),
    alphaVal: document.getElementById("alphaVal"),
    includeAugCheck: document.getElementById("includeAugCheck"),
    btnRunAnalysis: document.getElementById("btnRunAnalysis"),

    // Viewport Tabs
    tabGrid: document.getElementById("tabGrid"),
    tabSlider: document.getElementById("tabSlider"),
    tabHeatmap: document.getElementById("tabHeatmap"),
    tabError: document.getElementById("tabError"),

    viewGridContainer: document.getElementById("viewGridContainer"),
    viewSliderContainer: document.getElementById("viewSliderContainer"),
    viewHeatmapContainer: document.getElementById("viewHeatmapContainer"),
    viewErrorContainer: document.getElementById("viewErrorContainer"),

    // Images Grid
    imgInput: document.getElementById("imgInput"),
    imgRawMask: document.getElementById("imgRawMask"),
    imgPhysOverlay: document.getElementById("imgPhysOverlay"),
    imgGradcam: document.getElementById("imgGradcam"),

    // Slider Comparison
    comparatorContainer: document.getElementById("comparatorContainer"),
    comparatorBeforeImg: document.getElementById("comparatorBeforeImg"),
    comparatorAfterImg: document.getElementById("comparatorAfterImg"),
    comparatorDivider: document.getElementById("comparatorDivider"),
    comparatorAfter: document.getElementById("comparatorAfter"),
    sliderModeSelect: document.getElementById("sliderModeSelect"),

    // Attribution & Error
    imgHeatmapLarge: document.getElementById("imgHeatmapLarge"),
    imgErrorMapLarge: document.getElementById("imgErrorMapLarge"),
    errorLegendCard: document.getElementById("errorLegendCard"),

    // Metrics Telemetry
    warHeroVal: document.getElementById("warHeroVal"),
    warHeroSub: document.getElementById("warHeroSub"),
    distFillWet: document.getElementById("distFillWet"),
    wetPixelsVal: document.getElementById("wetPixelsVal"),
    totalPixelsVal: document.getElementById("totalPixelsVal"),
    absorptionHeightVal: document.getElementById("absorptionHeightVal"),
    noiseFilteredVal: document.getElementById("noiseFilteredVal"),

    gtSection: document.getElementById("gtSection"),
    warGtVal: document.getElementById("warGtVal"),
    warErrVal: document.getElementById("warErrVal"),
    diceVal: document.getElementById("diceVal"),
    iouVal: document.getElementById("iouVal"),
    precisionVal: document.getElementById("precisionVal"),
    recallVal: document.getElementById("recallVal"),

    latencyPreproc: document.getElementById("latencyPreproc"),
    latencyInference: document.getElementById("latencyInference"),
    latencyPhysics: document.getElementById("latencyPhysics"),

    // Loading
    loadingOverlay: document.getElementById("loadingOverlay"),
    loadingMessage: document.getElementById("loadingMessage"),

    // Export
    btnExportJson: document.getElementById("btnExportJson"),
  };

  // State
  let currentResult = null;
  let customImageFile = null;
  let customMaskFile = null;
  let isDraggingSlider = false;

  // Initialize
  initApp();

  async function initApp() {
    setupEventListeners();
    setupSliderInteraction();

    try {
      showLoading("Connecting to U-Net++ XAI server...");
      const status = await API.getStatus();
      renderStatus(status);

      const vData = await API.getVideos();
      populateVideos(vData.videos);

      // Default benchmark run: video 21 - frame 040 (or first frame available)
      await selectDefaultBenchmark();
    } catch (err) {
      console.error("Initialization error:", err);
      alert(`Backend connection warning: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  function renderStatus(status) {
    if (el.statusDot) el.statusDot.className = "indicator-dot";
    if (el.devicePill) {
      el.devicePill.textContent = status.cuda_available ? `GPU: ${status.device.toUpperCase()}` : "DEVICE: CPU";
      el.devicePill.className = "telemetry-pill status-online";
    }
    if (el.modelPill && status.model_metadata) {
      const bestIoU = status.model_metadata.best_val_iou ? ` (Val IoU ${status.model_metadata.best_val_iou.toFixed(3)})` : "";
      el.modelPill.textContent = `${status.model_metadata.architecture} [${status.model_metadata.encoder}]${bestIoU}`;
    }
  }

  function populateVideos(videos) {
    el.videoSelect.innerHTML = "";
    videos.forEach(v => {
      const opt = document.createElement("option");
      opt.value = v.video;
      const tag = v.is_test_benchmark ? "[TEST BENCHMARK]" : `[${v.split.toUpperCase()}]`;
      opt.textContent = `${tag} ${v.video} (${v.frame_count} frames)`;
      el.videoSelect.appendChild(opt);
    });

    // Populate quick benchmark chips
    const presets = [
      { vid: "video 21", stem: "ezgif-frame-040", label: "Video 21 - Frame 040 (Test)" },
      { vid: "video 1", stem: "ezgif-frame-001", label: "Video 1 - Frame 001 (Test)" },
      { vid: "video 24", stem: "ezgif-frame-064", label: "Video 24 - Frame 064 (Test)" },
    ];

    el.benchmarkChips.innerHTML = "";
    presets.forEach(p => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "benchmark-chip";
      btn.textContent = p.label;
      btn.dataset.video = p.vid;
      btn.dataset.stem = p.stem;
      btn.addEventListener("click", () => {
        el.sourceModeBenchmark.click();
        el.videoSelect.value = p.vid;
        onVideoChanged(p.vid).then(() => {
          el.frameSelect.value = p.stem;
          runCurrentAnalysis();
        });
      });
      el.benchmarkChips.appendChild(btn);
    });
  }

  async function selectDefaultBenchmark() {
    // Select video 21 if available
    const hasVid21 = Array.from(el.videoSelect.options).some(o => o.value === "video 21");
    const targetVid = hasVid21 ? "video 21" : el.videoSelect.options[0]?.value;

    if (targetVid) {
      el.videoSelect.value = targetVid;
      await onVideoChanged(targetVid);

      // Frame 040
      const hasFrame40 = Array.from(el.frameSelect.options).some(o => o.value.includes("040"));
      if (hasFrame40) {
        const opt = Array.from(el.frameSelect.options).find(o => o.value.includes("040"));
        el.frameSelect.value = opt.value;
      }
      await runCurrentAnalysis();
    }
  }

  async function onVideoChanged(video) {
    el.frameSelect.innerHTML = "<option>Loading frames...</option>";
    try {
      const res = await API.getFrames(video);
      el.frameSelect.innerHTML = "";
      res.frames.forEach(f => {
        const opt = document.createElement("option");
        opt.value = f.stem;
        const gtTag = f.has_mask ? " [GT Mask ✓]" : "";
        opt.textContent = `${f.stem}${gtTag}`;
        el.frameSelect.appendChild(opt);
      });
    } catch (err) {
      el.frameSelect.innerHTML = "<option>Error loading frames</option>";
      console.error(err);
    }
  }

  function setupEventListeners() {
    // Mode toggles
    el.sourceModeBenchmark.addEventListener("click", () => {
      el.sourceModeBenchmark.classList.add("active");
      el.sourceModeUpload.classList.remove("active");
      el.benchmarkSection.style.display = "block";
      el.uploadSection.style.display = "none";
    });

    el.sourceModeUpload.addEventListener("click", () => {
      el.sourceModeUpload.classList.add("active");
      el.sourceModeBenchmark.classList.remove("active");
      el.benchmarkSection.style.display = "none";
      el.uploadSection.style.display = "block";
    });

    el.videoSelect.addEventListener("change", () => onVideoChanged(el.videoSelect.value));

    // Sliders
    el.thresholdSlider.addEventListener("input", () => {
      el.thresholdVal.textContent = parseFloat(el.thresholdSlider.value).toFixed(2);
    });
    el.alphaSlider.addEventListener("input", () => {
      el.alphaVal.textContent = parseFloat(el.alphaSlider.value).toFixed(2);
    });

    // Run Button
    el.btnRunAnalysis.addEventListener("click", runCurrentAnalysis);

    // Viewport Tabs
    const tabs = [
      { btn: el.tabGrid, view: el.viewGridContainer },
      { btn: el.tabSlider, view: el.viewSliderContainer },
      { btn: el.tabHeatmap, view: el.viewHeatmapContainer },
      { btn: el.tabError, view: el.viewErrorContainer }
    ];

    tabs.forEach(({ btn, view }) => {
      btn.addEventListener("click", () => {
        tabs.forEach(t => {
          t.btn.classList.remove("active");
          t.view.style.display = "none";
        });
        btn.classList.add("active");
        view.style.display = "flex";
      });
    });

    // Slider Mode Switcher
    el.sliderModeSelect.addEventListener("change", updateComparatorImages);

    // Drag and drop for custom uploads
    setupDropzone(el.dropzone, el.fileInput, (file) => {
      customImageFile = file;
      el.dropzoneText.textContent = `Selected: ${file.name}`;
    });

    // Optional GT Mask dropzone
    const maskDropzone = document.getElementById("maskDropzone");
    if (maskDropzone && el.maskFileInput) {
      setupDropzone(maskDropzone, el.maskFileInput, (file) => {
        customMaskFile = file;
        if (el.maskDropzoneText) el.maskDropzoneText.textContent = `Selected GT: ${file.name}`;
      });
    }

    // Export JSON
    el.btnExportJson.addEventListener("click", () => {
      if (!currentResult) return;
      const exportData = {
        metadata: currentResult.metadata,
        specimen: currentResult.specimen_info,
        war_analysis: currentResult.war_analysis,
        ground_truth_evaluation: currentResult.ground_truth_evaluation,
        timestamp: new Date().toISOString()
      };
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `sorptivity_analysis_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  function setupDropzone(zone, input, onSelect) {
    zone.addEventListener("click", () => input.click());
    input.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        onSelect(e.target.files[0]);
      }
    });

    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });

    ["dragleave", "dragend"].forEach(ev => {
      zone.addEventListener(ev, () => zone.classList.remove("dragover"));
    });

    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onSelect(e.dataTransfer.files[0]);
      }
    });
  }

  async function runCurrentAnalysis() {
    showLoading("Running U-Net++ segmentation & Grad-CAM++ inference...");

    const isBenchmark = el.sourceModeBenchmark.classList.contains("active");
    const payloadParams = {
      target_layer: el.targetLayerSelect.value,
      threshold: parseFloat(el.thresholdSlider.value),
      apply_physics: el.applyPhysicsCheck.checked,
      bottom_margin: parseInt(el.bottomMarginInput.value) || 32,
      colormap: el.colormapSelect.value,
      heatmap_alpha: parseFloat(el.alphaSlider.value),
      include_simulated_augmentation: el.includeAugCheck.checked,
    };

    try {
      let result;
      if (isBenchmark) {
        const video = el.videoSelect.value;
        const stem = el.frameSelect.value;
        if (!video || !stem) throw new Error("Select a video and frame from the benchmark library.");

        result = await API.analyzeBenchmark({
          video,
          stem,
          ...payloadParams
        });
      } else {
        if (!customImageFile) throw new Error("Please select or drop an image file to analyze.");
        const fd = new FormData();
        fd.append("image", customImageFile);
        if (customMaskFile) fd.append("gt_mask", customMaskFile);
        Object.entries(payloadParams).forEach(([k, v]) => fd.append(k, v));

        result = await API.analyzeUpload(fd);
      }

      currentResult = result;
      renderAnalysis(result);
    } catch (err) {
      console.error("Analysis failure:", err);
      alert(`Analysis Failed: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  function renderAnalysis(data) {
    const { war_analysis, ground_truth_evaluation, images, metadata } = data;

    // Latency
    if (el.latencyPill) {
      el.latencyPill.textContent = `${metadata.total_latency_ms} ms`;
    }
    if (el.latencyPreproc) el.latencyPreproc.textContent = `${metadata.breakdown_latency_ms.preprocessing} ms`;
    if (el.latencyInference) el.latencyInference.textContent = `${metadata.breakdown_latency_ms.inference_and_xai} ms`;
    if (el.latencyPhysics) el.latencyPhysics.textContent = `${metadata.breakdown_latency_ms.physics} ms`;

    // WAR Hero
    const warPercent = war_analysis.war_physics_percent;
    el.warHeroVal.textContent = `${warPercent.toFixed(2)}%`;
    el.warHeroSub.textContent = `RAW: ${war_analysis.war_raw_percent.toFixed(2)}% | PHYS: ${warPercent.toFixed(2)}%`;

    // Distribution Bar
    el.distFillWet.style.width = `${Math.min(100, Math.max(0, warPercent))}%`;
    el.wetPixelsVal.textContent = war_analysis.physics_wet_pixels.toLocaleString();
    el.totalPixelsVal.textContent = war_analysis.total_specimen_pixels.toLocaleString();
    el.absorptionHeightVal.textContent = `${war_analysis.absorption_height_px} px`;
    el.noiseFilteredVal.textContent = war_analysis.filtered_floating_noise_pixels.toLocaleString();

    // Ground Truth Evaluation
    if (ground_truth_evaluation) {
      el.gtSection.style.display = "block";
      el.warGtVal.textContent = `${(ground_truth_evaluation.war_gt * 100).toFixed(2)}%`;
      el.warErrVal.textContent = `${(ground_truth_evaluation.war_abs_error_phys * 100).toFixed(2)}%`;
      el.diceVal.textContent = ground_truth_evaluation.phys.dice.toFixed(4);
      el.iouVal.textContent = ground_truth_evaluation.phys.iou.toFixed(4);
      el.precisionVal.textContent = ground_truth_evaluation.phys.precision.toFixed(4);
      el.recallVal.textContent = ground_truth_evaluation.phys.recall.toFixed(4);

      // Show error tab and render image
      el.tabError.style.display = "inline-block";
      if (images.error_map) {
        el.imgErrorMapLarge.src = images.error_map;
      }
    } else {
      el.gtSection.style.display = "none";
      el.tabError.style.display = "none";
    }

    // Grid Images
    el.imgInput.src = images.input_image;
    el.imgRawMask.src = images.raw_mask;
    el.imgPhysOverlay.src = images.physics_overlay;
    el.imgGradcam.src = images.gradcam_blended;

    // Large Heatmap
    el.imgHeatmapLarge.src = images.gradcam_blended;

    // Update Comparator
    updateComparatorImages();
  }

  function updateComparatorImages() {
    if (!currentResult) return;
    const { images } = currentResult;
    const mode = el.sliderModeSelect.value;

    const labelLeft = document.getElementById("labelBefore");
    const labelRight = document.getElementById("labelAfter");

    if (mode === "input_vs_physics") {
      el.comparatorBeforeImg.src = images.input_image;
      el.comparatorAfterImg.src = images.physics_overlay;
      if (labelLeft) labelLeft.textContent = "Input Specimen";
      if (labelRight) labelRight.textContent = "Physics Segmentation";
    } else if (mode === "input_vs_gradcam") {
      el.comparatorBeforeImg.src = images.input_image;
      el.comparatorAfterImg.src = images.gradcam_blended;
      if (labelLeft) labelLeft.textContent = "Input Specimen";
      if (labelRight) labelRight.textContent = "Grad-CAM++ Attribution";
    } else if (mode === "raw_vs_physics") {
      el.comparatorBeforeImg.src = images.raw_overlay;
      el.comparatorAfterImg.src = images.physics_overlay;
      if (labelLeft) labelLeft.textContent = "Raw U-Net++";
      if (labelRight) labelRight.textContent = "Physics Enforced";
    } else if (mode === "gt_vs_prediction" && images.ground_truth_mask) {
      el.comparatorBeforeImg.src = images.ground_truth_mask;
      el.comparatorAfterImg.src = images.physics_mask;
      if (labelLeft) labelLeft.textContent = "Ground Truth";
      if (labelRight) labelRight.textContent = "U-Net++ Prediction";
    }
  }

  // Interactive Split Slider Logic
  function setupSliderInteraction() {
    const container = el.comparatorContainer;
    const divider = el.comparatorDivider;
    const after = el.comparatorAfter;

    function setSliderPosition(x) {
      const rect = container.getBoundingClientRect();
      let percent = ((x - rect.left) / rect.width) * 100;
      percent = Math.max(0, Math.min(100, percent));

      divider.style.left = `${percent}%`;
      after.style.clipPath = `polygon(${percent}% 0, 100% 0, 100% 100%, ${percent}% 100%)`;
    }

    container.addEventListener("mousedown", (e) => {
      isDraggingSlider = true;
      setSliderPosition(e.clientX);
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDraggingSlider) return;
      setSliderPosition(e.clientX);
    });

    window.addEventListener("mouseup", () => {
      isDraggingSlider = false;
    });

    // Touch support
    container.addEventListener("touchstart", (e) => {
      if (e.touches.length > 0) {
        isDraggingSlider = true;
        setSliderPosition(e.touches[0].clientX);
      }
    });

    window.addEventListener("touchmove", (e) => {
      if (!isDraggingSlider || e.touches.length === 0) return;
      setSliderPosition(e.touches[0].clientX);
    });

    window.addEventListener("touchend", () => {
      isDraggingSlider = false;
    });
  }

  function showLoading(msg) {
    if (el.loadingMessage) el.loadingMessage.textContent = msg || "Processing...";
    if (el.loadingOverlay) el.loadingOverlay.style.display = "flex";
  }

  function hideLoading() {
    if (el.loadingOverlay) el.loadingOverlay.style.display = "none";
  }
});
