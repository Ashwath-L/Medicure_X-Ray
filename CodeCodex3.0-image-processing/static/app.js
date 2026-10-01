/**
 * RadPulse Clinical Intelligence Console - Controller
 * Connects medical PACS viewport, drag-and-drop ingestion,
 * sample showcase, Grad-CAM attention visualizer, and 5-fold ensemble telemetry.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentTask = "xray"; // 'xray' | 'ct'
  let currentImageSrc = null;
  let currentHeatmapSrc = null;
  let currentFile = null;
  let currentSamplePath = null;
  let activeViewMode = "raw"; // 'raw' | 'heatmap'
  let isNegativeInverted = false;
  let currentPredictionData = null;

  // DOM Elements
  const btnModalityXray = document.getElementById("btn-modality-xray");
  const btnModalityCt = document.getElementById("btn-modality-ct");
  const tabBtnConsole = document.getElementById("tab-btn-console");
  const tabBtnMetrics = document.getElementById("tab-btn-metrics");
  const viewConsole = document.getElementById("view-console");
  const viewMetrics = document.getElementById("view-metrics");

  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("file-input");
  const sampleContainer = document.getElementById("sample-container");
  const toggleEnsemble = document.getElementById("toggle-ensemble");
  const toggleClahe = document.getElementById("toggle-clahe");

  const viewportEmptyState = document.getElementById("viewport-empty-state");
  const imageWrapper = document.getElementById("image-wrapper");
  const displayImage = document.getElementById("display-image");
  const loadingOverlay = document.getElementById("loading-overlay");
  const activeFilename = document.getElementById("active-filename");
  const hudModality = document.getElementById("hud-modality");

  const btnViewRaw = document.getElementById("btn-view-raw");
  const btnViewHeatmap = document.getElementById("btn-view-heatmap");
  const btnInvert = document.getElementById("btn-invert");
  const btnResetZoom = document.getElementById("btn-reset-zoom");

  const sliderBrightness = document.getElementById("slider-brightness");
  const sliderContrast = document.getElementById("slider-contrast");

  const verdictCard = document.getElementById("verdict-card");
  const verdictClass = document.getElementById("verdict-class");
  const verdictConfidence = document.getElementById("verdict-confidence");
  const confidenceBarFill = document.getElementById("confidence-bar-fill");
  const class0Name = document.getElementById("class-0-name");
  const class0Pct = document.getElementById("class-0-pct");
  const class0Bar = document.getElementById("class-0-bar");
  const class1Name = document.getElementById("class-1-name");
  const class1Pct = document.getElementById("class-1-pct");
  const class1Bar = document.getElementById("class-1-bar");
  const foldVotesContainer = document.getElementById("fold-votes-container");
  const btnExportReport = document.getElementById("btn-export-report");

  const xrayCmImg = document.getElementById("xray-cm-img");
  const ctCmImg = document.getElementById("ct-cm-img");

  // ==========================================
  // Navigation & Modality Switching
  // ==========================================
  function setTask(task) {
    currentTask = task;
    if (task === "xray") {
      btnModalityXray.classList.add("active");
      btnModalityCt.classList.remove("active");
      hudModality.textContent = "CHEST_XRAY";
      class0Name.textContent = "NORMAL";
      class1Name.textContent = "PNEUMONIA";
    } else {
      btnModalityCt.classList.add("active");
      btnModalityXray.classList.remove("active");
      hudModality.textContent = "CT_TOMOGRAPHY";
      class0Name.textContent = "NO_CONTRAST";
      class1Name.textContent = "CONTRAST";
    }
    loadSamples();
  }

  btnModalityXray.addEventListener("click", () => setTask("xray"));
  btnModalityCt.addEventListener("click", () => setTask("ct"));

  tabBtnConsole.addEventListener("click", () => {
    tabBtnConsole.classList.add("active");
    tabBtnMetrics.classList.remove("active");
    viewConsole.classList.add("active");
    viewMetrics.classList.remove("active");
  });

  tabBtnMetrics.addEventListener("click", () => {
    tabBtnMetrics.classList.add("active");
    tabBtnConsole.classList.remove("active");
    viewMetrics.classList.add("active");
    viewConsole.classList.remove("active");
    loadMetrics();
  });

  // ==========================================
  // Sample Scans Loader
  // ==========================================
  async function loadSamples() {
    sampleContainer.innerHTML = '<div class="sample-skeleton">Loading benchmark studies...</div>';
    try {
      const res = await fetch("/api/samples");
      const samples = await res.json();
      const filtered = samples.filter(s => s.modality === currentTask);

      sampleContainer.innerHTML = "";
      filtered.forEach(sample => {
        const card = document.createElement("div");
        card.className = "sample-card";
        card.innerHTML = `
          <img src="/api/sample-image?path=${encodeURIComponent(sample.path)}" class="sample-thumb" alt="${sample.title}">
          <div class="sample-details">
            <div class="sample-title">${sample.title}</div>
            <div class="sample-sub">${sample.filename}</div>
          </div>
        `;
        card.addEventListener("click", () => {
          document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
          card.classList.add("active");
          runSamplePrediction(sample.path, sample.filename);
        });
        sampleContainer.appendChild(card);
      });
    } catch (err) {
      sampleContainer.innerHTML = '<div class="empty-telemetry">Failed to load samples</div>';
    }
  }

  // ==========================================
  // Image Ingestion: Drag & Drop / File Input
  // ==========================================
  dropArea.addEventListener("click", () => fileInput.click());

  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
  });

  dropArea.addEventListener("dragleave", () => dropArea.classList.remove("dragover"));

  dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    currentFile = file;
    currentSamplePath = null;
    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    executeInference(file, null, file.name);
  }

  function runSamplePrediction(path, filename) {
    currentFile = null;
    currentSamplePath = path;
    executeInference(null, path, filename);
  }

  // ==========================================
  // Inference Execution API Caller
  // ==========================================
  async function executeInference(fileObj, samplePathStr, filename) {
    activeFilename.textContent = filename;
    loadingOverlay.classList.add("active");
    viewportEmptyState.style.display = "none";

    const formData = new FormData();
    formData.append("task", currentTask);
    formData.append("use_ensemble", toggleEnsemble.checked);

    if (fileObj) {
      formData.append("file", fileObj);
    } else if (samplePathStr) {
      formData.append("sample_path", samplePathStr);
    }

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Inference failed: ${await response.text()}`);
      }

      const data = await response.json();
      currentPredictionData = data;
      renderPrediction(data);
    } catch (err) {
      alert("Error during model evaluation: " + err.message);
    } finally {
      loadingOverlay.classList.remove("active");
    }
  }

  // ==========================================
  // Diagnostic Rendering & HUD Updates
  // ==========================================
  function renderPrediction(data) {
    currentImageSrc = data.preprocessed_image;
    currentHeatmapSrc = data.heatmap_image;

    // Viewport image
    displayImage.src = activeViewMode === "heatmap" ? currentHeatmapSrc : currentImageSrc;
    imageWrapper.style.display = "flex";

    // Verdict card
    verdictClass.textContent = data.predicted_class;
    const isPositive = (data.predicted_class === "PNEUMONIA" || data.predicted_class === "CONTRAST");
    verdictClass.className = "verdict-class " + (isPositive ? "status-positive" : "status-negative");

    const confPct = (data.confidence * 100).toFixed(1);
    verdictConfidence.textContent = `${confPct}%`;
    confidenceBarFill.style.width = `${confPct}%`;
    confidenceBarFill.style.background = isPositive ? "var(--accent-red)" : "var(--accent-green)";

    // Probability breakdown
    const keys = Object.keys(data.probabilities);
    class0Name.textContent = keys[0];
    const p0 = (data.probabilities[keys[0]] * 100).toFixed(1);
    class0Pct.textContent = `${p0}%`;
    class0Bar.style.width = `${p0}%`;

    class1Name.textContent = keys[1];
    const p1 = (data.probabilities[keys[1]] * 100).toFixed(1);
    class1Pct.textContent = `${p1}%`;
    class1Bar.style.width = `${p1}%`;

    // 5-Fold Ensemble Breakdown
    foldVotesContainer.innerHTML = "";
    if (data.fold_breakdown && data.fold_breakdown.length > 0) {
      data.fold_breakdown.forEach(v => {
        const item = document.createElement("div");
        item.className = "fold-vote-item";
        item.innerHTML = `
          <span class="fold-name">Fold ${v.fold} Model</span>
          <span class="fold-decision ${v.prediction}">${v.prediction} (${(v.confidence * 100).toFixed(0)}%)</span>
        `;
        foldVotesContainer.appendChild(item);
      });
    } else {
      foldVotesContainer.innerHTML = '<div class="empty-telemetry">Single checkpoint inference</div>';
    }

    btnExportReport.disabled = false;
  }

  // ==========================================
  // Radiologist Viewport Image Adjustments
  // ==========================================
  function applyImageFilters() {
    const b = sliderBrightness.value;
    const c = sliderContrast.value;
    const inv = isNegativeInverted ? 1 : 0;
    displayImage.style.filter = `brightness(${b}%) contrast(${c}%) invert(${inv})`;
  }

  sliderBrightness.addEventListener("input", applyImageFilters);
  sliderContrast.addEventListener("input", applyImageFilters);

  btnInvert.addEventListener("click", () => {
    isNegativeInverted = !isNegativeInverted;
    btnInvert.classList.toggle("active", isNegativeInverted);
    applyImageFilters();
  });

  btnResetZoom.addEventListener("click", () => {
    sliderBrightness.value = 100;
    sliderContrast.value = 100;
    isNegativeInverted = false;
    btnInvert.classList.remove("active");
    applyImageFilters();
  });

  btnViewRaw.addEventListener("click", () => {
    activeViewMode = "raw";
    btnViewRaw.classList.add("active");
    btnViewHeatmap.classList.remove("active");
    if (currentImageSrc) displayImage.src = currentImageSrc;
  });

  btnViewHeatmap.addEventListener("click", () => {
    if (!currentHeatmapSrc) return;
    activeViewMode = "heatmap";
    btnViewHeatmap.classList.add("active");
    btnViewRaw.classList.remove("active");
    displayImage.src = currentHeatmapSrc;
  });

  // ==========================================
  // Export Clinical Summary Report
  // ==========================================
  btnExportReport.addEventListener("click", () => {
    if (!currentPredictionData) return;
    const reportText = `
================================================================================
RADPULSE MEDICAL INTELLIGENCE — CLINICAL DIAGNOSTIC REPORT
Generated: ${new Date().toLocaleString()}
================================================================================
PATIENT STUDY: ${currentPredictionData.filename}
MODALITY     : ${currentTask.toUpperCase()}
CLASSIFIER   : DenseNet-121 (Selective Freeze & 5-Fold Stratified Ensemble)
PREDICTION   : ${currentPredictionData.predicted_class}
CONFIDENCE   : ${(currentPredictionData.confidence * 100).toFixed(2)}%

PROBABILITY BREAKDOWN:
${Object.entries(currentPredictionData.probabilities).map(([k, v]) => `  - ${k.padEnd(16)}: ${(v * 100).toFixed(2)}%`).join("\n")}

ENSEMBLE VOTING TELEMETRY (5-FOLD):
${currentPredictionData.fold_breakdown.map(f => `  - Fold ${f.fold}: ${f.prediction} (${(f.confidence * 100).toFixed(1)}%)`).join("\n")}
================================================================================
NOTICE: This report is generated by RadPulse AI for clinical decision support.
`;
    const blob = new Blob([reportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `RadPulse_Report_${currentPredictionData.filename}.txt`;
    a.click();
  });

  // ==========================================
  // Metrics & Benchmark Inspector
  // ==========================================
  async function loadMetrics() {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();

      if (data.xray.confusion_matrix_b64) {
        xrayCmImg.src = `data:image/png;base64,${data.xray.confusion_matrix_b64}`;
      }
      if (data.ct.confusion_matrix_b64) {
        ctCmImg.src = `data:image/png;base64,${data.ct.confusion_matrix_b64}`;
      }
    } catch (err) {
      console.error("Failed to load metrics:", err);
    }
  }

  // Initial Load
  setTask("xray");
});
