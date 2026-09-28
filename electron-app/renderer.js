const $ = (sel) => document.querySelector(sel);

const startBtn = $("#startBtn");
const cancelBtn = $("#cancelBtn");
const clearBtn = $("#clearBtn");
const clearDlBtn = $("#clearDlBtn");
const logBox = $("#logBox");
const progressBar = $("#progressBar");
const statusBadge = $("#statusBadge");

// Inline progress-step labels (below bar)
const steps = {
  download: document.querySelectorAll(".step")[0],
  transcribe: document.querySelectorAll(".step")[1],
  translate: document.querySelectorAll(".step")[2],
  burn: document.querySelectorAll(".step")[3],
};

// Sidebar step items
const sidebarSteps = {
  download: $("#stepDownload"),
  transcribe: $("#stepTranscribe"),
  translate: $("#stepTranslate"),
  burn: $("#stepBurn"),
};

let isRunning = false;

// ── Theme Toggle ────────────────────────────────────────
const themeToggle = $("#themeToggle");
const themeLabel = $("#themeLabel");
const themeIcon = $("#themeIcon");

const MOON_ICON = `<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>`;
const SUN_ICON = `<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>`;

function getTheme() {
  const stored = localStorage.getItem("theme");
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  if (theme === "dark") {
    themeIcon.innerHTML = `<circle cx="12" cy="12" r="4"/><path d="M12 3a1 1 0 010 2 7 7 0 00-7 7 1 1 0 01-2 0 9 9 0 019-9z"/>`;
    themeLabel.textContent = "Light";
  } else {
    themeIcon.innerHTML = MOON_ICON;
    themeLabel.textContent = "Dark";
  }
}

applyTheme(getTheme());

// ── 恢复已保存的 API Key / Base URL / URL / Cookies ───
const savedApiKey = localStorage.getItem("apiKey");
const savedBaseUrl = localStorage.getItem("baseUrl");
const savedLlmModel = localStorage.getItem("llmModel");
const savedModelSize = localStorage.getItem("modelSize");
const savedTargetLang = localStorage.getItem("targetLang");
const savedUrl = localStorage.getItem("url");
const savedCookiesFile = localStorage.getItem("cookiesFile");
const savedBurnMode = localStorage.getItem("burnSubtitleMode");
const savedDownloadSeries = localStorage.getItem("downloadSeries");
if (savedApiKey) $("#apiKey").value = savedApiKey;
if (savedBaseUrl) $("#baseUrl").value = savedBaseUrl;
if (savedLlmModel) $("#llmModel").value = savedLlmModel;
if (savedModelSize) $("#modelSize").value = savedModelSize;
if (savedTargetLang) $("#targetLang").value = savedTargetLang;
if (savedUrl) $("#url").value = savedUrl;
if (savedCookiesFile) $("#cookiesFile").value = savedCookiesFile;
if (savedBurnMode) $("#burnSubtitleMode").value = savedBurnMode;
if (savedDownloadSeries === "true") $("#downloadSeries").checked = true;

themeToggle.addEventListener("click", () => {
  const next = getTheme() === "dark" ? "light" : "dark";
  localStorage.setItem("theme", next);
  applyTheme(next);
});

// ── Log Output ──────────────────────────────────────────
function appendLog(msg) {
  logBox.textContent += msg + "\n";
  logBox.scrollTop = logBox.scrollHeight;
}

// ── Highlight active step (both inline + sidebar) ───────
function setStepState(key, state) {
  // Inline progress labels
  if (steps[key]) {
    steps[key].classList.remove("active", "done");
    if (state) steps[key].classList.add(state);
  }
  // Sidebar items
  if (sidebarSteps[key]) {
    sidebarSteps[key].classList.remove("active", "done");
    if (state) sidebarSteps[key].classList.add(state);
  }
}

// ── IPC Listeners ───────────────────────────────────────
window.api.onLog((msg) => {
  appendLog(msg);
});

window.api.onProgress(({ step, percent }) => {
  // 系列模式：直接显示集数进度
  if (step === "series") {
    progressBar.style.width = Math.min(percent, 100) + "%";
    // 系列模式下所有步骤标记为进行中
    Object.keys(steps).forEach((k) => setStepState(k, null));
    setStepState("download", "active");
    setStepState("transcribe", "active");
    setStepState("translate", "active");
    if (percent === 100) {
      Object.keys(steps).forEach((k) => setStepState(k, "done"));
    }
    return;
  }

  const weights = { download: 0.25, transcribe: 0.25, translate: 0.25, burn: 0.25 };
  const offsets = { download: 0, transcribe: 25, translate: 50, burn: 75 };
  const overall = offsets[step] + (percent * weights[step]) / 100;
  progressBar.style.width = Math.min(overall, 100) + "%";

  // Highlight current step
  Object.keys(steps).forEach((k) => setStepState(k, null));
  setStepState(step, "active");
});

window.api.onStatus((status) => {
  statusBadge.className = "badge";
  switch (status) {
    case "running":
      statusBadge.textContent = "Running…";
      statusBadge.classList.add("running");
      startBtn.disabled = true;
      startBtn.style.display = "none";
      cancelBtn.style.display = "";
      break;
    case "done":
      statusBadge.textContent = "Done";
      statusBadge.classList.add("done");
      startBtn.disabled = false;
      startBtn.style.display = "";
      cancelBtn.style.display = "none";
      isRunning = false;
      progressBar.style.width = "100%";
      Object.keys(steps).forEach((k) => setStepState(k, "done"));
      break;
    case "error":
      statusBadge.textContent = "Error";
      statusBadge.classList.add("error");
      startBtn.disabled = false;
      startBtn.style.display = "";
      cancelBtn.style.display = "none";
      isRunning = false;
      break;
    case "cancelled":
      statusBadge.textContent = "Cancelled";
      statusBadge.classList.add("error");
      startBtn.disabled = false;
      startBtn.style.display = "";
      cancelBtn.style.display = "none";
      isRunning = false;
      progressBar.style.width = "0%";
      Object.keys(steps).forEach((k) => setStepState(k, null));
      break;
    default:
      statusBadge.textContent = "Ready";
      statusBadge.classList.add("idle");
      break;
  }
});

// ── Start Workflow ──────────────────────────────────────
startBtn.addEventListener("click", async () => {
  if (isRunning) return;

  const url = $("#url").value.trim();
  const modelSize = $("#modelSize").value;
  const targetLang = $("#targetLang").value;
  const apiKey = $("#apiKey").value.trim();
  const baseUrl = $("#baseUrl").value.trim();
  const llmModel = $("#llmModel").value.trim();
  const burnSubtitleMode = $("#burnSubtitleMode").value;
  const downloadSeries = $("#downloadSeries").checked;
  const cookiesFile = $("#cookiesFile").value.trim();

  if (!url) {
    appendLog("Please enter a video URL");
    return;
  }
  if (!apiKey) {
    appendLog("Please enter an LLM API Key");
    return;
  }

  localStorage.setItem("apiKey", apiKey);
  localStorage.setItem("baseUrl", baseUrl || "https://api.deepseek.com/anthropic");
  localStorage.setItem("llmModel", llmModel);
  localStorage.setItem("modelSize", modelSize);
  localStorage.setItem("targetLang", targetLang);
  localStorage.setItem("url", url);
  localStorage.setItem("burnSubtitleMode", burnSubtitleMode);
  localStorage.setItem("downloadSeries", downloadSeries);
  if (cookiesFile) localStorage.setItem("cookiesFile", cookiesFile);

  isRunning = true;
  logBox.textContent = "";
  progressBar.style.width = "0%";
  Object.keys(steps).forEach((k) => setStepState(k, null));

  appendLog("═══════════════════════════════════");
  appendLog("  VideoSub - Start");
  appendLog(`  URL: ${url}`);
  appendLog(`  Model: ${modelSize}`);
  appendLog("═══════════════════════════════════");

  const result = await window.api.startWorkflow({
    url,
    modelSize,
    targetLang,
    apiKey,
    baseUrl: baseUrl || "https://api.deepseek.com/anthropic",
    llmModel: llmModel || "deepseek-flash",
    burnSubtitleMode,
    cookiesFile,
    downloadSeries,
  });

  if (!result.success) {
    appendLog("");
    appendLog(`Failed: ${result.error}`);
  }
});

// ── Clear Downloads ─────────────────────────────────────
clearDlBtn.addEventListener("click", async () => {
  const count = await window.api.clearDownloads();
  if (count > 0) {
    appendLog(`Cleaned ${count} downloaded files`);
  }
});

// ── Clear Log ───────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  logBox.textContent = "";
  progressBar.style.width = "0%";
  Object.keys(steps).forEach((k) => {
    setStepState(k, null);
    if (steps[k]) {
      steps[k].classList.remove("active", "done");
    }
  });
  statusBadge.className = "badge idle";
  statusBadge.textContent = "Ready";
});

// ── Cancel Workflow ──────────────────────────────────────
cancelBtn.addEventListener("click", () => {
  appendLog("");
  appendLog("⏹ Cancelling...");
  window.api.cancelWorkflow();
});

// ── Series Mode Hint Toggle ───────────────────────────────
const downloadSeriesCheckbox = $("#downloadSeries");
const seriesHint = $("#seriesHint");
if (downloadSeriesCheckbox && seriesHint) {
  // 初始化显示状态
  seriesHint.style.display = downloadSeriesCheckbox.checked ? "block" : "none";
  downloadSeriesCheckbox.addEventListener("change", () => {
    seriesHint.style.display = downloadSeriesCheckbox.checked ? "block" : "none";
  });
}

// ── Browse Cookies File ──────────────────────────────────
$("#browseCookiesBtn").addEventListener("click", async () => {
  const filePath = await window.api.browseFile();
  if (filePath) {
    $("#cookiesFile").value = filePath;
    localStorage.setItem("cookiesFile", filePath);
  }
});

// ── Open Local Video Window ───────────────────────────────
$("#openLocalBtn").addEventListener("click", () => {
  window.api.openLocalWindow();
});
