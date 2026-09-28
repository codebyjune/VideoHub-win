const $ = (sel) => document.querySelector(sel);

const startBtn = $("#startBtn");
const cancelBtn = $("#cancelBtn");
const clearBtn = $("#clearBtn");
const browseBtn = $("#browseVideoBtn");
const logBox = $("#logBox");
const progressBar = $("#progressBar");
const statusBadge = $("#statusBadge");

const stepEls = document.querySelectorAll(".progress-steps .step");

let isRunning = false;

// ── 恢复已保存的设置 ─────────────────────────────────────
const savedKeys = ["apiKey", "baseUrl", "llmModel", "modelSize", "targetLang", "burnSubtitleMode"];
for (const k of savedKeys) {
  const v = localStorage.getItem(k);
  if (v && $(`#${k}`)) $(`#${k}`).value = v;
}

// ── 浏览视频文件 ─────────────────────────────────────────
browseBtn.addEventListener("click", async () => {
  const filePath = await window.api.browseVideo();
  if (filePath) {
    $("#videoPath").value = filePath;
  }
});

// ── Log ──────────────────────────────────────────────────
function appendLog(msg) {
  logBox.textContent += msg + "\n";
  logBox.scrollTop = logBox.scrollHeight;
}

// ── IPC Listeners ────────────────────────────────────────
window.api.onLog((msg) => appendLog(msg));

window.api.onProgress(({ step, percent }) => {
  const weights = { transcribe: 0.34, translate: 0.33, burn: 0.33 };
  const offsets = { transcribe: 0, translate: 34, burn: 67 };
  const overall = offsets[step] + (percent * weights[step]) / 100;
  progressBar.style.width = Math.min(overall, 100) + "%";

  stepEls.forEach((el) => el.classList.remove("active", "done"));
  const idx = { transcribe: 0, translate: 1, burn: 2 }[step];
  if (idx !== undefined) stepEls[idx].classList.add("active");
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
      stepEls.forEach((el) => { el.classList.remove("active"); el.classList.add("done"); });
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
      stepEls.forEach((el) => el.classList.remove("active", "done"));
      break;
    default:
      statusBadge.textContent = "Ready";
      statusBadge.classList.add("idle");
      break;
  }
});

// ── Start Workflow ───────────────────────────────────────
startBtn.addEventListener("click", async () => {
  if (isRunning) return;

  const videoPath = $("#videoPath").value.trim();
  const modelSize = $("#modelSize").value;
  const targetLang = $("#targetLang").value;
  const apiKey = $("#apiKey").value.trim();
  const baseUrl = $("#baseUrl").value.trim();
  const llmModel = $("#llmModel").value.trim();
  const burnSubtitleMode = $("#burnSubtitleMode").value;

  if (!videoPath) {
    appendLog("请先选择视频文件");
    return;
  }
  if (!apiKey) {
    appendLog("请输入 LLM API Key");
    return;
  }

  // 持久化设置
  localStorage.setItem("apiKey", apiKey);
  localStorage.setItem("baseUrl", baseUrl || "https://api.deepseek.com/anthropic");
  localStorage.setItem("llmModel", llmModel);
  localStorage.setItem("modelSize", modelSize);
  localStorage.setItem("targetLang", targetLang);
  localStorage.setItem("burnSubtitleMode", burnSubtitleMode);

  isRunning = true;
  logBox.textContent = "";
  progressBar.style.width = "0%";
  stepEls.forEach((el) => el.classList.remove("active", "done"));

  const result = await window.api.startLocalWorkflow({
    videoPath,
    modelSize,
    targetLang,
    apiKey,
    baseUrl: baseUrl || "https://api.deepseek.com/anthropic",
    llmModel: llmModel || "deepseek-flash",
    burnSubtitleMode,
  });

  if (!result.success) {
    isRunning = false;
    startBtn.disabled = false;
    startBtn.style.display = "";
    cancelBtn.style.display = "none";
    statusBadge.className = "badge error";
    statusBadge.textContent = "Error";
    appendLog("");
    appendLog(`失败: ${result.error}`);
  }
});

// ── Cancel ───────────────────────────────────────────────
cancelBtn.addEventListener("click", () => {
  appendLog("");
  appendLog("⏹ 正在取消...");
  window.api.cancelWorkflow();
});

// ── Clear Log ────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  logBox.textContent = "";
  progressBar.style.width = "0%";
  stepEls.forEach((el) => el.classList.remove("active", "done"));
  statusBadge.className = "badge idle";
  statusBadge.textContent = "Ready";
});
