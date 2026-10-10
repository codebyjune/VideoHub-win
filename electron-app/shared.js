const path = require("path");
const fs = require("fs");

let electronApp;
try {
  electronApp = require("electron").app;
} catch {}

const isPackaged = electronApp ? electronApp.isPackaged : false;
const ROOT_DIR = isPackaged
  ? process.resourcesPath
  : path.join(__dirname, "..");
const isMac = process.platform === "darwin";
const isWin = process.platform === "win32";

// 优先使用项目内 bin/ 的可执行文件，找不到则回退系统 PATH
function resolveTool(name) {
  const exe = isWin ? `${name}.exe` : name;
  const local = path.join(ROOT_DIR, "bin", exe);
  if (fs.existsSync(local)) return local;
  return exe; // 交给 PATH
}

// macOS 上优先用 homebrew 的 ffmpeg-full（带 libass，才能烧字幕）
const FFMPEG_FULL = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg";
const FFMPEG =
  isMac && fs.existsSync(FFMPEG_FULL) ? FFMPEG_FULL : resolveTool("ffmpeg");
const FFPROBE = resolveTool("ffprobe");
const YTDLP = resolveTool("yt-dlp");
const PYTHON_SCRIPT = path.join(ROOT_DIR, "main.py");
const DOWNLOADS_DIR = path.join(ROOT_DIR, "downloads");
const VENV_DIR = path.join(ROOT_DIR, ".venv");

// 虚拟环境里的 Python 解释器（Windows 为 Scripts/python.exe，POSIX 为 bin/python）
const VENV_BIN = isWin ? path.join(VENV_DIR, "Scripts") : path.join(VENV_DIR, "bin");
const VENV_PYTHON = path.join(VENV_BIN, isWin ? "python.exe" : "python");

// pip 安装的 nvidia-*-cu12 包内 DLL 目录。CTranslate2 不会自动搜索这些路径，
// 缺少时 CUDA 会静默回退 CPU（报 cublas64_12.dll not found）。
function cudaDllDirs() {
  if (!isWin) return [];
  const nvRoot = path.join(VENV_DIR, "Lib", "site-packages", "nvidia");
  if (!fs.existsSync(nvRoot)) return [];
  const dirs = [];
  for (const sub of fs.readdirSync(nvRoot)) {
    const binDir = path.join(nvRoot, sub, "bin");
    if (fs.existsSync(binDir)) dirs.push(binDir);
  }
  return dirs;
}

// 运行 Python 子进程时统一的环境变量
function buildPythonEnv(extra = {}) {
  const dirs = cudaDllDirs();
  return {
    ...process.env,
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
    PYTHONUNBUFFERED: "1",
    VIRTUAL_ENV: VENV_DIR,
    // Windows 上 HuggingFace 默认用符号链接，需管理员权限；关掉改为复制
    HF_HUB_DISABLE_SYMLINKS: "1",
    HF_HUB_DISABLE_SYMLINKS_WARNING: "1",
    HF_ENDPOINT: process.env.HF_ENDPOINT || "https://hf-mirror.com",
    PATH: [VENV_BIN, ...dirs, process.env.PATH].join(path.delimiter),
    ...extra,
  };
}

let mainWindow = null;
let currentProcess = null;
let isCancelled = false;
let isSeriesMode = false;
let isBusy = false;

// 当前活跃窗口上下文：本地视频窗口流程启动时设置，
// 使所有下游 sendLog/sendProgress 自动路由到对应窗口，无需逐函数透传。
// main 流程保持 currentWindowId = null（走 mainWindow 默认分支）。
let currentWindowId = null;

function setCurrentWindowId(id) {
  currentWindowId = id || null;
}

function getCurrentWindowId() {
  return currentWindowId;
}

// 活跃窗口注册表：windowId -> BrowserWindow
// main 窗口的 windowId 固定为 "main"，本地视频窗口用其 Electron 内置 id："local:<win.id>"
const activeWindows = new Map();

function setMainWindow(win) {
  mainWindow = win;
  if (win) {
    activeWindows.set("main", win);
    win.on("closed", () => {
      activeWindows.delete("main");
      if (mainWindow === win) mainWindow = null;
    });
  }
}

function registerLocalWindow(win) {
  const id = `local:${win.id}`;
  activeWindows.set(id, win);
  win.on("closed", () => activeWindows.delete(id));
  return id;
}

function setIsCancelled(v) {
  isCancelled = v;
}

function getIsCancelled() {
  return isCancelled;
}

function setIsSeriesMode(v) {
  isSeriesMode = v;
}

function setCurrentProcess(p) {
  currentProcess = p;
}

function getCurrentProcess() {
  return currentProcess;
}

function getIsBusy() {
  return isBusy;
}

function setIsBusy(v) {
  isBusy = v;
}

/**
 * 从 LLM 响应中取出真正的回答文本。
 *
 * ⚠ 只认 `text` 块，绝不回退到 `thinking` 块。
 *
 * 推理模型（deepseek-flash / reasoner 等）会先输出 thinking 再输出 text。
 * 当 max_tokens 不足时，预算会被 thinking 吃光，返回的 content 只有
 * thinking、没有 text —— 此时若把 thinking 当答案返回，整段「模型内心独白」
 * 就会被当成译文写进字幕（表现为字幕里出现 "We need answer only Chinese..."
 * 这类英文推理文字）。正确做法是判定为失败，由上层加大预算重试。
 */
function extractResponseText(response) {
  if (!response.content || !Array.isArray(response.content) || response.content.length === 0) {
    throw new Error(`Unexpected response structure: ${JSON.stringify(response).slice(0, 200)}`);
  }

  const textBlock = response.content.find((b) => b.type === "text" && b.text);
  if (textBlock) return textBlock.text;

  const hasThinking = response.content.some((b) => b.type === "thinking");
  const types = response.content.map((b) => b.type).join(", ");
  if (hasThinking) {
    // 打上机器可读标记：调用方（translation.js）据此识别「思考吃光预算」这种
    // 可恢复故障，进而抬高 max_tokens 或拆小批次重试，而不是直接把整个任务判死。
    const err = new Error(
      `模型只返回了思考内容(thinking)，没有返回最终答案(text)。` +
      `通常因为 max_tokens 被思考过程耗尽（stop_reason=${response.stop_reason || "?"}）。`
    );
    err.thinkingOnly = true;
    err.stopReason = response.stop_reason;
    throw err;
  }
  throw new Error(`响应中没有文本内容，block 类型: ${types}`);
}

/**
 * 判断一段文本是否更像「模型的推理过程」而非译文。
 *
 * 这是最后一道防线：万一模型把推理写进了 text 块，也能识别出来并丢弃，
 * 避免污染字幕。
 */
function looksLikeReasoning(text) {
  if (!text) return false;
  const t = text.trim();

  // 典型英文推理开头。
  // ⚠ 不要把 ^we need / ^let me / ^i need to 这类当规则 ——
  //   教学类字幕里 "we need to put do or does plus not" 是完全正常的英文内容，
  //   过宽的规则会误杀真实字幕（实测雅思语法课第 82 条踩过这个坑）。
  //   因此每一条都要求命中"谈论翻译任务本身"的元语言。
  const starts = [
    /^we need (to )?(translate|answer|output|determine|infer|parse|craft)/i,
    /^need (to )?(translate|answer|output)/i,
    /^the user (wants|is asking|asked|is requesting)/i,
    /^let me (translate|look|analyze|parse|craft|think|check)/i,
    /^i (need to|should|will|shall) (translate|answer|output|determine|craft)/i,
    /^looking at the (context|input|subtitle|line)/i,
    /^first,? (i|we) (need|should|will)/i,
    /^hmm,? (the|this|let)/i,
    /^context:/i,
    /^previous subtitles/i,
    /^the line to translate/i,
    /^let's (craft|parse|think|translate)/i,
    /^output only/i,
    /^here'?s? (the|my) (translation|reasoning|analysis)/i,
  ];
  if (starts.some((re) => re.test(t))) return true;

  // 出现明显的元语言（在讨论"翻译"本身），且整段较长 → 判定为推理
  const meta = [
    /translate (this|the) (english )?(subtitle )?line/i,
    /output only/i,
    /the user (wants|is asking|asked)/i,
    /(into|to) (natural,? )?(fluent )?(simplified )?chinese\b/i,
    /we need (answer|translate|output)/i,
    /no explanations?,? no/i,
  ];
  const hits = meta.filter((re) => re.test(t)).length;
  return hits >= 2 && t.length > 120;
}

/**
 * 从路径里解析出「集号」，用于系列模式按集数排序。
 *
 * 兼容两种形态：
 *   1. 原始下载文件 —— 标题形如 "<系列名> p73 <集名>.mp4"
 *   2. 归档后的文件 —— "<系列名>/73. <集名>/video.mp4"（父目录以集号开头）
 *
 * 解析不出时返回 null。
 */
function episodeNumberFromPath(p) {
  const base = path.basename(p);

  // 1) 原始下载：标题里带 " pNN "
  let m = base.match(/\sp(\d+)(?=[\s.]|$)/i);
  if (m) return parseInt(m[1], 10);

  // 2) 归档目录：父目录形如 "73. Java 2D animation/"
  const parent = path.basename(path.dirname(p));
  m = parent.match(/^(\d+)[\.\s]/);
  if (m) return parseInt(m[1], 10);

  // 3) 兜底：文件名本身以数字开头
  m = base.match(/^(\d+)[\.\s]/);
  if (m) return parseInt(m[1], 10);

  return null;
}

/**
 * 系列模式的处理顺序：按集号升序（第 1 集在前），保证从第一集开始，
 * 而不是被字符串排序把 "73." 排到 "【系列】 p01" 前面导致从中间/末尾开始。
 * 集号相同时回退到自然序比较。
 */
function compareByEpisode(a, b) {
  const na = episodeNumberFromPath(a);
  const nb = episodeNumberFromPath(b);
  if (na !== null && nb !== null && na !== nb) return na - nb;
  if (na !== null && nb === null) return -1;
  if (na === null && nb !== null) return 1;
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

function isFinalVideo(filename) {
  return (
    /\.(mp4|mkv|webm|mov|avi)$/i.test(filename) &&
    // 排除 yt-dlp 的中间格式文件（title.f137.mp4 等）
    !/\.f\d+\.(mp4|mkv|webm|mov|avi)$/i.test(filename) &&
    // 排除我们自己烧录出来的成品（video_subtitled.mp4）。
    // 否则下载的「兜底扫描」会把烧录成品当成源视频，再烧一遍 →
    // 产生 video_subtitled_subtitled.mp4 这种垃圾。
    !/_subtitled\.(mp4|mkv|webm|mov|avi)$/i.test(filename)
  );
}

function collectMediaFiles(dir) {
  const out = [];
  const walk = (d) => {
    if (!fs.existsSync(d)) return;
    for (const entry of fs.readdirSync(d)) {
      const full = path.join(d, entry);
      let st;
      try { st = fs.statSync(full); } catch { continue; }
      if (st.isDirectory()) { walk(full); continue; }
      if (/\.(mp4|mkv|webm|mov|avi|srt)$/i.test(entry) && !/\.f\d+\.(mp4|mkv|webm|mov|avi)$/i.test(entry)) {
        out.push(full);
      }
    }
  };
  walk(dir);
  return out;
}

function cleanEmptyDirs(dir) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir)) {
    const full = path.join(dir, entry);
    let st;
    try { st = fs.statSync(full); } catch { continue; }
    if (st.isDirectory()) {
      cleanEmptyDirs(full);
      try { if (fs.readdirSync(full).length === 0) fs.rmdirSync(full); } catch {}
    }
  }
}

function send(channel, data, windowId) {
  if (windowId) {
    const win = activeWindows.get(windowId);
    if (win && !win.isDestroyed()) {
      win.webContents.send(channel, data);
    }
    return;
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, data);
  }
}

function sendLog(message, windowId) {
  send("log", message, windowId !== undefined ? windowId : currentWindowId);
}

function sendProgress(step, percent, windowId) {
  if (isSeriesMode && step !== "series") return;
  send("progress", { step, percent }, windowId !== undefined ? windowId : currentWindowId);
}

function sendStatus(status, windowId) {
  send("status", status, windowId !== undefined ? windowId : currentWindowId);
}

module.exports = {
  ROOT_DIR,
  DOWNLOADS_DIR,
  VENV_DIR,
  VENV_BIN,
  VENV_PYTHON,
  PYTHON_SCRIPT,
  FFMPEG,
  FFPROBE,
  YTDLP,
  isMac,
  isWin,
  isPackaged,
  buildPythonEnv,
  setMainWindow,
  registerLocalWindow,
  setCurrentWindowId,
  getCurrentWindowId,
  setIsCancelled,
  getIsCancelled,
  setIsSeriesMode,
  setCurrentProcess,
  getCurrentProcess,
  getIsBusy,
  setIsBusy,
  extractResponseText,
  looksLikeReasoning,
  episodeNumberFromPath,
  compareByEpisode,
  isFinalVideo,
  collectMediaFiles,
  cleanEmptyDirs,
  sendLog,
  sendProgress,
  sendStatus,
};
