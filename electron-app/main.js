const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const {
  DOWNLOADS_DIR,
  setMainWindow,
  registerLocalWindow,
  setCurrentWindowId,
  getIsBusy,
  setIsBusy,
  setIsCancelled,
  getIsCancelled,
  setIsSeriesMode,
  setCurrentProcess,
  getCurrentProcess,
  compareByEpisode,
  episodeNumberFromPath,
  collectMediaFiles,
  cleanEmptyDirs,
  sendLog,
  sendProgress,
  sendStatus,
} = require("./shared");
const { downloadVideo, organizeIntoSeriesDir } = require("./download");
const { transcribe } = require("./transcribe");
const { isValidSrt, isUsableTranslation, getBurnSubtitlePath, burnSubtitlesIntoVideo } = require("./subtitles");
const { generateSubtitleForBurn } = require("./translation");

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    title: "视频字幕工具",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  setMainWindow(mainWindow);
  mainWindow.loadFile("index.html");
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// ── 本地视频窗口（跳过下载，直接转录→翻译→烧录）──────────────
function createLocalVideoWindow() {
  const win = new BrowserWindow({
    width: 820,
    height: 720,
    title: "本地视频转录",
    webPreferences: {
      preload: path.join(__dirname, "preload-local.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  registerLocalWindow(win);
  win.loadFile("local.html");
  return win;
}

async function processSingleVideo(
  videoPath,
  modelSize,
  apiKey,
  baseUrl,
  llmModel,
  burnSubtitleMode,
  targetLang,
) {
  const expectedSrt = videoPath.replace(/\.[^.]+$/, ".srt");
  let srtPath;
  if (isValidSrt(expectedSrt)) {
    sendLog("");
    sendLog("  ✓ 已存在转录字幕，跳过转录");
    sendProgress("transcribe", 100);
    srtPath = expectedSrt;
  } else {
    srtPath = await transcribe(videoPath, modelSize);
  }

  const mode = burnSubtitleMode || "none";
  let resultPath;

  if (mode === "none") {
    const langSuffix = targetLang === "zh" ? ".zh-only.srt" : `.${targetLang}-only.srt`;
    const translatedPath = srtPath.replace(/\.srt$/, langSuffix);
    if (isUsableTranslation(translatedPath)) {
      sendLog("");
      sendLog("  ✓ 已存在翻译字幕，跳过翻译");
      sendProgress("translate", 100);
      resultPath = translatedPath;
    } else {
      resultPath = await generateSubtitleForBurn(
        srtPath, apiKey, baseUrl, llmModel || "deepseek-flash", targetLang,
      );
    }
  } else {
    const burnSrtPath = getBurnSubtitlePath(srtPath, mode);
    let finalBurnSrt;
    if (isUsableTranslation(burnSrtPath)) {
      sendLog("");
      sendLog("  ✓ 已存在对应模式字幕，跳过翻译");
      sendProgress("translate", 100);
      finalBurnSrt = burnSrtPath;
    } else {
      finalBurnSrt = await generateSubtitleForBurn(
        srtPath, apiKey, baseUrl, llmModel || "deepseek-flash", mode,
      );
    }
    sendLog("");
    sendLog("═══════════════════════════════════");
    sendLog("▶ 步骤 4：硬嵌入字幕到视频");
    sendLog(`  视频: ${path.basename(videoPath)}`);
    sendLog(`  字幕: ${path.basename(finalBurnSrt)}`);
    sendLog(`  模式: ${mode}`);
    sendLog("═══════════════════════════════════");

    const burnedVideoPath = await burnSubtitlesIntoVideo(videoPath, finalBurnSrt);
    sendLog(`  ✓ 字幕已嵌入视频: ${path.basename(burnedVideoPath)}`);
    resultPath = burnedVideoPath;
  }

  return resultPath;
}

ipcMain.on("cancel-workflow", () => {
  setIsCancelled(true);
  sendLog("⚠ 正在取消...");
  const proc = getCurrentProcess();
  if (proc) {
    try { proc.kill("SIGTERM"); } catch {}
    setCurrentProcess(null);
  }
});

ipcMain.handle(
  "start-workflow",
  async (event, {
    url, modelSize, targetLang, apiKey, baseUrl,
    llmModel, burnSubtitleMode, cookiesFile, downloadSeries,
  }) => {
    if (getIsBusy()) {
      return { success: false, error: "已有任务正在运行，请等待完成或取消后再试" };
    }
    setIsBusy(true);
    try {
    setIsCancelled(false);
    setIsSeriesMode(!!downloadSeries);
    setCurrentProcess(null);

    try {
      if (!url || !url.trim()) throw new Error("请输入视频链接");
      if (!apiKey || !apiKey.trim()) throw new Error("请输入 LLM API Key");

      sendStatus("running");

      if (downloadSeries) {
        setIsSeriesMode(false);
        const videoFiles = await downloadVideo(url, cookiesFile, true);
        setIsSeriesMode(true);

        if (!Array.isArray(videoFiles) || videoFiles.length === 0) {
          throw new Error("未找到下载的视频文件（可能是播放列表为空或 URL 不是播放列表链接）");
        }

        videoFiles.sort(compareByEpisode);

        // 打印实际处理顺序，方便确认是「从第 1 集开始」
        const seq = videoFiles.map((p) => episodeNumberFromPath(p));
        sendLog("");
        sendLog(`  📋 实际处理顺序（共 ${seq.length} 集）: ${
          seq.slice(0, 10).map((n) => (n === null ? "?" : n)).join(", ")
        }${seq.length > 10 ? ", …" : ""}`);

        const results = [];
        for (let i = 0; i < videoFiles.length; i++) {
          if (getIsCancelled()) {
            sendLog("");
            sendLog("⚠ 用户取消整个系列任务，停止处理");
            break;
          }

          const rawPath = videoFiles[i];
          const videoPath = organizeIntoSeriesDir(rawPath);
          sendLog("");
          sendLog(`═══════════════════════════════════`);
          sendLog(`  处理第 ${i + 1}/${videoFiles.length} 集`);
          sendLog(`  文件: ${path.basename(videoPath)}`);
          sendLog(`═══════════════════════════════════`);

          const doneProgress = videoFiles.length > 1
            ? Math.round((i / videoFiles.length) * 100) : 0;
          sendProgress("series", doneProgress);

          try {
            const result = await processSingleVideo(
              videoPath, modelSize, apiKey, baseUrl, llmModel, burnSubtitleMode, targetLang,
            );
            results.push(result);
          } catch (err) {
            if (err.message === "任务已取消") {
              sendLog(`  ⚠ 第 ${i + 1} 集已取消，跳过`);
              continue;
            }
            throw err;
          }
        }

        sendProgress("series", 100);
        sendStatus("done");
        sendLog("");
        sendLog("═══════════════════════════════════");
        const total = videoFiles.length;
        const skipped = total - results.length;
        if (skipped > 0) {
          sendLog(`⚠ 系列处理完成（已跳过 ${skipped} 集）`);
          sendLog(`  成功处理 ${results.length} / ${total} 集`);
        } else {
          sendLog("🎉 系列全部完成！");
          sendLog(`  共处理 ${results.length} 集`);
        }
        sendLog("═══════════════════════════════════");

        return { success: true, output: results };
      }

      let videoPath = await downloadVideo(url, cookiesFile, false);
      videoPath = organizeIntoSeriesDir(videoPath);

      const result = await processSingleVideo(
        videoPath, modelSize, apiKey, baseUrl, llmModel, burnSubtitleMode, targetLang,
      );

      sendStatus("done");
      sendLog("");
      sendLog("═══════════════════════════════════");
      sendLog("🎉 全部完成！");
      sendLog(`  输出文件: ${result}`);
      sendLog("═══════════════════════════════════");

      return { success: true, output: result };
    } catch (err) {
      if (err.message === "任务已取消") {
        sendStatus("cancelled");
        sendLog("");
        sendLog("⏹ 任务已取消");
        return { success: false, error: "任务已取消" };
      }

      sendStatus("error");
      sendLog("");
      sendLog(`❌ 错误: ${err.message}`);

      const files = collectMediaFiles(DOWNLOADS_DIR);

      if (files.length > 0) {
        const displayNames = files.map((f) => path.relative(DOWNLOADS_DIR, f));
        const { response } = await dialog.showMessageBox(mainWindow, {
          type: "warning",
          title: "下载/转录失败",
          message: `错误: ${err.message}`,
          detail: `downloads 目录中有 ${files.length} 个文件。是否删除所有已下载的视频和字幕，以便重新下载？\n\n(${displayNames.slice(0, 5).join(", ")}${files.length > 5 ? ` ...等 ${files.length} 个文件` : ""})`,
          buttons: ["删除并清理", "跳过"],
          defaultId: 0,
          cancelId: 1,
        });

        if (response === 0) {
          let deleted = 0;
          files.forEach((f) => { try { fs.unlinkSync(f); deleted++; } catch {} });
          cleanEmptyDirs(DOWNLOADS_DIR);
          sendLog(`  🗑 已清理 ${deleted} 个文件，请重试`);
        }
      }

      return { success: false, error: err.message };
    }
    } finally {
      setIsBusy(false);
    }
  },
);

ipcMain.handle("browse-file", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "选择 Cookies 文件",
    filters: [
      { name: "Text Files", extensions: ["txt"] },
      { name: "All Files", extensions: ["*"] },
    ],
    properties: ["openFile"],
  });
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle("select-output-dir", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  });
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle("clear-downloads", async () => {
  if (!fs.existsSync(DOWNLOADS_DIR)) return 0;

  const files = collectMediaFiles(DOWNLOADS_DIR);
  const displayNames = files.map((f) => path.relative(DOWNLOADS_DIR, f));

  const { response } = await dialog.showMessageBox(mainWindow, {
    type: "question",
    title: "清理下载文件",
    message: `确定删除 downloads 目录中的 ${files.length} 个文件吗？`,
    detail:
      displayNames.length > 0
        ? displayNames.slice(0, 8).join("\n") +
          (displayNames.length > 8 ? `\n...等 ${displayNames.length} 个文件` : "")
        : "没有可清理的文件",
    buttons: ["确认删除", "取消"],
    defaultId: 1,
    cancelId: 1,
  });

  if (response === 0) {
    let deleted = 0;
    files.forEach((f) => { try { fs.unlinkSync(f); deleted++; } catch {} });
    cleanEmptyDirs(DOWNLOADS_DIR);
    sendLog(`  🗑 已清理 ${deleted} 个文件`);
    return deleted;
  }
  return 0;
});

// ── 打开本地视频窗口 ──────────────────────────────────────
ipcMain.on("open-local-window", () => {
  createLocalVideoWindow();
});

// ── 浏览本地视频文件 ──────────────────────────────────────
ipcMain.handle("browse-video", async () => {
  const focused = BrowserWindow.getFocusedWindow() || mainWindow;
  const result = await dialog.showOpenDialog(focused, {
    title: "选择视频文件",
    filters: [
      { name: "Video Files", extensions: ["mp4", "mkv", "webm", "mov", "avi", "m4v", "flv"] },
      { name: "All Files", extensions: ["*"] },
    ],
    properties: ["openFile"],
  });
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

// ── 本地视频工作流：跳过下载，直接 转录→翻译→烧录 ───────────
ipcMain.handle(
  "start-local-workflow",
  async (event, {
    videoPath, modelSize, targetLang, apiKey, baseUrl,
    llmModel, burnSubtitleMode,
  }) => {
    if (getIsBusy()) {
      return { success: false, error: "已有任务正在运行，请等待完成或取消后再试" };
    }
    setIsBusy(true);
    setIsCancelled(false);
    setIsSeriesMode(false);
    setCurrentProcess(null);

    const senderWin = BrowserWindow.fromWebContents(event.sender);
    const winId = senderWin
      ? `local:${senderWin.id}`
      : null;
    if (winId) setCurrentWindowId(winId);

    try {
      if (!videoPath || !videoPath.trim()) throw new Error("请选择视频文件");
      if (!apiKey || !apiKey.trim()) throw new Error("请输入 LLM API Key");
      if (!fs.existsSync(videoPath)) throw new Error(`文件不存在: ${videoPath}`);

      sendStatus("running");
      sendLog("═══════════════════════════════════");
      sendLog("  本地视频转录 - 开始");
      sendLog(`  文件: ${path.basename(videoPath)}`);
      sendLog(`  模型: ${modelSize}`);
      sendLog("═══════════════════════════════════");

      const result = await processSingleVideo(
        videoPath, modelSize, apiKey, baseUrl, llmModel, burnSubtitleMode, targetLang,
      );

      sendStatus("done");
      sendLog("");
      sendLog("═══════════════════════════════════");
      sendLog("🎉 全部完成！");
      sendLog(`  输出文件: ${result}`);
      sendLog("═══════════════════════════════════");

      return { success: true, output: result };
    } catch (err) {
      if (err.message === "任务已取消") {
        sendStatus("cancelled");
        sendLog("");
        sendLog("⏹ 任务已取消");
        return { success: false, error: "任务已取消" };
      }
      sendStatus("error");
      sendLog("");
      sendLog(`❌ 错误: ${err.message}`);
      return { success: false, error: err.message };
    } finally {
      setCurrentWindowId(null);
      setIsBusy(false);
    }
  },
);
