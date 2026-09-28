const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const {
  ROOT_DIR,
  DOWNLOADS_DIR,
  VENV_DIR,
  VENV_PYTHON,
  PYTHON_SCRIPT,
  isWin,
  buildPythonEnv,
  getIsCancelled,
  setCurrentProcess,
  sendLog,
  sendProgress,
} = require("./shared");

function transcribe(videoPath, modelSize) {
  return new Promise((resolve, reject) => {
    sendLog("");
    sendLog("═══════════════════════════════════");
    sendLog("▶ 步骤 2：faster-whisper 转录英文字幕");
    sendLog(`  模型: ${modelSize}`);
    sendLog(`  文件: ${path.basename(videoPath)}`);
    sendLog("═══════════════════════════════════");

    const pythonExe = isWin
      ? path.join(VENV_DIR, "Scripts", "python.exe")
      : path.join(VENV_DIR, "bin", "python");

    if (!fs.existsSync(pythonExe)) {
      reject(new Error(
        `未找到 Python 虚拟环境: ${pythonExe}\n` +
        `请先在项目根目录运行: uv sync  (或 python -m venv .venv 后 pip install faster-whisper)`
      ));
      return;
    }

    const env = buildPythonEnv();

    if (getIsCancelled()) throw new Error("任务已取消");

    const proc = spawn(pythonExe, ["-u", PYTHON_SCRIPT, videoPath, modelSize], {
      cwd: ROOT_DIR,
      windowsHide: true,
      env,
    });
    setCurrentProcess(proc);

    let lastLog = "";

    proc.stdout.on("data", (data) => {
      const text = data.toString().trim();
      if (text) {
        lastLog = text;
        sendLog(`  ${text}`);
        if (/GPU 加速已启用|cuda/i.test(text)) sendProgress("transcribe", 20);
        if (/模型加载完成/.test(text)) sendProgress("transcribe", 30);
        const pct = text.match(/转录进度\s*(\d+)%/);
        if (pct) sendProgress("transcribe", Math.min(30 + Number(pct[1]) * 0.6, 95));
      }
    });

    proc.stderr.on("data", (data) => {
      const text = data.toString().trim();
      if (text) sendLog(`  ${text}`);
    });

    proc.on("close", (code) => {
      setCurrentProcess(null);

      if (getIsCancelled()) {
        reject(new Error("任务已取消"));
        return;
      }

      const srtMatch = lastLog.match(/SRT_OUTPUT:(.+)/);
      let srtPath;

      if (srtMatch) {
        srtPath = srtMatch[1].trim();
      } else {
        srtPath = videoPath.replace(/\.[^.]+$/, ".srt");
      }

      if (fs.existsSync(srtPath)) {
        sendProgress("transcribe", 100);
        sendLog(`  ✓ 字幕文件: ${path.basename(srtPath)}`);
        resolve(srtPath);
      } else if (code === 0) {
        const altPath = path.join(
          DOWNLOADS_DIR,
          path.basename(videoPath).replace(/\.[^.]+$/, ".srt"),
        );
        if (fs.existsSync(altPath)) {
          resolve(altPath);
        } else {
          reject(new Error("转录完成但未生成 SRT 文件"));
        }
      } else {
        reject(new Error(`转录失败，退出码: ${code}`));
      }
    });

    proc.on("error", (err) => {
      setCurrentProcess(null);
      reject(new Error(`启动失败: ${err.message}\n请确认虚拟环境已安装 faster-whisper`));
    });
  });
}

module.exports = { transcribe };
