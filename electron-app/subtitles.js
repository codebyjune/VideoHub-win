const path = require("path");
const fs = require("fs");
const { spawn, spawnSync } = require("child_process");
const {
  ROOT_DIR,
  FFMPEG,
  looksLikeReasoning,
  getIsCancelled,
  setCurrentProcess,
  sendLog,
  sendProgress,
} = require("./shared");

function parseSrt(content) {
  const normalized = content.replace(/\r\n/g, "\n");
  const blocks = normalized.split(/\n\n+/).filter((b) => b.trim());
  return blocks
    .map((block) => {
      const lines = block.split("\n").filter((l) => l.trim() !== "");
      if (
        lines.length >= 3 &&
        /^\d+$/.test(lines[0].trim()) &&
        /-->/.test(lines[1])
      ) {
        return {
          seq: lines[0].trim(),
          timestamp: lines[1].trim(),
          text: lines.slice(2).join(" ").trim(),
        };
      }
      return null;
    })
    .filter(Boolean);
}

function isValidSrt(srtPath) {
  if (!fs.existsSync(srtPath)) return false;
  try {
    const content = fs.readFileSync(srtPath, "utf-8");
    return parseSrt(content).length > 0;
  } catch {
    return false;
  }
}

/**
 * 判断已存在的翻译字幕是否「可用」。
 *
 * 除了结构有效，还要确认里面没有混入模型的推理文本 —— 早期版本在
 * max_tokens 被 thinking 耗尽时会把推理当成译文写进文件，这种文件虽然
 * 结构合法，但内容是垃圾。检测到就返回 false，让上层重新翻译。
 */
function isUsableTranslation(srtPath) {
  if (!isValidSrt(srtPath)) return false;
  try {
    const entries = parseSrt(fs.readFileSync(srtPath, "utf-8"));
    const bad = entries.filter((e) => looksLikeReasoning(e.text));
    if (bad.length > 0) {
      sendLog(`  ⚠ 已有译文字幕含 ${bad.length}/${entries.length} 条推理垃圾，将重新翻译`);
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

function buildBilingualSrt(entries, translatedLines) {
  const result = [];
  for (let j = 0; j < entries.length; j++) {
    const engText = entries[j].text;
    const tgtText = j < translatedLines.length ? translatedLines[j] : "";
    result.push(
      [entries[j].seq, entries[j].timestamp, engText, tgtText].join("\n"),
    );
  }
  return result.join("\n\n") + "\n";
}

function buildMonolingualSrt(entries, translatedLines) {
  const result = [];
  for (let j = 0; j < entries.length; j++) {
    const text = j < translatedLines.length ? translatedLines[j] : "";
    result.push([entries[j].seq, entries[j].timestamp, text].join("\n"));
  }
  return result.join("\n\n") + "\n";
}

function getBurnSubtitlePath(srtPath, mode) {
  const baseName = srtPath.replace(/\.srt$/, "");
  const suffixMap = {
    en: ".srt",
    zh: ".zh-only.srt",
    "en-zh": ".en-zh.srt",
    tr: ".tr-only.srt",
    "en-tr": ".en-tr.srt",
  };
  return baseName + (suffixMap[mode] || ".srt");
}

/**
 * 把路径转成可安全放进 ffmpeg 滤镜参数的字符串。
 *
 * 在 filtergraph 里，滤镜参数以 ':' 分隔；Windows 盘符（C:\）里的冒号会被
 * 误当成参数分隔符，因此必须转义。由于整体已用单引号包裹，只需处理
 * '\' 和 ':' 以及单引号本身，空格无需转义。
 */
function escapeFilterPath(p) {
  return p
    .replace(/\\/g, "/")     // Windows 反斜杠 → 正斜杠
    .replace(/:/g, "\\:")    // 盘符冒号转义
    .replace(/'/g, "\\'");   // 单引号转义
}

// ── 烧录编码器选择 ────────────────────────────────────────
// 字幕烧录必须重编码。默认走 CPU 的 libx264（慢、吃满 CPU）；有 N 卡时改用
// NVENC 硬件编码可快数倍、CPU 占用大降。可用环境变量强制指定：
//   set VIDEOSUB_BURN_ENCODER=h264_nvenc    （或 libx264 / hevc_nvenc ...）
const ENCODER_EXTRA_ARGS = {
  libx264: [],
  libx265: [],
  h264_nvenc: ["-preset", "p4", "-cq", "24"],
  hevc_nvenc: ["-preset", "p4", "-cq", "26"],
  h264_qsv: ["-global_quality", "24"],
  hevc_qsv: ["-global_quality", "26"],
  h264_amf: ["-quality", "balanced", "-qp_i", "24", "-qp_p", "24"],
  hevc_amf: ["-quality", "balanced", "-qp_i", "26", "-qp_p", "26"],
};

const HARDWARE_ENCODERS = new Set([
  "h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_amf", "hevc_amf",
]);

// ── 烧录字体 ──────────────────────────────────────────────
// 内置字体放在 ROOT_DIR/fonts 下，通过 subtitles 滤镜的 fontsdir 直接加载，
// 无需用户把字体装到系统里。fontName 必须与字体内部的族名一致（不是文件名）。
const BURN_FONT_DIR = path.join(ROOT_DIR, "fonts");
const BURN_FONT_NAME = "LXGW WenKai Mono";

// 只在这些容器里替换编码器：webm 只认 VP8/VP9/AV1，塞不进 H.264
const ENCODER_CONTAINERS = new Set([".mp4", ".mkv", ".mov", ".avi"]);

let preferredEncoder; // undefined = 尚未探测

/** 真的用 ffmpeg 编一小段，确认编码器可用（比只查 -encoders 可靠，能发现驱动问题）。 */
function encoderWorks(name) {
  try {
    const r = spawnSync(FFMPEG, [
      "-hide_banner", "-loglevel", "error",
      // 用 320x240：部分编码器（如 NVENC）有最小分辨率限制，太小会被误判为不可用
      "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=15",
      "-c:v", name, "-pix_fmt", "yuv420p", "-f", "null", "-",
    ], { timeout: 10000, windowsHide: true });
    return r.status === 0;
  } catch {
    return false;
  }
}

/** 决定本次会话用哪个编码器（结果缓存，只探测一次）。 */
function resolveBurnEncoder() {
  if (preferredEncoder !== undefined) return preferredEncoder;

  const forced = (process.env.VIDEOSUB_BURN_ENCODER || "").trim();
  if (forced) {
    if (HARDWARE_ENCODERS.has(forced) && !encoderWorks(forced)) {
      sendLog(`  ⚠ 指定的编码器 ${forced} 不可用，回退 libx264`);
      preferredEncoder = "libx264";
    } else {
      sendLog(`  🎬 烧录编码器: ${forced}（来自 VIDEOSUB_BURN_ENCODER）`);
      preferredEncoder = forced;
    }
    return preferredEncoder;
  }

  if (encoderWorks("h264_nvenc")) {
    sendLog("  🚀 烧录编码器: h264_nvenc（NVIDIA GPU 硬件编码）");
    preferredEncoder = "h264_nvenc";
  } else {
    preferredEncoder = "libx264";
  }
  return preferredEncoder;
}

function burnSubtitlesIntoVideo(videoPath, srtPath) {
  return new Promise((resolve, reject) => {
    const ext = path.extname(videoPath);
    const outputPath = videoPath.replace(ext, `_subtitled${ext}`);

    if (fs.existsSync(outputPath)) {
      try {
        const srtMtime = fs.statSync(srtPath).mtimeMs;
        const outMtime = fs.statSync(outputPath).mtimeMs;
        // 只比 mtime 是不够的：若成品是用「含推理垃圾」的旧字幕烧出来的，
        // 即使它比字幕新也必须重烧，否则硬字幕会一直带着垃圾。
        if (outMtime >= srtMtime && isUsableTranslation(srtPath)) {
          sendLog(`  ✓ 已存在嵌入字幕的视频，跳过烧录: ${path.basename(outputPath)}`);
          sendProgress("burn", 100);
          resolve(outputPath);
          return;
        }
        if (outMtime >= srtMtime) {
          sendLog(`  ⚠ 已有成品但来源字幕不可信，重新烧录: ${path.basename(outputPath)}`);
        }
      } catch {}
    }

    // 把字幕复制到 downloads/ 下一个纯 ASCII 临时名：
    //   - Windows 上 fs.symlinkSync 需要管理员或开发者模式，普通用户会失败，故用复制
    //   - 中文/空格/特殊字符路径在 libass 滤镜里容易出问题，用纯 ASCII 名最稳
    const tmpDir = path.join(ROOT_DIR, "downloads");
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });
    const tmpSrt = path.join(tmpDir, `_sub_tmp_${Date.now()}_${process.pid}.srt`);
    try {
      fs.copyFileSync(srtPath, tmpSrt);
    } catch (e) {
      reject(new Error(`创建临时字幕文件失败: ${e.message}`));
      return;
    }

    const filterPath = escapeFilterPath(tmpSrt);
    // 若 fonts/ 目录存在，就用其中的 LXGW WenKai Mono；否则回退 libass 默认字体
    const hasBurnFont = fs.existsSync(BURN_FONT_DIR);
    const fontOpts = hasBurnFont
      ? `:fontsdir='${escapeFilterPath(BURN_FONT_DIR)}'`
      : "";
    const fontNameStyle = hasBurnFont ? `FontName=${BURN_FONT_NAME},` : "";
    const filter =
      `subtitles='${filterPath}'${fontOpts}:force_style='${fontNameStyle}FontSize=20,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,BorderStyle=1,Outline=1,Shadow=1'`;

    const encoder = ENCODER_CONTAINERS.has(ext.toLowerCase()) ? resolveBurnEncoder() : null;

    // 跑一次 ffmpeg；退出码 0 视为成功
    const runBurn = (enc) => new Promise((res, rej) => {
      if (getIsCancelled()) { rej(new Error("任务已取消")); return; }

      if (fs.existsSync(outputPath)) {
        try { fs.unlinkSync(outputPath); } catch {}
      }

      const args = ["-i", videoPath, "-vf", filter, "-c:a", "copy"];
      if (enc) args.push("-c:v", enc, ...(ENCODER_EXTRA_ARGS[enc] || []));
      args.push("-y", outputPath);

      sendLog(`  ffmpeg ${args.join(" ")}`);

      const ffmpeg = spawn(FFMPEG, args, {
        cwd: ROOT_DIR,
        windowsHide: true,
      });
      setCurrentProcess(ffmpeg);

      let lastErrLine = "";
      ffmpeg.stderr.on("data", (data) => {
        const text = data.toString().trim();
        if (text) {
          lastErrLine = text;
          sendLog(`  [FFmpeg] ${text}`);
        }
      });

      ffmpeg.on("close", (code) => {
        setCurrentProcess(null);
        if (getIsCancelled()) { rej(new Error("任务已取消")); return; }
        if (code === 0) { res(); return; }
        rej(new Error(`FFmpeg 退出码: ${code}\n${lastErrLine}`));
      });

      ffmpeg.on("error", (err) => {
        setCurrentProcess(null);
        rej(new Error(`启动 FFmpeg 失败: ${err.message}\n请确认 bin/ffmpeg.exe 存在或系统已安装 ffmpeg`));
      });
    });

    (async () => {
      try {
        try {
          await runBurn(encoder);
        } catch (err) {
          // 非 CPU 编码失败 → 回退 libx264 重试，并把本会话编码器降级，避免每集都白试
          if (!getIsCancelled() && encoder && encoder !== "libx264") {
            sendLog(`  ⚠ 硬件编码失败（${String(err.message).split("\n")[0]}），回退 libx264 重试...`);
            preferredEncoder = "libx264";
            await runBurn("libx264");
          } else {
            throw err;
          }
        }

        try { fs.unlinkSync(tmpSrt); } catch {}
        sendProgress("burn", 100);
        sendLog(`  ✓ 完成: ${path.basename(outputPath)}`);
        resolve(outputPath);
      } catch (err) {
        try { fs.unlinkSync(tmpSrt); } catch {}
        reject(err);
      }
    })();
  });
}

module.exports = {
  parseSrt,
  isValidSrt,
  isUsableTranslation,
  buildBilingualSrt,
  buildMonolingualSrt,
  getBurnSubtitlePath,
  escapeFilterPath,
  burnSubtitlesIntoVideo,
};
