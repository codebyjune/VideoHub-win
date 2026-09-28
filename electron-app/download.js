const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const {
  ROOT_DIR,
  DOWNLOADS_DIR,
  VENV_DIR,
  YTDLP,
  isMac,
  isWin,
  getIsCancelled,
  setCurrentProcess,
  compareByEpisode,
  isFinalVideo,
  sendLog,
  sendProgress,
} = require("./shared");

/**
 * 选择 yt-dlp 的启动方式。
 *
 * 优先用 bin/yt-dlp.exe（独立可执行，打包分发最省事）；
 * 如果不存在，则回退到虚拟环境里的 `python -m yt_dlp`
 * （打包时 .venv 会一起带上，所以这个回退在发行版里同样可用）。
 */
function resolveYtdlpCommand() {
  if (YTDLP && fs.existsSync(YTDLP)) {
    return { cmd: YTDLP, prefix: [], label: path.basename(YTDLP) };
  }
  const py = isWin
    ? path.join(VENV_DIR, "Scripts", "python.exe")
    : path.join(VENV_DIR, "bin", "python");
  return { cmd: py, prefix: ["-m", "yt_dlp"], label: "python -m yt_dlp" };
}

/**
 * 在指定目录里找归档后的成品视频（organizeIntoSeriesDir 会命名为 video.<ext>）。
 * 只认 `video.mp4` 这类，不认 `video_subtitled.mp4`。
 */
function findVideoInDir(dir) {
  if (!fs.existsSync(dir)) return null;
  for (const f of fs.readdirSync(dir)) {
    if (/^video\.(mp4|mkv|webm|mov|avi)$/i.test(f)) return path.join(dir, f);
  }
  return null;
}

/**
 * 归一化目录名，用于把 yt-dlp 给的原始 playlist_title 和磁盘上的实际目录名对上。
 *
 * yt-dlp 在 Windows 上清洗文件名时会把非法字符换成全角形式：
 *   > → ＞   < → ＜   : → ：   * → ＊   ? → ？   " → ＂   | → ｜   / → ／
 * 于是 "Java tutorial --> x" 落盘后变成 "Java tutorial --＞ x"。
 */
function normalizeName(s) {
  return String(s || "")
    .replace(/＞/g, ">").replace(/＜/g, "<").replace(/：/g, ":")
    .replace(/＊/g, "*").replace(/？/g, "?").replace(/＂/g, '"')
    .replace(/｜/g, "|").replace(/[／⁄]/g, "/")
    .replace(/\s+/g, " ")
    .replace(/[.\s]+$/, "")
    .trim()
    .toLowerCase();
}

/** 在 downloads 目录里按 playlist_title 找到对应的系列目录（找不到返回 null）。 */
function findSeriesDir(playlistTitle) {
  if (!playlistTitle || playlistTitle === "NA") return null;

  let dirs;
  try {
    dirs = fs.readdirSync(DOWNLOADS_DIR).filter((d) => {
      try { return fs.statSync(path.join(DOWNLOADS_DIR, d)).isDirectory(); } catch { return false; }
    });
  } catch { return null; }

  const want = normalizeName(playlistTitle);
  if (!want) return null;

  for (const d of dirs) if (normalizeName(d) === want) return path.join(DOWNLOADS_DIR, d);

  // 兜底：标题超过 150 字符时会被 .150s 截断，用前缀匹配
  if (want.length >= 10) {
    for (const d of dirs) {
      const n = normalizeName(d);
      if (n.length >= 10 && (want.startsWith(n) || n.startsWith(want))) {
        return path.join(DOWNLOADS_DIR, d);
      }
    }
  }
  return null;
}

/** 从标题里解析集号（形如 " p73 "）；拿不到返回 null。 */
function episodeNumFromTitle(title) {
  const m = String(title || "").match(/\sp(\d+)(?=[\s.]|$)/i);
  return m ? parseInt(m[1], 10) : null;
}

/**
 * 在系列目录里按「集号」找已归档的成品视频。
 *
 * 只按集号匹配、不比对集名：归档目录名同样可能被 yt-dlp 清洗/截断。
 */
function organizedVideoForEpisode(seriesDir, epNum) {
  if (!seriesDir || epNum === null || epNum === undefined) return null;
  const want = String(epNum).padStart(2, "0");

  let entries;
  try { entries = fs.readdirSync(seriesDir); } catch { return null; }

  for (const entry of entries) {
    const em = entry.match(/^(\d+)\b/);
    if (!em || String(em[1]).padStart(2, "0") !== want) continue;
    const full = path.join(seriesDir, entry);
    let st;
    try { st = fs.statSync(full); } catch { continue; }
    if (!st.isDirectory()) continue;
    const video = findVideoInDir(full);
    if (video) return video;
  }
  return null;
}

/**
 * 把待下载的集号压缩成 --playlist-items 能识别的区间串，如 "1-5,7,9-12"。
 * 避免集数多时命令行过长。
 */
function formatPlaylistItems(items) {
  const nums = [...new Set(items)].sort((a, b) => a - b);
  const parts = [];
  let start = null;
  let prev = null;
  for (const n of nums) {
    if (start === null) { start = n; prev = n; continue; }
    if (n === prev + 1) { prev = n; continue; }
    parts.push(start === prev ? `${start}` : `${start}-${prev}`);
    start = n;
    prev = n;
  }
  if (start !== null) parts.push(start === prev ? `${start}` : `${start}-${prev}`);
  return parts.join(",");
}

/**
 * 只列播放列表、不下载：拿到每一集的 playlist 序号与标题，
 * 用来判断哪些集已经在本地归档，避免重复下载同一集。
 * 任何失败都返回 []，由调用方回退到「全量下载」。
 */
function listPlaylistEntries(url, cookiesFile) {
  return new Promise((resolve) => {
    const args = [
      "--flat-playlist",
      "--ignore-errors",
      "--no-warnings",
      // 中文 Windows 上 yt-dlp 默认按 GBK 输出，中文会变成乱码；
      // 强制 UTF-8，否则下面的 playlist_title 根本匹配不上磁盘目录名。
      "--encoding", "utf-8",
      // 同时输出 playlist_title：B 站（多分 P）flat 模式下 title 是 "NA"，
      // 只能靠 playlist_title 定位系列目录、靠 playlist_index 匹配集号。
      "--print", "%(playlist_index)s\t%(playlist_title)s\t%(title)s",
    ];
    if (isWin || isMac) args.push("--js-runtimes", "node");
    if (cookiesFile && fs.existsSync(cookiesFile)) args.push("--cookies", cookiesFile);
    args.push(url);

    const env = { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" };
    const { cmd, prefix } = resolveYtdlpCommand();
    let child;
    try {
      child = spawn(cmd, [...prefix, ...args], { cwd: ROOT_DIR, windowsHide: true, env });
    } catch {
      resolve([]);
      return;
    }
    setCurrentProcess(child);

    let out = "";
    child.stdout.on("data", (d) => { out += d.toString(); });
    child.on("close", () => {
      setCurrentProcess(null);
      const entries = [];
      for (const line of out.split(/\r?\n/)) {
        const t = line.replace(/\s+$/, "");
        if (!t) continue;
        const cols = t.split("\t");
        if (cols.length < 2) continue;
        const index = parseInt(cols[0], 10);
        entries.push({
          index: Number.isFinite(index) ? index : null,
          playlistTitle: cols[1] ?? "",
          title: cols.length >= 3 ? cols.slice(2).join("\t") : "",
        });
      }
      resolve(entries);
    });
    child.on("error", () => { setCurrentProcess(null); resolve([]); });
  });
}

async function downloadVideo(url, cookiesFile, downloadSeries = false) {
  if (!fs.existsSync(DOWNLOADS_DIR)) {
    fs.mkdirSync(DOWNLOADS_DIR, { recursive: true });
  }

  sendLog("═══════════════════════════════════");
  sendLog("▶ 步骤 1：下载视频");
  sendLog(`  命令: yt-dlp -f "bestvideo+bestaudio/best" "${url}"`);
  if (cookiesFile) sendLog(`  Cookies: ${cookiesFile}`);
  if (downloadSeries) {
    sendLog(`  模式: 下载整个系列`);
  } else {
    sendLog(`  模式: 单集下载`);
  }
  sendLog("═══════════════════════════════════");

  if (getIsCancelled()) throw new Error("任务已取消");

  // 系列模式：先探明哪些集已经在本地归档，避免重复下载同一集。
  // 背景：文件会被 organizeIntoSeriesDir 从 yt-dlp 的输出路径搬进子目录，
  // 导致 --no-overwrites 认不出已下载的集数，第二次运行会白下载一遍。
  const alreadyOrganized = []; // 已归档的成品视频绝对路径
  let playlistItems = null;    // 仍需下载的 playlist 序号；null = 不限制（全量下载）

  // downloads 为空说明没有任何可复用的归档，省掉一次列表请求
  if (downloadSeries && fs.readdirSync(DOWNLOADS_DIR).length > 0) {
    const entries = await listPlaylistEntries(url, cookiesFile);
    if (getIsCancelled()) throw new Error("任务已取消");

    // 只有拿到「全部带序号」的条目才敢用 --playlist-items 精确挑集，
    // 否则宁可回退到全量下载，也不能漏集。
    const usable = entries.length > 0 && entries.every((e) => Number.isFinite(e.index));
    if (usable) {
      // 用 playlist_title 定位系列目录；B 站 flat 的 title 是 NA，但 playlist_title 可用
      const playlistTitle = entries.find((e) => e.playlistTitle && e.playlistTitle !== "NA")?.playlistTitle || "";
      const seriesDir = findSeriesDir(playlistTitle);
      if (seriesDir) sendLog(`  📁 系列目录: ${path.basename(seriesDir)}`);

      const missing = [];
      const seen = new Set();
      for (const e of entries) {
        // 集号优先取标题里的 " pNN "（YouTube 等）；拿不到就用 playlist_index（B 站 flat 标题为 NA）
        const epNum = episodeNumFromTitle(e.title) ?? e.index;
        const existing = organizedVideoForEpisode(seriesDir, epNum);
        if (existing) {
          if (!seen.has(existing)) { seen.add(existing); alreadyOrganized.push(existing); }
        } else {
          missing.push(e.index);
        }
      }
      playlistItems = missing;

      if (alreadyOrganized.length > 0) {
        sendLog(`  ⏩ 检测到 ${alreadyOrganized.length} 集已下载归档，跳过重复下载`);
      }
      if (missing.length === 0) {
        sendLog(`  ✓ 该系列 ${alreadyOrganized.length} 集全部已下载，无需重新下载`);
        alreadyOrganized.sort(compareByEpisode);
        return alreadyOrganized;
      }
    }
  }

  const outputTemplate = downloadSeries
    ? path.join(DOWNLOADS_DIR, "%(playlist_title).150s/%(title).150s.%(ext)s")
    : path.join(DOWNLOADS_DIR, "%(title).150s.%(ext)s");

  const args = [
    "-f", "bestvideo+bestaudio/best",
    "-o", outputTemplate,
    "--no-part",
    // 与列表接口保持一致：强制 UTF-8 输出，否则中文路径在 stdout 里是乱码，
    // 会导致 "Destination:" 解析失败（旧代码只能靠 mtime 兜底扫描救回来）。
    "--encoding", "utf-8",
    // 已存在同名成品时跳过下载。
    // yt-dlp 的 overwrites 默认是 True（见 downloader/common.py），
    // 不加这个参数的话，第二次运行会白下载一遍整个视频。
    // 加了之后 yt-dlp 会打印 "has already been downloaded"，下面的逻辑据此复用本地文件。
    "--no-overwrites",
  ];

  // Node 作为 JS 运行时用于解 YouTube 的 nsig（yt-dlp 需要）
  if (isWin || isMac) args.push("--js-runtimes", "node");

  if (!downloadSeries) args.push("--no-playlist");

  // 系列模式：只让 yt-dlp 下载缺失的集数，已归档的集数由 alreadyOrganized 补回结果
  if (downloadSeries && playlistItems) {
    args.push("--playlist-items", formatPlaylistItems(playlistItems));
  }

  if (cookiesFile && fs.existsSync(cookiesFile)) {
    args.push("--cookies", cookiesFile);
  }

  args.push(url);

  const ytdlpEnv = { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" };
  if (isMac) {
    const brewCert = "/opt/homebrew/opt/ca-certificates/share/cacert.pem";
    if (fs.existsSync(brewCert)) ytdlpEnv.SSL_CERT_FILE = brewCert;
  }

  if (getIsCancelled()) throw new Error("任务已取消");

  const { cmd, prefix, label } = resolveYtdlpCommand();
  sendLog(`  运行时: ${label}`);

  const ytdlp = spawn(cmd, [...prefix, ...args], {
    cwd: ROOT_DIR,
    windowsHide: true,
    env: ytdlpEnv,
  });
  setCurrentProcess(ytdlp);

  const stderrLines = [];
  const stdoutLines = [];
  const downloadedFiles = [];

  ytdlp.stdout.on("data", (data) => {
    const text = data.toString().trim();
    if (text) stdoutLines.push(text);
    sendLog(`  ${text}`);
  });

  ytdlp.stderr.on("data", (data) => {
    const text = data.toString().trim();
    if (text) {
      stderrLines.push(text);
      const pct = text.match(/(\d+\.\d+)%/);
      if (pct) sendProgress("download", parseFloat(pct[1]));
      sendLog(`  ${text}`);
    }
  });

  return new Promise((resolve, reject) => {
    ytdlp.on("close", (code) => {
      setCurrentProcess(null);

      if (getIsCancelled()) {
        reject(new Error("任务已取消"));
        return;
      }

      const fullOutput = [...stdoutLines, ...stderrLines].join("\n");

      if (downloadSeries) {
        const patterns = [
          /\[download\] Destination:\s*(.+?)$/gm,
          /\[download\]\s+(.+?)\s+has already been downloaded/g,
          /\[download\]\s+(.+?)\s+100% of/g,
          /\[Merger\] Merging formats into "(.+?)"/g,
        ];
        for (const regex of patterns) {
          let m;
          while ((m = regex.exec(fullOutput)) !== null) {
            const p = m[1].trim();
            if (fs.existsSync(p) && isFinalVideo(p)) downloadedFiles.push(p);
          }
        }

        // 补回因「已归档」而被 --playlist-items 排除的集数，
        // 保证它们仍进入后续转录 / 翻译 / 烧录流程（这些步骤会自行跳过已完成的部分）
        for (const p of alreadyOrganized) downloadedFiles.push(p);

        if (downloadedFiles.length === 0) {
          const now = Date.now();
          const walk = (dir) => {
            if (!fs.existsSync(dir)) return;
            for (const entry of fs.readdirSync(dir)) {
              const full = path.join(dir, entry);
              let st;
              try { st = fs.statSync(full); } catch { continue; }
              if (st.isDirectory()) { walk(full); continue; }
              if (!isFinalVideo(entry)) continue;
              if (now - st.mtimeMs > 5 * 60 * 1000) continue;
              downloadedFiles.push(full);
            }
          };
          walk(DOWNLOADS_DIR);
        }

        const uniqueFiles = [...new Set(downloadedFiles)];
        if (uniqueFiles.length > 0) {
          uniqueFiles.sort(compareByEpisode);
          sendLog(`  ✓ 系列下载完成，共 ${uniqueFiles.length} 个视频`);
          resolve(uniqueFiles);
          return;
        }

        reject(new Error("系列下载完成但未找到任何视频文件"));
        return;
      }

      const alreadyMsg = stderrLines.join("\n");
      const alreadyMatch = alreadyMsg.match(
        /\[download\]\s+(.+?)\s+has already been downloaded/,
      );
      if (alreadyMatch) {
        const cachedPath = alreadyMatch[1].trim();
        if (fs.existsSync(cachedPath)) {
          sendLog(`  ✓ 使用已下载文件: ${path.basename(cachedPath)}`);
          resolve(cachedPath);
          return;
        }
      }

      const mergeMatch = alreadyMsg.match(/\[Merger\] Merging formats into "(.+?)"/);
      if (mergeMatch) {
        const mergedPath = mergeMatch[1].trim();
        if (fs.existsSync(mergedPath)) {
          sendLog(`  ✓ 下载完成: ${path.basename(mergedPath)}`);
          resolve(mergedPath);
          return;
        }
      }

      if (code !== 0) {
        const detail = stderrLines.slice(-3).join(" | ");
        reject(new Error(`下载失败，退出码: ${code}${detail ? `\n${detail}` : ""}`));
        return;
      }

      const now = Date.now();
      let latest = null;
      let latestMtime = 0;
      const walk = (dir) => {
        if (!fs.existsSync(dir)) return;
        for (const entry of fs.readdirSync(dir)) {
          const full = path.join(dir, entry);
          let st;
          try { st = fs.statSync(full); } catch { continue; }
          if (st.isDirectory()) { walk(full); continue; }
          if (!isFinalVideo(entry)) continue;
          if (now - st.mtimeMs > 5 * 60 * 1000) continue;
          if (st.mtimeMs > latestMtime) { latestMtime = st.mtimeMs; latest = full; }
        }
      };
      walk(DOWNLOADS_DIR);

      if (latest) {
        const naDir = path.join(DOWNLOADS_DIR, "NA");
        if (latest.startsWith(naDir + path.sep)) {
          const dest = path.join(DOWNLOADS_DIR, path.basename(latest));
          fs.renameSync(latest, dest);
          try { fs.rmdirSync(naDir); } catch {}
          latest = dest;
        }
        sendLog(`  ✓ 下载完成: ${path.relative(DOWNLOADS_DIR, latest)}`);
        resolve(latest);
      } else {
        reject(new Error("下载完成但未找到视频文件"));
      }
    });

    ytdlp.on("error", (err) => {
      setCurrentProcess(null);
      reject(new Error(`启动 yt-dlp 失败: ${err.message}`));
    });
  });
}

function organizeIntoSeriesDir(videoPath) {
  const name = path.basename(videoPath);
  const match = name.match(/^(.+?)\s+p(\d+)\s+(.+?)(\.[^.]+)$/i);
  if (!match) return videoPath;
  const seriesName = match[1].trim();
  const epNum = match[2];
  const epName = match[3].replace(/^\d+[\.\、\)]\s*/, "").trim();
  const ext = match[4];

  const seriesDir = path.join(DOWNLOADS_DIR, seriesName);
  const epDir = path.join(seriesDir, `${epNum.padStart(2, "0")}. ${epName}`);
  if (!fs.existsSync(epDir)) fs.mkdirSync(epDir, { recursive: true });

  const newPath = path.join(epDir, `video${ext}`);

  const oldDir = path.dirname(videoPath);
  const oldBase = name.replace(/\.[^.]+$/, "");
  const normalize = (s) =>
    s.replace(/^\d+[\.\s]+/, "").trim().toLowerCase();
  const normalizedTarget = normalize(oldBase);
  for (const f of fs.readdirSync(oldDir)) {
    if (f === name) continue;
    if (!/\.(srt|zh\.srt|zh-only\.srt|en-zh\.srt|tr-only\.srt|en-tr\.srt)$/i.test(f)) continue;
    const fBase = f.replace(/\.(srt|zh\.srt|zh-only\.srt|en-zh\.srt|tr-only\.srt|en-tr\.srt)$/i, "");
    if (normalize(fBase) === normalizedTarget) {
      const subExt = f.slice(fBase.length);
      try { fs.renameSync(path.join(oldDir, f), path.join(epDir, `video${subExt}`)); } catch {}
    }
  }

  if (videoPath !== newPath) {
    fs.renameSync(videoPath, newPath);
    sendLog(`  📁 归档: ${seriesName}/${epNum.padStart(2, "0")}. ${epName}/`);
  }
  return newPath;
}

module.exports = { downloadVideo, organizeIntoSeriesDir };
