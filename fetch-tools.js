// 从 npmmirror registry 直接下载 npm tarball，提取 ffmpeg.exe / ffprobe.exe 到 bin/
// 不走 npm CLI（本机沙箱下 spawn cmd.exe 会 EPERM）
const fs = require("fs");
const path = require("path");
const os = require("os");
const { execFileSync } = require("child_process");

const BIN = path.join(__dirname, "bin");
fs.mkdirSync(BIN, { recursive: true });
const TMP = path.join(os.tmpdir(), "videosub-tools");
fs.mkdirSync(TMP, { recursive: true });

const REGISTRIES = [
  "https://registry.npmmirror.com",
  "https://registry.npmjs.org",
];

async function fetchT(url, ms = 300000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    return await fetch(url, { redirect: "follow", signal: ctl.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function getPackument(pkg) {
  for (const reg of REGISTRIES) {
    try {
      const r = await fetchT(`${reg}/${pkg}`, 60000);
      if (r.ok) return await r.json();
      console.log(`  HTTP ${r.status} ${reg}/${pkg}`);
    } catch (e) {
      console.log(`  FAIL ${e.message} ${reg}/${pkg}`);
    }
  }
  return null;
}

function findFile(dir, name) {
  const stack = [dir];
  const hits = [];
  while (stack.length) {
    const d = stack.pop();
    let entries;
    try { entries = fs.readdirSync(d, { withFileTypes: true }); } catch { continue; }
    for (const e of entries) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) stack.push(full);
      else if (e.name.toLowerCase() === name.toLowerCase()) hits.push(full);
    }
  }
  // 优先路径里带 win32 / x64 的
  hits.sort((a, b) => {
    const score = (p) => (/win32|win64|x64/i.test(p) ? 0 : 1);
    return score(a) - score(b);
  });
  return hits[0] || null;
}

async function grab(pkg, wantName, dstName, version) {
  console.log(`\n=== ${pkg}${version ? "@" + version : ""} ===`);
  const doc = await getPackument(pkg);
  if (!doc) { console.log("  !! 无法获取包信息"); return false; }

  const ver = version || doc["dist-tags"].latest;
  const meta = doc.versions[ver];
  if (!meta) { console.log(`  !! 版本不存在: ${ver}`); return false; }

  const url = meta.dist.tarball;
  console.log(`  version ${ver}`);
  console.log(`  tarball ${url}`);

  const t0 = Date.now();
  let res;
  try {
    res = await fetchT(url);
  } catch (e) {
    console.log(`  !! 下载失败: ${e.message}`);
    return false;
  }
  if (!res.ok) { console.log(`  !! HTTP ${res.status}`); return false; }

  const buf = Buffer.from(await res.arrayBuffer());
  const tgz = path.join(TMP, `${pkg.replace(/[@/]/g, "_")}-${ver}.tgz`);
  fs.writeFileSync(tgz, buf);
  console.log(`  下载完成 ${(buf.length / 1048576).toFixed(1)} MiB in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const dest = path.join(TMP, `${pkg.replace(/[@/]/g, "_")}-${ver}`);
  fs.rmSync(dest, { recursive: true, force: true });
  fs.mkdirSync(dest, { recursive: true });
  execFileSync("tar.exe", ["-xzf", tgz, "-C", dest], { stdio: "inherit" });

  const found = findFile(dest, wantName);
  if (!found) { console.log(`  !! tarball 内未找到 ${wantName}`); return false; }

  const dst = path.join(BIN, dstName);
  fs.copyFileSync(found, dst);
  console.log(`  ✓ ${dstName} (${fs.statSync(dst).size} bytes)  ← ${path.relative(dest, found)}`);
  return true;
}

async function grabGz(name, urls, dstName, minBytes) {
  console.log(`\n=== ${name} (gz) ===`);
  const zlib = require("zlib");
  for (const u of urls) {
    const t0 = Date.now();
    try {
      const res = await fetchT(u);
      if (!res.ok) { console.log(`  HTTP ${res.status}  ${u}`); continue; }
      const gz = Buffer.from(await res.arrayBuffer());
      const buf = zlib.gunzipSync(gz);
      if (buf.length < minBytes) { console.log(`  too small (${buf.length})  ${u}`); continue; }
      const dst = path.join(BIN, dstName);
      fs.writeFileSync(dst, buf);
      console.log(`  ✓ ${dstName} ${(buf.length / 1048576).toFixed(1)} MiB `
        + `in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
      return true;
    } catch (e) {
      console.log(`  FAIL ${e.message}  ${u}`);
    }
  }
  console.log(`  !! ${name} 全部失败`);
  return false;
}

(async () => {
  // ffmpeg：ffmpeg-static 包的 tarball 里只有下载器，真正的二进制在 GitHub release
  //   （见其 package.json: binary-release-tag = b6.1.1, ffmpeg-win32-x64.gz）
  if (!fs.existsSync(path.join(BIN, "ffmpeg.exe"))) {
    await grabGz("ffmpeg.exe", [
      "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-win32-x64.gz",
      "https://ghproxy.net/https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-win32-x64.gz",
    ], "ffmpeg.exe", 20 * 1048576);
  }

  // ffprobe：npm tarball 里直接带二进制
  if (!fs.existsSync(path.join(BIN, "ffprobe.exe"))) {
    await grab("ffprobe-static", "ffprobe.exe", "ffprobe.exe", "3.1.0");
  }

  console.log("\n=== bin/ 内容 ===");
  for (const f of fs.readdirSync(BIN)) {
    console.log("  ", f, fs.statSync(path.join(BIN, f)).size);
  }
})();
