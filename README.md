# VideoSub

下载视频 → Whisper 转录英文字幕（**NVIDIA GPU 加速**）→ LLM 翻译中文 → 硬嵌入视频，一站式桌面工具。

Download → Transcribe (CUDA) → Translate → Burn subtitles, all-in-one desktop tool.

![](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900)
![](https://img.shields.io/badge/license-MIT-green)

> 本项目已从 macOS / Apple Silicon（mlx-whisper + Metal）迁移到 **Windows + NVIDIA CUDA**。
> 转录引擎改为 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（CTranslate2 后端），
> 在 NVIDIA 显卡上以 `float16` 推理；无显卡时自动回退 CPU。

## 功能 / Features

- **视频下载** — 基于 yt-dlp，支持 YouTube 等主流视频网站，支持 Cookies 与整个播放列表
- **Whisper 转录（GPU 加速）** — faster-whisper + CTranslate2，CUDA `float16`，
  实测 RTX 3060 上 `large-v3-turbo` 约 **16x 实时**；显存不足自动降级到 `int8_float16` / CPU
- **转录后补漏** — 首次转录完成后自动扫描时间轴，对所有 ≥2.5s 的未覆盖间隙做二次补录，追求最高覆盖率，几乎不漏句子
- **LLM 翻译** — 字幕逐条翻译为中文/土耳其语，自动分批 + 降级重试，保证 1:1 行数对齐
- **硬嵌入字幕** — 可选，翻译完成后用 FFmpeg 将双语字幕直接烧录进视频画面
- **暗色主题** — 自动跟随系统，也可手动切换

## 环境要求 / Requirements

| 组件 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Windows 10 / 11 (x64) | |
| GPU | NVIDIA 显卡（推荐 ≥6GB 显存） | 本机验证：RTX 3060 Laptop 6GB |
| 显卡驱动 | 支持 CUDA 12 的驱动 | `nvidia-smi` 能正常输出即可 |
| Python | 3.10 – 3.12 | |
| Node.js | 18+ | |
| FFmpeg | **无需单独安装** | 已随 `bin/ffmpeg.exe` 提供（含 libass） |

> 没有 NVIDIA 显卡也能跑，会自动使用 CPU（`int8`），只是慢一些。

## 安装 / Setup

```powershell
# 1. 安装 Python 依赖（创建 .venv）
uv venv .venv
uv pip install --python .venv\Scripts\python.exe faster-whisper ctranslate2 av yt-dlp `
    nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12

# 2. 安装 Electron 依赖
cd electron-app
npm install

# 3. 启动
npm start
```

> **不需要 `nvidia-cudnn-cu12`。** 实测 CTranslate2 4.8.2 的 CUDA 路径只依赖
> `cublas64_12.dll`（移走它立刻报 `Library cublas64_12.dll is not found`），
> cuDNN 完全用不到 —— 移走 1092 MB 的 cuDNN 后，`large-v3-turbo` 仍以
> 13.4x 实时速度正常转录（GPU 利用率实测 42~49%）。
> 不装 cuDNN 可让 `.venv` 从 2264 MB 降到 1172 MB。

`bin/` 目录需要以下文件（仓库已包含，若缺失可用 `node fetch-tools.js` 重新获取）：

| 文件 | 用途 |
|------|------|
| `bin/ffmpeg.exe` | 字幕烧录（需 libass） |
| `bin/ffprobe.exe` | 媒体时长探测 |
| `bin/yt-dlp.exe` | 视频下载 |

> `bin/yt-dlp.exe` 若不可用，程序会自动回退到 `.venv` 里的 `python -m yt_dlp`。

### 首次运行 / First run

首次转录会自动下载模型到 `models/`：

| 模型 | 大小 | 说明 |
|------|------|------|
| `tiny.en` | ~75 MB | 最快，精度一般 |
| `small.en` | ~480 MB | 平衡 |
| `medium.en` | ~1.5 GB | 高精度 |
| `large-v3` | ~3 GB | 最高精度 |
| `large-v3-turbo` | ~1.6 GB | **推荐**：接近 large 精度，速度快数倍 |

已默认预下载 `large-v3-turbo`。

## GPU 加速说明 / GPU Notes

程序启动转录时会打印实际使用的设备：

```
🚀 GPU 加速已启用: NVIDIA GPU ×1 (device=cuda, compute_type=float16)
```

若显示 `⚠ 未检测到可用 CUDA 设备，使用 CPU 推理`，请依次检查：

1. **驱动**：`nvidia-smi` 是否正常
2. **CUDA 运行库**：`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` 是否已装进 `.venv`
3. **DLL 搜索路径**：这是最容易踩的坑 —— CTranslate2 **不会**像 PyTorch 那样自动搜索
   `site-packages\nvidia\*\bin`，找不到 `cublas64_12.dll` / `cudnn64_9.dll` 就会静默回退 CPU。
   `main.py` 的 `add_cuda_dll_dirs()` 已自动把这些目录加入 `PATH`；
   Electron 侧也通过 `buildPythonEnv()` 一并注入。

## 使用 / Usage

1. 填入视频链接（YouTube / 直链均可）
2. 选择 Whisper 模型（默认 `large-v3-turbo`，GPU 下推荐）
3. 填入 LLM API Key
4. （可选）勾选 "硬嵌入字幕到视频"
5. 点击 **Start**

生成的 SRT 字幕和视频文件在 `downloads/` 目录下。

## 转录后补漏机制 / Backfill

Whisper 首次转录时 `no_speech_threshold=0.6` 会把弱语音/低能量段误判为静音而跳过，导致偶尔漏掉整句话。本工具在转录完成后自动做二次补漏：

```
首次转录(0.6) → 过滤重复幻觉 → 扫描时间轴找 ≥2.5s 间隙
                                    ↓
                    对所有间隙二次补录(0.3，零门槛全补)
                                    ↓
                          双重去重合并 → 输出 SRT
```

**关键参数：**

| 参数 | 首次转录 | 补录 |
|------|---------|------|
| `no_speech_threshold` | 0.6（保准确） | 0.3（捕获弱语音） |
| `condition_on_previous_text` | True（减少边界丢词） | False（孤立片段不被带偏） |
| `temperature` | 0.0 | 0.0（防幻觉） |

**去重策略：** 文本相似度（≥0.9）+ 时间重叠 IoU（≥0.5），冲突时保留 `avg_logprob` 更高的条目。

> 时间代价约 +15-25%，换取几乎不漏句子的覆盖率。

## 架构 / Architecture

```
app/
├── main.py                  # Python: faster-whisper 转录(CUDA) + 转录后补漏
├── pyproject.toml           # Python 项目配置
├── fetch-tools.js           # 下载 ffmpeg / ffprobe / yt-dlp 到 bin/
├── bin/                     # 预置二进制（Windows）
│   ├── ffmpeg.exe           #   字幕烧录（含 libass）
│   ├── ffprobe.exe          #   时长探测
│   └── yt-dlp.exe           #   视频下载
├── models/                  # Whisper 模型缓存（首次运行自动下载）
├── downloads/               # 输出目录（自动创建）
├── .venv/                   # Python 虚拟环境
└── electron-app/
    ├── main.js              # Electron 主进程 — 流程编排
    ├── preload.js           # contextBridge — IPC 桥梁
    ├── renderer.js          # 渲染进程 — UI 逻辑
    ├── index.html           # 界面布局
    ├── style.css            # 样式（暗色/亮色主题）
    ├── shared.js            # 路径解析 / 日志 / GPU DLL 环境注入
    ├── download.js          # yt-dlp 下载
    ├── transcribe.js        # 调用 Python 转录
    ├── translation.js       # LLM 翻译
    ├── subtitles.js         # SRT 解析 + ffmpeg 烧录
    └── package.json
```

### 数据流 / Data Flow

```
UI (index.html)
  │ 用户点击 Start
  ▼
renderer.js ──IPC──▶ main.js
                        │
                        ├─ 1. spawn yt-dlp     → 下载视频
                        ├─ 2. spawn python     → faster-whisper 转录 (CUDA float16)
                        │     │
                        │     ├─ 首次转录（no_speech_threshold=0.6，保准确）
                        │     ├─ 过滤重复幻觉段
                        │     ├─ 扫描时间轴找 ≥2.5s 可疑间隙（logprob 加权）
                        │     ├─ 对所有间隙二次补录（threshold=0.3，捕获弱语音）
                        │     └─ 双重去重（文本相似 + 时间重叠 IoU）→ .srt
                        ├─ 3. call LLM API     → 翻译 → .zh.srt
                        └─ 4. spawn ffmpeg     → 硬嵌入 → _subtitled.mp4
                        │
                        ◀── IPC 实时推送日志/进度 ──
  ▼
UI 更新 logBox + progressBar
```

## 从 macOS 版本迁移的改动 / Windows Port Notes

| 问题 | 处理 |
|------|------|
| `mlx-whisper` 仅支持 Apple Silicon | 换成 `faster-whisper` + CTranslate2，CUDA `float16` |
| CTranslate2 找不到 CUDA DLL，静默回退 CPU | `add_cuda_dll_dirs()` 把 `nvidia/*/bin` 注入 `PATH` |
| HuggingFace 用符号链接，Windows 非管理员会失败 | 设 `HF_HUB_DISABLE_SYMLINKS=1`，改为复制文件 |
| 控制台默认 GBK，emoji 导致 `UnicodeEncodeError` | `stdout/stderr` 强制 UTF-8 |
| `fs.symlinkSync` 创建临时字幕需管理员权限 | 改为 `fs.copyFileSync` 到纯 ASCII 临时名 |
| ffmpeg `subtitles=` 滤镜不认 Windows 盘符 `C:` | 转义为 `subtitles='C\:/path/...'` |
| `ffprobe` 可能不存在 | 优先用 PyAV 解码探测时长，回退 `bin/ffprobe.exe` |
| PowerShell 的 `Invoke-WebRequest` TLS 不可用 | 工具下载统一走 Node `fetch` |

## 断点续跑 / 缓存复用

重复运行同一任务时，已完成的步骤会自动跳过，不会白费时间。判定规则：

| 环节 | 复用条件 | 不满足时 |
|------|---------|---------|
| **下载（单集）** | 目标文件名已存在（yt-dlp `--no-overwrites`） | 重新下载 |
| **下载（系列）** | 该集已在 `<系列>/<集号>. <集名>/` 下归档 | 仅重新下载缺失的集 |
| **转录** | 同名 `.srt` 存在且条目可解析 | 重新转录（GPU） |
| **翻译** | 同名译文存在、可解析、且**不含推理垃圾** | 重新翻译（调用 LLM） |
| **烧录** | `_subtitled.mp4` 比字幕新、**且来源字幕可信** | 重新烧录 |

> ⚠ 系列模式为什么单独判定：下载完成后 `organizeIntoSeriesDir()` 会把视频从
> yt-dlp 的输出路径搬进 `<系列>/<集号>. <集名>/video.<ext>`，导致
> `--no-overwrites` 在第二次运行时认不出已下载的集数、把同一集又下了一遍。
> 因此系列模式会先用 `yt-dlp --flat-playlist` 列出各集，反推归档位置，
> 只把缺失的集交给 `--playlist-items` 下载。
>
> 反推集号时的两个坑（已处理）：
> 1. **B 站多分 P** 在 `--flat-playlist` 下 `title` 恒为 `NA`，只有 `playlist_index`
>    可用；因此集号优先取标题里的 `pNN`，取不到就用 `playlist_index`，系列目录名
>    则用 `playlist_title` 定位。
> 2. **中文 Windows 上 yt-dlp 默认按 GBK 输出**，中文会变乱码导致目录名匹配不上；
>    所有 yt-dlp 调用都强制 `--encoding utf-8`。

> ⚠ 两个容易踩的点：
> 1. **烧录会检查来源字幕**。若旧字幕含推理垃圾，即使成品文件更新也会重烧 ——
>    否则硬字幕会一直带着垃圾文字。
> 2. **被污染的译文会自动重建**。首次运行会打印
>    `⚠ 已有译文字幕含 N/M 条推理垃圾，将重新翻译`，属正常修复行为。

## 翻译限流说明 / LLM Rate Limits

翻译走 Anthropic 兼容协议。为保护服务端配额，所有请求都经过**全局并发闸门 + RPM 限速器**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 每批条数 | 30 | 兼顾输出稳定性与前后文语境 |
| 全局并发上限 | 4 | **所有** HTTP 请求共用，含降级/补录路径 |
| RPM 上限 | 自动学习（默认不限） | 滑动窗口限速；从 429 报文识别，只降不升，留 10% 余量 |
| 重试 | 429 / 408 / 5xx / 网络抖动 | 指数退避，尊重服务端 `Retry-After`；429 可重试 8 次 |
| 429 冷却 | 收到即全局冷却 | 冷却期内所有请求排队，避免刚重试又撞上限 |
| 数量不匹配 | 严格提示 → 对半拆分 | 拆到 8 条以下才逐条兜底 |

实测（180 条字幕，顺利路径）：**6 次请求、峰值并发 4、零 429、26~45s 完成**。

**RPM 限流（针对 `exhausted local RPM limit 15` 这类报错）：**

有些中转/低价套餐按「每分钟请求数」限流（常见 15 RPM）。并发闸门只管「同时在飞」的请求数，
挡不住一分钟内发几十个请求。为此程序内置了一个**滑动窗口 RPM 限速器**：

- 所有请求（含逐条 / 补录）都记入 60s 时间窗，**无论是否已限速**；
- 一旦收到 429，自动从报文里识别上限并降速（如 `RPM limit 15` → 限到 13），**只降不升**，
  且因为此前的请求已被计入窗口，降速**立即生效**，不会白撞一整窗；
- 429 时额外做一次全局冷却，并最多重试 8 次，确保能等过一个完整窗口。

也可用环境变量手动指定（无需改代码）：

```powershell
$env:VIDEOSUB_LLM_CONCURRENCY = "2"   # 并发上限，默认 4
$env:VIDEOSUB_LLM_RPM         = "15"  # 每分钟请求上限；不设则按 429 自动学习
npm start
```

> 为什么需要全局闸门：降级路径（递归拆分 / 逐条翻译 / 空译文补录）也会发请求。
> 早期版本只限制了顶层批次并发，递归拆分时并发会逐层翻倍 ——
> 实测最坏情况峰值可达 **64 并发**，必然触发限流，而限流又引发更多重试。

## 烧录编码 / Burn Encoder

字幕烧录必须重编码。程序会**自动选择编码器**：

| 情况 | 编码器 | 说明 |
|------|--------|------|
| 检测到 NVIDIA 可用 | `h264_nvenc` | GPU 硬件编码，快数倍、CPU 占用大降 |
| 其它 | `libx264` | CPU 软件编码（慢但兼容性最好） |

- 启动第一次烧录时会**真实编码一小段**来探测（不是只看 `-encoders`，能发现驱动问题），结果缓存整场会话。
- 硬件编码中途失败会**自动回退 libx264**，不会导致任务失败。
- 只对 `.mp4/.mkv/.mov/.avi` 生效（`.webm` 塞不进 H.264，保持原样）。
- 也可手动指定：

```powershell
$env:VIDEOSUB_BURN_ENCODER = "h264_nvenc"   # 或 libx264 / hevc_nvenc ...
npm start
```

> 注：NVENC 只接管「编码」。**解码**和 **libass 画字幕**仍在 CPU 上，
> 所以 CPU 占用会明显下降但不会归零；字幕特别密时仍可能偏慢。

## License

MIT © [codebyjune](https://github.com/codebyjune)
