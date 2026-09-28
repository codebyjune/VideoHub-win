"""VideoSub 转录引擎 — Windows / NVIDIA GPU 版本

从 mlx-whisper (Apple Silicon / Metal) 迁移到 faster-whisper (CTranslate2)。
在 NVIDIA 显卡上使用 CUDA + float16 推理，无显卡时自动回退 CPU / int8。

用法: python main.py <video_path> [model_size]
"""

import os
import sys
import math

# ── Windows 控制台编码：默认 GBK 会让 emoji / 特殊字符直接抛异常 ──────────
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ── HuggingFace：Windows 上默认使用符号链接，需要管理员或开发者模式；
#    关闭后改为复制文件，普通用户权限即可正常下载模型 ────────────────────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# 国内网络下载模型加速（可用 HF_ENDPOINT 覆盖）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

HERE = os.path.dirname(os.path.abspath(__file__))


def add_cuda_dll_dirs():
    """
    把 pip 安装的 nvidia-*-cu12 包里的 DLL 目录加入 PATH。

    CTranslate2 不像 PyTorch 那样自动搜索这些目录，找不到 cublas64_12.dll /
    cudnn64_9.dll 就会报 "Library cublas64_12.dll is not found or cannot be loaded"
    并退回 CPU。这里手动补齐搜索路径。
    """
    if sys.platform != "win32":
        return []
    added = []
    for sp in sys.path:
        nv = os.path.join(sp, "nvidia")
        if not os.path.isdir(nv):
            continue
        for sub in os.listdir(nv):
            bin_dir = os.path.join(nv, sub, "bin")
            if os.path.isdir(bin_dir):
                added.append(bin_dir)
    if added:
        os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])
        # Python 3.8+ 需要显式声明 DLL 搜索目录
        try:
            for d in added:
                os.add_dll_directory(d)
        except Exception:
            pass
    return added


CUDA_DLL_DIRS = add_cuda_dll_dirs()

# ── 模型目录（支持手动放置模型） ─────────────────────────────────────────
MODELS_DIR = os.path.join(HERE, "models")

# faster-whisper 直接支持这些官方名称；同时兼容用户传入完整 HF repo / 本地路径
KNOWN_MODELS = {
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3",
}


def resolve_model(model_size):
    """把界面上的模型名转换为 faster-whisper 可用值（原样即可）。"""
    return model_size


# ── 设备 / 精度选择 ──────────────────────────────────────────────────────
def pick_device_and_compute(verbose=True):
    """
    选择推理设备与计算精度。

    优先级: CUDA(float16) → CUDA(int8_float16) → CPU(int8)
    float16 在 NVIDIA GPU 上最快；显存不足时退到 int8_float16。
    """
    import ctranslate2

    cuda_ok = False
    try:
        cuda_ok = ctranslate2.get_cuda_device_count() > 0
    except Exception as e:
        if verbose:
            print(f"  ⚠ 检测 CUDA 失败: {e}")

    if cuda_ok:
        try:
            supported = ctranslate2.get_supported_compute_types("cuda")
        except Exception:
            supported = set()
        if "float16" in supported:
            return "cuda", "float16"
        if "int8_float16" in supported:
            return "cuda", "int8_float16"
        if "int8" in supported:
            return "cuda", "int8"

    return "cpu", "int8"


def load_model(model_size, verbose=True):
    """
    加载 Whisper 模型。返回 (model, device, compute_type)。

    显存不足（6GB 卡跑 large-v3 float16 等）时自动降级精度重试。
    """
    from faster_whisper import WhisperModel

    device, compute_type = pick_device_and_compute(verbose)

    if device == "cuda":
        gpu_name = "NVIDIA GPU"
        try:
            import ctranslate2
            gpu_name = f"NVIDIA GPU ×{ctranslate2.get_cuda_device_count()}"
        except Exception:
            pass
        print(f"🚀 GPU 加速已启用: {gpu_name} (device=cuda, compute_type={compute_type})")
    else:
        print("⚠ 未检测到可用 CUDA 设备，使用 CPU 推理 (device=cpu, compute_type=int8)")

    # 显存不足时的降级顺序
    fallbacks = {
        "float16": ["int8_float16", "int8"],
        "int8_float16": ["int8"],
    }
    attempts = [compute_type] + fallbacks.get(compute_type, [])
    if device == "cuda":
        attempts.append("cpu")  # 最后的兜底

    last_err = None
    for attempt in attempts:
        use_device = "cpu" if attempt == "cpu" else device
        use_compute = "int8" if attempt == "cpu" else attempt
        try:
            model = WhisperModel(
                model_size,
                device=use_device,
                compute_type=use_compute,
                download_root=os.path.join(HERE, "models"),
                local_files_only=False,
            )
            if attempt != compute_type:
                print(f"  ⚠ 精度降级为 {use_compute}@{use_device} (原 {compute_type} 不可用: {last_err})")
            return model, use_device, use_compute
        except Exception as e:
            last_err = e
            msg = str(e)
            # 只有显存/精度类错误才值得降级重试；模型不存在等错误直接抛
            if any(k in msg for k in ("out of memory", "CUDA", "cublas", "cudnn",
                                      "compute type", "not supported")):
                continue
            raise

    raise RuntimeError(f"模型加载失败: {last_err}")


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ─── 字幕断句：把过长的 Whisper segment 拆成短 cue ────────────────────────
# Whisper 的 segment 最长可达 30s，直接一条写一条 SRT 会让字幕长时间停留在
# 画面上（且一行超长）。下面利用词级时间戳把它切成 ≤MAX_CUE_SEC 的短句，
# 断点优先落在标点上，保证可读性。
MAX_CUE_SEC = 6.0        # 单条字幕最长显示时长（秒）
MAX_CUE_CHARS = 84       # 单条字幕最多字符数（约 42 字/行 × 2 行）
MIN_CUE_SEC = 1.0        # 切分后单条最短时长，低于此值尽量并入相邻条
WRAP_WIDTH = 42          # 英文自动折行宽度（字符）
WRAP_MAX_LINES = 2       # 最多折成几行

_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
_CLAUSE_END = (",", ";", ":", "—", "–", "、", "，", "；", "：")


def wrap_text(text, width=WRAP_WIDTH, max_lines=WRAP_MAX_LINES):
    """把一行字幕按单词边界折成最多 max_lines 行；超出部分并入最后一行。"""
    text = " ".join((text or "").split())
    if not text or len(text) <= width:
        return text

    lines = []
    cur = ""
    for word in text.split(" "):
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)

    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines - 1]
    tail = " ".join(lines[max_lines - 1:])
    return "\n".join(head + [tail])


def _words_of(segment):
    """把 segment 的词级时间戳统一成 [(start, end, text), ...]。"""
    raw = segment.get("words") or []
    words = []
    for w in raw:
        try:
            if isinstance(w, dict):
                ws, we, wt = w["start"], w["end"], w["word"]
            else:
                ws, we, wt = w[0], w[1], w[2]
            words.append((float(ws), float(we), str(wt)))
        except (TypeError, KeyError, IndexError):
            continue
    return words


def _split_text_evenly(text, n):
    """兜底切分用：按词数把文本尽量平均切成 n 份。"""
    words = (text or "").split()
    if n <= 1 or len(words) <= 1:
        return [text] if text else []
    parts = []
    for i in range(n):
        lo = round(i * len(words) / n)
        hi = round((i + 1) * len(words) / n)
        chunk = " ".join(words[lo:hi]).strip()
        if chunk:
            parts.append(chunk)
    return parts or [text]


def _split_by_chars(segment, max_sec, max_chars):
    """兜底切分：没有词级时间戳时，按字符数/时长估算切点，时间按比例分配。"""
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", 0.0))
    text = (segment.get("text") or "").strip()
    if not text or end <= start:
        return [dict(segment)] if text else []

    n = max(
        1,
        math.ceil(len(text) / max_chars),
        math.ceil((end - start) / max_sec),
    )
    # 按字符比例分配时，长句会被分到更长时间；逐步增加份数直到每份都不超时
    parts = _split_text_evenly(text, n)
    for _ in range(4):
        if len(parts) <= 1:
            break
        weights = [max(1, len(p)) for p in parts]
        total_w = sum(weights)
        if max((end - start) * w / total_w for w in weights) <= max_sec:
            break
        n += 1
        parts = _split_text_evenly(text, n)
    if len(parts) <= 1:
        return [dict(segment)]

    weights = [max(1, len(p)) for p in parts]
    total_w = sum(weights)
    out = []
    t = start
    for i, part in enumerate(parts):
        cue_end = end if i == len(parts) - 1 else t + (end - start) * weights[i] / total_w
        out.append({
            "start": t,
            "end": cue_end,
            "text": part,
            "avg_logprob": segment.get("avg_logprob", float("nan")),
        })
        t = cue_end
    return out


def split_segment(segment, max_sec=MAX_CUE_SEC, max_chars=MAX_CUE_CHARS,
                  min_sec=MIN_CUE_SEC):
    """
    把一条 segment 拆成多条短 cue。

    优先用词级时间戳精确切分；没有词级时间戳时退回按字符估算。
    切分点优先级：句末标点 → 从句标点（且本条已接近上限）→ 到时/字数上限。
    """
    words = _words_of(segment)
    if not words:
        return _split_by_chars(segment, max_sec, max_chars)

    n = len(words)
    # 前缀字符数：快速计算任意 words[a:b] 的可见字符数（含词间空格）
    char_run = [0] * (n + 1)
    for i, w in enumerate(words):
        char_run[i + 1] = char_run[i] + len(w[2].strip())

    def cue_chars(a, b):
        return char_run[b] - char_run[a] + max(0, b - a - 1)

    spans = []
    i = 0
    while i < n:
        start_t = words[i][0]
        # 在不超时长/字数的前提下，尽可能多地纳入单词
        b = i
        while b < n:
            if (words[b][1] - start_t) > max_sec or cue_chars(i, b + 1) > max_chars:
                break
            b += 1
        if b == i:          # 单个词就超限，至少纳入它，避免死循环
            b = i + 1

        # 到达上限时，回退到窗口内最后一个标点断点，让句子更自然
        if b < n:
            best = None
            for k in range(b, i, -1):
                if words[k - 1][2].strip().endswith(_SENTENCE_END):
                    best = k
                    break
            if best is None or (words[best - 1][1] - start_t) < min_sec:
                for k in range(b, i, -1):
                    if words[k - 1][2].strip().endswith(_CLAUSE_END):
                        best = k
                        break
                if best is not None and (words[best - 1][1] - start_t) < min_sec:
                    best = None
            if best is not None:
                b = best

        spans.append((i, b))
        i = b

    out = []
    for a, b in spans:
        ctext = "".join(x[2] for x in words[a:b]).strip()
        if not ctext:
            continue
        out.append({
            "start": words[a][0],
            "end": words[b - 1][1],
            "text": ctext,
            "avg_logprob": segment.get("avg_logprob", float("nan")),
        })
    # 首尾对齐原段，避免切分后与相邻段之间出现小空洞
    if out:
        out[0]["start"] = min(float(segment.get("start", out[0]["start"])), out[0]["start"])
        out[-1]["end"] = max(float(segment.get("end", out[-1]["end"])), out[-1]["end"])
    return out


def refine_cues(cues, min_sec=MIN_CUE_SEC, max_sec=MAX_CUE_SEC,
                max_chars=MAX_CUE_CHARS):
    """合并过短 cue，并保证时间轴单调、不重叠、不短于 min_sec。"""
    merged = []
    for c in cues:
        if c["end"] - c["start"] <= 0.01:
            continue
        if merged and (c["end"] - c["start"]) < min_sec:
            prev = merged[-1]
            if ((c["end"] - prev["start"]) <= max_sec * 1.5
                    and len(prev["text"]) + 1 + len(c["text"]) <= max_chars * 1.5):
                prev["text"] = f'{prev["text"]} {c["text"]}'.strip()
                prev["end"] = c["end"]
                continue
        merged.append(dict(c))

    for i, c in enumerate(merged):
        if i > 0:
            c["start"] = max(c["start"], merged[i - 1]["end"])
        limit = merged[i + 1]["start"] if i + 1 < len(merged) else None
        if c["end"] - c["start"] < min_sec:
            target = c["start"] + min_sec
            c["end"] = target if limit is None else min(target, limit)
    return [c for c in merged if c["end"] - c["start"] > 0.01]


def split_segments(segments):
    """对所有 segment 做断句，再合并过短 cue。"""
    cues = []
    for seg in segments:
        cues.extend(split_segment(seg))
    return refine_cues(cues)


# ─── 转录后补漏：检测 + 二次补录 + 合并 ───────────────────────────────────
GAP_THRESHOLD = 2.5        # 间隙≥2.5s 视为可疑遗漏
BACKFILL_CHUNK = 30.0      # 单次补录窗口上限（Whisper 单窗口即 30s）
LOGPROB_SURE = -0.8        # 前后段 avg_logprob ≥ 此值 → 确信有说话，间隙更可疑


def get_media_duration(video_file):
    """
    获取媒体总时长（秒）。

    优先用 faster-whisper 自带的 PyAV 解码（无需系统 ffmpeg）；
    失败时回退 ffprobe / ffmpeg。全部失败返回 None。
    """
    try:
        from faster_whisper.audio import decode_audio
        audio = decode_audio(video_file, sampling_rate=16000)
        return len(audio) / 16000.0
    except Exception:
        pass

    import subprocess
    import shutil

    candidates = []
    local = os.path.join(HERE, "bin", "ffprobe.exe")
    if os.path.exists(local):
        candidates.append([local])
    else:
        exe = shutil.which("ffprobe")
        if exe:
            candidates.append([exe])

    for base in candidates:
        try:
            out = subprocess.check_output(
                base + [
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_file,
                ],
                stderr=subprocess.DEVNULL,
            )
            return float(out.decode().strip())
        except Exception:
            continue
    return None


def find_suspicious_gaps(segments, total_duration=None):
    """
    扫描 segment 时间轴，找出可疑的遗漏间隙。

    判定逻辑（logprob 加权）：
      - 时间间隔 ≥ GAP_THRESHOLD
      - 且 前后段 avg_logprob 较高（说明前后确实在说话，中间突然断了更可疑）

    total_duration 用于检测片尾间隙：最后一个 segment 之后到视频结尾
      若仍有 ≥ GAP_THRESHOLD 的空白，也会纳入补录（防止片尾重复幻觉
      吞掉真实内容）。若为 None 则仅检查 segment 之间的间隙。

    返回 [(gap_start, gap_end, score), ...]
    """
    gaps = []
    n = len(segments)
    for i in range(n - 1):
        cur_end = segments[i].get("end", 0)
        nxt_start = segments[i + 1].get("start", 0)
        gap = nxt_start - cur_end
        if gap < GAP_THRESHOLD:
            continue

        lp_before = segments[i].get("avg_logprob", float("nan"))
        lp_after = segments[i + 1].get("avg_logprob", float("nan"))
        if math.isnan(lp_before):
            lp_before = LOGPROB_SURE
        if math.isnan(lp_after):
            lp_after = LOGPROB_SURE

        sure = (lp_before >= LOGPROB_SURE) + (lp_after >= LOGPROB_SURE)
        time_factor = min(gap / 10.0, 1.0)
        score = (sure / 2.0) * 0.6 + time_factor * 0.4

        gaps.append((cur_end, nxt_start, score))

    # 片尾补录
    if total_duration is not None and n > 0:
        last_end = segments[-1].get("end", 0)
        tail_gap = total_duration - last_end
        if tail_gap >= GAP_THRESHOLD:
            lp_before = segments[-1].get("avg_logprob", float("nan"))
            if math.isnan(lp_before):
                lp_before = LOGPROB_SURE
            sure = 1 if lp_before >= LOGPROB_SURE else 0
            time_factor = min(tail_gap / 10.0, 1.0)
            score = (sure / 2.0) * 0.6 + time_factor * 0.4
            gaps.append((last_end, total_duration, score))

    return gaps


def transcribe_clip(model, video_file, start, end, lang="en",
                    no_speech_threshold=0.3, condition_on_previous_text=False):
    """
    对 [start, end] 区间做一次转录，返回 segment dict 列表。

    faster-whisper 的 clip_timestamps 会保持绝对时间戳（内部加 time_offset），
    因此这里的时间戳可直接使用。
    """
    segments, _info = model.transcribe(
        video_file,
        language=lang,
        temperature=0.0,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        # 补录段同样要能断句，因此也开启词级时间戳
        word_timestamps=True,
        clip_timestamps=[float(start), float(end)],
        compression_ratio_threshold=2.4,
        # 注意：faster-whisper 的参数名是 log_prob_threshold（带下划线）。
        # 写成 logprob_threshold 会抛 TypeError；而调用方有 try/except 兜底，
        # 异常会被吞掉，导致补漏功能静默失效 —— 这个坑踩过一次。
        log_prob_threshold=-1.0,
        beam_size=5,
        vad_filter=False,
    )

    out = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        if seg.end <= start or seg.start >= end:
            continue
        out.append({
            "start": max(seg.start, start),
            "end": min(seg.end, end),
            "text": txt,
            "avg_logprob": getattr(seg, "avg_logprob", float("nan")),
            "words": [
                (float(w.start), float(w.end), str(w.word))
                for w in (getattr(seg, "words", None) or [])
            ],
        })
    return out


def backfill_gap(model, video_file, gap_start, gap_end, lang="en"):
    """
    对单个可疑间隙做二次补录。

    采用更激进的 no_speech_threshold（0.3）捕获弱语音，
    关闭 condition_on_previous_text 避免被前文带偏，
    并启用 logprob / compression_ratio 防幻觉阈值。
    超过 30s 的间隙自动分块。
    """
    backfilled = []
    t = gap_start
    while t < gap_end - 0.05:
        chunk_end = min(t + BACKFILL_CHUNK, gap_end)
        try:
            got = transcribe_clip(
                model, video_file, t, chunk_end, lang=lang,
                no_speech_threshold=0.3, condition_on_previous_text=False,
            )
            for seg in got:
                seg["_backfilled"] = True
                backfilled.append(seg)
        except Exception as e:
            print(f"  ⚠ 补录 [{t:.1f}-{chunk_end:.1f}] 失败: {e}")
        t = chunk_end
    return backfilled


def merge_segments(original, backfilled, is_near_duplicate_fn):
    """
    把补录段合并进原 segment 列表：
      - 按 start 时间排序
      - 双重去重：文本相似 + 时间区间重叠（IoU ≥ 0.5）
      - 冲突时保留 avg_logprob 更高（更可信）的那条
    """
    merged = list(original) + backfilled
    merged.sort(key=lambda s: s.get("start", 0))

    def _iou(a, b):
        as_, ae = a.get("start", 0), a.get("end", 0)
        bs, be = b.get("start", 0), b.get("end", 0)
        if ae <= as_ or be <= bs:
            return 0.0
        inter = min(ae, be) - max(as_, bs)
        union = max(ae, be) - min(as_, bs)
        return inter / union if union > 0 else 0.0

    deduped = []
    for seg in merged:
        txt = seg.get("text", "").strip()
        if not txt:
            continue
        is_dup = False
        for i_prev, prev in enumerate(deduped[-2:]):
            if is_near_duplicate_fn(txt, prev.get("text", "")) or _iou(seg, prev) >= 0.5:
                if seg.get("avg_logprob", -9) > prev.get("avg_logprob", -9):
                    actual_idx = len(deduped) - 2 + i_prev
                    deduped[actual_idx] = seg
                is_dup = True
                break
        if not is_dup:
            deduped.append(seg)
    return deduped


# ──────────────────────────────────────────────────────────────────────


def is_repetition_hallucination(text, max_repeat=5):
    words = text.strip().rstrip(".").split()
    if not words:
        return False
    counts = {}
    for w in words:
        w_clean = w.rstrip(".,;:!?")
        counts[w_clean] = counts.get(w_clean, 0) + 1
    return any(c >= max_repeat for c in counts.values())


def is_near_duplicate(text, prev_text, threshold=0.9):
    """检测跨 30s 窗口的重复幻觉（condition_on_previous_text=True 时偶发）。"""
    if not prev_text:
        return False
    a = text.lower().strip().rstrip(".,;:!?")
    b = prev_text.lower().strip().rstrip(".,;:!?")
    if not a or not b:
        return False
    if a == b:
        return True

    def _strip_punct(ws):
        return [w.rstrip(".,;:!?") for w in ws]

    wa, wb = _strip_punct(a.split()), _strip_punct(b.split())
    if len(wa) < 4 or len(wb) < 4:
        return False
    sa, sb = set(wa), set(wb)
    return len(sa & sb) / max(len(sa), len(sb)) >= threshold


def main():
    if len(sys.argv) < 2:
        print("用法: python main.py <video_path> [model_size]")
        print("  model_size: tiny.en, small.en, medium.en, large-v3, large-v3-turbo (默认: medium.en)")
        sys.exit(1)

    video_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium.en"

    if not os.path.exists(video_file):
        print(f"错误: 文件不存在: {video_file}")
        sys.exit(1)

    model_path = resolve_model(model_size)
    print(f"加载模型: {model_size}")

    import time
    t0 = time.time()
    model, device, compute_type = load_model(model_path)
    print(f"模型加载完成，耗时 {time.time() - t0:.1f}s")

    # ── 首次转录 ──────────────────────────────────────────────────────
    # no_speech_threshold=0.6：0.3 过于激进，会把含背景音/停顿的真实语音
    #   误判为静音而整段跳过，是"偶尔漏几句话"的主因。
    # condition_on_previous_text=True：保留 30s 窗口之间的上下文，减少边界丢词；
    #   由此可能产生的跨窗口重复幻觉由下方 is_near_duplicate 过滤。
    print("开始转录...")
    t0 = time.time()
    raw_segments, info = model.transcribe(
        video_file,
        language="en",
        temperature=0.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        # 词级时间戳：断句时需要用它精确切分长 segment
        word_timestamps=True,
        beam_size=5,
        vad_filter=False,
    )

    audio_duration = getattr(info, "duration", None) or 0
    segments = []
    for seg in raw_segments:  # 生成器，必须迭代才会真正计算
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": (seg.text or "").strip(),
            "avg_logprob": getattr(seg, "avg_logprob", float("nan")),
            "words": [
                (float(w.start), float(w.end), str(w.word))
                for w in (getattr(seg, "words", None) or [])
            ],
        })
        if audio_duration:
            pct = min(99, int(seg.end / audio_duration * 100))
            print(f"  转录进度 {pct}% [{format_timestamp(seg.end)}/{format_timestamp(audio_duration)}]")

    elapsed = time.time() - t0

    detected_lang = getattr(info, "language", "") or "en"
    print(f"转录完成！检测到语言: {detected_lang}")
    if audio_duration and elapsed > 0:
        print(f"  音频时长 {format_timestamp(audio_duration)}，"
              f"耗时 {elapsed:.1f}s（{audio_duration / elapsed:.1f}x 实时）")

    srt_filename = os.path.splitext(video_file)[0] + ".srt"

    # 先过滤掉重复幻觉段，得到干净的初始 segment 列表
    clean_segments = []
    skipped_repetition = 0
    prev_text = ""
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        if is_repetition_hallucination(text):
            skipped_repetition += 1
            print(f"[跳过重复] {text[:80]}...")
            continue
        if is_near_duplicate(text, prev_text):
            skipped_repetition += 1
            print(f"[跳过跨窗口重复] {text[:80]}...")
            continue
        prev_text = text
        clean_segments.append(segment)

    print(f"初始转录段: {len(clean_segments)} 条（跳过重复 {skipped_repetition} 条）")

    # ─── 转录后补漏：检测可疑间隙并二次补录 ─────────────────────────────
    total_duration = get_media_duration(video_file)
    if total_duration is None:
        print("⚠ 未获取到媒体总时长，仍可继续补漏")
    gaps = find_suspicious_gaps(clean_segments, total_duration)
    if gaps:
        print(f"\n🔎 检测到 {len(gaps)} 个可疑遗漏间隙（≥{GAP_THRESHOLD}s）:")
        for gs, ge, sc in gaps:
            print(f"   [{format_timestamp(gs)} → {format_timestamp(ge)}] "
                  f"时长 {ge-gs:.1f}s  可疑度 {sc:.2f}")

        all_backfilled = []
        for idx, (gs, ge, sc) in enumerate(gaps, 1):
            print(f"   [{idx}/{len(gaps)}] 补录中...")
            bf = backfill_gap(model, video_file, gs, ge, lang="en")
            if bf:
                print(f"        ✓ 补回 {len(bf)} 段:")
                for b in bf:
                    print(f"          [{format_timestamp(b['start'])} → "
                          f"{format_timestamp(b['end'])}] {b['text'][:60]}")
                all_backfilled.extend(bf)
            else:
                print("        - 该间隙无语音（确属静音）")

        if all_backfilled:
            clean_segments = merge_segments(
                clean_segments, all_backfilled, is_near_duplicate
            )
            print(f"\n✅ 补漏完成：新增 {len(all_backfilled)} 段，"
                  f"合并后共 {len(clean_segments)} 段")
    else:
        print("\n✓ 未检测到可疑遗漏间隙")

    # ─── 断句：把过长的 segment 拆成短 cue ────────────────────────────────
    raw_count = len(clean_segments)
    clean_segments = split_segments(clean_segments)
    print(f"断句：{raw_count} 段 → {len(clean_segments)} 条字幕")

    # ─── 输出 SRT ──────────────────────────────────────────────────────
    written = 0
    with open(srt_filename, "w", encoding="utf-8") as f:
        for segment in clean_segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            if not text:
                continue
            if not any(c.isalnum() for c in text):
                continue
            written += 1
            f.write(f"{written}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{wrap_text(text)}\n\n")

            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")

    print(f"\n✅ 字幕文件已保存：{srt_filename}")
    if skipped_repetition > 0:
        print(f"   跳过重复幻觉段: {skipped_repetition} 条")
    # 标记输出，方便 Electron 解析
    print(f"SRT_OUTPUT:{os.path.abspath(srt_filename)}")


if __name__ == "__main__":
    main()
