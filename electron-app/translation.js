const fs = require("fs");
const path = require("path");
const Anthropic = require("@anthropic-ai/sdk");
const {
  extractResponseText,
  looksLikeReasoning,
  getIsCancelled,
  sendLog,
  sendProgress,
} = require("./shared");
const { parseSrt, buildBilingualSrt, buildMonolingualSrt } = require("./subtitles");

const LANG_CONFIG = {
  zh: {
    label: "中文",
    suffix: ".zh.srt",
    target: "Simplified Chinese",
    style: "(适合中文字幕风格)",
    langName: "Chinese",
    logStep: "LLM 翻译中文字幕",
  },
  tr: {
    label: "Türkçe",
    suffix: ".tr.srt",
    target: "Turkish",
    style: "(Türkçe altyazı stili)",
    langName: "Turkish",
    logStep: "LLM 翻译土耳其语字幕",
  },
};

function buildTranslationSystemPrompt(targetLang) {
  const lang = LANG_CONFIG[targetLang] || LANG_CONFIG.zh;

  const basePrompt = `You are an expert subtitle translator specializing in natural, fluent ${lang.target} translations.

SUBTITLE TRANSLATION PRINCIPLES:
1. Read ALL input lines first to understand the overall topic and domain before translating.
2. Each translated subtitle must read as natural, conversational ${lang.langName} — avoid stiff, word-for-word translationese.
3. Each subtitle should feel like a self-contained, semantically complete phrase that flows naturally.

CRITICAL — CONTEXT-DEPENDENT WORD CHOICE (多义词辨析):
The same English word can have very different meanings. Always determine the correct meaning from context.

PRESERVATION RULES:
- Keep proper names accurately
- Preserve ALL non-English letters, special characters, and notations exactly as-is
- Keep technical terminology accurate throughout`;

  if (targetLang === "tr") {
    return `${basePrompt}

TURKISH TRANSLATION GUIDELINES:
- Use natural, conversational Turkish ("sen" / "siz" form appropriate to context)
- Prefer shorter Turkish expressions to match subtitle timing — Turkish often requires more characters than English
- For technical/linguistics terms: "letter" → harf, "alphabet" → alfabe, "consonant" → ünsüz, "vowel" → ünlü, "pronunciation" → telaffuz, "syllable" → hece, "suffix" → ek
- Proper names should use Turkish phonetic conventions (e.g., "John" → "Con" if historically used, otherwise keep as-is)
- Maintain Turkish word order (SOV) naturally — rephrase English (SVO) sentences completely`;
  }

  return `${basePrompt}
- "letter" in language/alphabet context → 字母 (NEVER translate as 信/信件 in this context)
- "letter" in postal/correspondence context → 信件
- "pose" in speech/pronunciation context → 停顿 (NEVER translate as 姿势/姿态 in this context)
- "writing" referring to script/characters → 书写/写法 (NOT 文笔)
- "sound" in phonetics/linguistics → 发音/音 (NOT 声音)
- "character" in text/script context → 字符/字母 (NOT 角色/性格)
- "soft sign" / "hard sign" → 软音符号 / 硬音符号 (Russian linguistic terms)
- "consonant" → 辅音, "vowel" → 元音
- "pronounce" / "pronunciation" → 发音
- "alphabet" → 字母表
- Preserve ALL Cyrillic letters (А, Б, В, Г, Д, Е, Ё, Ж, З, И, Й, К, Л, М, Н, О, П, Р, С, Т, У, Ф, Х, Ц, Ч, Ш, Щ, Ъ, Ы, Ь, Э, Ю, Я) and phonetic notations exactly as-is — do not replace them with Latin equivalents`;
}

/**
 * 判断一批解析结果是否其实是推理文本（而非译文）。
 * 只要有一条像推理，就认为整批不可信。
 */
function batchLooksLikeReasoning(arr) {
  if (!Array.isArray(arr)) return false;
  return arr.some((s) => looksLikeReasoning(String(s)));
}

function parseTranslationResponse(raw, expectedCount) {
  const cleaned = raw
    .replace(/```json\s*/gi, "")
    .replace(/```\s*/g, "")
    .trim();
  let lines;
  try {
    lines = JSON.parse(cleaned);
    if (!Array.isArray(lines)) return null;
  } catch {
    lines = raw.split("\n").filter((l) => {
      const t = l.trim();
      return t !== "" && !t.startsWith("```");
    });
  }
  if (lines.length !== expectedCount) return null;
  const cleanedLines = lines.map((l) => String(l).replace(/^\d+[\.\、\)]\s*/, "").trim());
  // 防止把推理文本当成译文收下
  if (batchLooksLikeReasoning(cleanedLines)) return null;
  return cleanedLines;
}

const TRANSLATE_CONCURRENCY = 4;

// 单次请求最多翻译多少条字幕。
// 实测：200 条一起发时模型只返回 170 条（长输出容易漏行/提前收尾），
// 所以宁可分成小批，也不要赌一次大请求。
const MAX_LINES_PER_REQUEST = 60;

// 每批 30 条：输出规模适中，且保留一定的前后文语境
const BATCH_SIZE = 30;

// 429/5xx/网络抖动时的重试次数与退避基数（毫秒）
const RETRY_MAX = 4;
// 限流(429)单独给更多重试次数：至少要能等一个 60s 的 RPM 窗口转过去
const RETRY_MAX_429 = 8;
const RETRY_BASE_MS = 1500;
// 退避上限（有 Retry-After 时以服务端为准，不受此上限约束）
const RETRY_WAIT_CAP_MS = 30000;

// ── 限流保护 ────────────────────────────────────────────────────────────
// TRANSLATE_CONCURRENCY 只能限制「顶层批次」的并发。递归拆分时每层
// Promise.all([left, right]) 会让并发翻倍：实测 180 条在最坏情况下峰值可达
// 64 并发 —— 必然触发服务端限流，而限流又会引发更多重试，形成正反馈。
// 因此所有真实 HTTP 请求都必须经过这个全局信号量，且必须收口在 createMessage。
//
// 若你的 API 套餐并发上限更低，可用环境变量调小（无需改代码）：
//   set VIDEOSUB_LLM_CONCURRENCY=2
const MAX_INFLIGHT_REQUESTS = (() => {
  const v = parseInt(process.env.VIDEOSUB_LLM_CONCURRENCY || "", 10);
  return Number.isFinite(v) && v > 0 ? v : 4;
})();

// 每分钟请求数上限（RPM）。很多中转/低价套餐按 RPM 限流（例如 15 RPM），
// 并发闸门挡不住「一分钟内发太多请求」，所以另设一个滑动窗口限速器。
//   默认 0 = 不主动限制，只在实际收到 429 时自动学习并降速；
//   也可用环境变量显式指定：set VIDEOSUB_LLM_RPM=15
const ENV_RPM_LIMIT = (() => {
  const v = parseInt(process.env.VIDEOSUB_LLM_RPM || "", 10);
  return Number.isFinite(v) && v > 0 ? v : 0;
})();

// 拆分下限：小于等于这个条数就不再继续拆，直接逐条翻译。
// 拆到 2 条会产生 ~16 层递归、每行一次请求，放大过于严重。
const MIN_SPLIT_SIZE = 8;

class Semaphore {
  constructor(max) {
    this.max = max;
    this.active = 0;
    this.queue = [];
  }
  async acquire() {
    if (this.active >= this.max) {
      await new Promise((resolve) => this.queue.push(resolve));
    }
    this.active++;
  }
  release() {
    this.active--;
    const next = this.queue.shift();
    if (next) next();
  }
  async run(fn) {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

const requestGate = new Semaphore(MAX_INFLIGHT_REQUESTS);

/**
 * 滑动窗口限速器（RPM）。
 *
 * 目的：服务端常按「每分钟请求数」限流（如 ZenMux 的 15 RPM）。并发闸门只能
 * 限制「同时在飞」的请求数，挡不住一分钟内发几十个请求 → 必撞 429。
 *
 * 关键设计：
 *   - 无论是否启用限制，每个请求都会记入时间窗；这样一旦从 429 学到真实上限，
 *     限速立刻生效，且能看见「刚才那波已经发出去的请求」，不用再白撞一窗。
 *   - rpm <= 0 时不限制突发，只记录时间。
 *   - penalize() 用于 429：在服务端给出的冷却期内拦住所有请求。
 */
class SlidingWindowLimiter {
  constructor(rpm) {
    this.windowMs = 60000;
    this.stamps = [];
    this.queue = [];
    this.timer = null;
    this.blockedUntil = 0;
    this.rpm = 0;
    this.setRpm(rpm);
  }

  setRpm(rpm) {
    const n = Math.floor(Number(rpm));
    this.rpm = Number.isFinite(n) && n > 0 ? n : 0;
    this._flush();
  }

  penalize(ms) {
    const until = Date.now() + Math.max(0, ms);
    if (until > this.blockedUntil) this.blockedUntil = until;
    this._flush();
  }

  _prune(now) {
    const cutoff = now - this.windowMs;
    while (this.stamps.length && this.stamps[0] <= cutoff) this.stamps.shift();
  }

  _flush() {
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    const now = Date.now();
    this._prune(now);

    while (this.queue.length) {
      const hasRoom = this.rpm <= 0 || this.stamps.length < this.rpm;
      if (!hasRoom || now < this.blockedUntil) break;
      const resolve = this.queue.shift();
      this.stamps.push(now);
      resolve();
    }

    if (this.queue.length) {
      let nextAt = this.blockedUntil;
      if (this.rpm > 0 && this.stamps.length >= this.rpm) {
        nextAt = Math.max(nextAt, this.stamps[0] + this.windowMs);
      }
      const wait = Math.max(5, nextAt - now + 5);
      this.timer = setTimeout(() => { this.timer = null; this._flush(); }, wait);
    }
  }

  acquire() {
    return new Promise((resolve) => {
      this.queue.push(resolve);
      this._flush();
    });
  }
}

const requestRateLimiter = new SlidingWindowLimiter(ENV_RPM_LIMIT);

/**
 * 从 429 报文里学习服务端的 RPM 上限并降速（只降不升）。
 * 典型报文：`channel ZenMux Sub exhausted local RPM limit 15`。
 */
function learnRpmFromError(err) {
  const msg = String((err && err.message) || "");
  const patterns = [
    /RPM\s*limit[^\d]{0,12}(\d+)/i,
    /requests?\s*per\s*minute[^\d]{0,12}(\d+)/i,
    /(\d+)\s*(?:\/\s*min\b|rpm\b|requests?\s*per\s*minute)/i,
    /limit[^\d]{0,12}(\d+)\s*(?:\/\s*min\b|rpm\b|per\s*minute)/i,
  ];
  let raw = null;
  for (const re of patterns) {
    const m = msg.match(re);
    if (m) { raw = parseInt(m[1], 10); break; }
  }
  if (!Number.isFinite(raw) || raw <= 0) return false;

  // 留 10% 余量：贴着上限跑仍会偶发 429
  let next = Math.max(1, Math.floor(raw * 0.9));
  if (ENV_RPM_LIMIT > 0) next = Math.min(next, ENV_RPM_LIMIT);
  if (requestRateLimiter.rpm > 0) next = Math.min(next, requestRateLimiter.rpm);

  if (next !== requestRateLimiter.rpm) {
    requestRateLimiter.setRpm(next);
    sendLog(`  ℹ 检测到服务端限流 (RPM≈${raw})，已自动限速为 ${next} 次/分钟`);
  }
  return true;
}

/**
 * 所有 LLM 请求的唯一出口，强制经过全局并发闸门 + RPM 限速器。
 *
 * 必须在这里收口，而不是散在 callWithRetry 里：逐条翻译与补录路径也是
 * 直接调 API 的，任何一处漏掉闸门，真实并发就会突破上限。
 */
async function createMessage(client, params) {
  await requestRateLimiter.acquire();
  return requestGate.run(() => client.messages.create(params));
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** 把数组均分成两半（用于数量不匹配时的拆分重试） */
function splitInHalf(arr) {
  const mid = Math.ceil(arr.length / 2);
  return [arr.slice(0, mid), arr.slice(mid)];
}

/**
 * 带指数退避的 API 调用（并发闸门已由 createMessage 统一保证）。
 *
 * SDK 自身不会对 429/5xx 自动重试，而并发过高又很容易触发限流。
 */
async function callWithRetry(fn, label) {
  let lastErr = null;
  for (let attempt = 0; ; attempt++) {
    if (getIsCancelled()) throw new Error("任务已取消");
    try {
      return await fn();
    } catch (e) {
      lastErr = e;
      const status = e && (e.status || e.statusCode);
      const isRateLimit = status === 429;
      const retriable = isRateLimit || status === 408 ||
        (status >= 500 && status < 600) ||
        /fetch failed|ENOTFOUND|ECONNREFUSED|ETIMEDOUT|socket hang up|ECONNRESET/i.test(e.message || "");
      const maxRetries = isRateLimit ? RETRY_MAX_429 : RETRY_MAX;
      if (!retriable || attempt >= maxRetries) throw e;

      // 优先尊重服务端给出的 Retry-After
      let wait = RETRY_BASE_MS * Math.pow(2, attempt);
      const ra = e.headers && (e.headers["retry-after"] || e.headers["Retry-After"]);
      const hasRA = ra !== undefined && !isNaN(Number(ra));
      if (hasRA) wait = Math.max(wait, Number(ra) * 1000);
      else wait = Math.min(wait, RETRY_WAIT_CAP_MS);

      if (isRateLimit) {
        // ① 学到服务端 RPM 上限并降速；② 对全局请求做一次冷却，避免刚重试又撞上
        learnRpmFromError(e);
        requestRateLimiter.penalize(wait);
      }

      sendLog(`    ↻ ${label} 触发限流/网络抖动 (${status || e.message.slice(0, 40)})，${(wait / 1000).toFixed(1)}s 后重试 (${attempt + 1}/${maxRetries})`);
      await sleep(wait);
    }
  }
}

/**
 * 清洗「单条」翻译的返回值。
 *
 * 逐条翻译与补录路径拿到的是一整段模型输出，没有 parseTranslationResponse 那层
 * JSON 保护。实测模型经常不听话，会以这些形式返回：
 *
 *   '["译文"]'                          → 方括号和引号会原样进字幕
 *   '"译文"'                            → 引号进字幕
 *   '```json\n["译文"]\n```'            → 代码块进字幕
 *   'Here is the translation:\n["译文"]' → 说明文字进字幕，译文丢失
 *
 * 这里把它们统一还原成纯译文。若实在提取不出干净文本，返回空串让上层重试，
 * 绝不把包装符号塞进字幕。
 */
function sanitizeSingleTranslation(raw, targetLang) {
  if (!raw) return "";
  let t = String(raw).trim();

  // ① 剥掉 markdown 代码块围栏
  t = t.replace(/```[a-zA-Z]*\s*/g, "").replace(/```/g, "").trim();

  // ② 整段就是 JSON 数组 → 取第一项
  const tryJson = (s) => {
    try {
      const v = JSON.parse(s);
      if (Array.isArray(v)) return v.length ? String(v[0]) : "";
      if (typeof v === "string") return v;
    } catch { /* 不是 JSON，继续 */ }
    return null;
  };
  const direct = tryJson(t);
  if (direct !== null) t = direct.trim();

  // ③ 逐行挑出真正的译文行。
  //    要点：包装说明（"Here is the translation:"）里也可能出现"译文"二字，
  //    所以不能靠"是否含中文"来判定整段，必须逐行比较后择优。
  const splitLines = t.split("\n").map((l) => l.trim()).filter(Boolean);
  if (splitLines.length > 1) {
    const isCjkLang = targetLang === "zh";
    let best = "";
    let bestScore = -1;
    for (const l of splitLines) {
      const v = (tryJson(l) ?? l).trim();
      if (!v) continue;
      const hasCJK = /[\u4e00-\u9fff]/.test(v);
      // 目标为中文时，含 CJK 的行优先；同优先级下取更长的（说明性文字通常更短）
      const score = (isCjkLang && hasCJK ? 1000 : 0) + v.length;
      if (score > bestScore) { bestScore = score; best = v; }
    }
    t = best;
  }

  // ④ 去掉序号前缀（"1. 译文" / "2、译文"）
  t = t.replace(/^\d+[\.\、\)]\s*/, "").trim();

  // ⑤ 去掉成对的首尾引号
  t = t.replace(/^["'“”‘’]+/, "").replace(/["'“”‘’]+$/, "").trim();

  // ⑥ 去掉 "译文：" / "Translation:" 之类的前缀
  t = t.replace(/^(translation|译文|翻译)\s*[:：]\s*/i, "").trim();

  return t;
}

async function translateOneBatch(batch, batchIndex, batchCount, client, llmModel, targetLang) {
  const lang = LANG_CONFIG[targetLang] || LANG_CONFIG.zh;
  const n = batch.length;
  const systemPrompt = buildTranslationSystemPrompt(targetLang);

  const request = (items, strict) => {
    const json = JSON.stringify(items.map((e) => e.text));
    const head = `Translate the following ${items.length} English subtitle lines into natural, fluent ${lang.langName}.`;
    const rules = `CRITICAL:
- Output EXACTLY ${items.length} ${lang.langName} strings — same count as input, no exceptions.
- Each line is a SEPARATE subtitle with its own timestamp. Do NOT merge adjacent lines.
- Preserve EXACT order.
- Apply the polysemy and domain rules from the system instructions carefully.`;
    const strictRules = `COUNT CHECK FAILED. You returned the wrong number of translations. This is a CRITICAL error.

You MUST output EXACTLY ${items.length} ${lang.langName} strings — not ${items.length - 1}, not ${items.length + 1}. EXACTLY ${items.length}.
Each line below is a SEPARATE subtitle timestamp entry. NEVER merge adjacent lines, even if they look like sentence fragments.`;
    return `${head}\n\n${strict ? strictRules : rules}\n\nOutput ONLY a JSON array of ${items.length} ${lang.langName} strings. No explanations, no markdown.\n\n${json}`;
  };

  const ask = async (items, strict, tag) => {
    // 注意：把「取文本」这一步也放进 callWithRetry ——
    // 模型只返回 thinking（token 预算被思考耗尽）是可重试的瞬时故障。
    const raw = await callWithRetry(async () => {
      const response = await createMessage(client, {
        model: llmModel,
        system: systemPrompt,
        messages: [{ role: "user", content: request(items, strict) }],
        temperature: strict ? 0.1 : 0.3,
        max_tokens: 16000,
      });
      return extractResponseText(response).trim();
    }, tag);
    return parseTranslationResponse(raw, items.length);
  };

  // ① 先按宽松提示请求；数量不匹配时用更严格的提示再问一次
  let arr = await ask(batch, false, `第 ${batchIndex + 1} 批`);
  if (!arr) {
    sendLog(`    ⚠ 第 ${batchIndex + 1} 批数量不匹配，用更严格的提示重试...`);
    arr = await ask(batch, true, `第 ${batchIndex + 1} 批(严格)`);
  }

  // ② 仍不匹配 → 对半拆分，而不是重发同样的输入。
  //    同样的输入大概率得到同样的错误结果，拆小才能保证收敛。
  //    但拆分有下限：拆到 2 条会产生 ~16 层递归、每行一次请求，放大过猛。
  if (!arr) {
    if (n <= MIN_SPLIT_SIZE) {
      sendLog(`    ⚠ 第 ${batchIndex + 1} 批（${n} 条）无法对齐，降级为逐条翻译...`);
      return translateOneByOne(batch, client, llmModel, targetLang);
    }
    const [left, right] = splitInHalf(batch);
    sendLog(`    ⚠ 第 ${batchIndex + 1} 批仍不匹配，拆分为 ${left.length}+${right.length} 条重试...`);
    const [a, b] = await Promise.all([
      translateOneBatch(left, batchIndex, batchCount, client, llmModel, targetLang),
      translateOneBatch(right, batchIndex, batchCount, client, llmModel, targetLang),
    ]);
    return [...a, ...b];
  }

  return arr;
}

/**
 * 逐条翻译（批量的降级兜底路径）。
 *
 * 这条路径最容易出问题，因为：
 *   1. 每条只翻译一行，模型倾向于「想很久」，reasoning 模型很容易把 token 预算
 *      全花在 thinking 上，导致没有 text 块（详见 extractResponseText 的注释）；
 *   2. 把「前几条译文」作为 context 回灌时，一旦某条已经泄漏了推理文本，
 *      后续请求会把它当成范例，形成自我强化的污染循环。
 *
 * 因此这里：不回灌 context、用大 token 预算、并对结果做推理文本检测 + 重试。
 */
async function translateOneByOne(entries, client, llmModel, targetLang) {
  const lang = LANG_CONFIG[targetLang] || LANG_CONFIG.zh;
  const systemPrompt = buildTranslationSystemPrompt(targetLang);
  const translations = [];

  const userPromptFor = (text) =>
    `Translate this English subtitle line into natural, fluent ${lang.target}. ` +
    `Pay close attention to context — determine the correct domain-specific word meaning ` +
    `(e.g., in language lessons "letter" = 字母, not 信; "pose" in speech = 停顿, not 姿势). ` +
    `Output ONLY the ${lang.langName} text — no quotes, no prefixes, no explanations, ` +
    `no reasoning, no notes about the task.\n\n${text}`;

  for (let i = 0; i < entries.length; i++) {
    if (getIsCancelled()) throw new Error("任务已取消");

    let cleaned = "";
    let lastErr = null;

    // 最多尝试 2 次：逐条路径本来请求数就多，重试次数不能再放大
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await createMessage(client, {
          model: llmModel,
          system: systemPrompt,
          messages: [{ role: "user", content: userPromptFor(entries[i].text) }],
          temperature: attempt === 0 ? 0.3 : 0.1,
          // reasoning 模型会先输出 thinking，预算必须给足，否则拿不到 text
          max_tokens: 8192,
        });

        const raw = sanitizeSingleTranslation(
          extractResponseText(response),
          targetLang,
        );

        if (!raw) throw new Error("返回内容为空");
        if (looksLikeReasoning(raw)) {
          throw new Error(`检测到返回的是推理文本而非译文（前 40 字: ${raw.slice(0, 40)}...）`);
        }
        cleaned = raw;
        break;
      } catch (e) {
        lastErr = e;
        if (getIsCancelled()) throw new Error("任务已取消");
        sendLog(`    ↻ 第 ${i + 1} 条翻译重试 (${attempt + 1}/2): ${e.message.slice(0, 120)}`);
      }
    }

    if (!cleaned) {
      // 两次都失败：宁可留空，也绝不把推理文本写进字幕
      sendLog(`    ⚠ 第 ${i + 1} 条翻译失败，已留空: ${lastErr ? lastErr.message.slice(0, 80) : "未知错误"}`);
      cleaned = "";
    }
    translations.push(cleaned);
  }
  return translations;
}

async function translateEntriesInBatchesCore(
  entries,
  client,
  llmModel,
  targetLang,
  progressBase,
  progressRange,
) {
  const batches = [];
  for (let i = 0; i < entries.length; i += BATCH_SIZE) {
    batches.push(entries.slice(i, i + BATCH_SIZE));
  }

  sendLog(`  共 ${batches.length} 批，每批 ${BATCH_SIZE} 条（${TRANSLATE_CONCURRENCY} 路并发）`);

  const results = new Array(batches.length);
  let completed = 0;
  let cursor = 0;
  let aborted = false;
  let firstErr = null;

  const worker = async () => {
    while (cursor < batches.length && !aborted) {
      if (getIsCancelled()) {
        aborted = true;
        const err = new Error("任务已取消");
        if (!firstErr) firstErr = err;
        throw err;
      }
      const i = cursor++;
      try {
        results[i] = await translateOneBatch(
          batches[i], i, batches.length, client, llmModel, targetLang,
        );
        completed++;
        sendLog(`  ✓ 完成 ${completed}/${batches.length} 批`);
        const pct = progressBase + Math.round((completed / batches.length) * progressRange);
        sendProgress("translate", Math.min(pct, progressBase + progressRange));
      } catch (e) {
        if (!firstErr) firstErr = e;
        aborted = true;
        throw e;
      }
    }
  };

  const workers = Array.from(
    { length: Math.min(TRANSLATE_CONCURRENCY, batches.length) },
    () => worker(),
  );
  try {
    await Promise.all(workers);
  } catch (e) {
    if (firstErr && firstErr !== e) throw firstErr;
    throw e;
  }

  return results.flat();
}

/**
 * 统一的翻译入口。
 *
 * 无论多少条，都按固定批量切分后走 translateOneBatch ——
 * 该函数内置「严格提示重试 → 对半拆分」的收敛策略。
 *
 * （旧实现有一条「条数不多就一次性全发」的分支：实测 200 条会返回 170 条，
 *   且重试是重发同样输入、几乎必然再失败，最终级联到最慢的逐条路径。）
 */
async function translateEntriesCore(
  entries,
  client,
  llmModel,
  targetLang,
  progressBase = 0,
  progressRange = 100,
) {
  if (entries.length === 0) return [];

  if (entries.length <= MAX_LINES_PER_REQUEST) {
    // 小规模：单次请求即可，但同样享受拆分兜底
    sendLog(`  共 ${entries.length} 条，单批翻译`);
    try {
      return await translateOneBatch(entries, 0, 1, client, llmModel, targetLang);
    } catch (err) {
      sendLog(`  ❌ 翻译失败: ${err.message}`);
      throw err;
    }
  }

  try {
    return await translateEntriesInBatchesCore(
      entries, client, llmModel, targetLang, progressBase, progressRange,
    );
  } catch (err) {
    sendLog(`  ❌ 翻译失败: ${err.message}`);
    throw err;
  }
}

/**
 * 把 SDK 抛出的 HTTP 错误翻译成人能看懂、能动手修的提示。
 *
 * 最常见的是 404：baseURL 少了 Anthropic 兼容子路径（例如 DeepSeek 需要 /anthropic）。
 * SDK 对 404 常常只报 "404 status code (no body)"，完全没有线索，很容易卡住。
 */
function explainApiError(err, baseUrl, llmModel) {
  const status = err && (err.status || err.statusCode);
  const where = `${baseUrl}  (SDK 实际请求: ${String(baseUrl).replace(/\/+$/, "")}/v1/messages)`;

  if (status === 404) {
    return new Error(
      `API 返回 404 —— 地址不存在，通常是 Base URL 少了 Anthropic 兼容子路径。\n` +
      `  当前 Base URL: ${where}\n` +
      `  • DeepSeek 官方正确地址: https://api.deepseek.com/anthropic\n` +
      `  • 其它中转服务请确认其 Anthropic 兼容入口（一般形如 https://xxx/v1 或 https://xxx/anthropic）`
    );
  }
  if (status === 401 || status === 403) {
    return new Error(
      `API 鉴权失败 (${status}) —— 请检查 API Key 是否正确、是否已充值。\n` +
      `  Base URL: ${baseUrl}`
    );
  }
  if (status === 400) {
    return new Error(
      `API 返回 400 —— 请求被拒，通常与 Base URL 协议不匹配或模型名无效有关。\n` +
      `  Base URL: ${baseUrl}\n  模型: ${llmModel}\n  原始信息: ${err.message}`
    );
  }
  if (status === 429) {
    return new Error(`API 限流 (429) —— 请稍后重试或降低并发。\n  原始信息: ${err.message}`);
  }
  if (err && /fetch failed|ENOTFOUND|ECONNREFUSED|ETIMEDOUT|getaddrinfo/i.test(err.message || "")) {
    return new Error(
      `无法连接到 API 服务 —— 请检查网络或代理。\n  Base URL: ${baseUrl}\n  原始信息: ${err.message}`
    );
  }
  return err;
}

const DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic";

/**
 * 规范化 Anthropic 兼容 API 的 baseURL。
 *
 * 代码走的是 Anthropic SDK（会自动在 baseURL 后拼 `/v1/messages`），但很多
 * 服务商的 Anthropic 兼容接口挂在子路径下，用户很容易只填站点根地址：
 *
 *   https://api.deepseek.com          → 实际请求 /v1/messages          → 404
 *   https://api.deepseek.com/anthropic → 实际请求 /anthropic/v1/messages → 200
 *
 * 这里对已知的站点根地址自动补上子路径，避免用户踩坑。
 */
function normalizeBaseUrl(raw) {
  const url = (raw || "").trim();
  if (!url) return DEFAULT_BASE_URL;

  // 去掉结尾斜杠，便于判断
  const trimmed = url.replace(/\/+$/, "");
  const lower = trimmed.toLowerCase();

  // DeepSeek 官方站点：Anthropic 兼容接口固定在 /anthropic 子路径下。
  // 用户可能填根地址、/v1 或 /anthropic，统一归一到 /anthropic。
  const dsMatch = lower.match(/^https?:\/\/api\.deepseek\.com(\/.*)?$/);
  if (dsMatch) {
    const origin = trimmed.slice(0, trimmed.length - (dsMatch[1] || "").length);
    return `${origin}/anthropic`;
  }

  return trimmed;
}

/**
 * 对空译文做一次廉价补录。
 *
 * 批量路径若最终仍对不齐会留空，逐条路径重试 2 次后也会留空 ——
 * 留空比写入推理垃圾好，但直接输出会得到空白字幕行。这里用「精简系统提示 +
 * 单条请求」再补一次，成本很低，能救回绝大多数瞬时失败。
 */
async function recoverEmptyTranslations(entries, lines, client, llmModel, targetLang) {
  const lang = LANG_CONFIG[targetLang] || LANG_CONFIG.zh;
  const missing = [];
  for (let i = 0; i < entries.length; i++) {
    if (!lines[i] || !String(lines[i]).trim()) missing.push(i);
  }
  if (missing.length === 0) return lines;

  sendLog(`  ⚠ 有 ${missing.length}/${entries.length} 条译文为空，尝试补录...`);
  const conciseSystem =
    `You are a professional subtitle translator. Translate English subtitle lines into ` +
    `natural, fluent ${lang.target}. Output ONLY the translation — no quotes, no explanations, ` +
    `no reasoning, no notes.`;

  const out = lines.slice();
  let recovered = 0;

  for (const i of missing) {
    if (getIsCancelled()) throw new Error("任务已取消");
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const text = await callWithRetry(async () => {
          const r = await createMessage(client, {
            model: llmModel,
            system: conciseSystem,
            messages: [{ role: "user", content: `Translate into ${lang.target}:\n\n${entries[i].text}` }],
            temperature: 0.2,
            max_tokens: 8192,
          });
          return sanitizeSingleTranslation(extractResponseText(r), targetLang);
        }, `补录第 ${i + 1} 条`);

        if (text && !looksLikeReasoning(text)) {
          out[i] = text;
          recovered++;
          break;
        }
      } catch (e) {
        if (getIsCancelled()) throw new Error("任务已取消");
      }
    }
  }

  const stillMissing = missing.length - recovered;
  sendLog(`  ✓ 补录成功 ${recovered} 条${stillMissing > 0 ? `，仍有 ${stillMissing} 条为空（对应字幕将只显示原文/留空）` : ""}`);
  return out;
}

async function generateSubtitleForBurn(srtPath, apiKey, baseUrl, llmModel, mode) {
  const srtContent = fs.readFileSync(srtPath, "utf-8");
  const entries = parseSrt(srtContent);
  sendLog(`  解析到 ${entries.length} 条字幕`);

  if (entries.length === 0) {
    throw new Error("SRT 文件解析失败，无有效条目");
  }

  const resolvedBaseUrl = normalizeBaseUrl(baseUrl);
  if (baseUrl && resolvedBaseUrl !== baseUrl.trim().replace(/\/+$/, "")) {
    sendLog(`  ℹ Base URL 已自动补全: ${baseUrl} → ${resolvedBaseUrl}`);
  }
  sendLog(`  API: ${resolvedBaseUrl}  模型: ${llmModel}`);

  const client = new Anthropic({
    apiKey: apiKey,
    baseURL: resolvedBaseUrl,
  });

  if (mode === "en") {
    sendLog("  ✓ 使用原始英语字幕");
    sendProgress("translate", 100);
    return srtPath;
  }

  if (mode === "zh") {
    sendLog("  正在翻译为中文...");
    sendProgress("translate", 20);
    const zhLines = await translateEntriesCore(entries, client, llmModel, "zh", 20, 75)
      .then((l) => recoverEmptyTranslations(entries, l, client, llmModel, "zh"))
      .catch((e) => { throw explainApiError(e, resolvedBaseUrl, llmModel); });
    const outputPath = srtPath.replace(/\.srt$/, ".zh-only.srt");
    const monoZh = buildMonolingualSrt(entries, zhLines);
    fs.writeFileSync(outputPath, monoZh, "utf-8");
    sendProgress("translate", 100);
    sendLog(`  ✓ 中文字幕已保存: ${path.basename(outputPath)} (${zhLines.length}/${entries.length} 条)`);
    return outputPath;
  }

  if (mode === "en-zh") {
    sendLog("  正在翻译为中文（中英双语）...");
    sendProgress("translate", 20);
    const zhLines = await translateEntriesCore(entries, client, llmModel, "zh", 20, 75)
      .then((l) => recoverEmptyTranslations(entries, l, client, llmModel, "zh"))
      .catch((e) => { throw explainApiError(e, resolvedBaseUrl, llmModel); });
    const outputPath = srtPath.replace(/\.srt$/, ".en-zh.srt");
    const bilingual = buildBilingualSrt(entries, zhLines);
    fs.writeFileSync(outputPath, bilingual, "utf-8");
    sendProgress("translate", 100);
    sendLog(`  ✓ 中英双语字幕已保存: ${path.basename(outputPath)} (${zhLines.length}/${entries.length} 条)`);
    return outputPath;
  }

  if (mode === "tr") {
    sendLog("  正在翻译为土耳其语...");
    sendProgress("translate", 20);
    const trLines = await translateEntriesCore(entries, client, llmModel, "tr", 20, 75)
      .then((l) => recoverEmptyTranslations(entries, l, client, llmModel, "tr"))
      .catch((e) => { throw explainApiError(e, resolvedBaseUrl, llmModel); });
    const outputPath = srtPath.replace(/\.srt$/, ".tr-only.srt");
    const monoTr = buildMonolingualSrt(entries, trLines);
    fs.writeFileSync(outputPath, monoTr, "utf-8");
    sendProgress("translate", 100);
    sendLog(`  ✓ 土耳其语字幕已保存: ${path.basename(outputPath)} (${trLines.length}/${entries.length} 条)`);
    return outputPath;
  }

  if (mode === "en-tr") {
    sendLog("  正在翻译为土耳其语（英土双语）...");
    sendProgress("translate", 20);
    const trLines = await translateEntriesCore(entries, client, llmModel, "tr", 20, 75)
      .then((l) => recoverEmptyTranslations(entries, l, client, llmModel, "tr"))
      .catch((e) => { throw explainApiError(e, resolvedBaseUrl, llmModel); });
    const outputPath = srtPath.replace(/\.srt$/, ".en-tr.srt");
    const bilingual = buildBilingualSrt(entries, trLines);
    fs.writeFileSync(outputPath, bilingual, "utf-8");
    sendProgress("translate", 100);
    sendLog(`  ✓ 英土双语字幕已保存: ${path.basename(outputPath)} (${trLines.length}/${entries.length} 条)`);
    return outputPath;
  }

  throw new Error(`未知的烧录模式: ${mode}`);
}

module.exports = {
  LANG_CONFIG,
  buildTranslationSystemPrompt,
  translateEntriesCore,
  generateSubtitleForBurn,
};
