/**
 * AIProviderRouter — Multi-provider text completion router.
 *
 * The ONLY module responsible for selecting an AI provider and enforcing
 * failover. Text chat completions flow:
 *
 *   Gemini 2.5 Flash (Primary)
 *      ↓  (on timeout / network error / 429 / 5xx / invalid / empty response)
 *   NVIDIA NIM — deepseek-ai/deepseek-v4-pro
 *      ↓  (on failure)
 *   OpenRouter — model from OPENROUTER_MODEL env
 *      ↓
 *   Surface the upstream error in the SAME shape callers already use.
 *
 * Scope: TEXT CHAT COMPLETIONS ONLY.
 * Modal-specific Gemini features (Live, TTS, Image gen, googleSearch grounding)
 * are intentionally NOT routed here — they keep using Gemini directly.
 *
 * Design constraints enforced:
 *  - AbortController per provider attempt. Never leaks pending requests.
 *  - Independent timeout per provider. On timeout: abort, release, move on.
 *  - The first VALID provider response wins; remaining providers are not tried.
 *  - Conversation history is forwarded EXACTLY as received (untouched).
 *  - Providers never know about each other; caller never knows which answered.
 *  - API keys come ONLY from environment variables. Never hardcoded.
 *  - Streaming support preserved (router exposes a streaming entrypoint that
 *    mirrors the existing onChunk(text, sources) contract).
 */

import { GoogleGenAI } from "@google/genai";
import { getSystemInstruction } from "../constants";
import { withRetry } from "./retryUtils.ts";

// ─── Types ───────────────────────────────────────────────────────────

export type ChatRole = "user" | "model" | "assistant";
export interface ChatPart {
  text: string;
  inlineData?: { data: string; mimeType: string };
}
export interface HistoryTurn {
  role: ChatRole | string;
  parts: ChatPart[];
}
export interface CitationSource {
  title: string;
  url: string;
}

interface CompletionRequest {
  userMessage: string;
  history: HistoryTurn[];
  systemInstruction: string;
  temperature: number;
  /** Gemini-only tool — ignored by non-Gemini providers. */
  enableGoogleSearch?: boolean;
}

interface StreamRequest extends CompletionRequest {
  onChunk: (text: string, sources?: CitationSource[]) => void;
  /** Caller-controlled abort (e.g. ChatManager). */
  abortSignal?: AbortSignal;
}

type CompletionResult = string;
interface StreamResult {
  text: string;
  sources: CitationSource[] | undefined;
}

/** Generic transport-level error so the router can decide to failover. */
class ProviderError extends Error {
  constructor(message: string, public readonly reason: string) {
    super(message);
    this.name = "ProviderError";
  }
}

// ─── Provider timeouts (ms), each provider independent ───────────────

const GEMINI_TIMEOUT_MS = 60_000;
const NVIDIA_TIMEOUT_MS = 60_000;
const OPENROUTER_TIMEOUT_MS = 60_000;

// ─── Environment helpers ────────────────────────────────────────────

function readEnv(name: string): string {
  try {
    // Vite injects process.env.* via define; fall back to import.meta.env.
    const fromProcess = (globalThis as any).process?.env?.[name];
    if (fromProcess) return fromProcess;
  } catch {
    /* ignore — process may be undefined in some sandboxes */
  }
  try {
    const v = (import.meta as any).env?.[name];
    if (v) return v;
  } catch {
    /* import.meta.env may be unavailable in some runtimes */
  }
  return "";
}

export function getGeminiApiKey(): string {
  // Existing key resolution order is preserved exactly.
  try {
    const v = (import.meta as any).env?.VITE_GEMINI_API_KEY;
    if (v) return v;
  } catch {
    /* ignore */
  }
  return readEnv("GEMINI_API_KEY") || readEnv("API_KEY") || "";
}

/**
 * NVIDIA NIM API key. Vite-injected `VITE_NVIDIA_API_KEY` takes precedence
 * (matching getGeminiApiKey's pattern); falls back to the unprefixed form
 * for non-Vite runtimes.
 */
export function getNvidiaApiKey(): string {
  try {
    const v = (import.meta as any).env?.VITE_NVIDIA_API_KEY;
    if (v) return v;
  } catch {
    /* ignore */
  }
  return readEnv("NVIDIA_API_KEY");
}

/**
 * OpenRouter API key + model. Same resolution pattern as the other providers.
 */
export function getOpenRouterApiKey(): string {
  try {
    const v = (import.meta as any).env?.VITE_OPENROUTER_API_KEY;
    if (v) return v;
  } catch {
    /* ignore */
  }
  return readEnv("OPENROUTER_API_KEY");
}

export function getOpenRouterModel(): string {
  try {
    const v = (import.meta as any).env?.VITE_OPENROUTER_MODEL;
    if (v) return v;
  } catch {
    /* ignore */
  }
  return readEnv("OPENROUTER_MODEL");
}

const NVIDIA_MODEL = "deepseek-ai/deepseek-v4-pro";
const NVIDIA_BASE_URL =
  "https://integrate.api.nvidia.com/v1/chat/completions";
const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions";

// ─── Abort utilities ────────────────────────────────────────────────

/**
 * Create an AbortController that fires after `timeoutMs`, AND is linked to an
 * optional external abort signal. If the external signal aborts, our controller
 * aborts too. Returns the controller and a cleanup fn to clear the timer.
 */
function makeScopedController(
  timeoutMs: number,
  externalAbort?: AbortSignal
): { controller: AbortController; cleanup: () => void } {
  const controller = new AbortController();

  let timer: ReturnType<typeof setTimeout> | null = null;
  const onExternalAbort = () => controller.abort();

  if (externalAbort) {
    if (externalAbort.aborted) {
      controller.abort();
    } else {
      externalAbort.addEventListener("abort", onExternalAbort, { once: true });
    }
  }

  timer = setTimeout(() => controller.abort(), timeoutMs);

  const cleanup = () => {
    if (timer) clearTimeout(timer);
    if (externalAbort) {
      externalAbort.removeEventListener("abort", onExternalAbort);
    }
  };

  return { controller, cleanup };
}

// ─── Failover predicate ─────────────────────────────────────────────

/**
 * A provider's response is considered a failover-trigger if it represents a
 * transport/availability problem — never a valid AI answer.  A VALID response
 * must NEVER trigger fallback.
 */
function isFailoverError(err: unknown): boolean {
  if (!err) return true;

  const name = (err as any)?.name ?? "";
  const msg = (err instanceof Error ? err.message : String(err)) || "";
  const combined = `${name} ${msg}`.toLowerCase();

  if (name === "AbortError") return true; // timeout or external abort
  if (/timeout/.test(combined)) return true;
  if (/networkerror|network (request )?failed|failed to fetch/.test(combined))
    return true;
  if (/dns|enamee_not_found|enotfound/.test(combined)) return true;
  if (/429/.test(combined)) return true; // rate limit
  if (/\b5\d\d\b|500|502|503|504/.test(combined)) return true; // server errors
  if (/unavailable|provider unavailable|service unavailable/.test(combined))
    return true;
  if (/connection reset|econnreset|socket hang up/.test(combined)) return true;
  if (/invalid json|unexpected token|json parse/.test(combined)) return true;
  if (/empty (response )?content|no content|nothing received/.test(combined))
    return true;
  return false;
}

/** Validate that a completion actually returned usable text. */
function hasContent(text: string | null | undefined): boolean {
  return typeof text === "string" && text.trim().length > 0;
}

/** True if either the scoped controller or the external caller aborted. */
function isAborted(
  controller: AbortController,
  external?: AbortSignal
): boolean {
  return controller.signal.aborted || Boolean(external?.aborted);
}

// ─── Conversation shape translation ────────────────────────────────

type OpenAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

interface OpenAIMessage {
  role: "system" | "user" | "assistant";
  content: string | OpenAIContentPart[];
}

/**
 * Translate Gemini-style history (role: user|model, parts with text/inlineData)
 * into the OpenAI-style chat format used by NVIDIA NIM and OpenRouter.
 *
 * The conversation is forwarded EXACTLY — no edits, no summarization.
 */
function toOpenAIMessages(req: CompletionRequest): OpenAIMessage[] {
  const messages: OpenAIMessage[] = [
    { role: "system", content: req.systemInstruction },
  ];

  for (const turn of req.history) {
    const role: "user" | "assistant" =
      String(turn.role) === "model" || String(turn.role) === "assistant"
        ? "assistant"
        : "user";

    if (!turn.parts || turn.parts.length === 0) continue;

    const parts: OpenAIContentPart[] = [];
    for (const p of turn.parts) {
      if (p.inlineData?.data && p.inlineData?.mimeType) {
        parts.push({
          type: "image_url",
          image_url: {
            url: `data:${p.inlineData.mimeType};base64,${p.inlineData.data}`,
          },
        });
      }
      if (p.text && p.text.trim().length > 0) {
        parts.push({ type: "text", text: p.text });
      }
    }

    if (parts.length === 0) continue;

    const content =
      parts.length === 1 && parts[0].type === "text"
        ? (parts[0] as { text: string }).text
        : parts;

    messages.push({ role, content });
  }

  // The final user message is always appended verbatim.
  messages.push({ role: "user", content: req.userMessage });

  return messages;
}

// ─── Shared fetch helper (NVIDIA / OpenRouter) ──────────────────────

interface FetchOpts {
  url: string;
  apiKey: string;
  body: Record<string, unknown>;
  timeoutMs: number;
  abortSignal?: AbortSignal;
  extraHeaders?: Record<string, string>;
}

async function postChat(opts: FetchOpts): Promise<string> {
  const { controller, cleanup } = makeScopedController(
    opts.timeoutMs,
    opts.abortSignal
  );

  try {
    const res = await fetch(opts.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${opts.apiKey}`,
        ...(opts.extraHeaders ?? {}),
      },
      body: JSON.stringify(opts.body),
      signal: controller.signal,
    });

    if (res.status === 429 || res.status >= 500) {
      throw new ProviderError(
        `upstream HTTP ${res.status}`,
        `HTTP_${res.status}`
      );
    }
    if (!res.ok) {
      // Non-failover HTTP error (4xx other than 429). Treat as a hard failure
      // of this provider but DO surface it via the standard failover path,
      // since a malformed request to one provider may still succeed on another.
      throw new ProviderError(`upstream HTTP ${res.status}`, `HTTP_${res.status}`);
    }

    const raw = await res.text();
    if (!raw) throw new ProviderError("empty response body", "EMPTY");

    let parsed: any;
    try {
      parsed = JSON.parse(raw);
    } catch {
      throw new ProviderError("invalid JSON from upstream", "INVALID_JSON");
    }

    const content =
      parsed?.choices?.[0]?.message?.content ??
      parsed?.choices?.[0]?.delta?.content ??
      "";
    if (typeof content !== "string" || !content.trim()) {
      throw new ProviderError("missing content in response", "NO_CONTENT");
    }
    return content;
  } catch (err) {
    // Translate fetch-level abort into the router's vocabulary.
    if ((err as any)?.name === "AbortError") {
      throw new ProviderError("request aborted / timed out", "TIMEOUT");
    }
    if (err instanceof ProviderError) throw err;
    // Network-level fetch failures (DNS, reset, etc.)
    throw new ProviderError(
      err instanceof Error ? err.message : String(err),
      "NETWORK"
    );
  } finally {
    cleanup();
  }
}

// ─── Provider implementations ───────────────────────────────────────

/**
 * Gemini — primary provider. Uses the official @google/genai SDK so the
 * existing gemini-2.5-flash model + grounding behavior stays identical when
 * Gemini is healthy. Returns plain text.
 */
async function geminiComplete(
  req: CompletionRequest,
  abortSignal?: AbortSignal
): Promise<string> {
  const apiKey = getGeminiApiKey();
  if (!apiKey) {
    throw new ProviderError("Gemini API key missing", "NO_KEY");
  }

  const ai = new GoogleGenAI({ apiKey });
  const userParts: any[] = [{ text: req.userMessage }];

  for (const turn of req.history) {
    for (const p of turn.parts) {
      if (p.inlineData?.data) {
        userParts.push({
          inlineData: { data: p.inlineData.data, mimeType: p.inlineData.mimeType },
        });
      }
    }
  }

  const { controller, cleanup } = makeScopedController(
    GEMINI_TIMEOUT_MS,
    abortSignal
  );

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [
        ...req.history.map((h) => ({ role: h.role, parts: h.parts })),
        { role: "user", parts: userParts },
      ],
      config: {
        systemInstruction: req.systemInstruction,
        temperature: req.temperature,
        tools: req.enableGoogleSearch ? [{ googleSearch: {} }] : undefined,
        abortSignal: controller.signal,
      },
    });

    const text = response?.text;
    if (!hasContent(text)) {
      throw new ProviderError("Gemini returned no content", "NO_CONTENT");
    }
    return text as string;
  } catch (err) {
    if ((err as any)?.name === "AbortError") {
      throw new ProviderError("Gemini aborted/timeout", "TIMEOUT");
    }
    if (err instanceof ProviderError) throw err;
    throw new ProviderError(
      err instanceof Error ? err.message : String(err),
      "GEMINI_ERROR"
    );
  } finally {
    cleanup();
  }
}

/**
 * NVIDIA NIM — secondary provider. deepseek-ai/deepseek-v4-pro.
 */
async function nvidiaComplete(
  req: CompletionRequest,
  abortSignal?: AbortSignal
): Promise<string> {
  const apiKey = getNvidiaApiKey();
  if (!apiKey) {
    throw new ProviderError("NVIDIA API key missing", "NO_KEY");
  }

  return postChat({
    url: NVIDIA_BASE_URL,
    apiKey,
    timeoutMs: NVIDIA_TIMEOUT_MS,
    abortSignal,
    body: {
      model: NVIDIA_MODEL,
      messages: toOpenAIMessages(req),
      temperature: req.temperature,
      stream: false,
      max_tokens: 4096,
    },
  });
}

/**
 * OpenRouter — tertiary provider. Model taken from env.
 */
async function openRouterComplete(
  req: CompletionRequest,
  abortSignal?: AbortSignal
): Promise<string> {
  const apiKey = getOpenRouterApiKey();
  const model = getOpenRouterModel();
  if (!apiKey || !model) {
    throw new ProviderError("OpenRouter key or model missing", "NO_KEY");
  }

  return postChat({
    url: OPENROUTER_BASE_URL,
    apiKey,
    timeoutMs: OPENROUTER_TIMEOUT_MS,
    abortSignal,
    body: {
      model,
      messages: toOpenAIMessages(req),
      temperature: req.temperature,
      stream: false,
    },
  });
}

// ─── Public: non-streaming router ───────────────────────────────────

/**
 * Run providers in failover order, returning the FIRST successful result.
 */
async function routeCompletion(
  req: CompletionRequest,
  providers: Array<{
    name: string;
    fn: (r: CompletionRequest, a?: AbortSignal) => Promise<string>;
  }>,
  externalAbort?: AbortSignal
): Promise<string> {
  let lastError: unknown = null;

  for (let i = 0; i < providers.length; i++) {
    const { name, fn } = providers[i];

    if (externalAbort?.aborted) {
      throw new ProviderError("aborted by caller", "ABORTED");
    }

    try {
      const text = await fn(req, externalAbort);
      if (hasContent(text)) {
        console.log(`[AIProviderRouter] ${name} success`);
        return text;
      }
      lastError = new ProviderError(`${name} empty content`, "NO_CONTENT");
    } catch (err) {
      lastError = err;

      if (!isFailoverError(err)) {
        // A non-failover error means the provider reached a definitive answer —
        // just not a successful one. We do NOT fall back here per spec: a valid
        // AI response must never trigger fallback, and conversely a definitive
        // refusal is not a transport failure. Re-throw so the caller sees the
        // real error in the standard shape.
        console.warn(`[AIProviderRouter] ${name} non-failover error:`, err);
        throw err;
      }

      const isLast = i === providers.length - 1;
      if (isLast) {
        console.warn(`[AIProviderRouter] ${name} failed (last provider)`, err);
      } else {
        const next = providers[i + 1].name;
        console.warn(
          `[AIProviderRouter] ${name} failed, switching to ${next}`,
          err
        );
      }
    }
  }

  // Every provider failed — rethrow the last error; the caller maps it to its
  // existing error shape (no new error types invented).
  throw lastError ?? new ProviderError("all providers failed", "ALL_FAILED");
}

/**
 * Public non-streaming entrypoint. Returns plain text exactly like the
 * existing getGeminiResponse contract.
 */
export async function routeText(req: {
  userMessage: string;
  history: HistoryTurn[];
  city: string;
  lang: "en" | "ta";
  location?: { lat: number; lng: number };
  temperature?: number;
  enableGoogleSearch?: boolean;
  attachments?: { data: string; mimeType: string }[];
  abortSignal?: AbortSignal;
}): Promise<string> {
  const systemInstruction = getSystemInstruction(
    req.city,
    req.lang,
    req.location
  );

  // Fold attachments into the final user turn (Gemini-shaped).
  const history: HistoryTurn[] = [...req.history];
  const finalParts: ChatPart[] = [{ text: req.userMessage }];
  if (req.attachments?.length) {
    for (const att of req.attachments) {
      finalParts.push({
        text: "",
        inlineData: { data: att.data, mimeType: att.mimeType },
      });
    }
  }

  const completionReq: CompletionRequest = {
    userMessage: req.userMessage,
    history,
    systemInstruction,
    temperature: req.temperature ?? 0.7,
    enableGoogleSearch: req.enableGoogleSearch,
  };

  // Override the final user turn with parts (incl attachments) for Gemini.
  // For OpenAI-shape providers, toOpenAIMessages builds the user message from
  // history's last user turn + the standalone userMessage; to keep behavior
  // uniform we attach the final user turn into history with attachments.
  if (req.attachments?.length) {
    completionReq.history = [
      ...history,
      { role: "user", parts: finalParts },
    ];
    completionReq.userMessage = req.userMessage;
  }

  const providers = [
    { name: "Gemini", fn: geminiComplete },
    { name: "NVIDIA", fn: nvidiaComplete },
    { name: "OpenRouter", fn: openRouterComplete },
  ];

  return routeCompletion(completionReq, providers, req.abortSignal);
}

// ─── Public: streaming router ───────────────────────────────────────

/**
 * Stream a chat completion. Honors the existing contract:
 *   - onChunk(accumulatedText, sources?)  is called as text arrives.
 *   - Returns { text, sources }.
 *   - abortSignal (from ChatManager) is honored at every stage.
 *
 * Streaming is supported natively by Gemini (SDK iterator) and by
 * OpenAI-shaped providers via SSE. On a provider failure, the router silently
 * moves to the next provider WITHOUT re-emitting partial data already shown;
 * each provider starts the stream from the beginning so the visible text never
 * merges across providers.
 */
export async function routeTextStream(req: {
  userMessage: string;
  history: HistoryTurn[];
  city: string;
  lang: "en" | "ta";
  location?: { lat: number; lng: number };
  temperature?: number;
  enableGoogleSearch?: boolean;
  attachments?: { data: string; mimeType: string }[];
  onChunk: (text: string, sources?: CitationSource[]) => void;
  abortSignal?: AbortSignal;
}): Promise<StreamResult> {
  if (req.abortSignal?.aborted) {
    return { text: "", sources: undefined };
  }

  const systemInstruction = getSystemInstruction(
    req.city,
    req.lang,
    req.location
  );

  const history: HistoryTurn[] = [...req.history];
  const finalParts: ChatPart[] = [{ text: req.userMessage }];
  if (req.attachments?.length) {
    for (const att of req.attachments) {
      finalParts.push({
        text: "",
        inlineData: { data: att.data, mimeType: att.mimeType },
      });
    }
  }

  const completionReq: CompletionRequest = {
    userMessage: req.userMessage,
    history,
    systemInstruction,
    temperature: req.temperature ?? 0.7,
    enableGoogleSearch: req.enableGoogleSearch,
  };

  if (req.attachments?.length) {
    completionReq.history = [
      ...history,
      { role: "user", parts: finalParts },
    ];
  }

  const providers: Array<{
    name: string;
    fn: (r: StreamRequest) => Promise<StreamResult>;
  }> = [
    { name: "Gemini", fn: streamGemini },
    { name: "NVIDIA", fn: streamNvidia },
    { name: "OpenRouter", fn: streamOpenRouter },
  ];

  let lastError: unknown = null;

  for (let i = 0; i < providers.length; i++) {
    const { name, fn } = providers[i];

    if (req.abortSignal?.aborted) {
      return { text: "", sources: undefined };
    }

    try {
      const result = await fn({
        ...completionReq,
        onChunk: req.onChunk,
        abortSignal: req.abortSignal,
      });

      if (hasContent(result.text)) {
        console.log(`[AIProviderRouter] ${name} stream success`);
        return result;
      }
      lastError = new ProviderError(`${name} empty stream`, "NO_CONTENT");
    } catch (err) {
      if (req.abortSignal?.aborted || (err as any)?.name === "AbortError") {
        // External abort — never fall over to the next provider.
        return { text: "", sources: undefined };
      }
      lastError = err;

      if (!isFailoverError(err)) {
        console.warn(
          `[AIProviderRouter] ${name} non-failover stream error:`,
          err
        );
        throw err;
      }

      const isLast = i === providers.length - 1;
      if (isLast) {
        console.warn(`[AIProviderRouter] ${name} stream failed (last)`, err);
      } else {
        const next = providers[i + 1].name;
        console.warn(
          `[AIProviderRouter] ${name} stream failed, switching to ${next}`,
          err
        );
      }
    }
  }

  throw lastError ?? new ProviderError(
    "all streaming providers failed",
    "ALL_FAILED"
  );
}

/**
 * Gemini streaming. Uses the official SDK contract (withRetry + grounding
 * source extraction) — identical to the prior implementation when Gemini is
 * the successful provider, so sources/citations continue to work.
 */
async function streamGemini(req: StreamRequest): Promise<StreamResult> {
  const apiKey = getGeminiApiKey();
  if (!apiKey) throw new ProviderError("Gemini API key missing", "NO_KEY");

  const ai = new GoogleGenAI({ apiKey });
  const userParts: any[] = [{ text: req.userMessage }];
  for (const turn of req.history) {
    for (const p of turn.parts) {
      if (p.inlineData?.data) {
        userParts.push({
          inlineData: { data: p.inlineData.data, mimeType: p.inlineData.mimeType },
        });
      }
    }
  }

  const { controller, cleanup } = makeScopedController(
    GEMINI_TIMEOUT_MS,
    req.abortSignal
  );

  try {
    const result = await withRetry(
      async () => {
        return await ai.models.generateContentStream({
          model: "gemini-2.5-flash",
          contents: [
            ...req.history.map((h) => ({ role: h.role, parts: h.parts })),
            { role: "user", parts: userParts },
          ],
          config: {
            systemInstruction: req.systemInstruction,
            temperature: req.temperature,
            tools: req.enableGoogleSearch ? [{ googleSearch: {} }] : undefined,
            abortSignal: controller.signal,
          },
        });
      },
      { maxRetries: 2, baseDelayMs: 1500 }
    );

    if (!result) throw new ProviderError("Gemini stream unavailable", "UNAVAILABLE");

    let fullText = "";
    const sources: CitationSource[] = [];
    const seenUrls = new Set<string>();

    for await (const chunk of result) {
      if (req.abortSignal?.aborted || controller.signal.aborted) break;

      const chunkText = chunk.text;
      if (chunkText) {
        fullText += chunkText;

        const gm = chunk.candidates?.[0]?.groundingMetadata;
        if (gm?.groundingChunks) {
          for (const c of gm.groundingChunks) {
            if (c.web?.uri && c.web?.title && !seenUrls.has(c.web.uri)) {
              seenUrls.add(c.web.uri);
              sources.push({ title: c.web.title, url: c.web.uri });
            }
          }
        }

        req.onChunk(fullText, sources.length > 0 ? sources : undefined);
      }
    }

    if (!hasContent(fullText)) {
      throw new ProviderError("Gemini stream produced no content", "NO_CONTENT");
    }

    return { text: fullText, sources: sources.length > 0 ? sources : undefined };
  } catch (err) {
    if ((err as any)?.name === "AbortError") {
      throw new ProviderError("Gemini stream aborted/timeout", "TIMEOUT");
    }
    if (err instanceof ProviderError) throw err;
    throw new ProviderError(
      err instanceof Error ? err.message : String(err),
      "GEMINI_STREAM_ERROR"
    );
  } finally {
    cleanup();
  }
}

/**
 * Generic SSE reader for OpenAI-shaped /chat/completions streams (NVIDIA NIM &
 * OpenRouter). Parses `data:` lines, accumulates delta.content, and forwards
 * via onChunk. Honors external abort via the scoped controller.
 */
async function streamOpenAICompatible(
  opts: {
    url: string;
    apiKey: string;
    body: Record<string, unknown>;
    timeoutMs: number;
    onChunk: (text: string, sources?: CitationSource[]) => void;
    abortSignal?: AbortSignal;
  }
): Promise<StreamResult> {
  const { controller, cleanup } = makeScopedController(
    opts.timeoutMs,
    opts.abortSignal
  );

  let fullText = "";

  try {
    const res = await fetch(opts.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${opts.apiKey}`,
        Accept: "text/event-stream",
      },
      body: JSON.stringify({ ...opts.body, stream: true }),
      signal: controller.signal,
    });

    if (res.status === 429 || res.status >= 500) {
      throw new ProviderError(`upstream HTTP ${res.status}`, `HTTP_${res.status}`);
    }
    if (!res.ok) {
      throw new ProviderError(`upstream HTTP ${res.status}`, `HTTP_${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) {
      throw new ProviderError("no response body for stream", "NO_CONTENT");
    }

    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    const pump = async (): Promise<void> => {
      while (true) {
        if (isAborted(controller, opts.abortSignal)) return;
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let nl: number;
        while ((nl = buffer.indexOf("\n")) !== -1) {
          let line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);

          if (!line) continue;
          if (line.startsWith(":")) continue; // SSE comment / keep-alive
          if (line.startsWith("data:")) line = line.slice(5).trim();
          else continue;

          if (line === "[DONE]") return;

          try {
            const evt = JSON.parse(line);
            const delta =
              evt?.choices?.[0]?.delta?.content ??
              evt?.choices?.[0]?.message?.content ??
              "";
            if (typeof delta === "string" && delta.length > 0) {
              fullText += delta;
              opts.onChunk(fullText, undefined);
            }
          } catch {
            // partial JSON across chunk boundary — ignore; next pump completes it
          }
        }
      }
    };

    await pump();

    if (!hasContent(fullText)) {
      throw new ProviderError("stream produced no content", "NO_CONTENT");
    }
    return { text: fullText, sources: undefined };
  } catch (err) {
    if ((err as any)?.name === "AbortError") {
      throw new ProviderError("stream aborted/timeout", "TIMEOUT");
    }
    if (err instanceof ProviderError) throw err;
    throw new ProviderError(
      err instanceof Error ? err.message : String(err),
      "NETWORK"
    );
  } finally {
    cleanup();
  }
}

async function streamNvidia(req: StreamRequest): Promise<StreamResult> {
  const apiKey = getNvidiaApiKey();
  if (!apiKey) throw new ProviderError("NVIDIA API key missing", "NO_KEY");

  return streamOpenAICompatible({
    url: NVIDIA_BASE_URL,
    apiKey,
    timeoutMs: NVIDIA_TIMEOUT_MS,
    onChunk: req.onChunk,
    abortSignal: req.abortSignal,
    body: {
      model: NVIDIA_MODEL,
      messages: toOpenAIMessages(req),
      temperature: req.temperature,
      max_tokens: 4096,
    },
  });
}

async function streamOpenRouter(req: StreamRequest): Promise<StreamResult> {
  const apiKey = getOpenRouterApiKey();
  const model = getOpenRouterModel();
  if (!apiKey || !model) {
    throw new ProviderError("OpenRouter key or model missing", "NO_KEY");
  }

  return streamOpenAICompatible({
    url: OPENROUTER_BASE_URL,
    apiKey,
    timeoutMs: OPENROUTER_TIMEOUT_MS,
    onChunk: req.onChunk,
    abortSignal: req.abortSignal,
    body: {
      model,
      messages: toOpenAIMessages(req),
      temperature: req.temperature,
    },
  });
}

// ─── Public: prompt-improvement router ──────────────────────────────

/**
 * improvePrompt is a single-turn text completion. Routed through
 * Gemini → NVIDIA → OpenRouter identically to chat. Same failover rules apply.
 * Returns plain text exactly like the existing improvePrompt contract.
 */
export async function routeImprovePrompt(req: {
  prompt: string;
  city: string;
  abortSignal?: AbortSignal;
}): Promise<string> {
  if (!req.prompt.trim()) return "";

  const instruction = `Transform this short query about "${req.city}" into a highly detailed, professional, and descriptive prompt for a travel assistant. Make it clear, smart, and aimed at getting the best recommendations. Keep it under 60 words: "${req.prompt}"`;

  const completionReq: CompletionRequest = {
    userMessage: instruction,
    history: [],
    systemInstruction: "",
    temperature: 0.7,
  };

  const providers = [
    { name: "Gemini", fn: geminiComplete },
    { name: "NVIDIA", fn: nvidiaComplete },
    { name: "OpenRouter", fn: openRouterComplete },
  ];

  try {
    const raw = await routeCompletion(completionReq, providers, req.abortSignal);
    return raw?.trim().replace(/^"|"$/g, "") || req.prompt;
  } catch {
    // Caller already has graceful degradation for improvePrompt failures.
    throw new ProviderError("improvePrompt failed across providers", "ALL_FAILED");
  }
}

// Exported for limited test/inspection clarity; not used by callers.
export const __routerInternals = {
  isFailoverError,
  toOpenAIMessages,
  GEMINI_TIMEOUT_MS,
  NVIDIA_TIMEOUT_MS,
  OPENROUTER_TIMEOUT_MS,
  NVIDIA_MODEL,
};

// Type-only re-exports so callers see a stable public surface.
export type {
  CompletionResult as _CompletionResult,
  StreamResult as _StreamResult,
};
