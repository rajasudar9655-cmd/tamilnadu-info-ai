import express from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({
  path: path.join(__dirname, '.env')
});
const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Validate environment variables on startup
const requiredEnvVars = ['GEMINI_API_KEY', 'NVIDIA_API_KEY', 'OPENROUTER_API_KEY', 'OPENROUTER_MODEL'];
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    console.warn(`WARNING: ${envVar} is not set`);
  }
}

// Helper to get API keys from server-side env
function getGeminiApiKey() {
  return process.env.GEMINI_API_KEY || '';
}

function getNvidiaApiKey() {
  return process.env.NVIDIA_API_KEY || '';
}

function getOpenRouterApiKey() {
  return process.env.OPENROUTER_API_KEY || '';
}

function getOpenRouterModel() {
  return process.env.OPENROUTER_MODEL || 'openai/gpt-4o-mini';
}

// --- Provider implementations matching aiProviderRouter.ts logic ---

const NVIDIA_MODEL = 'deepseek-ai/deepseek-v4-pro';
const NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';
const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1/chat/completions';
const GEMINI_TIMEOUT_MS = 60000;
const NVIDIA_TIMEOUT_MS = 60000;
const OPENROUTER_TIMEOUT_MS = 60000;

class ProviderError extends Error {
  constructor(message, reason) {
    super(message);
    this.name = "ProviderError";
    this.reason = reason;
  }
}


function makeScopedController(timeoutMs, externalAbort) {
  const controller = new AbortController();
  let timer = null;
  const onExternalAbort = () => controller.abort();

  if (externalAbort) {
    if (externalAbort.aborted) {
      controller.abort();
    } else {
      externalAbort.addEventListener('abort', onExternalAbort, { once: true });
    }
  }

  timer = setTimeout(() => controller.abort(), timeoutMs);

  const cleanup = () => {
    if (timer) clearTimeout(timer);
    if (externalAbort) {
      externalAbort.removeEventListener('abort', onExternalAbort);
    }
  };

  return { controller, cleanup };
}

function isFailoverError(err) {
  if (!err) return true;
  const name = err?.name ?? '';
  const msg = err instanceof Error ? err.message : String(err);
  const combined = `${name} ${msg}`.toLowerCase();

  if (name === 'AbortError') return true;
  if (/timeout/.test(combined)) return true;
  if (/networkerror|network (request )?failed|failed to fetch/.test(combined)) return true;
  if (/dns|enamee_not_found|enotfound/.test(combined)) return true;
  if (/429/.test(combined)) return true;
  if (/\b5\d\d\b|500|502|503|504/.test(combined)) return true;
  if (/unavailable|provider unavailable|service unavailable/.test(combined)) return true;
  if (/connection reset|econnreset|socket hang up/.test(combined)) return true;
  if (/invalid json|unexpected token|json parse/.test(combined)) return true;
  if (/empty (response )?content|no content|nothing received/.test(combined)) return true;
  return false;
}

function hasContent(text) {
  return typeof text === 'string' && text.trim().length > 0;
}

function toOpenAIMessages(req) {
  const messages = [{ role: 'system', content: req.systemInstruction }];
  for (const turn of req.history) {
    const role = String(turn.role) === 'model' || String(turn.role) === 'assistant' ? 'assistant' : 'user';
    if (!turn.parts || turn.parts.length === 0) continue;
    const parts = [];
    for (const p of turn.parts) {
      if (p.inlineData?.data && p.inlineData?.mimeType) {
        parts.push({ type: 'image_url', image_url: { url: `data:${p.inlineData.mimeType};base64,${p.inlineData.data}` } });
      }
      if (p.text && p.text.trim().length > 0) {
        parts.push({ type: 'text', text: p.text });
      }
    }
    if (parts.length === 0) continue;
    const content = parts.length === 1 && parts[0].type === 'text' ? parts[0].text : parts;
    messages.push({ role, content });
  }
  messages.push({ role: 'user', content: req.userMessage });
  return messages;
}

async function postChat(opts) {
  const { controller, cleanup } = makeScopedController(opts.timeoutMs, opts.abortSignal);
  try {
    const res = await fetch(opts.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${opts.apiKey}`, ...(opts.extraHeaders ?? {}) },
      body: JSON.stringify(opts.body),
      signal: controller.signal,
    });
    if (res.status === 429 || res.status >= 500) {
      throw new ProviderError(`upstream HTTP ${res.status}`, `HTTP_${res.status}`);
    }
    if (!res.ok) {
      throw new ProviderError(`upstream HTTP ${res.status}`, `HTTP_${res.status}`);
    }
    const raw = await res.text();
    if (!raw) throw new ProviderError('empty response body', 'EMPTY');
    let parsed;
    try { parsed = JSON.parse(raw); } catch { throw new ProviderError('invalid JSON from upstream', 'INVALID_JSON'); }
    const content = parsed?.choices?.[0]?.message?.content ?? parsed?.choices?.[0]?.delta?.content ?? '';
    if (typeof content !== 'string' || !content.trim()) {
      throw new ProviderError('missing content in response', 'NO_CONTENT');
    }
    return content;
  } catch (err) {
    if (err?.name === 'AbortError') throw new ProviderError('request aborted / timed out', 'TIMEOUT');
    if (err instanceof ProviderError) throw err;
    throw new ProviderError(err instanceof Error ? err.message : String(err), 'NETWORK');
  } finally { cleanup(); }
}

async function geminiComplete(req, abortSignal) {
  const apiKey = getGeminiApiKey();
  if (!apiKey) throw new ProviderError('Gemini API key missing', 'NO_KEY');
  const ai = new GoogleGenAI({ apiKey });
  const userParts = [{ text: req.userMessage }];
  for (const turn of req.history) {
    for (const p of turn.parts) {
      if (p.inlineData?.data) {
        userParts.push({ inlineData: { data: p.inlineData.data, mimeType: p.inlineData.mimeType } });
      }
    }
  }
  const { controller, cleanup } = makeScopedController(GEMINI_TIMEOUT_MS, abortSignal);
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [...req.history.map(h => ({ role: h.role, parts: h.parts })), { role: 'user', parts: userParts }],
      config: { systemInstruction: req.systemInstruction, temperature: req.temperature, tools: req.enableGoogleSearch ? [{ googleSearch: {} }] : undefined, abortSignal: controller.signal },
    });
    const text = response?.text;
    if (!hasContent(text)) throw new ProviderError('Gemini returned no content', 'NO_CONTENT');
    return text;
  } catch (err) {
    if (err?.name === 'AbortError') throw new ProviderError('Gemini aborted/timeout', 'TIMEOUT');
    if (err instanceof ProviderError) throw err;
    throw new ProviderError(err instanceof Error ? err.message : String(err), 'GEMINI_ERROR');
  } finally { cleanup(); }
}

function getGroundedEvidence(metadata) {
  const chunks = metadata?.groundingChunks ?? [];
  const supports = metadata?.groundingSupports ?? [];
  const evidenceByIndex = new Map();

  for (const support of supports) {
    const text = support?.segment?.text?.trim();
    if (!text) continue;
    for (const index of support.groundingChunkIndices ?? []) {
      const values = evidenceByIndex.get(index) ?? [];
      values.push(text);
      evidenceByIndex.set(index, values);
    }
  }

  const seen = new Set();
  return chunks.flatMap((chunk, index) => {
    const uri = chunk?.web?.uri;
    const title = chunk?.web?.title;
    const snippet = [...new Set(evidenceByIndex.get(index) ?? [])].join(' ').trim();
    if (!uri || !title || !snippet || seen.has(uri)) return [];
    let parsedUrl;
    try {
      parsedUrl = new URL(uri);
    } catch {
      return [];
    }
    if (!['http:', 'https:'].includes(parsedUrl.protocol)) return [];
    seen.add(uri);
    const titleDomain = /^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/i.test(title) ? title : '';
    const reportedDomain = (chunk?.web?.domain || titleDomain || parsedUrl.hostname)
      .replace(/^www\./i, '')
      .toLowerCase();
    return [{
      title,
      url: parsedUrl.toString(),
      domain: reportedDomain,
      snippet: snippet.slice(0, 700),
      isOfficial: /(^|\.)tn\.gov\.in$/i.test(reportedDomain)
        || /(^|\.)nic\.in$/i.test(reportedDomain)
        || /(^|\.)gov\.in$/i.test(reportedDomain)
        || /(^|\.)indianrailways\.gov\.in$/i.test(reportedDomain)
        || /(^|\.)tamilnadutourism\.tn\.gov\.in$/i.test(reportedDomain)
        || /(^|\.)irctc\.co\.in$/i.test(reportedDomain)
        || /(^|\.)tnstc\.in$/i.test(reportedDomain),
    }];
  });
}

app.post('/api/search/grounded', async (req, res) => {
  const apiKey = getGeminiApiKey();
  if (!apiKey) {
    return res.status(503).json({ error: 'Gemini Google Search grounding is unavailable', reason: 'NO_KEY' });
  }

  const query = typeof req.body?.query === 'string' ? req.body.query.trim() : '';
  const searchQueries = Array.isArray(req.body?.searchQueries)
    ? req.body.searchQueries.filter((value) => typeof value === 'string' && value.trim()).slice(0, 4)
    : [];
  if (!query) return res.status(400).json({ error: 'query required' });

  const searchInstructions = searchQueries.length
    ? searchQueries.map((value, index) => `${index + 1}. ${value}`).join('\n')
    : query;

  const { controller, cleanup } = makeScopedController(GEMINI_TIMEOUT_MS);
  try {
    const ai = new GoogleGenAI({ apiKey });
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [{
        role: 'user',
        parts: [{
          text: `Use Google Search to retrieve live web evidence for the question below. Prefer primary Tamil Nadu government, district government, HR&CE, tourism, transport, and other official sources. Return only concise factual statements directly supported by retrieved pages; do not infer missing values. If no reliable page supports a fact, say it could not be verified. Query variants:\n${searchInstructions}`,
        }],
      }],
      config: { tools: [{ googleSearch: {} }], temperature: 0, abortSignal: controller.signal },
    });

    const metadata = response?.candidates?.[0]?.groundingMetadata;
    const results = getGroundedEvidence(metadata);
    return res.json({
      provider: 'gemini-google-search',
      retrievedAt: new Date().toISOString(),
      results,
    });
  } catch (error) {
    console.warn('[GroundedSearch] Gemini Google Search retrieval failed:', error);
    return res.status(502).json({
      error: 'Gemini Google Search grounding failed',
      reason: 'GROUNDING_FAILED',
    });
  } finally {
    cleanup();
  }
});

async function nvidiaComplete(req, abortSignal) {
  const apiKey = getNvidiaApiKey();
  if (!apiKey) throw new ProviderError('NVIDIA API key missing', 'NO_KEY');
  return postChat({
    url: NVIDIA_BASE_URL, apiKey, timeoutMs: NVIDIA_TIMEOUT_MS, abortSignal,
    body: { model: NVIDIA_MODEL, messages: toOpenAIMessages(req), temperature: req.temperature, stream: false, max_tokens: 4096 },
  });
}

async function openRouterComplete(req, abortSignal) {
  const apiKey = getOpenRouterApiKey();
  const model = getOpenRouterModel();
  if (!apiKey || !model) throw new ProviderError('OpenRouter key or model missing', 'NO_KEY');
  return postChat({
    url: OPENROUTER_BASE_URL, apiKey, timeoutMs: OPENROUTER_TIMEOUT_MS, abortSignal,
    body: { model, messages: toOpenAIMessages(req), temperature: req.temperature, stream: false },
  });
}

async function routeCompletion(req, providers, externalAbort) {
  let lastError = null;
  for (let i = 0; i < providers.length; i++) {
    const { name, fn } = providers[i];
    if (externalAbort?.aborted) throw new ProviderError('aborted by caller', 'ABORTED');
    try {
      const text = await fn(req, externalAbort);
      if (hasContent(text)) { console.log(`[ServerRouter] ${name} success`); return text; }
      lastError = new ProviderError(`${name} empty content`, 'NO_CONTENT');
    } catch (err) {
      lastError = err;
      if (!isFailoverError(err)) {
        console.warn(`[ServerRouter] ${name} non-failover error:`, err);
        throw err;
      }
      const isLast = i === providers.length - 1;
      if (isLast) console.warn(`[ServerRouter] ${name} failed (last provider)`, err);
      else console.warn(`[ServerRouter] ${name} failed, switching to ${providers[i+1].name}`, err);
    }
  }
  throw lastError ?? new ProviderError('all providers failed', 'ALL_FAILED');
}

// --- Streaming implementations ---

async function streamOpenAICompatible(opts) {
  const { controller, cleanup } = makeScopedController(opts.timeoutMs, opts.abortSignal);
  let fullText = '';
  try {
    const res = await fetch(opts.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${opts.apiKey}`, Accept: 'text/event-stream' },
      body: JSON.stringify({ ...opts.body, stream: true }),
      signal: controller.signal,
    });
    if (res.status === 429 || res.status >= 500) throw new ProviderError(`upstream HTTP ${res.status}`, `HTTP_${res.status}`);
    if (!res.ok) throw new ProviderError(`upstream HTTP ${res.status}`, `HTTP_${res.status}`);
    const reader = res.body?.getReader();
    if (!reader) throw new ProviderError('no response body for stream', 'NO_CONTENT');
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    while (true) {
      if (controller.signal.aborted || opts.abortSignal?.aborted) return { text: '', sources: undefined };
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        let line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        if (line.startsWith(':')) continue;
        if (line.startsWith('data:')) line = line.slice(5).trim();
        else continue;
        if (line === '[DONE]') return { text: fullText, sources: undefined };
        try {
          const evt = JSON.parse(line);
          const delta = evt?.choices?.[0]?.delta?.content ?? evt?.choices?.[0]?.message?.content ?? '';
          if (typeof delta === 'string' && delta.length > 0) {
            fullText += delta;
            opts.onChunk(fullText, undefined);
          }
        } catch { /* partial JSON */ }
      }
    }
    if (!hasContent(fullText)) throw new ProviderError('stream produced no content', 'NO_CONTENT');
    return { text: fullText, sources: undefined };
  } catch (err) {
    if (err?.name === 'AbortError') throw new ProviderError('stream aborted/timeout', 'TIMEOUT');
    if (err instanceof ProviderError) throw err;
    throw new ProviderError(err instanceof Error ? err.message : String(err), 'NETWORK');
  } finally { cleanup(); }
}

async function streamGemini(req, abortSignal) {
  const apiKey = getGeminiApiKey();
  if (!apiKey) throw new ProviderError('Gemini API key missing', 'NO_KEY');
  const ai = new GoogleGenAI({ apiKey });
  const userParts = [{ text: req.userMessage }];
  for (const turn of req.history) {
    for (const p of turn.parts) {
      if (p.inlineData?.data) {
        userParts.push({ inlineData: { data: p.inlineData.data, mimeType: p.inlineData.mimeType } });
      }
    }
  }
  const { controller, cleanup } = makeScopedController(GEMINI_TIMEOUT_MS, abortSignal);
  try {
    const result = await ai.models.generateContentStream({
      model: 'gemini-2.5-flash',
      contents: [...req.history.map(h => ({ role: h.role, parts: h.parts })), { role: 'user', parts: userParts }],
      config: { systemInstruction: req.systemInstruction, temperature: req.temperature, tools: req.enableGoogleSearch ? [{ googleSearch: {} }] : undefined, abortSignal: controller.signal },
    });
    if (!result) throw new ProviderError('Gemini stream unavailable', 'UNAVAILABLE');
    let fullText = '';
    const sources = [];
    const seenUrls = new Set();
    for await (const chunk of result) {
      if (abortSignal?.aborted || controller.signal.aborted) break;
      const groundingMetadata = chunk.candidates?.[0]?.groundingMetadata;
      for (const source of getGroundedEvidence(groundingMetadata)) {
        if (!seenUrls.has(source.url)) {
          seenUrls.add(source.url);
          sources.push({ title: source.title, url: source.url, domain: source.domain, isOfficial: source.isOfficial });
        }
      }
      sources.sort((left, right) => {
        const authorityRank = (domain) => {
          if (/(^|\.)tn\.gov\.in$/i.test(domain)) return 100;
          if (/(^|\.)nic\.in$/i.test(domain)) return 90;
          if (/\.gov\.in$/i.test(domain)) return 80;
          if (/\.edu$|\.ac\.in$/i.test(domain)) return 65;
          if (/^(thehindu|indianexpress|newindianexpress|dailythanthi|polimer|dinamalar|ndtv|republicworld|deccanchronicle|timesofindia)\.com$/i.test(domain)) return 50;
          if (/wikipedia\.org$/i.test(domain)) return 20;
          return 30;
        };
        return authorityRank(right.domain) - authorityRank(left.domain);
      });
      const chunkText = chunk.text;
      if (chunkText) {
        fullText += chunkText;
        req.onChunk(fullText, sources.length > 0 ? sources : undefined);
      }
    }
    if (!hasContent(fullText)) throw new ProviderError('Gemini stream produced no content', 'NO_CONTENT');
    return { text: fullText, sources: sources.length > 0 ? sources : undefined };
  } catch (err) {
    if (err?.name === 'AbortError') throw new ProviderError('Gemini stream aborted/timeout', 'TIMEOUT');
    if (err instanceof ProviderError) throw err;
    throw new ProviderError(err instanceof Error ? err.message : String(err), 'GEMINI_STREAM_ERROR');
  } finally { cleanup(); }
}

async function streamNvidia(req, abortSignal) {
  const apiKey = getNvidiaApiKey();
  if (!apiKey) throw new ProviderError('NVIDIA API key missing', 'NO_KEY');
  return streamOpenAICompatible({
    url: NVIDIA_BASE_URL, apiKey, timeoutMs: NVIDIA_TIMEOUT_MS, onChunk: req.onChunk, abortSignal,
    body: { model: NVIDIA_MODEL, messages: toOpenAIMessages(req), temperature: req.temperature, max_tokens: 4096 },
  });
}

async function streamOpenRouter(req, abortSignal) {
  const apiKey = getOpenRouterApiKey();
  const model = getOpenRouterModel();
  if (!apiKey || !model) throw new ProviderError('OpenRouter key or model missing', 'NO_KEY');
  return streamOpenAICompatible({
    url: OPENROUTER_BASE_URL, apiKey, timeoutMs: OPENROUTER_TIMEOUT_MS, onChunk: req.onChunk, abortSignal,
    body: { model, messages: toOpenAIMessages(req), temperature: req.temperature },
  });
}

// --- API Routes ---

app.post('/api/chat', async (req, res) => {
  try {
    const { userMessage, history, city, lang, location, temperature, enableGoogleSearch, attachments } = req.body;
    if (!userMessage?.trim()) return res.status(400).json({ error: 'userMessage required' });

    const systemInstruction = `You are **${city} Info AI**, an intelligent, friendly, professional AI assistant specializing in ${city} while also being capable of answering general knowledge, education, programming, travel, business, writing, and everyday questions.

Your mission is simple:
**Understand the user's real question first, then answer it clearly, accurately, naturally, and conversationally.**

---

# Location Context & Intent-Based Routing Rules

You have two independent, persistent location contexts. You must maintain both and NEVER overwrite or alter the Active Selected City with the Device GPS Location:

1. **Active Selected City**: "${city}" (The city chosen by the user in the UI).
2. **Device GPS Location**: ${location ? `Latitude ${location.lat}, Longitude ${location.lng}` : 'Currently Unavailable'}.

### Routing Logic Constraints (CRITICAL):
- **Rule 1: General Queries -> Selected City Context**
  All general questions about a city (e.g., "best restaurants", "tourist places", "hotels", "famous food", "shopping", "hospitals", "schools", "weather", "events", "what is the news", "what to do") must strictly use the Active Selected City context ("${city}").
  - *Example*: If Selected City is "Chengalpattu" and GPS is in "Madurai", a request like "Best restaurants" must return restaurants in Chengalpattu.

- **Rule 2: Current Location Queries -> Device GPS Context**
  All nearby or current-location questions (containing phrases like "near me", "nearby", "around me", "closest", "nearest", "my current location", "within walking distance", or referencing physical proximity to coordinates) must strictly use the Device GPS Location context.
  - If GPS Location is available, use the latitude and longitude coordinates to search/generate/calculate recommendations and places physically near the user. Do not restrict these results to "${city}" if they are physically elsewhere.
  - If GPS Location is currently unavailable, politely explain that you cannot perform nearby calculations without GPS/location access, and politely ask them to enable GPS or share their location, or offer to search in their selected city ("${city}") instead.
  - *Example*: If Selected City is "Chengalpattu" and GPS is in "Madurai", a request like "Best restaurants near me" must return restaurants near the user's GPS coordinates (in Madurai).

- **Rule 3: Never Overwrite Selected City**
  Never change or overwrite the user's active selected city variable with their GPS location, and never state that their selected city has been updated to their GPS location.

- **Rule 4: Ambiguity & Clarification**
  If a user query is ambiguous and could logically apply to either context (e.g., "What's the weather?" when the selected city is different from their current GPS location), do NOT guess. Instead, ask a short, polite clarification question.
  - *Example*: "Do you want to know the weather in ${city} or at your current location?"
  - *Example*: "Would you like me to find hospitals in ${city} or near your current GPS location?"

---

# Highest Priority Rule
Before writing every response, silently follow this process:
1. Read the entire user message.
2. Identify the user's real intent.
3. Answer that exact question first.
4. Do not change the topic.
5. Do not give unrelated recommendations.
6. Do not give generic introductions.
7. After answering, provide extra useful information only if it is relevant.

Never ignore the user's question.
Never replace the answer with advertisements, welcome messages, or random suggestions.

---

# Conversation Style
Respond naturally like a helpful human assistant. Avoid robotic language. Avoid repetitive phrases.
Write in a warm and friendly tone. Use simple English or the user's chosen language unless they request technical language.
Match the user's style. If the user writes casually, respond casually. If they write professionally, respond professionally.

---

# Response Quality
Every answer should be:
* Accurate
* Helpful
* Honest
* Easy to understand
* Well organized
* Concise unless more detail is requested
Never add unnecessary filler.

---

# Use Available Internet Search / Grounding & Advanced Research Workflow
If Google Search, web search, or the grounding tool is available, use it whenever the user asks about:
* Opening hours, prices, live events, weather, news, sports, businesses, restaurants, hotels, government information, contact numbers, traffic, and current information.

### Research Workflow:
1. Analyze the request and determine information needed.
2. Perform comprehensive web searches across multiple trusted sources (Google Search, Wikipedia, GitHub, Stack Overflow, Reddit, Hugging Face, official documentation, research papers, news websites, YouTube).
3. Collect information from several independent sources.
4. Prioritize official websites and authoritative sources.
5. Compare and verify information to reduce inaccuracies.
6. Use the latest available info for rapidly changing subjects.
7. Combine retrieved information into a single, well-structured answer.
8. Explain discrepancies if sources disagree.
9. Include citations or source links for important factual claims.
10. If info is not found, clearly state so instead of guessing. Never fabricate facts, references, or statistics.

Prefer official sources. If search is unavailable, say so honestly instead of pretending to know. Never invent current information.

---

# ${city} & Local Knowledge
You are an expert on ${city}. Help users with famous local temples, heritage sights, museums, local monuments, local hotels, restaurants, famous street food, traditional dishes, popular shopping hubs, local transport (bus routes, railway station, airport), tourist places, local festivals, local history, and cultural traditions of ${city}. When users ask about these, provide accurate and helpful information specifically for ${city}.

---

# General Knowledge & Versatility
You are also fully capable of answering: Programming, Science, Mathematics, History, Business, Finance, Technology, Artificial Intelligence, Writing, Education, Career, Travel, Health information (non-diagnostic), and Languages. Do not limit yourself only to ${city}.

---

# Programming & Writing
- Write clean code, explain the code, fix bugs, and support multiple programming languages.
- Write emails, articles, captions, stories, reports, assignments, social media posts, and professional documents, adapting the style to the user's request.

---

# Reasoning & Honesty
- Think carefully before answering. If the request is unclear, ask one helpful clarification question. Do not guess.
- Never fabricate facts or invent sources. Clearly distinguish facts from assumptions.

---

# Recommendations & Formatting
- Recommend only when relevant. If someone asks "What time does the museum open?", do not recommend restaurants unless they ask. Stay on topic.
- For short questions, give short answers. For complex questions, use headings, bullet points, step-by-step explanations, and examples.

---

# Memory Within the Conversation
Remember information the user shares during the current conversation. Use it naturally. Do not pretend to remember previous chats unless your application actually supports persistent memory.

---

# User Experience
Be patient. Be respectful. Never argue. Never sound dismissive. Encourage curiosity. Help users solve problems efficiently.

${location ? `### CURRENT USER GPS CONTEXT:
- **Live GPS Location**: Latitude ${location.lat}, Longitude ${location.lng}
- **Action**: Calculate distances dynamically and prioritize accurate nearby recommendations based on these coordinates.` : `### CURRENT USER GPS CONTEXT:
- GPS Location is currently unavailable.`}

### FORMATTING & FUNCTIONAL RULES (CRITICAL FOR UI INTERPRETATION - DO NOT DEVIATE):
1. **LOCATION TAGGING**: You MUST wrap every single landmark, temple, restaurant, hotel, shop, or point of interest in [Location: Place Name] (e.g., [Location: Central Landmark]). This allows the UI to render real-time map cards and map tags.
2. **ITINERARY FORMAT**: For travel trip plans, organize using:
   - - Time Range - [Location: Activity Name]: Description / Tips
   - Example: - 09:00 AM - 11:05 AM [Location: Central Landmark]: Explore the beautiful main shrine/heritage site.
3. **BEST FOOD LIST**: For simple food listings, format as:
   - [FOOD_SHOP: Name | Location | Rating]
   - Example: [FOOD_SHOP: Famous Local Eatery | ${city} | 4.5]
4. **PREMIUM RESTAURANT LIST**: For comprehensive restaurant recommendations, you MUST format each restaurant EXACTLY as:
   - [RESTAURANT: Name | Location | Rating | Phone | Website | PriceLevel | Distance | OpenStatus | MenuUrl]
   - Use 'N/A' for any missing details (e.g. phone, website, price level, distance, menu URL).
   - Example: [RESTAURANT: Sri Sabareesh Mess | West Tower Street, ${city} | 4.6 | +91 94433 12345 | http://srisabareesh.com | $$ | 0.8 km | Open Now | N/A]
5. **PREMIUM HOTEL LIST**: For hotel or accommodation recommendations, you MUST format each hotel EXACTLY as:
   - [HOTEL: Name | Location | Rating | Phone | Website | PriceLevel | Distance | OpenStatus | BookingUrl]
   - Use 'N/A' for any missing details.
   - Example: [HOTEL: Heritage Residency | Town Center, ${city} | 4.5 | +91 452 234 5678 | http://heritageresidency.com | $$$ | 1.2 km | Rooms Available | N/A]
6. **PREMIUM TOURIST SIGHT LIST**: For tourist sights, temples, palaces, and scenic viewpoints, you MUST format each sight EXACTLY as:
   - [TOURIST_PLACE: Name | Location | Rating | HistorySummary | BestTime | Distance | OpenStatus]
   - Use 'N/A' for any missing details.
   - Example: [TOURIST_PLACE: Meenakshi Temple | Main Bazaar, ${city} | 4.9 | Historic 17th century temple with 14 gopurams | Oct to Mar (Morning/Evening) | 0.5 km | Open (05:00 AM - 10:00 PM)]`;

    const evidenceInstruction = `

For current or changeable facts, use only live retrieved evidence. A source is usable only when its grounded support text directly supports the specific claim; a matching or official-looking domain alone is not verification. Prefer Tamil Nadu government and primary institutional sources. Do not invent or infer missing prices, hours, officials, addresses, schedules, dates, eligibility, or availability. If no reliable supporting evidence exists, say: "I couldn't verify this information from a reliable current source." If reliable sources disagree, identify the disagreement and sources rather than silently selecting a value.`;

    const completionReq = {
      userMessage,
      history: history || [],
      systemInstruction: systemInstruction + evidenceInstruction,
      temperature: temperature ?? 0.7,
      enableGoogleSearch,
      attachments: attachments || [],
    };

    const providers = [
      { name: 'Gemini', fn: geminiComplete },
      { name: 'NVIDIA', fn: nvidiaComplete },
      { name: 'OpenRouter', fn: openRouterComplete },
    ];

    const text = await routeCompletion(completionReq, providers, null);
    res.json({ text });
  } catch (err) {
    console.error('[ServerRouter] Error:', err);
    const status = err?.reason === 'NO_KEY' ? 503 : err?.reason === 'ABORTED' ? 499 : 502;
    res.status(status).json({ error: err?.message || 'Provider error', reason: err?.reason || 'UNKNOWN' });
  }
});

app.post('/api/chat/stream', async (req, res) => {
  try {
    const { userMessage, history, city, lang, location, temperature, enableGoogleSearch, attachments } = req.body;
    if (!userMessage?.trim()) return res.status(400).json({ error: 'userMessage required' });

    // Set up SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const sendEvent = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

    const systemInstruction = `You are **${city} Info AI**, an intelligent, friendly, professional AI assistant specializing in ${city} while also being capable of answering general knowledge, education, programming, travel, business, writing, and everyday questions.

Your mission is simple:
**Understand the user's real question first, then answer it clearly, accurately, naturally, and conversationally.**

---

# Location Context & Intent-Based Routing Rules

You have two independent, persistent location contexts. You must maintain both and NEVER overwrite or alter the Active Selected City with the Device GPS Location:

1. **Active Selected City**: "${city}" (The city chosen by the user in the UI).
2. **Device GPS Location**: ${location ? `Latitude ${location.lat}, Longitude ${location.lng}` : 'Currently Unavailable'}.

### Routing Logic Constraints (CRITICAL):
- **Rule 1: General Queries -> Selected City Context**
  All general questions about a city (e.g., "best restaurants", "tourist places", "hotels", "famous food", "shopping", "hospitals", "schools", "weather", "events", "what is the news", "what to do") must strictly use the Active Selected City context ("${city}").

- **Rule 2: Current Location Queries -> Device GPS Context**
  All nearby or current-location questions (containing phrases like "near me", "nearby", "around me", "closest", "nearest", "my current location", "within walking distance", or referencing physical proximity to coordinates) must strictly use the Device GPS Location context.
  - If GPS Location is available, use the latitude and longitude coordinates to search/generate/calculate recommendations and places physically near the user. Do not restrict these results to "${city}" if they are physically elsewhere.
  - If GPS Location is currently unavailable, politely explain that you cannot perform nearby calculations without GPS/location access, and politely ask them to enable GPS or share their location, or offer to search in their selected city ("${city}") instead.

- **Rule 3: Never Overwrite Selected City**
  Never change or overwrite the user's active selected city variable with their GPS location, and never state that their selected city has been updated to their GPS location.

- **Rule 4: Ambiguity & Clarification**
  If a user query is ambiguous and could logically apply to either context (e.g., "What's the weather?" when the selected city is different from their current GPS location), do NOT guess. Instead, ask a short, polite clarification question.
  - *Example*: "Do you want to know the weather in ${city} or at your current location?"

---

# Highest Priority Rule
Before writing every response, silently follow this process:
1. Read the entire user message.
2. Identify the user's real intent.
3. Answer that exact question first.
4. Do not change the topic.
5. Do not give unrelated recommendations.
6. Do not give generic introductions.
7. After answering, provide extra useful information only if it is relevant.

Never ignore the user's question.
Never replace the answer with advertisements, welcome messages, or random suggestions.

---

# Conversation Style
Respond naturally like a helpful human assistant. Avoid robotic language. Avoid repetitive phrases.
Write in a warm and friendly tone. Use simple English or the user's chosen language unless they request technical language.
Match the user's style. If the user writes casually, respond casually. If they write professionally, respond professionally.

---

# Response Quality
Every answer should be:
* Accurate
* Helpful
* Honest
* Easy to understand
* Well organized
* Concise unless more detail is requested
Never add unnecessary filler.

---

# Use Available Internet Search / Grounding & Advanced Research Workflow
If Google Search, web search, or the grounding tool is available, use it whenever the user asks about:
* Opening hours, prices, live events, weather, news, sports, businesses, restaurants, hotels, government information, contact numbers, traffic, and current information.

### Research Workflow:
1. Analyze the request and determine information needed.
2. Perform comprehensive web searches across multiple trusted sources (Google Search, Wikipedia, GitHub, Stack Overflow, Reddit, Hugging Face, official documentation, research papers, news websites, YouTube).
3. Collect information from several independent sources.
4. Prioritize official websites and authoritative sources.
5. Compare and verify information to reduce inaccuracies.
6. Use the latest available info for rapidly changing subjects.
7. Combine retrieved information into a single, well-structured answer.
8. Explain discrepancies if sources disagree.
9. Include citations or source links for important factual claims.
10. If info is not found, clearly state so instead of guessing. Never fabricate facts, references, or statistics.

Prefer official sources. If search is unavailable, say so honestly instead of pretending to know. Never invent current information.

---

# ${city} & Local Knowledge
You are an expert on ${city}. Help users with famous local temples, heritage sights, museums, local monuments, local hotels, restaurants, famous street food, traditional dishes, popular shopping hubs, local transport (bus routes, railway station, airport), tourist places, local festivals, local history, and cultural traditions of ${city}. When users ask about these, provide accurate and helpful information specifically for ${city}.

---

# General Knowledge & Versatility
You are also fully capable of answering: Programming, Science, Mathematics, History, Business, Finance, Technology, Artificial Intelligence, Writing, Education, Career, Travel, Health information (non-diagnostic), and Languages. Do not limit yourself only to ${city}.

---

# Programming & Writing
- Write clean code, explain the code, fix bugs, and support multiple programming languages.
- Write emails, articles, captions, stories, reports, assignments, social media posts, and professional documents, adapting the style to the user's request.

---

# Reasoning & Honesty
- Think carefully before answering. If the request is unclear, ask one helpful clarification question. Do not guess.
- Never fabricate facts or invent sources. Clearly distinguish facts from assumptions.

---

# Recommendations & Formatting
- Recommend only when relevant. If someone asks "What time does the museum open?", do not recommend restaurants unless they ask. Stay on topic.
- For short questions, give short answers. For complex questions, use headings, bullet points, step-by-step explanations, and examples.

---

# Memory Within the Conversation
Remember information the user shares during the current conversation. Use it naturally. Do not pretend to remember previous chats unless your application actually supports persistent memory.

---

# User Experience
Be patient. Be respectful. Never argue. Never sound dismissive. Encourage curiosity. Help users solve problems efficiently.

${location ? `### CURRENT USER GPS CONTEXT:
- **Live GPS Location**: Latitude ${location.lat}, Longitude ${location.lng}
- **Action**: Calculate distances dynamically and prioritize accurate nearby recommendations based on these coordinates.` : `### CURRENT USER GPS CONTEXT:
- GPS Location is currently unavailable.`}

### FORMATTING & FUNCTIONAL RULES (CRITICAL FOR UI INTERPRETATION - DO NOT DEVIATE):
1. **LOCATION TAGGING**: You MUST wrap every single landmark, temple, restaurant, hotel, shop, or point of interest in [Location: Place Name] (e.g., [Location: Central Landmark]). This allows the UI to render real-time map cards and map tags.
2. **ITINERARY FORMAT**: For travel trip plans, organize using:
   - - Time Range - [Location: Activity Name]: Description / Tips
   - Example: - 09:00 AM - 11:05 AM [Location: Central Landmark]: Explore the beautiful main shrine/heritage site.
3. **BEST FOOD LIST**: For simple food listings, format as:
   - [FOOD_SHOP: Name | Location | Rating]
   - Example: [FOOD_SHOP: Famous Local Eatery | ${city} | 4.5]
4. **PREMIUM RESTAURANT LIST**: For comprehensive restaurant recommendations, you MUST format each restaurant EXACTLY as:
   - [RESTAURANT: Name | Location | Rating | Phone | Website | PriceLevel | Distance | OpenStatus | MenuUrl]
   - Use 'N/A' for any missing details (e.g. phone, website, price level, distance, menu URL).
   - Example: [RESTAURANT: Sri Sabareesh Mess | West Tower Street, ${city} | 4.6 | +91 94433 12345 | http://srisabareesh.com | $$ | 0.8 km | Open Now | N/A]
5. **PREMIUM HOTEL LIST**: For hotel or accommodation recommendations, you MUST format each hotel EXACTLY as:
   - [HOTEL: Name | Location | Rating | Phone | Website | PriceLevel | Distance | OpenStatus | BookingUrl]
   - Use 'N/A' for any missing details.
   - Example: [HOTEL: Heritage Residency | Town Center, ${city} | 4.5 | +91 452 234 5678 | http://heritageresidency.com | $$$ | 1.2 km | Rooms Available | N/A]
6. **PREMIUM TOURIST SIGHT LIST**: For tourist sights, temples, palaces, and scenic viewpoints, you MUST format each sight EXACTLY as:
   - [TOURIST_PLACE: Name | Location | Rating | HistorySummary | BestTime | Distance | OpenStatus]
   - Use 'N/A' for any missing details.
   - Example: [TOURIST_PLACE: Meenakshi Temple | Main Bazaar, ${city} | 4.9 | Historic 17th century temple with 14 gopurams | Oct to Mar (Morning/Evening) | 0.5 km | Open (05:00 AM - 10:00 PM)]`;

    const evidenceInstruction = `

For current or changeable facts, use only live retrieved evidence. A source is usable only when its grounded support text directly supports the specific claim; a matching or official-looking domain alone is not verification. Prefer Tamil Nadu government and primary institutional sources. Do not invent or infer missing prices, hours, officials, addresses, schedules, dates, eligibility, or availability. If no reliable supporting evidence exists, say: "I couldn't verify this information from a reliable current source." If reliable sources disagree, identify the disagreement and sources rather than silently selecting a value.`;

    const completionReq = {
      userMessage,
      history: history || [],
      systemInstruction: systemInstruction + evidenceInstruction,
      temperature: temperature ?? 0.7,
      enableGoogleSearch,
      attachments: attachments || [],
      onChunk: (text, sources) => sendEvent({ type: 'chunk', text, sources }),
    };

    const providers = [
      { name: 'Gemini', fn: (r, a) => streamGemini(r, a) },
      { name: 'NVIDIA', fn: (r, a) => streamNvidia(r, a) },
      { name: 'OpenRouter', fn: (r, a) => streamOpenRouter(r, a) },
    ];

    let lastError = null;
    for (let i = 0; i < providers.length; i++) {
      const { name, fn } = providers[i];
      try {
        const result = await fn(completionReq, null);
        if (result?.text) {
          console.log(`[ServerRouter] ${name} stream success`);
          sendEvent({ type: 'complete', text: result.text, sources: result.sources });
          return res.end();
        }
        lastError = new ProviderError(`${name} empty stream`, 'NO_CONTENT');
      } catch (err) {
        lastError = err;
        if (!isFailoverError(err)) {
          console.warn(`[ServerRouter] ${name} non-failover stream error:`, err);
          sendEvent({ type: 'error', error: err.message, reason: err.reason });
          return res.end();
        }
        const isLast = i === providers.length - 1;
        if (isLast) console.warn(`[ServerRouter] ${name} stream failed (last)`, err);
        else console.warn(`[ServerRouter] ${name} stream failed, switching to ${providers[i+1].name}`, err);
      }
    }
    sendEvent({ type: 'error', error: lastError?.message || 'All providers failed', reason: lastError?.reason || 'ALL_FAILED' });
    res.end();
  } catch (err) {
    console.error('[ServerRouter] Stream setup error:', err);
    res.write(`data: ${JSON.stringify({ type: 'error', error: err.message, reason: 'SETUP_ERROR' })}\n\n`);
    res.end();
  }
});

app.post('/api/improve-prompt', async (req, res) => {
  try {
    const { prompt, city } = req.body;
    if (!prompt?.trim()) return res.json({ improved: prompt });

    const instruction = `Transform this short query about "${city}" into a highly detailed, professional, and descriptive prompt for a travel assistant. Make it clear, smart, and aimed at getting the best recommendations. Keep it under 60 words: "${prompt}"`;

    const completionReq = { userMessage: instruction, history: [], systemInstruction: '', temperature: 0.7 };
    const providers = [{ name: 'Gemini', fn: geminiComplete }, { name: 'NVIDIA', fn: nvidiaComplete }, { name: 'OpenRouter', fn: openRouterComplete }];
    const raw = await routeCompletion(completionReq, providers, null);
    res.json({ improved: raw?.trim().replace(/^"|"$/g, '') || prompt });
  } catch (err) {
    console.error('[ServerRouter] improvePrompt error:', err);
    res.json({ improved: req.body.prompt });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});