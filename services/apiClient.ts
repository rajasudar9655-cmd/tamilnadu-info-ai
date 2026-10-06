export interface ChatRequest {
  userMessage: string;
  history: { role: 'user' | 'model'; parts: { text?: string; inlineData?: { data: string; mimeType: string } }[] }[];
  city: string;
  lang: 'en' | 'ta';
  location?: { lat: number; lng: number };
  temperature?: number;
  enableGoogleSearch?: boolean;
  attachments?: { data: string; mimeType: string }[];
}

export interface StreamCallbacks {
  onChunk: (text: string, sources?: { title: string; url: string; domain?: string; isOfficial?: boolean }[]) => void;
  onComplete: (text: string, sources?: { title: string; url: string; domain?: string; isOfficial?: boolean }[]) => void;
  onError: (error: string, reason: string) => void;
}

function getApiBaseUrl(): string {
  try {
    const v = (import.meta as any).env?.VITE_API_BASE_URL;
    if (v) return v;
  } catch { }
  return '';
}

async function fetchWithTimeout(url: string, options: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function sendChatRequest(req: ChatRequest): Promise<string> {
  const baseUrl = getApiBaseUrl();
  const response = await fetchWithTimeout(`${baseUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  }, 120000);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed', reason: 'UNKNOWN' }));
    throw new Error(error.error || `HTTP ${response.status}`);
  }
  const data = await response.json();
  return data.text;
}

export function streamChatRequest(req: ChatRequest, callbacks: StreamCallbacks): () => void {
  const baseUrl = getApiBaseUrl();
  let aborted = false;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180000);

  (async () => {
    try {
      const response = await fetch(`${baseUrl}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
        signal: controller.signal,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Stream failed', reason: 'UNKNOWN' }));
        callbacks.onError(error.error || `HTTP ${response.status}`, error.reason || 'UNKNOWN');
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError('No response body', 'NO_CONTENT');
        return;
      }

      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        if (aborted) return;
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
          try {
            const evt = JSON.parse(line);
            if (evt.type === 'chunk') {
              callbacks.onChunk(evt.text, evt.sources);
            } else if (evt.type === 'complete') {
              callbacks.onComplete(evt.text, evt.sources);
              return;
            } else if (evt.type === 'error') {
              callbacks.onError(evt.error, evt.reason);
              return;
            }
          } catch { }
        }
      }
    } catch (err) {
      if (aborted) return;
      callbacks.onError(err instanceof Error ? err.message : 'Stream error', 'NETWORK');
    } finally {
      clearTimeout(timer);
    }
  })();

  return () => {
    aborted = true;
    controller.abort();
    clearTimeout(timer);
  };
}

export async function improvePrompt(prompt: string, city: string): Promise<string> {
  const baseUrl = getApiBaseUrl();
  const response = await fetchWithTimeout(`${baseUrl}/api/improve-prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, city }),
  }, 30000);

  if (!response.ok) return prompt;
  const data = await response.json();
  return data.improved || prompt;
}