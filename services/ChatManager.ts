/**
 * ChatManager — Centralized AI request / stream lifecycle manager.
 *
 * ── Single responsibility ─────────────────────────────────────────────
 *   • Owns AI requests (ONE at a time — never overlapping)
 *   • Owns the AbortController lifecycle
 *   • Mints monotonic request IDs (the "generation token" for the chat)
 *   • Streams chunked text updates to React
 *   • Resolves final message state (text + sources + followUps)
 *   • Tracks loading flag for the UI
 *
 * ── Concurrency model ───────────────────────────────────────────────
 *   • There is AT MOST ONE active AI request at any moment.
 *     Starting a new request FIRST aborts & disposes the previous one.
 *   • Each request mints an ID (incrementing integer). Every async
 *     callback verifies its ID is still the active one before touching
 *     UI state. Stale requests exit silently.
 *   • abort() cancels the in-flight request and bumps the request ID
 *     so late callbacks cannot fire.
 *   • reset() is the "city switch" reset: abort + clear slate + ready
 *     for a fresh request with no pollution from the old city.
 *
 * ── Voice binding (Bug 1, 2, 3, 4 fix) ──────────────────────────────
 *   ChatManager holds a reference to VoiceManager. When a NEW request
 *   starts, ChatManager calls `voiceManager.bindRequest(requestId)`.
 *   This guarantees:
 *     • Any TTS tied to the previous request is cancelled immediately.
 *     • The only request whose response may ever produce speech is the
 *       current one. Stale responses cannot speak even if they slip
 *       through some other gate.
 *
 *   This is the synchronization primitive that makes text and voice
 *   always come from the same generation.
 *
 * ── Integration ──────────────────────────────────────────────────────
 *   - Calls the existing `getGeminiResponseStream` service function
 *     (the underlying API/prompt/streaming behavior is unchanged).
 *   - Notifies React through callbacks: onStart, onChunk, onComplete,
 *     onError, onLoadingChange. The React layer owns message state.
 *
 * The React layer is the source of truth for the messages array;
 * ChatManager only STREAMS state events and never mutates React
 * state directly.
 */

import { getGeminiResponseStream, generateLocalFollowUps } from './geminiService.ts';
import { VoiceManager } from './VoiceManager.ts';

// ── Types ───────────────────────────────────────────────────────────

export interface ChatRequestContext {
  /** Previous assistant-style messages converted to API history. */
  history: { role: 'user' | 'model'; parts: any[] }[];
  /** Active selected city name. */
  city: string;
  /** Active language. */
  language: 'en' | 'ta';
  /** Optional device GPS location. */
  location?: { lat: number; lng: number };
  /** Optional attachments from the current user message. */
  attachments?: { data: string; mimeType: string }[];
  /** Deep search toggle. */
  isDeepSearch: boolean;
}

export interface ChatRequestCallbacks {
  /** Fires (with the new request ID) when a fresh request starts. */
  onStart?: (requestId: number) => void;
  /** Fires for every streamed text chunk. */
  onChunk?: (
    requestId: number,
    assistantMsgId: string,
    text: string,
    sources?: { title: string; url: string; domain?: string; isOfficial?: boolean }[]
  ) => void;
  /** Fires when the request completes successfully. */
  onComplete?: (
    requestId: number,
    result: {
      assistantMsgId: string;
      text: string;
      sources?: { title: string; url: string; domain?: string; isOfficial?: boolean }[];
      followUps: string[];
      isDeepSearch: boolean;
    }
  ) => void;
  /** Fires (synchronously) when loading state should change. */
  onLoadingChange?: (loading: boolean) => void;
  /** Fires if the request errors (abortions are NOT errors). */
  onError?: (
    requestId: number,
    assistantMsgId: string,
    message: string
  ) => void;
}

// ── ChatManager ──────────────────────────────────────────────────────

export class ChatManager {
  // Active AbortController for the in-flight request (if any).
  private controller: AbortController | null = null;

  // Monotonic request ID. Only the latest ID is "current".
  private currentRequestId = 0;

  // The currently-active assistant message ID (set when a request
  // starts; cleared when the request resolves/aborts). Used so we can
  // emit onError for the right message even if state moved on.
  private currentAssistantMsgId: string | null = null;

  // Single-flight guard. Prevents a second sendMessage() call from racing
  // with the synchronous preamble of the first before the first reaches
  // the abortInternal() step. Without this, two rapid Enter presses can
  // both pass the React `isLoading` gate (which is async) and start two
  // requests "simultaneously" from the manager's perspective.
  private inFlight = false;

  private callbacks: ChatRequestCallbacks;

  // VoiceManager binding. Set externally via setVoiceManager(). When a
  // new request starts, ChatManager calls voiceManager.bindRequest() so
  // any TTS from a previous request is cancelled and can never speak.
  private voiceManager: VoiceManager | null = null;

  constructor(callbacks: ChatRequestCallbacks = {}) {
    this.callbacks = callbacks;
  }

  /**
   * Bind a VoiceManager instance. ChatManager will notify it whenever a
   * new request begins so that TTS for stale requests is cancelled and
   * ignored. This is the synchronization point that fixes Bugs 1/2/3/4.
   */
  setVoiceManager(vm: VoiceManager): void {
    this.voiceManager = vm;
  }

  setCallbacks(callbacks: ChatRequestCallbacks): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  /** Current request ID (ChatManager consumers can read this). */
  getCurrentRequestId(): number {
    return this.currentRequestId;
  }

  /** True iff a request is currently running. */
  isRunning(): boolean {
    return this.controller !== null;
  }

  // ── Public API ────────────────────────────────────────────────────

  /**
   * Start a new chat request.
   *
   * 1. Abort any in-flight request and dispose its controller.
   * 2. Mint a fresh request ID + AbortController.
   * 3. Notify VoiceManager of the new request id (binds TTS to this gen).
   * 4. Stream chunks to onChunk (only if still current).
   * 5. Resolve to onComplete (only if still current).
   * 6. On error → onError (only if still current and NOT aborted).
   *
   * @param userMessage     Text the user just sent.
   * @param assistantMsgId  Pre-allocated ID for the assistant message.
   * @param ctx             Request context (history, city, language, …).
   * @returns The request ID assigned to this request. Use it to ignore
   *          late results from the React side if needed.
   */
  async sendMessage(
    userMessage: string,
    assistantMsgId: string,
    ctx: ChatRequestContext
  ): Promise<number> {
    // ── Single-flight guard ──
    // If a previous sendMessage() call is still in its synchronous
    // preamble (pre-abort), wait for it to release. This guards against
    // rapid Enter/send spam where two calls both see `isLoading === false`.
    if (this.inFlight) {
      // We're already mid-start; bump the request id and bail out so the
      // already-running start is what wins. (The caller's UI will reflect
      // the in-flight request.)
      this.currentRequestId++;
      return this.currentRequestId;
    }
    this.inFlight = true;

    try {
      // ── Step 1: Abort & dispose any in-flight request. ──
      this.abortInternal();

      // ── Step 2: Mint a fresh request ID + controller. ──
      const requestId = ++this.currentRequestId;
      const controller = new AbortController();
      this.controller = controller;
      this.currentAssistantMsgId = assistantMsgId;

      // ── Step 3: Bind VoiceManager to this request. ──
      // Any TTS tied to a previous request is cancelled here. After this
      // point, ONLY this requestId may produce speech. This is the fix
      // for "stale response speaks after city change / rapid send".
      if (this.voiceManager) {
        this.voiceManager.bindRequest(requestId);
      }

      this.callbacks.onStart?.(requestId);
      this.callbacks.onLoadingChange?.(true);

      // Guard closure: a request is "current" only if its ID matches the
      // running ID AND it hasn't been aborted.
      const isCurrent = (): boolean =>
        requestId === this.currentRequestId &&
        this.controller === controller &&
        !controller.signal.aborted;

      try {
        const streamResult = await getGeminiResponseStream(
          userMessage,
          ctx.history,
          (chunk, chunkSources) => {
            // Streaming chunk → only forward if we're still current.
            if (!isCurrent()) return;
            this.callbacks.onChunk?.(requestId, assistantMsgId, chunk, chunkSources);
          },
          ctx.city,
          ctx.language,
          ctx.location,
          ctx.attachments,
          ctx.isDeepSearch,
          controller.signal
        );

        // If aborted mid-flight → silently exit (don't fire onComplete).
        if (!isCurrent()) return requestId;

        const responseText = streamResult?.text ?? '';
        const responseSources = streamResult?.sources;
        const followUps = generateLocalFollowUps(
          responseText,
          ctx.city,
          ctx.language
        );

        // Last stale-check before notifying React.
        if (!isCurrent()) return requestId;

        this.callbacks.onComplete?.(requestId, {
          assistantMsgId,
          text: responseText,
          sources: responseSources,
          followUps,
          isDeepSearch: ctx.isDeepSearch,
        });
      } catch (error) {
        // AbortError → silent exit (not an error condition).
        if (
          (error as any)?.name === 'AbortError' ||
          controller.signal.aborted ||
          !isCurrent()
        ) {
          return requestId;
        }
        console.error('ChatManager streaming error:', error);
        this.callbacks.onError?.(
          requestId,
          assistantMsgId,
          'I encountered an error while generating a response. Please try again.'
        );
      } finally {
        // Only clear controller/loading if we are still the active one.
        if (this.controller === controller) {
          this.controller = null;
          this.currentAssistantMsgId = null;
          this.callbacks.onLoadingChange?.(false);
        }
      }

      return requestId;
    } finally {
      this.inFlight = false;
    }
  }

  /**
   * Abort the in-flight request (if any) and bump the request ID so
   * stale callbacks can never fire. Does NOT throw. Idempotent.
   */
  abort(): void {
    this.abortInternal();
  }

  /**
   * "City switch / fresh slate" reset.
   *
   * Aborts the in-flight request + bumps the request ID + clears the
   * assistant message tracking. Safe to call repeatedly. After this
   * returns, no callback from a prior request can ever fire.
   */
  reset(): void {
    this.abortInternal();
    this.currentAssistantMsgId = null;
  }

  /**
   * Full teardown. Call on component unmount. After this, the manager
   * is unusable; construct a new one if needed.
   */
  dispose(): void {
    this.abortInternal();
    this.callbacks = {};
    this.voiceManager = null;
  }

  // ── Internal ──────────────────────────────────────────────────────

  private abortInternal(): void {
    // Bump the request ID FIRST so any in-flight async callback that
    // resolves later sees itself as stale and exits.
    this.currentRequestId++;

    if (this.controller) {
      try {
        this.controller.abort();
      } catch (_) {
        /* no-op */
      }
      this.controller = null;
    }
  }
}
