/**
 * VoiceManager — Centralized, race-condition-free TTS / audio / mic lifecycle service.
 *
 * ── Single responsibility ─────────────────────────────────────────────
 *   • Gemini TTS (gemini-2.5-flash-tts-preview) playback
 *   • Web Speech API fallback playback
 *   • AudioContext lifecycle (warm / resume / suspend / close)
 *   • Per-message audio cache (instant replay, cleared on city switch)
 *   • Microphone (MediaStream) lifecycle for voice input / live mode
 *   • SpeechRecognition lifecycle for voice typing
 *   • Generation-token + request-id binding so TTS can only ever
 *     arise from the CURRENT AI generation — never a stale one
 *   • Instant, total, idempotent cancellation of every active resource
 *
 * ── Concurrency model (the core invariant) ────────────────────────────
 *   The "current generation" is identified by a monotonic `generationToken`.
 *   Every async operation captures the token at the start and re-validates
 *   it after EVERY `await`. If the token is no longer current, the
 *   operation exits silently and mutates NO state.
 *
 *   `chatRequestId` binds TTS to a specific AI generation. TTS for an
 *   outdated request is rejected even if its token passes — this is what
 *   fixes the "stale response speaks after city change" bug. ChatManager
 *   calls `bindRequest()` when a new request starts so that any in-flight
 *   TTS tied to the previous request is cancelled and ignored.
 *
 * ── Resource ownership (no orphans) ──────────────────────────────────
 *   VoiceManager tracks every live resource in class fields:
 *     • currentSource            (AudioBufferSourceNode)
 *     • activeUtterance          (SpeechSynthesisUtterance)
 *     • micStream                (MediaStream — tracks stopped on cancel)
 *     • recognition              (SpeechRecognition — stopped on cancel)
 *     • liveAudioContexts        (input/output AudioContexts for Live mode)
 *     • liveSources              (Set of AudioBufferSourceNode for Live mode)
 *
 *   `cancelAll()` is the single tear-down path. Every reset/city-switch/
 *   voice-disable unmount flows through it. After it returns, no callback
 *   from a prior generation can mutate state or produce sound.
 */

import { GoogleGenAI, Modality } from '@google/genai';

// ── Small audio helpers ──────────────────────────────────────────────

const decode = (base64: string): Uint8Array => {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};

const decodeAudioData = async (
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number
): Promise<AudioBuffer> => {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
};

const cleanTextForSpeech = (text: string): string =>
  text
    .replace(/\[Location: (.*?)\]/g, '$1')
    .replace(/\[FOOD_SHOP:[^\]]*\]/g, ' ')
    .replace(/\[RESTAURANT:[^\]]*\]/g, ' ')
    .replace(/\[HOTEL:[^\]]*\]/g, ' ')
    .replace(/\[TOURIST_PLACE:[^\]]*\]/g, ' ')
    .replace(/\*\*/g, '')
    .replace(/#/g, '')
    .replace(/`/g, '')
    .replace(/\s+/g, ' ')
    .trim();

const getClientApiKey = (): string =>
  (import.meta as any).env?.VITE_GEMINI_API_KEY ||
  process.env.GEMINI_API_KEY ||
  process.env.API_KEY ||
  '';

// ── Types ────────────────────────────────────────────────────────────

export interface VoiceManagerCallbacks {
  /** Fires when the currently-speaking message ID changes (or becomes null). */
  onSpeakingChange: (messageId: string | null) => void;
  /** Fires when the loading-for-TTS message ID changes (or becomes null). */
  onLoadingChange: (messageId: string | null) => void;
}

// ── VoiceManager ──────────────────────────────────────────────────────

export class VoiceManager {
  // -- Audio nodes / context --
  private audioContext: AudioContext | null = null;
  private currentSource: AudioBufferSourceNode | null = null;

  /** Per-message audio cache (key = message id). Survives stop()/start(). */
  private audioCache: Map<string, AudioBuffer> = new Map();

  // -- React-facing state --
  private speakingMessageId: string | null = null;
  private loadingMessageId: string | null = null;

  // -- Generation token (monotonically increasing) --
  // Every speak() attempt mints a token. Only the holder of the current
  // token may mutate state. Anyone else must exit immediately.
  private generationToken = 0;

  // -- Request-id binding --
  // The AI generation ID that "owns" the right to speak right now. Set by
  // ChatManager via bindRequest(). TTS for any other request is rejected.
  private chatRequestId: number = 0;

  // -- Master "voice enabled" flag --
  // When false, NO new TTS can start AND any in-flight TTS that resolves
  // later must NOT speak. Toggled by setEnabled().
  private enabled = true;

  private callbacks: VoiceManagerCallbacks;

  // Track the active Web Speech utterance so cancelled tokens can be
  // compared against a stale one.
  private activeUtterance: SpeechSynthesisUtterance | null = null;

  // ── Microphone (MediaStream) ownership ──
  // VoiceManager is the SOLE owner of the mic. Cancelling voice MUST
  // stop every MediaStreamTrack so the browser releases the mic tab.
  private micStream: MediaStream | null = null;

  // ── SpeechRecognition ownership (voice typing) ──
  private recognition: any = null;

  // ── Live mode resources ──
  // (Kept here so cancelAll() can tear them down too.)
  private liveSession: any = null;
  private liveInputCtx: AudioContext | null = null;
  private liveOutputCtx: AudioContext | null = null;
  private liveSources: Set<AudioBufferSourceNode> = new Set();
  private liveScriptProcessor: ScriptProcessorNode | null = null;
  private liveStream: MediaStream | null = null;

  constructor(callbacks: VoiceManagerCallbacks) {
    this.callbacks = callbacks;

    // Chrome pauses speechSynthesis when the tab loses focus; some builds
    // also leave queued utterances stuck. Cancelling on visibilitychange
    // guarantees no orphaned queued speech survives a tab hide/show cycle.
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.handleVisibility);
    }
  }

  // ── Public API ────────────────────────────────────────────────────

  /** Pre-warm AudioContext on a user gesture. Idempotent. */
  warmAudioContext(): void {
    if (!this.enabled) return;
    if (!this.audioContext) {
      const Ctor = window.AudioContext || (window as any).webkitAudioContext;
      if (!Ctor) return;
      this.audioContext = new Ctor({ sampleRate: 24000 });
    }
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume().catch(() => undefined);
    }
  }

  /**
   * Master voice-enable toggle.
   *
   * When set to false: Immediately stop EVERYTHING and prevent any
   * future TTS from starting (or finishing) until re-enabled. Any
   * response that finishes while disabled will NEVER speak — this is
   * the fix for Bug 3 (delayed speech after voice off).
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      // Disabling voice = instant, total cancellation of every audio
      // resource. Late-resolving promises will see enabled === false and
      // exit silently (see checks inside speak()).
      this.cancelAll();
    } else {
      // Re-enabling pre-warms the AudioContext so the next speak() is
      // low-latency.
      this.warmAudioContext();
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Bind the CURRENT ChatManager request to TTS. Any in-flight TTS tied
   * to a previous request is cancelled and forgotten. TTS for any other
   * request id is rejected by speakForRequest().
   *
   * Called by ChatManager when a new AI request starts (so a response
   * streaming from a previous request can never produce speech).
   */
  bindRequest(requestId: number): void {
    if (this.chatRequestId !== requestId) {
      // Cancel speech tied to the OLD request — but keep AudioContext warm
      // for the new request and keep the cache (instant replay is safe).
      this.stopSpeechOnly();
    }
    this.chatRequestId = requestId;
  }

  /** Returns the current generation token (used by ChatManager). */
  getGenerationToken(): number {
    return this.generationToken;
  }

  /** Returns the request-id currently allowed to speak. */
  getChatRequestId(): number {
    return this.chatRequestId;
  }

  /** True iff `token` is still the active generation. */
  isCurrentToken(token: number): boolean {
    return token === this.generationToken;
  }

  /** True iff `requestId` is the request currently allowed to speak. */
  isCurrentRequest(requestId: number): boolean {
    return requestId === this.chatRequestId;
  }

  /** Returns true iff a message is currently speaking. */
  isSpeaking(): boolean {
    return this.speakingMessageId !== null;
  }

  /** Returns the currently-speaking message ID (or null). */
  getSpeakingMessageId(): string | null {
    return this.speakingMessageId;
  }

  /**
   * Speak a message — but only if the owning AI request is still current.
   *
   * If this exact message is already speaking, the call toggles it OFF
   * (instant stop). Always cancels prior playback first so only the
   * latest request can speak.
   *
   * @param text        Raw assistant response text (will be cleaned).
   * @param id          Message ID (cache key + toggle detection).
   * @param language    'en' | 'ta'.
   * @param requestId   The AI generation that owns this speak call. If it
   *                    no longer matches the bound request, the call is
   *                    a no-op (prevents stale-response speech).
   */
  async speakForRequest(
    text: string,
    id: string,
    language: 'en' | 'ta',
    requestId: number
  ): Promise<void> {
    // 1. Toggle off if this message is currently playing.
    if (this.speakingMessageId === id) {
      this.stop();
      return;
    }

    // 2. Never start if voice is disabled.
    if (!this.enabled) return;

    // 3. Reject speech from a non-current AI request. This is the key
    //    synchronization fix: TTS can NEVER arise from a stale generation.
    if (!this.isCurrentRequest(requestId)) return;

    // 4. Cancel EVERYTHING previous (start clean). Don't bump the token
    //    here — we mint our own below.
    this.stopSpeechOnly();

    // 5. Mint the generation token for this attempt.
    const token = ++this.generationToken;

    // 6. AudioContext must be alive.
    this.warmAudioContext();
    if (!this.audioContext) return;
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    // Re-check after await: a newer call may have pre-empted us, or
    // voice may have been disabled, or the owning request may have
    // changed mid-resume. Any of these → silent exit.
    if (
      !this.isCurrentToken(token) ||
      !this.enabled ||
      !this.isCurrentRequest(requestId)
    ) {
      this.setLoading(null);
      return;
    }

    // 7. Cache hit → instant playback.
    const cachedBuffer = this.audioCache.get(id);
    if (cachedBuffer) {
      this.setLoading(null);
      this.playBuffer(cachedBuffer, id, token, requestId);
      return;
    }

    // 8. Miss → fetch from Gemini TTS.
    this.setLoading(id);

    try {
      const ai = new GoogleGenAI({ apiKey: getClientApiKey() });
      const cleanText = cleanTextForSpeech(text);

      // Defensive: an empty cleaned text would crash TTS. Bail out
      // gracefully instead of playing silence.
      if (!cleanText) {
        this.setLoading(null);
        return;
      }

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-tts-preview',
        contents: [{ parts: [{ text: cleanText }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Aoede' },
            },
          },
        },
      });

      // Stale-check #1 (post-fetch).
      if (
        !this.isCurrentToken(token) ||
        !this.enabled ||
        !this.isCurrentRequest(requestId)
      ) {
        this.setLoading(null);
        return;
      }

      const base64Audio =
        response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;

      if (base64Audio) {
        const audioBuffer = await decodeAudioData(
          decode(base64Audio),
          this.audioContext!,
          24000,
          1
        );

        // Stale-check #2 (post-decode).
        if (
          !this.isCurrentToken(token) ||
          !this.enabled ||
          !this.isCurrentRequest(requestId)
        ) {
          this.setLoading(null);
          return;
        }

        this.audioCache.set(id, audioBuffer);
        this.setLoading(null);
        this.playBuffer(audioBuffer, id, token, requestId);
      } else {
        // No audio returned — bounce to Web Speech fallback if still
        // the active request.
        if (
          !this.isCurrentToken(token) ||
          !this.enabled ||
          !this.isCurrentRequest(requestId)
        ) {
          this.setLoading(null);
          return;
        }
        this.fallbackWebSpeech(text, id, language, token, requestId);
      }
    } catch (e) {
      console.error('TTS Error, falling back to Web Speech Synthesis API:', e);

      // Stale-check before fallback.
      if (
        !this.isCurrentToken(token) ||
        !this.enabled ||
        !this.isCurrentRequest(requestId)
      ) {
        this.setLoading(null);
        return;
      }
      this.fallbackWebSpeech(text, id, language, token, requestId);
    }
  }

  /**
   * Backwards-compatible speak() — uses the currently-bound request id.
   * Prefer speakForRequest() when the caller has the request id.
   */
  async speak(text: string, id: string, language: 'en' | 'ta'): Promise<void> {
    return this.speakForRequest(text, id, language, this.chatRequestId);
  }

  /**
   * Instantly cancel ALL speech / audio. Idempotent. Safe from anywhere.
   *
   * This increments the generation token (so any in-flight async TTS
   * sees itself as stale and bails), stops the AudioBufferSourceNode,
   * cancels Web Speech, and clears speaking/loading state. Does NOT
   * touch the mic, recognition, or live resources (use cancelAll() for
   * the full teardown).
   */
  stop(): void {
    // Bump the generation token so any in-flight speakForRequest() that is
    // still resolving its Gemini TTS fetch / decode sees itself as stale and
    // bails out instead of playing audio AFTER the user toggled the
    // speaker off or a city switch reset ran. Previously this passed
    // bumpToken=false, which contradicted the documented contract above
    // ("increments the generation token") and let a late TTS for the same
    // request speak even after the user stopped it (Bugs 3 & 6).
    this.stopSpeechOnly(true);
  }

  /**
   * "City switch / fresh slate" reset.
   *
   * Cancels speech + bumps token + clears loading state. Does NOT
   * close the AudioContext (we want it warm for the next message).
   */
  reset(): void {
    this.stopSpeechOnly();
  }

  /**
   * Stop TTS only (speech + audio nodes + web speech), bumping the
   * generation token so in-flight async TTS exits. Leaves mic /
   * recognition / live resources untouched.
   *
   * @param bumpToken If true (default), increments the generation token
   *                  so any in-flight speak() promise sees itself as
   *                  stale. Set to false for an internal "stop" that
   *                  should NOT invalidate the caller's own token.
   */
  private stopSpeechOnly(bumpToken: boolean = true): void {
    if (bumpToken) {
      // Bumping the token invalidates any in-flight speak() promise.
      this.generationToken++;
    }

    // Stop AudioBufferSourceNode.
    if (this.currentSource) {
      try {
        this.currentSource.onended = null;
        this.currentSource.stop();
      } catch (_) {
        /* already stopped */
      }
      try {
        this.currentSource.disconnect();
      } catch (_) {
        /* already disconnected */
      }
      this.currentSource = null;
    }

    // Stop Web Speech API.
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (_) {
        /* no-op */
      }
    }
    this.activeUtterance = null;

    // Clear React-facing state.
    this.setSpeaking(null);
    this.setLoading(null);
  }

  /**
   * INSTANT, TOTAL cancellation of EVERY voice/audio resource. This is
   * the canonical teardown path used by:
   *   • voice toggle OFF  (Bug 3 fix)
   *   • manual stop      (cancel voice conversation)
   *   • city switch
   *   • new chat
   *   • unmount
   *
   * After this returns:
   *   • no AudioBufferSourceNode is running
   *   • speechSynthesis queue is empty
   *   • AudioContext is suspended (kept warm) OR closed (on dispose)
   *   • microphone is released (every MediaStreamTrack.stop())
   *   • SpeechRecognition is stopped
   *   • Live session + live audio contexts + live sources are gone
   *   • generation token is bumped (in-flight promises bail)
   *   • speaking/loading React state is cleared
   *
   * No callback from a prior generation can mutate state afterwards.
   *
   * @param closeAudioCtx If true, also close() the AudioContext. Use on
   *                      full unmount. For voice-toggle-off / city-switch
   *                      we SUSPEND (cheaper, keeps it warm for re-enable).
   */
  cancelAll(closeAudioCtx: boolean = false): void {
    // Bump the token FIRST so any in-flight async callback that resolves
    // later sees itself as stale and exits without touching state.
    this.generationToken++;

    // ── 1. Speech + audio nodes + web speech ──
    // (stopSpeechOnly calls setSpeaking/setLoading — we re-run below to
    //  guarantee state clears even if it short-circuits.)
    this.stopSpeechOnly(false);

    // ── 2. SpeechRecognition (voice typing) ──
    this.stopRecognition();

    // ── 3. Microphone (MediaStream) ──
    // Stop every track so the browser releases the mic indicator.
    this.releaseMic();

    // ── 4. Live mode resources ──
    this.stopLiveModeInternal();

    // ── 5. AudioContext ──
    if (this.audioContext) {
      if (closeAudioCtx) {
        if (this.audioContext.state !== 'closed') {
          this.audioContext.close().catch(() => undefined);
        }
        this.audioContext = null;
      } else {
        // Suspend (cheaper than close) so re-enabling voice is instant.
        if (this.audioContext.state === 'running') {
          this.audioContext.suspend().catch(() => undefined);
        }
      }
    }

    // ── 6. Final state clear ──
    this.setSpeaking(null);
    this.setLoading(null);
  }

  /**
   * Full teardown. Call on unmount: close AudioContext, clear cache,
   * remove listeners, cancel everything.
   */
  dispose(): void {
    this.cancelAll(true);
    this.clearCache();
    this.activeUtterance = null;
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.handleVisibility);
    }
    this.callbacks = { onSpeakingChange: () => undefined, onLoadingChange: () => undefined };
  }

  /**
   * Clear the per-message audio cache. Call on city switch so stale
   * audio from a different city context can never be replayed.
   */
  clearCache(): void {
    this.audioCache.clear();
  }

  // ── Microphone management ─────────────────────────────────────────

  /**
   * Acquire the microphone. VoiceManager becomes the sole owner of the
   * resulting MediaStream so it can reliably release it later. The
   * caller must NOT hold its own reference; use releaseMic() to free.
   */
  async acquireMic(): Promise<MediaStream | null> {
    if (!navigator.mediaDevices?.getUserMedia) return null;
    // If we already have a live mic, reuse it.
    if (
      this.micStream &&
      this.micStream.getTracks().some((t) => t.readyState === 'live')
    ) {
      return this.micStream;
    }
    try {
      // Release any dead reference first.
      this.releaseMic();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.micStream = stream;
      return stream;
    } catch (e) {
      console.warn('Microphone acquisition failed', e);
      return null;
    }
  }

  /**
   * Stop every MediaStreamTrack and drop the reference. Idempotent.
   * The browser releases the mic indicator immediately.
   */
  releaseMic(): void {
    if (this.micStream) {
      this.micStream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_) {
          /* no-op */
        }
      });
      this.micStream = null;
    }
    // Also release any live-mode stream (kept separately because Live
    // mode allocates its own getUserMedia for a 16 kHz input).
    if (this.liveStream) {
      this.liveStream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_) {
          /* no-op */
        }
      });
      this.liveStream = null;
    }
  }

  /** True iff a live mic MediaStream is currently held. */
  hasMic(): boolean {
    return (
      !!this.micStream &&
      this.micStream.getTracks().some((t) => t.readyState === 'live')
    );
  }

  // ── SpeechRecognition management ──────────────────────────────────

  /**
   * Start a SpeechRecognition session for voice typing. Any prior
   * session is stopped first. The supplied callbacks drive UI state.
   * VoiceManager owns the recognition instance so cancelAll() can
   * reliably stop it and prevent stale onresult callbacks.
   */
  startRecognition(opts: {
    language: 'en' | 'ta';
    onResult: (transcript: string) => void;
    onStart?: () => void;
    onEnd?: () => void;
    onError?: () => void;
  }): boolean {
    const Ctor =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!Ctor) return false;

    // Stop any prior recognition so we never have two running.
    this.stopRecognition();

    const recognition = new Ctor();
    recognition.lang = opts.language === 'ta' ? 'ta-IN' : 'en-IN';
    recognition.interimResults = false;
    recognition.continuous = false;

    // Capture the token at start; only forward results if still current.
    const startToken = this.generationToken;

    recognition.onstart = () => opts.onStart?.();
    recognition.onend = () => {
      this.recognition = null;
      opts.onEnd?.();
    };
    recognition.onerror = () => {
      this.recognition = null;
      opts.onError?.();
    };
    recognition.onresult = (event: any) => {
      // Drop results from a cancelled generation.
      if (!this.isCurrentToken(startToken)) return;
      const transcript = event.results[0][0].transcript;
      opts.onResult(transcript);
    };

    this.recognition = recognition;
    try {
      recognition.start();
    } catch (e) {
      console.warn('SpeechRecognition start failed', e);
      this.recognition = null;
      return false;
    }
    return true;
  }

  /** Stop the active SpeechRecognition (if any) and drop the reference. */
  stopRecognition(): void {
    if (this.recognition) {
      try {
        // Set handlers to no-ops first so onend/onerror don't fire
        // post-cancel and confuse the UI.
        this.recognition.onresult = null;
        this.recognition.onerror = null;
        this.recognition.onend = null;
        this.recognition.onstart = null;
        this.recognition.stop();
      } catch (_) {
        /* no-op */
      }
      this.recognition = null;
    }
  }

  // ── Live mode management ──────────────────────────────────────────
  // Live mode uses Gemini's realtime API with a 16 kHz input AudioContext
  // and a 24 kHz output AudioContext. VoiceManager holds these so
  // cancelAll() can fully tear them down — no orphaned audio contexts
  // or mic streams survive a voice cancel.

  /** Store the live session handle so cancelAll() can close it. */
  setLiveSession(session: any): void {
    // If a previous session exists, close it first.
    if (this.liveSession && this.liveSession !== session) {
      try {
        this.liveSession.close();
      } catch (_) {
        /* no-op */
      }
    }
    this.liveSession = session;
  }

  getLiveSession(): any {
    return this.liveSession;
  }

  /** Store the live-mode audio contexts so cancelAll() can close them. */
  setLiveAudioContexts(input: AudioContext, output: AudioContext): void {
    this.liveInputCtx = input;
    this.liveOutputCtx = output;
  }

  /** Store the live-mode MediaStream so cancelAll() can stop its tracks. */
  setLiveStream(stream: MediaStream): void {
    this.liveStream = stream;
  }

  /** Track a live-mode AudioBufferSourceNode so it can be stopped on cancel. */
  addLiveSource(source: AudioBufferSourceNode): void {
    this.liveSources.add(source);
    source.onended = () => {
      this.liveSources.delete(source);
    };
  }

  /** Return the live output AudioContext (or null). */
  getLiveOutputCtx(): AudioContext | null {
    return this.liveOutputCtx;
  }

  /** True iff a live session is currently active. */
  isLiveActive(): boolean {
    return this.liveSession !== null;
  }

  /**
   * Stop live mode: close the session, stop every live source, close
   * both audio contexts, stop the live mic stream. Idempotent.
   *
   * Public alias for stopLiveModeInternal() — exposed so App.tsx can
   * tear down live mode without touching the main TTS AudioContext.
   */
  stopLiveMode(): void {
    this.stopLiveModeInternal();
  }

  /**
   * Stop live mode: close the session, stop every live source, close
   * both audio contexts, stop the live mic stream. Idempotent.
   */
  stopLiveModeInternal(): void {
    if (this.liveSession) {
      try {
        this.liveSession.close();
      } catch (_) {
        /* no-op */
      }
      this.liveSession = null;
    }
    this.liveSources.forEach((s) => {
      try {
        s.onended = null;
        s.stop();
      } catch (_) {
        /* no-op */
      }
      try {
        s.disconnect();
      } catch (_) {
        /* no-op */
      }
    });
    this.liveSources.clear();

    if (this.liveScriptProcessor) {
      try {
        this.liveScriptProcessor.disconnect();
      } catch (_) {
        /* no-op */
      }
      this.liveScriptProcessor = null;
    }

    if (this.liveInputCtx) {
      try {
        if (this.liveInputCtx.state !== 'closed') {
          this.liveInputCtx.close().catch(() => undefined);
        }
      } catch (_) {
        /* no-op */
      }
      this.liveInputCtx = null;
    }
    if (this.liveOutputCtx) {
      try {
        if (this.liveOutputCtx.state !== 'closed') {
          this.liveOutputCtx.close().catch(() => undefined);
        }
      } catch (_) {
        /* no-op */
      }
      this.liveOutputCtx = null;
    }

    // Live mic stream is released via releaseMic() in cancelAll(); we
    // also handle it here for standalone stopLiveMode calls.
    if (this.liveStream) {
      this.liveStream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_) {
          /* no-op */
        }
      });
      this.liveStream = null;
    }
  }

  // ── Private helpers ───────────────────────────────────────────────

  /** Play an AudioBuffer and wire up the onended handler. */
  private playBuffer(
    buffer: AudioBuffer,
    messageId: string,
    token: number,
    requestId: number
  ): void {
    if (!this.audioContext) return;
    // Re-validate EVERY precondition right before scheduling the source
    // node — this is the last line of defense against a race that raced
    // between speak()'s last await and the synchronous playBuffer call.
    if (
      !this.isCurrentToken(token) ||
      !this.enabled ||
      !this.isCurrentRequest(requestId)
    ) {
      return;
    }

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    try {
      source.connect(this.audioContext.destination);
    } catch (_) {
      // AudioContext may have been suspended/closed concurrently.
      return;
    }

    source.onended = () => {
      // Only clear state if still the active generation AND still
      // playing this same message AND same request.
      if (
        this.isCurrentToken(token) &&
        this.isCurrentRequest(requestId) &&
        this.speakingMessageId === messageId
      ) {
        this.setSpeaking(null);
      }
      if (this.currentSource === source) {
        this.currentSource = null;
      }
    };

    this.currentSource = source;
    this.setSpeaking(messageId);
    try {
      source.start();
    } catch (_) {
      /* already started */
    }
  }

  /** Web Speech API fallback (when Gemini TTS fails). */
  private fallbackWebSpeech(
    text: string,
    id: string,
    language: 'en' | 'ta',
    token: number,
    requestId: number
  ): void {
    if (
      !this.isCurrentToken(token) ||
      !this.enabled ||
      !this.isCurrentRequest(requestId)
    ) {
      this.setLoading(null);
      this.setSpeaking(null);
      return;
    }
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      this.setLoading(null);
      this.setSpeaking(null);
      return;
    }

    this.setLoading(null);
    this.setSpeaking(id);

    const cleanText = cleanTextForSpeech(text);
    if (!cleanText) {
      this.setSpeaking(null);
      return;
    }

    // Clear any queued speech first — we don't want a previously-queued
    // utterance from a stale request to play ahead of ours.
    try {
      window.speechSynthesis.cancel();
    } catch (_) {
      /* no-op */
    }

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = language === 'ta' ? 'ta-IN' : 'en-IN';

    utterance.onend = () => {
      if (
        this.isCurrentToken(token) &&
        this.isCurrentRequest(requestId) &&
        this.speakingMessageId === id &&
        this.activeUtterance === utterance
      ) {
        this.setSpeaking(null);
        this.activeUtterance = null;
      }
    };
    utterance.onerror = () => {
      if (
        this.isCurrentToken(token) &&
        this.isCurrentRequest(requestId) &&
        this.speakingMessageId === id &&
        this.activeUtterance === utterance
      ) {
        this.setSpeaking(null);
        this.activeUtterance = null;
      }
    };

    const voices = window.speechSynthesis.getVoices();
    const langPrefix = language === 'ta' ? 'ta' : 'en';
    const preferredVoice =
      voices.find(
        (v) =>
          v.lang.startsWith(langPrefix) &&
          v.name.toLowerCase().includes('female')
      ) || voices.find((v) => v.lang.startsWith(langPrefix));

    if (preferredVoice) utterance.voice = preferredVoice;

    this.activeUtterance = utterance;
    try {
      window.speechSynthesis.speak(utterance);
    } catch (_) {
      this.setSpeaking(null);
      this.activeUtterance = null;
    }
  }

  private setSpeaking(id: string | null): void {
    if (this.speakingMessageId === id) return;
    this.speakingMessageId = id;
    this.callbacks.onSpeakingChange(id);
  }

  private setLoading(id: string | null): void {
    if (this.loadingMessageId === id) return;
    this.loadingMessageId = id;
    this.callbacks.onLoadingChange(id);
  }

  /**
   * Chrome sometimes pauses / leaves stranded speechSynthesis utterances
   * when the tab is hidden. On visibility regain, cancel queued speech
   * so a hidden tab can never leak stale audio into the active session.
   */
  private handleVisibility = (): void => {
    if (document.hidden) {
      // Tab hidden — prevent queued speech from surprising the user later.
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel();
        } catch (_) {
          /* no-op */
        }
      }
    }
  };
}
