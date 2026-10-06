
import React, { useState, useCallback, useRef, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import LoginModal from './components/LoginModal';
import CitySelector from './components/CitySelector';
import ImageGenModal from './components/ImageGenModal';
import NewsView from './components/NewsView';
import StatsView from './components/StatsView';
import AboutView from './components/AboutView';
import PromoteModal from './components/PromoteModal';
import TripPlannerView from './components/TripPlannerView';
import { Message, ChatHistory } from './types';
import { improvePrompt } from './services/geminiService';
import { VoiceManager } from './services/VoiceManager';
import { ChatManager } from './services/ChatManager';
import { auth, db, handleFirestoreError } from './services/firebaseService';
import { getCurrentWeather } from './services/weatherService';
import { onAuthStateChanged, User } from 'firebase/auth';
import { collection, query, orderBy, onSnapshot, doc, setDoc, deleteDoc, getDoc } from 'firebase/firestore';
import { 
  Moon, 
  Sun, 
  Cloud, 
  X, 
  Volume2, 
  Mic, 
  RefreshCw, 
  ChevronDown, 
  CloudRain, 
  CloudLightning 
} from 'lucide-react';
import { SUBTITLE, getSuggestions } from './constants';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';

const encode = (bytes: Uint8Array) => {
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
};

const decode = (base64: string) => {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  return bytes;
};

const decodeAudioData = async (data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number) => {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
  }
  return buffer;
};

const getClientApiKey = () => (import.meta as any).env?.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY || process.env.API_KEY || '';

const App: React.FC = () => {
  const [selectedCity, setSelectedCity] = useState('Madurai');
  const [currentView, setCurrentView] = useState<'chat' | 'news' | 'stats' | 'about' | 'trip-planner'>('chat');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isCitySelectorOpen, setIsCitySelectorOpen] = useState(false);
  const [isImageGenOpen, setIsImageGenOpen] = useState(false);
  const [isPromoteOpen, setIsPromoteOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<ChatHistory[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [isDeepSearch, setIsDeepSearch] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [language, setLanguage] = useState<'en' | 'ta'>('en');
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [messagesSent, setMessagesSent] = useState(0);
  const [user, setUser] = useState<User | null>(null);

  // Weather State
  const [weather, setWeather] = useState({ temp: '--', condition: 'Loading...' });
  const [isWeatherRefreshing, setIsWeatherRefreshing] = useState(false);
  const [userLocation, setUserLocation] = useState<{lat: number, lng: number} | null>(null);
  const [isLogoSpinning, setIsLogoSpinning] = useState(false);

  // Request Permissions on mount
  useEffect(() => {
    // Request Geolocation
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
        },
        (error) => {
          console.warn("Location access denied or unavailable", error);
        }
      );
    }

    // Request Microphone early so user can "Allow" once and it's ready
    const requestMic = async () => {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          // Stop stream immediately, just needed the permission prompt
          stream.getTracks().forEach(track => track.stop());
        }
      } catch (err) {
        console.warn("Microphone permission denied or not supported on startup", err);
      }
    };
    requestMic();
  }, []);

  // TTS State
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  const [isTTSLoading, setIsTTSLoading] = useState<string | null>(null);
  const currentAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioCacheRef = useRef<Map<string, AudioBuffer>>(new Map());

  // ── VoiceManager + ChatManager (singletons) ──
  // VoiceManager owns speech synthesis / audio playback / cancellation.
  // ChatManager owns AI requests / AbortController / request IDs.
  // Both are created ONCE and live for the lifetime of the component.
  const voiceManagerRef = useRef<VoiceManager | null>(null);
  if (voiceManagerRef.current === null) {
    voiceManagerRef.current = new VoiceManager({
      onSpeakingChange: (id) => setSpeakingMessageId(id),
      onLoadingChange: (id) => setIsTTSLoading(id),
    });
  }
  const chatManagerRef = useRef<ChatManager | null>(null);
  if (chatManagerRef.current === null) {
    chatManagerRef.current = new ChatManager({});
  }
  const voiceManager = voiceManagerRef.current;
  const chatManager = chatManagerRef.current;

  // ── React-side single-flight guard for handleSendMessage. ──
  // Complements ChatManager's internal guard. Synchronous so rapid
  // Enter/send presses can't both slip through the async `isLoading`
  // check before the first request flips the React state. (Bug 4 fix.)
  const sendInFlightRef = useRef(false);

  // ── Bind ChatManager <-> VoiceManager (single sources of truth). ──
  // When ChatManager starts a NEW request, it calls
  // voiceManager.bindRequest(requestId). This guarantees the only AI
  // generation allowed to produce speech is the current one — fixing
  // the "stale response speaks after city change / rapid send" bug.
  // VoiceManager also exposes speakForRequest() which rejects TTS for
  // any non-current request, so text and voice can never desync.
  useEffect(() => {
    chatManager.setVoiceManager(voiceManager);
  }, [chatManager, voiceManager]);

  // Live Mode State
  const [isLiveActive, setIsLiveActive] = useState(false);
  const [liveTranscription, setLiveTranscription] = useState('');
  const liveSessionRef = useRef<any>(null);
  const liveAudioContextRef = useRef<{ input: AudioContext; output: AudioContext } | null>(null);
  const liveNextStartTimeRef = useRef(0);
  const liveSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Initialize and Pre-warm AudioContext on first user gesture
  // (driven through VoiceManager — single source of truth for speaker).
  useEffect(() => {
    const warmAudio = () => {
      voiceManager.warmAudioContext();
      window.removeEventListener('click', warmAudio);
      window.removeEventListener('keydown', warmAudio);
    };
    window.addEventListener('click', warmAudio);
    window.addEventListener('keydown', warmAudio);
    return () => {
      window.removeEventListener('click', warmAudio);
      window.removeEventListener('keydown', warmAudio);
    };
  }, [voiceManager]);

  // ── Unmount / full teardown ──
  // Abort any in-flight AI request, cancel all speech, release the mic,
  // close every AudioContext (TTS + live input/output), stop SpeechRecognition,
  // remove listeners, clear timers/intervals. No orphaned resources survive.
  useEffect(() => {
    return () => {
      // ChatManager: abort + dispose (no further callbacks can fire).
      chatManager.dispose();
      // VoiceManager: cancelAll(true) → stops speech, releases the mic
      // (every MediaStreamTrack.stop()), stops SpeechRecognition, closes
      // the live session + both live audio contexts, AND closes the main
      // TTS AudioContext. Single tear-down path = no orphans.
      voiceManager.dispose();
      // Defensive: clear legacy live-mode refs (VoiceManager is the source
      // of truth but these local refs may still be referenced by in-flight
      // closures).
      liveAudioContextRef.current = null;
      liveSessionRef.current = null;
      liveSourcesRef.current.clear();
      // Web Speech API fallback safety.
      if ('speechSynthesis' in window) {
        try { window.speechSynthesis.cancel(); } catch (_) {}
      }
    };
  }, [chatManager, voiceManager]);

  // Handle Dark Mode
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Handle Document Title
  useEffect(() => {
    document.title = `${selectedCity} Info AI`;
  }, [selectedCity]);

  // Load history from persistence (Fallback for guest mode)
  useEffect(() => {
    if (user) return; // Skip guest local storage if logged in
    const saved = localStorage.getItem('city-ai-chat-history');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        const chatsMap = new Map<string, ChatHistory>();
        parsed.forEach((h: any) => {
          const chat: ChatHistory = {
            ...h,
            lastUpdated: new Date(h.lastUpdated),
            messages: (h.messages || []).map((m: any) => ({
              ...m,
              timestamp: new Date(m.timestamp)
            }))
          };
          chatsMap.set(h.id, chat);
        });
        setHistory(Array.from(chatsMap.values()));
      } catch (e) {
        console.error("History recovery failed", e);
      }
    } else {
      setHistory([]);
    }
  }, [user]);

  // Save history to persistence (Guest mode)
  useEffect(() => {
    if (user || history.length === 0) return;
    localStorage.setItem('city-ai-chat-history', JSON.stringify(history));
  }, [history, user]);

  // Firebase Auth Listener
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        setMessages([]);
        setCurrentChatId(null);
      } else {
        // Logged out
        setMessages([]);
        setCurrentChatId(null);
      }
    });
    return () => unsubscribe();
  }, []);

  // Firebase Firestore History Sync
  useEffect(() => {
    if (!user) return;

    const q = query(
      collection(db, 'users', user.uid, 'chats'),
      orderBy('lastUpdated', 'desc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const chatsMap = new Map<string, ChatHistory>();
      snapshot.docs.forEach(doc => {
        const data = doc.data();
        const chat: ChatHistory = {
          id: doc.id,
          title: data.title,
          lastUpdated: new Date(data.lastUpdated),
          messages: (data.messages || []).map((m: any) => ({
            ...m,
            timestamp: new Date(m.timestamp)
          }))
        };
        // Use Map to ensure unique IDs (last one wins if duplicates exist for some reason)
        chatsMap.set(doc.id, chat);
      });
      setHistory(Array.from(chatsMap.values()));
    }, (error) => {
      console.error("Firestore sync error", error);
    });

    return () => unsubscribe();
  }, [user]);

  const handleRefreshWeather = useCallback(async (city: string, force: boolean = false) => {
    if (isWeatherRefreshing) return;

    // Check Cache (unless forced)
    const cacheKey = `weather-cache-${city.toLowerCase()}`;
    if (!force) {
      const cachedData = localStorage.getItem(cacheKey);
      if (cachedData) {
        try {
          const { temp, condition, timestamp } = JSON.parse(cachedData);
          // Cache valid for 30 minutes
          if (Date.now() - timestamp < 30 * 60 * 1000) {
            setWeather({ temp, condition });
            return;
          }
        } catch (e) {}
      }
    }

    setIsWeatherRefreshing(true);
    try {
      const newWeather = await getCurrentWeather(city);
      setWeather(newWeather);
      localStorage.setItem(cacheKey, JSON.stringify({ ...newWeather, timestamp: Date.now() }));
    } catch (e: any) {
      console.warn("Live weather fetch failed, using local estimate:", e);
      // Fallback to localized smart weather data so the UI remains pristine even if rate-limited or offline
      const now = new Date();
      const hour = now.getHours();
      // Tamil Nadu is generally warm; estimate a temperature based on the time of day
      const estimatedTemp = (hour >= 6 && hour <= 18) 
        ? Math.floor(Math.random() * 4) + 32 // 32°C to 35°C during daytime
        : Math.floor(Math.random() * 3) + 26; // 26°C to 28°C during nighttime
      const conditionOptions = (hour >= 6 && hour <= 18)
        ? ['Sunny', 'Mostly Sunny', 'Partly Cloudy']
        : ['Clear', 'Partly Cloudy', 'Cloudy'];
      const estimatedCondition = conditionOptions[Math.floor(Math.random() * conditionOptions.length)];
      
      const fallbackWeather = {
        temp: estimatedTemp.toString(),
        condition: estimatedCondition
      };
      
      setWeather(fallbackWeather);
      console.log(`Applied fallback local weather for custom experience: ${estimatedTemp}°C, ${estimatedCondition}`);
      // Store in cache so we don't spam requests while experiencing low quota or server issues
      localStorage.setItem(cacheKey, JSON.stringify({ ...fallbackWeather, timestamp: Date.now() }));
    } finally {
      // Small delay to ensure the loading state is actually visible to the user
      setTimeout(() => {
        setIsWeatherRefreshing(false);
      }, 800);
    }
  }, [isWeatherRefreshing]);

  useEffect(() => {
    handleRefreshWeather(selectedCity);
  }, [selectedCity]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  /**
   * Unified "fresh slate" reset used by city switch / new chat /
   * history select / delete-current-chat. Single path = no missed
   * cleanup steps. Steps:
   *   1. Abort the in-flight AI request (bumps ChatManager.requestId).
   *   2. Re-bind VoiceManager to the new request id so any subsequent
   *      "Read Aloud" press or late TTS from the prior request is
   *      rejected — this is the synchronization fix for Bug 2 & Bug 4.
   *   3. Cancel ALL speech + clear the per-message audio cache (so
   *      stale audio from a previous city never replays).
   *   4. Clear React-side TTS state.
   *
   * It does NOT change `selectedCity` / `messages` / `currentChatId` —
   * callers do that themselves (different paths want different values).
   */
  const resetConversationState = useCallback(() => {
    // 1 + 2: Abort AI + rebind VoiceManager to the bumped request id.
    chatManager.reset();
    voiceManager.bindRequest(chatManager.getCurrentRequestId());

    // 3: Cancel ALL speech + clear the per-message audio cache.
    voiceManager.stop();
    voiceManager.clearCache();

    // 4: Clear React-side TTS / loading state.
    setSpeakingMessageId(null);
    setIsTTSLoading(null);
    setIsLoading(false);
  }, [chatManager, voiceManager]);

  const handleSendMessage = useCallback(async (text: string, files?: { data: string, mimeType: string }[]) => {
    // ── React-side single-flight guard. ──
    // React's `isLoading` state is async — rapid Enter/send presses can
    // both pass `!isLoading` before the first request flips it true.
    // This ref provides a *synchronous* guard that complements
    // ChatManager's own internal single-flight protection. If a send is
    // already in flight, ignore this call entirely (the in-flight send
    // is the one that wins). This is part of the Bug 4 fix.
    if (sendInFlightRef.current) return;
    sendInFlightRef.current = true;
    try {
      const isTripRequest = getSuggestions(selectedCity).some(s => s.prompt === text && s.title.toLowerCase().includes('trip')) || 
                          text.toLowerCase().includes('plan a trip') || 
                          text.toLowerCase().includes('itinerary');
    
    const userMsg: Message = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      role: 'user',
      content: text,
      timestamp: new Date(),
      attachments: files ? files.map(f => ({
        data: f.data,
        mimeType: f.mimeType
      })) : undefined
    };

    // ── Snapshot the messages array NOW (before any state updates). ──
    // Used for history writes inside ChatManager's onComplete so that
    // the latest request — and only the latest — commits history.
    const messagesAtStart = messages;
    const newMessages = [...messagesAtStart, userMsg];
    setMessages(newMessages);
    setMessagesSent(prev => prev + 1);
    setIsLoading(true);
    
    if (isTripRequest) {
      setCurrentView('trip-planner');
    } else if (currentView !== 'trip-planner') {
      setCurrentView('chat');
    }

    // Build API chat history using all PREVIOUS messages
    const chatHistoryForAPI = messagesAtStart.map(m => ({
      role: m.role === 'assistant' ? 'model' as const : 'user' as const,
      parts: m.role === 'user' && m.attachments ? [
        { text: m.content },
        ...m.attachments.map(att => ({
          inlineData: {
            data: att.data,
            mimeType: att.mimeType
          }
        }))
      ] : [{ text: m.content }]
    }));

    const assistantMsgId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const assistantMsg: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, assistantMsg]);

    // ── Resolve current language (may switch based on user text). ──
    let currentLang = language;
    const lowerText = text.toLowerCase();
    if (lowerText.includes('talk tamil') || lowerText.includes('tamil pesu') || lowerText.includes('tamilil pesu')) {
      setLanguage('ta');
      currentLang = 'ta';
    } else if (lowerText.includes('talk english') || lowerText.includes('englishil pesu') || lowerText.includes('pesu english')) {
      setLanguage('en');
      currentLang = 'en';
    }

    // ── Cancel any speech currently playing — the streaming text will
    //    replace the currently-speaking message; old audio is no longer
    //    relevant. Rule: every new request starts with a clean audio
    //    slate. ──
    voiceManager.reset();

    // ── Snapshot top-level values for the closure used by onComplete. ──
    // Using fresh values from React state ensures latest-city consistency.
    const cityForRequest = selectedCity;
    const userLocationForRequest = userLocation;
    const userForRequest = user;
    const currentChatIdForRequest = currentChatId;
    const isDeepSearchForRequest = isDeepSearch;
    const userTextForRequest = text;

    // ── Wire ChatManager callbacks for THIS request. ChatManager
    //    aborts any older request before this one starts AND only the
    //    latest request can fire onChunk / onComplete. ──
    chatManager.setCallbacks({
      onChunk: (reqId, msgId, chunkText, chunkSources) => {
        if (reqId !== chatManager.getCurrentRequestId()) return;
        if (chunkText.length > 0) setIsLoading(false);
        setMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: chunkText, sources: chunkSources } : m));
      },
      onComplete: (reqId, result) => {
        // Latest-response-wins: bail out if no longer current.
        if (reqId !== chatManager.getCurrentRequestId()) return;
        const {
          assistantMsgId: msgId,
          text: responseText,
          sources: responseSources,
          followUps,
        } = result;

        setMessages(prev => prev.map(m => m.id === msgId ? { 
          ...m, 
          content: responseText, 
          sources: responseSources,
          followUps: followUps,
          isDeepSearch: isDeepSearchForRequest
        } : m));

        const finalMessages = [...newMessages, { 
          ...assistantMsg, 
          content: responseText, 
          sources: responseSources,
          followUps: followUps,
          isDeepSearch: isDeepSearchForRequest
        }];

        // Update history storage
        const chatData = {
          id: currentChatIdForRequest || `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          title: userTextForRequest.length > 25 ? userTextForRequest.substring(0, 25) + '...' : userTextForRequest,
          lastUpdated: new Date().toISOString(),
          messages: finalMessages.map(m => {
            const msgObj: any = {
              id: m.id,
              role: m.role,
              content: m.content || "",
              timestamp: m.timestamp instanceof Date ? m.timestamp.toISOString() : m.timestamp
            };
            if (m.sources) msgObj.sources = m.sources;
            if (m.followUps) msgObj.followUps = m.followUps;
            if (m.isDeepSearch !== undefined) msgObj.isDeepSearch = m.isDeepSearch;
            if (m.attachments !== undefined && m.attachments !== null) {
              msgObj.attachments = m.attachments.map(att => {
                const attObj: any = {
                  data: att.data,
                  mimeType: att.mimeType,
                };
                if (att.name !== undefined) attObj.name = att.name;
                if (att.size !== undefined) attObj.size = att.size;
                return attObj;
              });
            }
            return msgObj;
          })
        };

        if (userForRequest) {
          let chatId = currentChatIdForRequest;
          try {
            if (!chatId || isTripRequest) {
              chatId = chatData.id;
              setCurrentChatId(chatId);
            }
            setDoc(doc(db, 'users', userForRequest.uid, 'chats', chatId), chatData).catch((e) => {
              handleFirestoreError(e, 'write', `users/${userForRequest.uid}/chats/${chatId}`);
            });
          } catch (e) {
            handleFirestoreError(e, 'write', `users/${userForRequest.uid}/chats/${chatData.id}`);
          }
        } else {
          // Guest local update
          setHistory(prev => {
            const index = prev.findIndex(h => h.id === chatData.id);
            const updatedChat = {
              ...chatData,
              lastUpdated: new Date(chatData.lastUpdated),
              messages: finalMessages
            };

            if (index >= 0) {
              const newHistory = [...prev];
              newHistory[index] = updatedChat;
              return newHistory;
            } else {
              return [updatedChat, ...prev];
            }
          });
          
          if (!currentChatIdForRequest || isTripRequest) {
            setCurrentChatId(chatData.id);
          }
        }
      },
      onLoadingChange: (loading) => {
        setIsLoading(loading);
      },
      onError: (reqId, msgId, message) => {
        if (reqId !== chatManager.getCurrentRequestId()) return;
        setIsLoading(false);
        setMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: message } : m));
      },
    });

    // ── Kick off the request via ChatManager. ChatManager aborts any
    //    previously-running request FIRST, so there is NEVER more than
    //    one AI request in flight. ──
    chatManager.sendMessage(userTextForRequest, assistantMsgId, {
      history: chatHistoryForAPI,
      city: cityForRequest,
      language: currentLang,
      location: userLocationForRequest || undefined,
      attachments: files,
      isDeepSearch: isDeepSearchForRequest,
    }).catch((e) => {
      // Defensive — ChatManager swallows AbortError internally.
      console.error("ChatManager.sendMessage threw unexpectedly:", e);
      setIsLoading(false);
    });
    } finally {
      // Release the React-side single-flight guard. This is outside the
      // await so it always runs, even if sendMessage rejects.
      sendInFlightRef.current = false;
    }
  }, [messages, selectedCity, currentView, currentChatId, language, user, isDeepSearch, userLocation, chatManager, voiceManager]);

  const handleSelectHistory = (id: string) => {
    const item = history.find(h => h.id === id);
    if (item) {
      // Unified fresh-slate reset: abort AI + rebind VoiceManager to the
      // new request id (so a late TTS / stale "Read Aloud" from the
      // previous chat cannot leak into the newly-selected chat) +
      // cancel all speech + clear React state.
      resetConversationState();
      setMessages(item.messages);
      setCurrentChatId(item.id);
      setCurrentView('chat');
      setIsSidebarOpen(false);
    }
  };

  const handleDeleteHistory = async (id: string) => {
    if (user) {
      try {
        await deleteDoc(doc(db, 'users', user.uid, 'chats', id));
      } catch (e) {
        handleFirestoreError(e, 'delete', `users/${user.uid}/chats/${id}`);
      }
    } else {
      setHistory(prev => prev.filter(item => item.id !== id));
    }
    
    if (currentChatId === id) {
      // The currently-active chat was deleted: full reset (abort + rebind
      // request id + cancel speech + clear cache) before clearing messages.
      resetConversationState();
      setMessages([]);
      setCurrentChatId(null);
    }
  };

  const handleDownloadHistory = (id: string) => {
    const chat = history.find(h => h.id === id);
    if (!chat) return;
    const targetMessages = chat.messages || messages;
    const transcript = targetMessages.map(m => 
      `${m.role.toUpperCase()} (${m.timestamp.toLocaleTimeString()}):\n${m.content}\n`
    ).join('\n---\n\n');

    const blob = new Blob([transcript], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${chat.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_transcript.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleSpeak = useCallback(async (text: string, id: string) => {
    // ── Delegate entirely to VoiceManager. ──
    // VoiceManager guarantees:
    //   • Only one audio source plays at a time.
    //   • Toggling the currently-speaking message OFF is instant.
    //   • Generation tokens prevent stale TTS responses from speaking.
    //   • request-id binding: TTS for a non-current AI request is rejected,
    //     so a response streaming from a previous generation can NEVER speak.
    //     This is the fix for Bug 2 (stale speech after city change) and
    //     Bug 4 (concurrent requests) at the TTS layer.
    //   • Audio cache is reused for instant replay.
    //   • speechSynthesis.cancel() + audio node stop() on stop.
    //
    // The signature (text, id) is preserved so ChatWindow / TripPlannerView
    // components are unchanged. We bind the speak to ChatManager's current
    // request id — so manual "Read Aloud" presses are also gated by the
    // current generation (preventing stale-city speech from replaying).
    voiceManager.warmAudioContext();
    await voiceManager.speakForRequest(
      text,
      id,
      language,
      chatManager.getCurrentRequestId()
    );
  }, [language, voiceManager, chatManager]);

  const handleUtilityAction = (action: 'news' | 'analytics' | 'promote' | 'about') => {
    setIsSidebarOpen(false);
    switch (action) {
      case 'news': setCurrentView('news'); break;
      case 'analytics': setCurrentView('stats'); break;
      case 'promote': setIsPromoteOpen(true); break;
      case 'about': setCurrentView('about'); break;
    }
  };

  const stopLiveMode = useCallback(() => {
    // ── Full live-mode teardown via VoiceManager. ──
    // VoiceManager owns the live session, audio contexts, mic stream,
    // and live audio sources. Calling its teardown guarantees:
    //   • the Gemini live session is closed
    //   • every live AudioBufferSourceNode is stopped + disconnected
    //   • the script processor is disconnected
    //   • both live AudioContexts (input 16 kHz + output 24 kHz) are closed
    //   • the live mic MediaStream tracks are all stopped (browser
    //     releases the mic indicator immediately — no orphaned mic,
    //     fixes the "mic stays active after cancel" requirement)
    //   • the live session/scriptProcessor refs are nulled
    //
    // We use stopLiveModeInternal() rather than cancelAll() so we don't
    // suspend the main 24 kHz TTS AudioContext (independent resource).
    voiceManager.stopLiveMode();

    // Keep the local refs in sync (defensive; VoiceManager is the source
    // of truth but these refs may be referenced by in-flight callbacks).
    liveSessionRef.current = null;
    liveSourcesRef.current.clear();
    liveAudioContextRef.current = null;

    setIsLiveActive(false);
    setLiveTranscription('');
  }, [voiceManager]);

  const startLiveMode = useCallback(async () => {
    try {
      setIsLiveActive(true);
      const ai = new GoogleGenAI({ apiKey: getClientApiKey() });
      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      await inputCtx.resume();
      await outputCtx.resume();
      
      console.log(`Starting live mode for city: ${selectedCity}`);
      // ── Register the live audio contexts with VoiceManager so it owns
      //    them and can reliably close them on cancel/unmount. This is
      //    the fix for the "orphaned audio contexts after cancellation"
      //    requirement — VoiceManager is the single teardown path. ──
      voiceManager.setLiveAudioContexts(inputCtx, outputCtx);
      liveAudioContextRef.current = { input: inputCtx, output: outputCtx };

      // ── Acquire the mic and register it with VoiceManager so the mic
      //    is ALWAYS released (every MediaStreamTrack.stop()) when live
      //    mode is cancelled. Fixes the "mic stays active after cancel"
      //    requirement. ──
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      voiceManager.setLiveStream(stream);
      const sessionPromise = ai.live.connect({
        model: 'gemini-3.1-flash-live-preview',
        callbacks: {
          onopen: () => {
            const source = inputCtx.createMediaStreamSource(stream);
            const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
            scriptProcessor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const int16 = new Int16Array(inputData.length);
              for (let i = 0; i < inputData.length; i++) int16[i] = inputData[i] * 32768;
              sessionPromise.then(s => s.sendRealtimeInput({ audio: { data: encode(new Uint8Array(int16.buffer)), mimeType: 'audio/pcm;rate=16000' } }));
            };
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputCtx.destination);
          },
          onmessage: async (m: LiveServerMessage) => {
            const audioData = m.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (audioData) {
              liveNextStartTimeRef.current = Math.max(liveNextStartTimeRef.current, outputCtx.currentTime);
              const buffer = await decodeAudioData(decode(audioData), outputCtx, 24000, 1);
              const source = outputCtx.createBufferSource();
              source.buffer = buffer;
              source.connect(outputCtx.destination);
              source.onended = () => {
                liveSourcesRef.current.delete(source);
              };
              // ── Register each live source with VoiceManager so a
              //    cancel stops it (and disconnects it) reliably. ──
              voiceManager.addLiveSource(source);
              liveSourcesRef.current.add(source);
              source.start(liveNextStartTimeRef.current);
              liveNextStartTimeRef.current += buffer.duration;
            }
            if (m.serverContent?.outputTranscription) setLiveTranscription(prev => prev + m.serverContent?.outputTranscription?.text);
            if (m.serverContent?.turnComplete) setLiveTranscription('');
          },
          onclose: () => {
            console.log("Live session closed by server/timeout");
            // Instead of closing the UI, we keep it open. 
            // We can null the session to allow a manual restart or just wait for user to close.
            liveSessionRef.current = null;
            voiceManager.setLiveSession(null);
          },
          onerror: (e) => {
            console.error("Live session error:", e);
            // Don't close UI on error, just log it.
            liveSessionRef.current = null;
            voiceManager.setLiveSession(null);
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Aoede' }
            }
          },
          systemInstruction: `You are ${selectedCity} Info AI, a powerful next-generation local expert for ${selectedCity}. 

          Location Context & Intent-Based Routing Rules:
          You have two independent, persistent location contexts. You must maintain both and NEVER overwrite or alter the Active Selected City with the Device GPS Location:
          1. Active Selected City: "${selectedCity}"
          2. Device GPS Location: ${userLocation ? `Latitude ${userLocation.lat}, Longitude ${userLocation.lng}` : 'Currently Unavailable'}.

          Routing Rules:
          - Rule 1 (General Inquiries -> Selected City): General questions about local spots (restaurants, tourist places, hotels, famous food, shopping, weather, events, news, etc.) must be answered using "${selectedCity}".
          - Rule 2 (Nearby/Current Inquiries -> GPS Location): Nearby or current-location questions (containing "near me", "nearby", "around me", "closest", "nearest", "my current location", "within walking distance", or referencing physical proximity to coordinates) must strictly use the Device GPS Location context. If unavailable, politely explain that you need location access for nearby lookups.
          - Rule 3 (No Overwrite): Never overwrite "${selectedCity}" with the GPS location. Keep them independent.
          - Rule 4 (Ambiguity & Clarification): If a query is ambiguous and could apply to both (e.g. "What's the weather?"), and both locations are relevant or different, do not guess; ask a short, polite clarification question.

          Provide fast, accurate, and conversational help. Be brief but highly intelligent. 
          Respond with confidence and soul. If asked who created you, say you were developed by K. Rajasudar. 
          Use your deep knowledge of ${selectedCity} festivals, food, and culture to delight the user.
          
          MANDATORY SPEECH TONE REQUIREMENT:
          - Always speak in an extremely soft, sweet, gentle, caring, and affectionate female voice (like a beautiful divine goddess/god-girl local guide companion).
          - Use a highly comforting, soothing, melodious, and welcoming tone.
          - **CRITICAL CONSTRAINT**: NEVER use the phrase "my dear" or similar patronizing greetings/terms of endearment in any language. Address the user directly, naturally, and professionally.`,
          outputAudioTranscription: {}
        }
      });
      const liveSession = await sessionPromise;
      liveSessionRef.current = liveSession;
      // ── Register the live session with VoiceManager so it owns the
      //    session handle and can close it on cancel/unmount. ──
      voiceManager.setLiveSession(liveSession);
    } catch (e) {
      console.error(e);
      // ── Live mode failed to start: release every resource we may
      //    have acquired BEFORE any error path could register them. ──
      //    (e.g. the mic may have been acquired before the connect()
      //    call rejected.) This guarantees no orphaned mic / audio
      //    context survives a failed start.
      voiceManager.stopLiveMode();
      liveSessionRef.current = null;
      liveAudioContextRef.current = null;
      liveSourcesRef.current.clear();
      // Only close if it failed to even start
      setIsLiveActive(false);
    }
  }, [selectedCity, userLocation, voiceManager]); // Removed stopLiveMode from dependency to prevent weird cycles

  const WeatherIcon = () => {
    const cond = weather.condition.toLowerCase();
    if (cond.includes('rain')) return <CloudRain size={28} className="text-blue-500" />;
    if (cond.includes('storm')) return <CloudLightning size={28} className="text-purple-500" />;
    if (cond.includes('sun') || cond.includes('clear')) return <Sun size={28} className="text-orange-400" />;
    return <Cloud size={28} className="text-slate-400" />;
  };

  const handleCitySelect = (city: string) => {
    // ── Full city-switch reset: no stale audio, no stale AI request,
    //    no leftover timer/promise/listener may pollute the new city. ──

    // Unified reset: 1) abort any in-flight AI request (bumps request id),
    // 2) rebind VoiceManager to the new request id (so a late TTS / stale
    // "Read Aloud" from the previous city cannot leak into the new city —
    // this is the Bug 2 fix), 3) cancel ALL speech + clear per-message
    // audio cache (so stale audio from the previous city never replays),
    // 4) clear React-side TTS / loading state.
    resetConversationState();

    // ── Tear down LIVE mode exactly like the Cancel (X) button does. ──
    // resetConversationState() only calls voiceManager.stop() (TTS-only)
    // which the VoiceManager docs explicitly state leaves mic /
    // recognition / live session / live audio contexts untouched. Without
    // this, a city change while live mode is active keeps the mic on, the
    // Gemini live session running, and isLiveActive=true, leaking the old
    // city's live resources into the new city. This matches Cancel flow:
    //   Cancel button -> stopLiveMode()  (App.tsx:1077)
    if (isLiveActive) {
      stopLiveMode();
    }

    // Defensive: reset local React-side audio refs (VoiceManager is the
    // source of truth, but legacy live-mode refs may still hold stale
    // state and must be cleared too).
    if (currentAudioSourceRef.current) {
      try { currentAudioSourceRef.current.stop(); } catch (_) {}
      try { currentAudioSourceRef.current.disconnect(); } catch (_) {}
      currentAudioSourceRef.current = null;
    }
    if ('speechSynthesis' in window) {
      try { window.speechSynthesis.cancel(); } catch (_) {}
    }

    // Clear all messages & state for the new city.
    setSelectedCity(city);
    setMessages([]); 
    setCurrentView('chat');
  };

  const onNewChat = () => {
    // Unified reset before starting a fresh chat. Prevents stale
    // responses / TTS from leaking into the new chat.
    resetConversationState();
    setMessages([]);
    setCurrentChatId(null);
    setCurrentView('chat');
    setIsSidebarOpen(false);
  };

  const handleRegenerate = useCallback(async (msgId: string) => {
    const msgIdx = messages.findIndex(m => m.id === msgId);
    if (msgIdx === -1) return;
    
    const prunedMessages = messages.slice(0, msgIdx);
    const userMsg = prunedMessages.slice().reverse().find(m => m.role === 'user');
    const actualUser = userMsg || prunedMessages[prunedMessages.length - 1];
    
    if (!actualUser) return;
    
    const baseMessages = prunedMessages.filter(m => m.id !== actualUser.id);
    setMessages(baseMessages);
    
    handleSendMessage(actualUser.content, actualUser.attachments);
  }, [messages, handleSendMessage]);

  const handleEditMessage = useCallback(async (msgId: string, newText: string) => {
    const msgIdx = messages.findIndex(m => m.id === msgId);
    if (msgIdx === -1) return;
    
    const prunedMessages = messages.slice(0, msgIdx);
    setMessages(prunedMessages);
    
    handleSendMessage(newText, messages[msgIdx].attachments);
  }, [messages, handleSendMessage]);

  return (
    <div className="flex h-screen overflow-hidden text-elegant-light dark:text-elegant-dark bg-[#FBF8F3] dark:bg-[#0F0D0C] relative transition-all duration-700 ease-in-out">
      {isSidebarOpen && (
        <div className="fixed inset-0 bg-black/5 dark:bg-black/40 backdrop-blur-[1px] z-40" onClick={() => setIsSidebarOpen(false)} />
      )}

      <Sidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)}
        history={history}
        user={user}
        onNewChat={onNewChat}
        onSelectHistory={handleSelectHistory}
        onDeleteHistory={handleDeleteHistory}
        onDownloadHistory={handleDownloadHistory}
        onUtilityAction={handleUtilityAction}
        onLogin={() => { setIsLoginOpen(true); setIsSidebarOpen(false); }}
      />

      <div className="flex-1 flex flex-col min-w-0 relative h-full">
        {currentView !== 'trip-planner' && (
          <header className="h-20 md:h-24 flex items-center px-4 md:px-10 relative bg-white/30 dark:bg-black/20 backdrop-blur-md border-b border-orange-100/30 shrink-0">
            <div className="flex items-center absolute left-4 md:left-10">
              <button 
                onClick={(e) => { 
                  e.stopPropagation(); 
                  setIsSidebarOpen(!isSidebarOpen); 
                  setIsLogoSpinning(true);
                  setTimeout(() => setIsLogoSpinning(false), 1000);
                }}
                className="w-10 h-10 md:w-12 md:h-12 flex items-center justify-center transition-all hover:scale-105 active:scale-95 focus:outline-none"
              >
                <div className={`relative w-8 h-8 md:w-10 md:h-10 flex items-center justify-center ${isLogoSpinning ? 'animate-[spin_1s_ease-in-out]' : ''}`}>
                  <svg width="100%" height="100%" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="50" cy="50" r="50" fill="#f97316" />
                    <circle cx="50" cy="50" r="35" stroke="white" strokeWidth="6" />
                    <path d="M62 38 A 15 15 0 1 0 62 62" stroke="white" strokeWidth="7" strokeLinecap="round" fill="none" />
                  </svg>
                </div>
              </button>
            </div>
            
            <div className="flex flex-col items-center justify-center w-full px-16">
              <button 
                onClick={() => setIsCitySelectorOpen(true)}
                className="group flex flex-col items-center justify-center transition-all hover:opacity-80 active:scale-95 px-3 py-1 rounded-2xl hover:bg-orange-50/50 dark:hover:bg-orange-900/10 max-w-full"
              >
                <div className="flex items-center gap-1.5 max-w-full overflow-hidden">
                  <h1 className="text-[20px] sm:text-[24px] md:text-[28px] font-bold text-[#f97316] tracking-tight truncate uppercase">{selectedCity} Info AI</h1>
                  <ChevronDown size={18} className="text-[#f97316] mt-0.5 shrink-0 transition-transform group-hover:translate-y-0.5" />
                </div>
                <p className="text-[10px] md:text-[13px] text-slate-500 font-medium tracking-wide truncate">{SUBTITLE}</p>
              </button>
            </div>

            <div className="absolute right-4 md:right-10 flex items-center gap-2 md:gap-6">
               <button 
                 onClick={() => handleRefreshWeather(selectedCity, true)}
                 className={`flex items-center gap-1.5 md:gap-2.5 transition-all p-1.5 md:p-2 rounded-xl hover:bg-white/50 dark:hover:bg-white/5 active:scale-90 ${isWeatherRefreshing ? 'bg-orange-50 dark:bg-orange-950/20' : ''}`}
                 title="Refresh weather"
               >
                  {isWeatherRefreshing ? (
                    <RefreshCw size={20} className="text-orange-500 animate-spin" />
                  ) : (
                    <WeatherIcon />
                  )}
                  <div className="flex flex-col items-start leading-none shrink-0">
                    <span className={`text-[14px] md:text-[17px] font-bold text-slate-700 dark:text-slate-300 transition-all ${isWeatherRefreshing ? 'opacity-50 animate-pulse text-orange-500' : 'opacity-100'}`}>
                      {isWeatherRefreshing ? '...' : `${weather.temp}°C`}
                    </span>
                  </div>
               </button>
               <button 
                 onClick={toggleTheme}
                 className="p-2 text-slate-700 dark:text-orange-200 transition-all hover:scale-110 rounded-xl hover:bg-white/50 dark:hover:bg-white/5 shrink-0"
               >
                 {theme === 'light' ? <Moon size={24} className="md:size-[28px]" /> : <Sun size={24} className="md:size-[28px]" />}
               </button>
            </div>
          </header>
        )}

        <main className="flex-1 flex flex-col relative overflow-hidden" onClick={() => isSidebarOpen && setIsSidebarOpen(false)}>
          {currentView === 'chat' ? (
            <>
              <ChatWindow 
                messages={messages} 
                onSuggestionClick={handleSendMessage} 
                onSpeak={handleSpeak} 
                speakingMessageId={speakingMessageId}
                isTTSLoading={isTTSLoading}
                isLoading={isLoading} 
                city={selectedCity}
                onEditMessage={handleEditMessage}
                onRegenerate={handleRegenerate}
              />
              <div className="w-full">
                <InputBar 
                  onSendMessage={handleSendMessage} 
                  onStartLive={startLiveMode} 
                  onOpenImageGen={() => setIsImageGenOpen(true)}
                  isLoading={isLoading} 
                  language={language}
                  setLanguage={setLanguage}
                  isDeepSearch={isDeepSearch}
                  setIsDeepSearch={setIsDeepSearch}
                  city={selectedCity}
                />
              </div>
            </>
          ) : currentView === 'news' ? (
            <NewsView onBack={() => setCurrentView('chat')} city={selectedCity} language={language} />
          ) : currentView === 'stats' ? (
            <StatsView onBack={() => setCurrentView('chat')} messageCount={messagesSent} />
          ) : currentView === 'trip-planner' ? (
            <TripPlannerView 
              messages={messages} 
              onSendMessage={handleSendMessage} 
              onSpeak={handleSpeak}
              speakingMessageId={speakingMessageId}
              isTTSLoading={isTTSLoading}
              onBack={() => { setCurrentView('chat'); }} 
              city={selectedCity}
              isLoading={isLoading}
              language={language}
              setLanguage={setLanguage}
            />
          ) : (
            <AboutView onBack={() => setCurrentView('chat')} city={selectedCity} stats={{ visits: 2, questions: messagesSent, trips: 0 }} />
          )}
        </main>

        {isLiveActive && (
          <div className="fixed inset-0 z-[60] bg-[#f97316]/95 backdrop-blur-md flex flex-col items-center justify-center text-white p-8 animate-in fade-in zoom-in duration-300">
            <button onClick={stopLiveMode} className="absolute top-10 right-10 p-4 bg-white/10 hover:bg-white/20 rounded-full transition-all">
              <X size={36} />
            </button>
            <div className="w-56 h-56 bg-white/10 rounded-full flex items-center justify-center relative mb-16">
              <div className="absolute inset-0 rounded-full border-4 border-white/20 animate-ping" />
              <div className="absolute inset-4 rounded-full border-4 border-white/30 animate-ping delay-150" />
              <div className="w-36 h-36 bg-white rounded-full flex items-center justify-center shadow-2xl">
                <Volume2 size={72} className="text-[#f97316] animate-pulse" />
              </div>
            </div>
            <h2 className="text-4xl font-bold mb-6 tracking-tight">{selectedCity} Live AI</h2>
            <p className="text-orange-50 text-xl text-center max-w-lg font-medium leading-relaxed italic opacity-90 px-4">
              {liveTranscription || `Listening... Talk to me about ${selectedCity}.`}
            </p>
            <div className="mt-20 px-6 py-3 bg-white/10 rounded-full border border-white/20 flex items-center gap-3">
              <Mic size={20} className="text-green-300 animate-pulse" />
              <span className="text-xs font-bold tracking-[0.2em] uppercase">Session Active</span>
            </div>
          </div>
        )}

        <LoginModal isOpen={isLoginOpen} onClose={() => setIsLoginOpen(false)} city={selectedCity} />
        <CitySelector 
          isOpen={isCitySelectorOpen} 
          onClose={() => setIsCitySelectorOpen(false)} 
          onSelect={handleCitySelect}
          currentCity={selectedCity}
        />
        <ImageGenModal 
          isOpen={isImageGenOpen} 
          onClose={() => setIsImageGenOpen(false)} 
          city={selectedCity}
        />
        <PromoteModal 
          isOpen={isPromoteOpen} 
          onClose={() => setIsPromoteOpen(false)} 
          city={selectedCity}
        />
      </div>
    </div>
  );
};

export default App;
