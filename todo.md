# AI Internet Search & Grounding - Implementation Plan

## Analysis Summary
The project already has the foundation for web search (Google Search grounding via `tools: [{ googleSearch: {} }]`, Wikipedia search, `shouldSearchWeb` decision engine), but several critical gaps exist:

1. **`getGeminiResponse` (non-streaming)** doesn't use pre-search context while streaming version does
2. **`getGeminiFallbackResponse`** returns only static keyword-matched text — no internet search attempted when API fails
3. **`NewsView`** relies on hardcoded mock data for Madurai, no real web news search
4. **Weather** uses Gemini-generated mock data, not real weather API
5. **`searchService.ts`** only has Wikipedia as a search source — needs more sources
6. **Hardcoded fallback API key** in `vite.config.ts` is a security concern

## Task Plan

- [x] Read and analyze all project files
- [ ] **Fix 1: `searchService.ts`** — Add DuckDuckGo web search as additional source alongside Wikipedia
- [ ] **Fix 2: `geminiService.ts`** — Make `getGeminiResponse` (non-streaming) also use pre-search; enhance fallback to attempt web search before returning static data
- [ ] **Fix 3: `services/weatherService.ts`** — Create real weather service using wttr.in (free, no API key)
- [ ] **Fix 4: `App.tsx`** — Connect real weather service
- [ ] **Fix 5: `NewsView.tsx`** — Rewrite to dynamically fetch news via Gemini search grounding
- [ ] **Fix 6: `constants.tsx`** — Strengthen search-first system instruction language
- [ ] **Fix 7: `vite.config.ts`** — Remove hardcoded API key, use env-only
- [ ] **Fix 8: `services/geminiService.ts`** — Remove hardcoded API key fallback in App.tsx/TTS calls
- [ ] Verify all changes compile with `tsc --noEmit`
- [ ] Test the application
