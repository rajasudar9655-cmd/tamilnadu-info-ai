/**
 * Intelligent Search Service with multi-source support.
 *
 * Provides:
 * - Official-source filtering for Tamil Nadu government and institutional sites
 * - Gemini Google Search grounding as primary retrieval, with web-search fallback
 * - Wikipedia as supporting material only for non-volatile queries
 * - Search decision engine to avoid unnecessary queries
 * - Evidence ranking and conflict detection for current/factual questions
 * - Citation-aware prompt formatting and cache reuse
 */

import { withRetry } from './retryUtils.ts';

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
  source: 'wikipedia' | 'google' | 'duckduckgo' | 'cache' | 'knowledge';
  relevanceScore?: number;
  domain?: string;
  publishedAt?: string;
  retrievedAt?: string;
  sourceType?: 'official' | 'government' | 'institution' | 'news' | 'wikipedia' | 'general';
  isOfficial?: boolean;
  evidenceDirectness?: 'grounded-support' | 'search-snippet';
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
  usedSearch: boolean;
  source: string;
}

export interface SearchDecision {
  shouldSearch: boolean;
  reason: string;
  suggestedSearchTerms?: string[];
}

interface CacheEntry {
  results: SearchResult[];
  timestamp: number;
}

interface WikiSearchResponse {
  query?: {
    search?: Array<{ title: string; pageid: number; snippet: string }>;
  };
}

interface WikiPageResponse {
  query?: {
    pages?: Record<string, { title: string; fullurl?: string; extract?: string }>;
  };
}

const searchCache = new Map<string, CacheEntry>();
const CACHE_TTL = 5 * 60 * 1000;
const VOLATILE_QUERY_PATTERN = /\b(current|today|latest|recent|now|tonight|tomorrow|weather|news|price|cost|ticket|fee|availability|available|booking|timing|hours|opening|closing|schedule|event|festival|collector|official|scheme|law|regulation|traffic|train|bus|route|statistics)\b/i;
const OFFICIAL_DOMAIN_PATTERNS = [
  /(^|\.)tn\.gov\.in$/i,
  /(^|\.)nic\.in$/i,
  /(^|\.)gov\.in$/i,
  /(^|\.)indianrailways\.gov\.in$/i,
  /(^|\.)tamilnadutourism\.tn\.gov\.in$/i,
  /(^|\.)irctc\.co\.in$/i,
  /(^|\.)tnstc\.in$/i,
];

function normalizeWhitespace(value: string): string {
  return (value || '').replace(/\s+/g, ' ').trim();
}

function stripHtml(html: string): string {
  return (html || '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, ' & ')
    .replace(/&[^;]+;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function decodeHtmlEntities(value: string): string {
  const entityMap: Record<string, string> = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'",
  };
  return value.replace(/&(?:amp|lt|gt|quot|#39);/g, (match) => entityMap[match] || match);
}

function getDomainFromUrl(url: string): string {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return '';
    return parsed.hostname.replace(/^www\./i, '').toLowerCase();
  } catch {
    return '';
  }
}

function normalizeDomain(domain: string): string {
  return normalizeWhitespace(domain)
    .replace(/^https?:\/\//i, '')
    .replace(/\/.*$/, '')
    .replace(/^www\./i, '')
    .toLowerCase();
}

function isOfficialDomain(domain: string): boolean {
  const normalized = normalizeDomain(domain);
  return OFFICIAL_DOMAIN_PATTERNS.some((pattern) => pattern.test(normalized));
}

export function isOfficialSource(url?: string): boolean {
  if (!url) return false;
  const domain = getDomainFromUrl(url);
  if (!domain) return false;
  return OFFICIAL_DOMAIN_PATTERNS.some((pattern) => pattern.test(domain));
}

function getSourceType(url?: string, domainOverride?: string): SearchResult['sourceType'] {
  if (!url) return 'general';
  const domain = normalizeDomain(domainOverride || getDomainFromUrl(url));
  if (!domain) return 'general';
  if (/\.tn\.gov\.in$/.test(domain) || /\.nic\.in$/.test(domain) || domain.endsWith('.gov.in') || domain === 'gov.in') {
    return 'official';
  }
  if (domain.endsWith('.gov') || domain.endsWith('.gov.uk') || domain.endsWith('.edu') || domain.endsWith('.ac.in')) {
    return 'institution';
  }
  if (domain.includes('wikipedia.org')) return 'wikipedia';
  if (/^(thehindu|indianexpress|newindianexpress|dailythanthi|polimer|dinamalar|ndtv|republicworld|deccanchronicle|timesofindia)\.com$/i.test(domain)) {
    return 'news';
  }
  return 'general';
}

function getAuthorityRank(result: Partial<SearchResult>): number {
  const domain = normalizeDomain(result.domain || getDomainFromUrl(result.url || ''));
  if (/(^|\.)tn\.gov\.in$/i.test(domain)) return 100;
  if (/(^|\.)nic\.in$/i.test(domain)) return 90;
  if (isOfficialDomain(domain) || isOfficialSource(result.url)) return 85;
  if (getSourceType(result.url, domain) === 'official') return 80;
  if (getSourceType(result.url, domain) === 'institution') return 65;
  if (getSourceType(result.url, domain) === 'news') return 50;
  if (getSourceType(result.url, domain) === 'wikipedia') return 20;
  return 30;
}

function scoreSearchResult(result: Partial<SearchResult>, query: string): number {
  let score = getAuthorityRank(result);
  const chunk = `${result.title ?? ''} ${result.snippet ?? ''}`.toLowerCase();
  const queryLower = query.toLowerCase();

  const queryTokens = new Set(
    queryLower
      .replace(/[^a-z0-9\s]/gi, ' ')
      .split(/\s+/)
      .filter((token) => token.length > 2)
  );

  for (const token of queryTokens) {
    if (chunk.includes(token)) score += 5;
  }

  return Math.min(100, score);
}

function normalizeSearchResult(result: Partial<SearchResult>, query: string): SearchResult {
  const title = normalizeWhitespace(result.title || getDomainFromUrl(result.url || '') || 'Search result');
  const url = result.url ? result.url.trim() : '';
  const snippet = normalizeWhitespace(stripHtml(decodeHtmlEntities(result.snippet || ''))).slice(0, 500);
  const domain = normalizeDomain(result.domain || getDomainFromUrl(url));
  const sourceType = getSourceType(url, domain);
  const isOfficial = isOfficialDomain(domain) || isOfficialSource(url);
  const score = scoreSearchResult({ ...result, title, url, snippet }, query);

  return {
    title,
    url,
    snippet,
    source: result.source || 'google',
    relevanceScore: score,
    domain,
    sourceType,
    isOfficial,
    publishedAt: result.publishedAt,
    retrievedAt: result.retrievedAt || new Date().toISOString(),
    evidenceDirectness: result.evidenceDirectness,
  };
}

function getCachedResults(query: string): SearchResult[] | null {
  const key = query.toLowerCase().trim();
  const entry = searchCache.get(key);
  const ttl = /(weather|news|availability|available|traffic|today|now|live|booking|train|bus|current|latest)/i.test(key)
    ? 30 * 1000
    : /(price|ticket|timing|hour|schedule|event|festival|collector|scheme|official)/i.test(key)
      ? 2 * 60 * 1000
      : CACHE_TTL;
  if (entry && Date.now() - entry.timestamp < ttl) {
    return entry.results;
  }
  if (entry) searchCache.delete(key);
  return null;
}

function setCachedResults(query: string, results: SearchResult[]): void {
  const key = query.toLowerCase().trim();
  searchCache.set(key, { results, timestamp: Date.now() });
  if (searchCache.size > 200) {
    const oldest = [...searchCache.entries()].sort(([, a], [, b]) => a.timestamp - b.timestamp)[0];
    if (oldest) searchCache.delete(oldest[0]);
  }
}

export function detectLanguage(text: string): 'en' | 'ta' | 'mixed' {
  const cleaned = (text || '').trim();
  if (!cleaned) return 'en';
  const hasTamil = /[\u0B80-\u0BFF]/.test(cleaned);
  const hasLatin = /[A-Za-z]/.test(cleaned);
  if (hasTamil && hasLatin) return 'mixed';
  if (hasTamil) return 'ta';
  return 'en';
}

export function buildSearchQueries(
  query: string,
  locationContext?: { city?: string; lat?: number; lng?: number }
): string[] {
  const sanitized = normalizeWhitespace(query || '');
  if (!sanitized) return [];

  const city = normalizeWhitespace(locationContext?.city || '');
  const gpsContext = locationContext?.lat !== undefined && locationContext?.lng !== undefined
    ? ` near ${locationContext.lat},${locationContext.lng}`
    : '';
  const variants = new Set<string>();
  const base = city ? `${sanitized} ${city} Tamil Nadu${gpsContext}` : `${sanitized} Tamil Nadu${gpsContext}`;
  const official = city ? `${sanitized} official ${city} Tamil Nadu${gpsContext}` : `${sanitized} official Tamil Nadu${gpsContext}`;
  variants.add(base);
  variants.add(official);
  variants.add(sanitized);

  if (/(today|current|today|latest|weather|ticket|price|timing|opening|closing|festival|news|collector|bus|train|schedule|hotel|restaurant)/i.test(sanitized)) {
    variants.add(`${sanitized} official ${city || 'Tamil Nadu'}`);
  }

  if (city && /(nearby|around|best|temple|hotel|restaurant|weather)/i.test(sanitized)) {
    variants.add(`${sanitized} in ${city} Tamil Nadu`);
  }

  return [...variants].filter(Boolean).slice(0, 4);
}

export function rankSearchResults(results: SearchResult[], query = ''): SearchResult[] {
  return [...results]
    .filter((result) => result && result.url)
    .sort((a, b) => {
      const authorityDelta = getAuthorityRank(b) - getAuthorityRank(a);
      if (authorityDelta !== 0) return authorityDelta;
      const relevanceDelta = (b.relevanceScore ?? 0) - (a.relevanceScore ?? 0);
      if (relevanceDelta !== 0) return relevanceDelta;
      return (a.title || '').localeCompare(b.title || '');
    })
    .map((result) => normalizeSearchResult(result, query));
}

export function detectSourceConflict(results: SearchResult[], query = ''): { hasConflict: boolean; conflictingSources: string[] } {
  const queryLower = query.toLowerCase();
  const type = /(price|cost|fee|ticket)/i.test(queryLower)
    ? 'price'
    : /(time|timing|hour|open|close|schedule)/i.test(queryLower)
      ? 'time'
      : /(date|festival|event)/i.test(queryLower)
        ? 'date'
        : null;
  if (!type) return { hasConflict: false, conflictingSources: [] };

  const valuePatterns: Record<'price' | 'time' | 'date', RegExp> = {
    price: /(?:₹|rs\.?|inr)\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:₹|rs\.?|inr)\b/gi,
    time: /\b(?:[01]?\d|2[0-3])(?::[0-5]\d)\s?(?:a\.?m\.?|p\.?m\.?)?\b/gi,
    date: /\b(?:\d{1,2}(?:st|nd|rd|th)?[ -](?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[ ,/-]+\d{2,4})?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b/gi,
  };
  const valuesByCategory = new Map<string, Map<string, Set<string>>>();
  for (const result of results) {
    const text = result.snippet || '';
    const matches = [...text.matchAll(valuePatterns[type])];
    if (type === 'price' && !matches.length && /(free entry|no (?:general )?entry fee|no admission fee)/i.test(text)) {
      const valuesBySource = valuesByCategory.get('general entry') ?? new Map<string, Set<string>>();
      const values = valuesBySource.get(result.url) ?? new Set<string>();
      values.add('0');
      valuesBySource.set(result.url, values);
      valuesByCategory.set('general entry', valuesBySource);
    }

    for (const match of matches) {
      const value = normalizeWhitespace(match[0]).toLowerCase();
      const before = text.slice(Math.max(0, (match.index ?? 0) - 80), match.index ?? 0).toLowerCase();
      const after = text.slice((match.index ?? 0) + match[0].length, (match.index ?? 0) + match[0].length + 30).toLowerCase();
      const context = `${before} ${after}`;
      const category = type === 'price'
        ? /museum/.test(context) ? 'museum'
          : /camera|photograph/.test(context) ? 'photography'
            : /vip/.test(context) ? 'vip darshan'
              : /special.{0,15}darshan|darshan.{0,15}special/.test(context) ? 'special darshan'
                : /general.{0,20}(entry|admission)|(?:entry|admission).{0,20}general/.test(context) ? 'general entry'
                  : /ticket/.test(context) ? 'ticket'
                    : 'price'
        : type === 'time'
          ? /p\.?m\.?/.test(value) || /evening/.test(context) ? 'evening'
            : /a\.?m\.?/.test(value) || /morning/.test(context) ? 'morning'
              : 'time'
          : 'date';
      const valuesBySource = valuesByCategory.get(category) ?? new Map<string, Set<string>>();
      const values = valuesBySource.get(result.url) ?? new Set<string>();
      values.add(value);
      valuesBySource.set(result.url, values);
      valuesByCategory.set(category, valuesBySource);
    }
  }

  const conflictingSources = new Set<string>();
  for (const valuesBySource of valuesByCategory.values()) {
    const sources = [...valuesBySource.entries()];
    let hasSourceDisagreement = false;
    for (let index = 0; index < sources.length; index += 1) {
      const [sourceUrl, sourceValues] = sources[index];
      for (let compareIndex = index + 1; compareIndex < sources.length; compareIndex += 1) {
        const [otherUrl, otherValues] = sources[compareIndex];
        if (![...sourceValues].some((value) => otherValues.has(value))) {
          hasSourceDisagreement = true;
          conflictingSources.add(sourceUrl);
          conflictingSources.add(otherUrl);
        }
      }
    }
    if (!hasSourceDisagreement) continue;
  }
  if (!conflictingSources.size) return { hasConflict: false, conflictingSources: [] };

  return {
    hasConflict: true,
    conflictingSources: [...conflictingSources].filter(Boolean),
  };
}

function isRelevantEvidence(result: SearchResult, query: string): boolean {
  if (!result.url || !result.snippet.trim()) return false;
  const stopWords = new Set(['the', 'and', 'for', 'with', 'from', 'what', 'when', 'where', 'who', 'how', 'today', 'current', 'latest', 'official', 'please', 'tell', 'about', 'price', 'cost', 'ticket', 'timing', 'opening', 'hours']);
  const terms = query.toLowerCase().match(/[a-z0-9\u0B80-\u0BFF]{3,}/g) || [];
  const content = `${result.title} ${result.snippet}`.toLowerCase();
  const meaningfulTerms = [...new Set(terms.filter((term) => !stopWords.has(term)))];
  if (!meaningfulTerms.length) return true;
  const requiredMatches = meaningfulTerms.length > 1 ? 2 : 1;
  const matchedTerms = meaningfulTerms.filter((term) => content.includes(term));
  if (matchedTerms.length < requiredMatches) return false;

  if (/(price|cost|fee|ticket)/i.test(query) && !/(?:₹|rs\.?|inr)\s?\d|\bfree entry\b|\bno (?:general )?entry fee\b|\bno admission fee\b/i.test(result.snippet)) {
    return false;
  }
  if (/(timing|hours|opening|closing|schedule)/i.test(query) && !/\b(?:[01]?\d|2[0-3])(?::[0-5]\d)\s?(?:a\.?m\.?|p\.?m\.?)?\b/i.test(result.snippet)) {
    return false;
  }
  if (/(eligibility|eligible|qualify)/i.test(query) && !/(eligib|qualif|criteria|must be|requirements?)/i.test(result.snippet)) {
    return false;
  }
  return true;
}

async function searchGeminiGrounding(query: string, searchQueries: string[]): Promise<SearchResult[]> {
  const apiBaseUrl = (import.meta as any).env?.VITE_API_BASE_URL
    || (typeof process !== 'undefined' ? process.env.VITE_API_BASE_URL : '')
    || '';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(`${apiBaseUrl}/api/search/grounded`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, searchQueries }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Gemini grounded search failed: HTTP ${response.status}`);
    }
    const data = await response.json();
    if (!Array.isArray(data?.results)) return [];
    return data.results
      .map((result: Partial<SearchResult>) => normalizeSearchResult({
        ...result,
        source: 'google',
        evidenceDirectness: 'grounded-support',
        retrievedAt: data.retrievedAt,
      }, query))
      .filter((result: SearchResult) => isRelevantEvidence(result, query));
  } finally {
    clearTimeout(timer);
  }
}

async function searchWikipedia(query: string): Promise<SearchResult[]> {
  const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&origin=*&srlimit=5`;

  const searchData = await withRetry<WikiSearchResponse>(
    async () => {
      const response = await fetch(searchUrl);
      if (!response.ok) throw new Error(`Wikipedia search error: ${response.status}`);
      return response.json();
    },
    { maxRetries: 2, baseDelayMs: 500 }
  );

  if (!searchData?.query?.search?.length) return [];

  const pageIds = searchData.query.search.map((s) => s.pageid).join('|');
  const contentUrl = `https://en.wikipedia.org/w/api.php?action=query&pageids=${pageIds}&prop=extracts|info&exintro&exlimit=5&inprop=url&format=json&origin=*`;

  const contentData = await withRetry<WikiPageResponse>(
    async () => {
      const response = await fetch(contentUrl);
      if (!response.ok) throw new Error(`Wikipedia content error: ${response.status}`);
      return response.json();
    },
    { maxRetries: 2, baseDelayMs: 500 }
  );

  if (!contentData?.query?.pages) return [];

  const results: SearchResult[] = [];
  for (const page of Object.values(contentData.query.pages)) {
    if (!page) continue;
    const url = page.fullurl || `https://en.wikipedia.org/wiki/${encodeURIComponent(page.title.replace(/ /g, '_'))}`;
    const match = searchData.query.search.find((s) => s.title === page.title);
    results.push(
      normalizeSearchResult(
        {
          title: page.title,
          url,
          snippet: match?.snippet ? stripHtml(match.snippet) : page.extract?.substring(0, 300) || '',
          source: 'wikipedia',
        },
        query
      )
    );
  }

  return results;
}

async function searchDuckDuckGo(query: string): Promise<SearchResult[]> {
  const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;

  const payload = await withRetry(
    async () => {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
          Accept: 'text/html,application/xhtml+xml',
        },
      });
      if (!response.ok) throw new Error(`DuckDuckGo error: ${response.status}`);
      return response.text();
    },
    { maxRetries: 1, baseDelayMs: 500 }
  );

  if (!payload) return [];

  const results: SearchResult[] = [];
  const seen = new Set<string>();
  const linkPattern = /<a rel="nofollow" class="result-link" href="([^"]+)"[^>]*>(.*?)<\/a>/gi;
  const snippetPattern = /<a rel="nofollow" class="result-link" href="[^"]+"[^>]*>.*?<\/a>\s*<a class="result-snippet"[^>]*>(.*?)<\/a>/gi;

  const snippets = [...payload.matchAll(snippetPattern)].map((match) => stripHtml(decodeHtmlEntities(match[1] || '')));
  const links = [...payload.matchAll(linkPattern)].map((match) => ({
    url: decodeHtmlEntities(match[1] || '').replace(/^\//, 'https://'),
    title: stripHtml(decodeHtmlEntities(match[2] || '')),
  }));

  for (let i = 0; i < links.length; i += 1) {
    const link = links[i];
    const normalizedUrl = (() => {
      try {
        const parsed = new URL(link.url);
        const destination = parsed.searchParams.get('uddg');
        return new URL(destination || link.url, 'https://duckduckgo.com').toString();
      } catch {
        return link.url;
      }
    })();
    if (!normalizedUrl || seen.has(normalizedUrl)) continue;
    seen.add(normalizedUrl);
    results.push(
      normalizeSearchResult(
        {
          title: link.title || getDomainFromUrl(normalizedUrl),
          url: normalizedUrl,
          snippet: snippets[i] || '',
          source: 'duckduckgo',
        },
        query
      )
    );
  }

  return results.slice(0, 5);
}

export function shouldSearchWeb(
  query: string,
  isDeepSearchEnabled: boolean
): SearchDecision {
  const trimmed = normalizeWhitespace(query).toLowerCase();
  if (!trimmed) {
    return { shouldSearch: false, reason: 'Empty query' };
  }

  const alwaysSearchPatterns = [
    /\b(current|today|latest|recent|now|tonight|tomorrow|yesterday|this week|this month)\b/i,
    /\b(weather|temperature|rain|forecast|humidity)\b/i,
    /\b(news|headlines|announcement|update)\b/i,
    /\b(price|cost|rate|fee|ticket|booking)\b/i,
    /\b(open(ing)?\s+hours?|timings?|schedule|when does|closing(?:\s+hours?)?|visiting hours?)\b/i,
    /\b(phone|contact|number|email|address|location|direction)\b/i,
    /\b(menu|dish|special|signature|cuisine)\b/i,
    /\b(rate|rating|review|star|popular|best|top|recommend)\b/i,
    /\b(event|festival|celebration|fair|conference)\b/i,
    /\b(how to reach|how do I|map|distance|route)\b/i,
    /\b(near me|nearby|closest|nearest|around)\b/i,
    /\b(bus|train|flight|travel|transport|airport|station)\b/i,
    /\b(hotel|resort|lodge|accommodation|stay|room|guest house)\b/i,
    /\b(distance|km|kilometer|minutes away)\b/i,
    /\b(availability|available|government scheme|scheme eligibility|government service|district official|collector|current official|law|laws|regulation|regulations|statistics|traffic|booking status|holiday|holidays)\b/i,
    /(இன்று|தற்போதைய|இப்போது|சமீபத்திய|செய்தி|வானிலை|விலை|நேரம்|திறப்பு|மூடல்|விடுமுறை|திருவிழா|அறிவிப்பு|அருகில்|போக்குவரத்து|பேருந்து|ரயில்|முன்பதிவு|திட்டம்|அரசு சேவை|தேர்தல்)/i,
  ];

  if (alwaysSearchPatterns.some((pattern) => pattern.test(trimmed))) {
    return { shouldSearch: true, reason: 'Query requires live or time-sensitive information' };
  }

  if (!isDeepSearchEnabled) {
    return { shouldSearch: false, reason: 'Deep search disabled by user' };
  }

  const knowledgePatterns = [
    /\b(meaning|define|definition|explain|concept)\b/i,
    /\b(translate|grammar|vocabulary|language)\b/i,
    /\b(history of|origin of|background of)\b/i,
    /\b(math|equation|formula|calculate|solve)\b/i,
    /\b(code|program|function|algorithm|syntax)\b/i,
    /\b(write|draft|compose|create|generate|essay|story|poem)\b/i,
    /\b(வரலாறு|மரபு|விளக்கம்|அர்த்தம்|பொருள்)\b/i,
  ];

  if (knowledgePatterns.some((pattern) => pattern.test(trimmed))) {
    return { shouldSearch: false, reason: 'Query is suited for model knowledge or general explanation' };
  }

  if (/^(what|where|when|who|how|why|which|tell|share|give|list|show|find|search)/i.test(trimmed)) {
    return { shouldSearch: true, reason: 'Question likely benefits from live web grounding' };
  }

  const questionWordsCount = (trimmed.match(/\b(what|where|when|who|how|why|which)\b/g) || []).length;
  const creativeWordsCount = (trimmed.match(/\b(write|create|compose|draft|imagine|suggest)\b/g) || []).length;

  if (questionWordsCount > creativeWordsCount) {
    return { shouldSearch: true, reason: 'Predominantly fact-seeking query' };
  }

  return { shouldSearch: false, reason: 'Query appears creative or general' };
}

export function extractGroundingSources(groundingMetadata: any): { title: string; url: string }[] {
  const sources: { title: string; url: string }[] = [];
  if (!groundingMetadata?.groundingChunks) return sources;

  const seen = new Set<string>();
  for (const chunk of groundingMetadata.groundingChunks) {
    if (chunk.web?.uri && chunk.web?.title) {
      const url = chunk.web.uri;
      if (!seen.has(url)) {
        seen.add(url);
        sources.push({ title: chunk.web.title, url });
      }
    }
  }
  return sources;
}

export function extractSearchSuggestions(groundingMetadata: any): string[] {
  return (
    groundingMetadata?.searchEntryPoint?.renderedContent
      ?.match(/href="([^"]+)"/g)
      ?.map((m: string) => m.replace(/href="/, '').replace(/"/g, ''))
      ?.filter(Boolean) || []
  );
}

export async function performIntelligentSearch(
  query: string,
  locationContext?: { city?: string; lat?: number; lng?: number }
): Promise<SearchResponse> {
  const trimmedQuery = normalizeWhitespace(query);
  if (!trimmedQuery) {
    return { results: [], query, usedSearch: false, source: 'none' };
  }

  const searchQueries = buildSearchQueries(trimmedQuery, locationContext);
  const primaryQuery = searchQueries[0] || trimmedQuery;
  const cacheKey = `${trimmedQuery}|${normalizeWhitespace(locationContext?.city || '')}|${locationContext?.lat ?? ''}|${locationContext?.lng ?? ''}`;
  const cached = getCachedResults(cacheKey);
  if (cached) {
    return { results: cached, query: trimmedQuery, usedSearch: true, source: 'cache' };
  }

  try {
    const groundedResults = await searchGeminiGrounding(trimmedQuery, searchQueries);
    const rankedGrounding = rankSearchResults(groundedResults, `${trimmedQuery} ${locationContext?.city || ''}`);
    if (rankedGrounding.length > 0) {
      setCachedResults(cacheKey, rankedGrounding);
      return {
        results: rankedGrounding,
        query: primaryQuery,
        usedSearch: true,
        source: 'gemini-google-search',
      };
    }
  } catch (error) {
    console.warn('[SearchService] Gemini Google Search grounding unavailable; trying fallback retrieval:', error);
  }

  const [wikiOutcome, ddgOutcome] = await Promise.allSettled([
    VOLATILE_QUERY_PATTERN.test(trimmedQuery) ? Promise.resolve([]) : searchWikipedia(primaryQuery),
    searchDuckDuckGo(primaryQuery),
  ]);
  const wikiResults = wikiOutcome.status === 'fulfilled' ? wikiOutcome.value : [];
  const ddgResults = ddgOutcome.status === 'fulfilled' ? ddgOutcome.value : [];

  if (wikiOutcome.status === 'rejected') console.warn('[SearchService] Wikipedia fallback failed:', wikiOutcome.reason);
  if (ddgOutcome.status === 'rejected') console.warn('[SearchService] DuckDuckGo fallback failed:', ddgOutcome.reason);

  const merged = [...ddgResults, ...wikiResults];
  const byUrl = new Map<string, SearchResult>();
  for (const result of merged) {
    if (!result?.url) continue;
    const key = result.url.toLowerCase();
    if (!byUrl.has(key)) byUrl.set(key, result);
  }

  const relevantResults = [...byUrl.values()]
    .map((result) => normalizeSearchResult({ ...result, evidenceDirectness: 'search-snippet' }, `${trimmedQuery} ${locationContext?.city || ''}`))
    .filter((result) => isRelevantEvidence(result, `${trimmedQuery} ${locationContext?.city || ''}`));
  const ranked = rankSearchResults(relevantResults, `${trimmedQuery} ${locationContext?.city || ''}`);
  setCachedResults(cacheKey, ranked);

  return {
    results: ranked,
    query: primaryQuery,
    usedSearch: ranked.length > 0,
    source: ranked.length === 0 ? 'failure' : ranked.some((item) => item.isOfficial) ? 'official-web' : (ddgResults.length > 0 ? 'duckduckgo' : 'wikipedia'),
  };
}

export function formatSearchContextForPrompt(results: SearchResult[], maxResults = 5): string {
  if (!results.length) return '';

  const top = results.slice(0, maxResults);
  const lines = top.map((result, index) => {
    const officialTag = result.isOfficial ? ' [Official]' : '';
    return `[${index + 1}] ${result.title}${officialTag}\n   Domain: ${result.domain || getDomainFromUrl(result.url)}\n   URL: ${result.url}\n   ${result.snippet || 'No snippet available.'}`;
  });

  return `\n\n--- Web Search Results ---\n${lines.join('\n\n')}\n--- End Search Results ---\n`;
}

export function formatCitations(sources: { title: string; url: string }[]): string {
  if (!sources?.length) return '';
  return sources.map((source) => `[${source.title}](${source.url})`).join(', ');
}
