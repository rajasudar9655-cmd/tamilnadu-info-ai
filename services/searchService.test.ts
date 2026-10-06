import test from 'node:test';
import assert from 'node:assert/strict';

import {
  shouldSearchWeb,
  buildSearchQueries,
  rankSearchResults,
  detectSourceConflict,
  detectLanguage,
  performIntelligentSearch,
  isOfficialSource,
} from './searchService.ts';

test('Current price question triggers retrieval', () => {
  const decision = shouldSearchWeb('What is the ticket price today for Meenakshi temple?', true);
  assert.equal(decision.shouldSearch, true);
});

test('Current temple timing triggers retrieval', () => {
  const decision = shouldSearchWeb('What are the current opening timings of Meenakshi Amman Temple?', true);
  assert.equal(decision.shouldSearch, true);
});

test('Current government official query triggers retrieval', () => {
  const decision = shouldSearchWeb('Who is the current Madurai district collector?', true);
  assert.equal(decision.shouldSearch, true);
});

test('Time-sensitive requests still retrieve when Deep Search is turned off', () => {
  assert.equal(shouldSearchWeb('What is the weather in Coimbatore today?', false).shouldSearch, true);
});

test('Opening hours, holidays, and time-sensitive Tamil requests trigger retrieval', () => {
  assert.equal(shouldSearchWeb('Meenakshi Temple opening hours', false).shouldSearch, true);
  assert.equal(shouldSearchWeb('Tamil Nadu public holidays', false).shouldSearch, true);
  assert.equal(shouldSearchWeb('மதுரை கோவில் திறப்பு நேரம் இன்று', false).shouldSearch, true);
});

test('General historical question can skip live search', () => {
  const decision = shouldSearchWeb('Tell me about the history of Thanjavur Brihadeeswarar Temple', true);
  assert.equal(decision.shouldSearch, false);
});

test('Official .tn.gov.in sources outrank general blog sources', () => {
  const ranked = rankSearchResults([
    { title: 'Random temple blog', url: 'https://example.com/madurai-temple', snippet: 'Temple info referenced from a blog', source: 'google', isOfficial: false, relevanceScore: 40 },
    { title: 'Tamil Nadu Government', url: 'https://www.tn.gov.in/department/temple', snippet: 'Official temple timing and fee update', source: 'google', isOfficial: true, relevanceScore: 85 },
  ], 'Madurai temple timings');

  assert.equal(ranked[0].url, 'https://www.tn.gov.in/department/temple');
  assert.equal(ranked[0].isOfficial, true);
});

test('Tamil Nadu government ranks ahead of NIC, reputable news, and Wikipedia', () => {
  const ranked = rankSearchResults([
    { title: 'Wikipedia', url: 'https://en.wikipedia.org/wiki/Meenakshi_Temple', snippet: 'Meenakshi Temple', source: 'wikipedia' },
    { title: 'News report', url: 'https://www.thehindu.com/news/madurai', snippet: 'Madurai temple', source: 'google' },
    { title: 'NIC district page', url: 'https://madurai.nic.in/temples', snippet: 'Madurai temple', source: 'google' },
    { title: 'Tamil Nadu department', url: 'https://hrce.tn.gov.in/temples', snippet: 'Madurai temple', source: 'google' },
  ], 'Madurai temple');

  assert.deepEqual(ranked.map((result) => result.domain), [
    'hrce.tn.gov.in',
    'madurai.nic.in',
    'thehindu.com',
    'en.wikipedia.org',
  ]);
});

test('City-like domains are not automatically treated as official', () => {
  assert.equal(isOfficialSource('https://madurai.example.com/temple'), false);
  assert.equal(isOfficialSource('https://www.tn.gov.in/temple'), true);
});

test('Gemini grounding is the preferred retrieval source and unrelated official pages are excluded', async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = '';
  globalThis.fetch = async (input) => {
    requestedUrl = String(input);
    return new Response(JSON.stringify({
        retrievedAt: '2026-10-01T12:00:00.000Z',
        results: [
          {
            title: 'Madurai temple information',
            url: 'https://hrce.tn.gov.in/madurai-temple-tickets',
            snippet: 'Meenakshi Amman Temple ticket price is ₹50 according to HR&CE.',
          },
          {
            title: 'Tamil Nadu department contact directory',
            url: 'https://www.tn.gov.in/departments',
            snippet: 'Department office directory and administrative contacts.',
          },
        ],
      }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await performIntelligentSearch(
      'Meenakshi Amman Temple ticket price',
      { city: 'Madurai' },
    );
    assert.ok(requestedUrl.endsWith('/api/search/grounded'));
    assert.equal(result.source, 'gemini-google-search');
    assert.equal(result.results.length, 1);
    assert.equal(result.results[0].isOfficial, true);
    assert.equal(result.results[0].retrievedAt, '2026-10-01T12:00:00.000Z');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('Conflicting sources are detected for time-sensitive facts', () => {
  const result = detectSourceConflict([
    { title: 'Source A', url: 'https://a.example/fees', snippet: 'The ticket price is ₹50.', source: 'google' },
    { title: 'Source B', url: 'https://b.example/fees', snippet: 'The ticket price is ₹75.', source: 'google' },
  ], 'Temple ticket price today');

  assert.equal(result.hasConflict, true);
  assert.equal(result.conflictingSources.length, 2);
});

test('Different ticket categories in one source are not reported as a conflict', () => {
  const result = detectSourceConflict([
    {
      title: 'Temple guide',
      url: 'https://guide.example/fees',
      snippet: 'General entry is free. Special darshan ticket price is ₹50. Camera fee is ₹100.',
      source: 'google',
    },
  ], 'Temple ticket price today');

  assert.equal(result.hasConflict, false);
});

test('Multiple hours within one source are not reported as source disagreement', () => {
  const result = detectSourceConflict([
    {
      title: 'Temple hours',
      url: 'https://temple.example/hours',
      snippet: 'Morning hours are 5:00 AM to 12:30 PM, evening hours are 4:00 PM to 9:00 PM.',
      source: 'google',
    },
  ], 'Current temple opening timings');

  assert.equal(result.hasConflict, false);
});

test('Missing evidence does not fabricate facts', () => {
  const result = buildSearchQueries('Madurai temple hours today', { city: 'Madurai' });
  assert.ok(Array.isArray(result));
  assert.ok(result.some((query) => query.includes('Madurai')));
});

test('District context modifies search query', () => {
  const queries = buildSearchQueries('best temple nearby', { city: 'Madurai' });
  assert.ok(queries.some((query) => query.includes('Madurai')));
  assert.ok(queries.some((query) => query.includes('Tamil Nadu')));
});

test('Tamil questions remain Tamil', () => {
  const language = detectLanguage('மதுரையில் இன்று வானிலை என்ன?');
  assert.equal(language, 'ta');
});

test('English questions remain English', () => {
  const language = detectLanguage('What is the weather in Coimbatore today?');
  assert.equal(language, 'en');
});

test('Search failure returns a safe response', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error('network failure');
  };

  try {
    const result = await performIntelligentSearch('latest Tamil Nadu government updates', { city: 'Chennai' });
    assert.equal(result.usedSearch, false);
    assert.ok(result.results.length === 0 || result.source === 'duckduckgo');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
