const memoryBuckets = new Map();

export default {
  async fetch(request, env) {
    const origin = request.headers.get('origin') || '';
    const allowedOrigin = env.ALLOWED_ORIGIN || 'https://adcs.restack.tech';
    const corsHeaders = {
      'access-control-allow-origin': allowedOrigin,
      'access-control-allow-methods': 'POST, OPTIONS',
      'access-control-allow-headers': 'content-type',
      vary: 'Origin',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    if (request.method !== 'POST') {
      return json({ error: 'Method not allowed' }, 405, corsHeaders);
    }

    if (origin && origin !== allowedOrigin) {
      return json({ error: 'Origin not allowed' }, 403, corsHeaders);
    }

    const rate = await checkRateLimit(request, env);
    if (!rate.ok) {
      return json(
        { error: 'Rate limit exceeded. 잠시 후 다시 시도해주세요.' },
        429,
        {
          ...corsHeaders,
          'retry-after': String(rate.retryAfter),
          'x-ratelimit-limit': String(rate.limit),
          'x-ratelimit-remaining': '0',
        },
      );
    }

    if (!env.GITHUB_MODELS_TOKEN) {
      return json({ error: 'GITHUB_MODELS_TOKEN is not configured' }, 500, corsHeaders);
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return json({ error: 'Invalid JSON body' }, 400, corsHeaders);
    }

    const question = normalizeText(payload?.question || '').slice(0, 800);
    const sources = normalizeSources(payload?.sources);

    if (!question) {
      return json({ error: 'Question is required' }, 400, corsHeaders);
    }
    if (sources.length === 0) {
      return json({ error: 'At least one source is required' }, 400, corsHeaders);
    }

    const answer = await askGitHubModels({ env, question, sources });
    return json(
      { answer },
      200,
      {
        ...corsHeaders,
        'x-ratelimit-limit': String(rate.limit),
        'x-ratelimit-remaining': String(rate.remaining),
      },
    );
  },
};

async function askGitHubModels({ env, question, sources }) {
  const context = sources
    .map((source, index) => {
      return [
        `[${index + 1}] ${source.title}`,
        `URL: ${source.url}`,
        source.excerpt,
      ].join('\n');
    })
    .join('\n\n---\n\n');

  const response = await fetch(env.GITHUB_MODELS_ENDPOINT, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${env.GITHUB_MODELS_TOKEN}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model: env.GITHUB_MODELS_MODEL || 'openai/gpt-4.1-mini',
      temperature: 0.2,
      max_tokens: 1400,
      messages: [
        {
          role: 'system',
          content: [
            'You are a patient technical tutor for the AI Data Center Systems documentation.',
            'Answer in Korean unless the user asks for another language.',
            'Use the provided documentation sources as the anchor for the answer, but you may add generally accepted background knowledge when it helps a beginner understand the topic.',
            'Clearly separate what the document says from extra explanatory context when the distinction matters.',
            'Do not fabricate document-specific claims, numbers, commands, product behavior, or citations that are not in the sources.',
            'Prefer teaching: define key terms, explain why they matter, connect cause and effect, then give practical checks or examples.',
            'If the sources are thin, say that the document only gives limited evidence and then provide a careful general explanation.',
          ].join(' '),
        },
        {
          role: 'user',
          content: [
            `Question:\n${question}`,
            `Sources:\n${context}`,
            [
              'Write a helpful teaching answer.',
              'Structure it as:',
              '1. Short direct answer.',
              '2. Explanation for a reader who is new to the topic.',
              '3. How this connects to the provided document, citing source numbers like [1] where useful.',
              '4. Practical checks, examples, or caveats when relevant.',
              'Keep it focused, but do not be so terse that the reader has to already know the background.',
            ].join('\n'),
          ].join('\n\n'),
        },
      ],
    }),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data?.error?.message || data?.message || `GitHub Models error: ${response.status}`;
    throw new Error(message);
  }

  const answer = data?.choices?.[0]?.message?.content;
  if (!answer) throw new Error('GitHub Models returned an empty answer');
  return answer;
}

async function checkRateLimit(request, env) {
  const limit = parsePositiveInt(env.RATE_LIMIT_REQUESTS, 20);
  const windowSeconds = parsePositiveInt(env.RATE_LIMIT_WINDOW_SECONDS, 3600);
  const now = Math.floor(Date.now() / 1000);
  const ip = request.headers.get('cf-connecting-ip') || 'unknown';
  const key = `ask:${ip}`;

  if (env.ASK_DOCS_RATE_LIMIT) {
    const existing = await env.ASK_DOCS_RATE_LIMIT.get(key, 'json');
    const bucket = existing && existing.resetAt > now ? existing : { count: 0, resetAt: now + windowSeconds };
    if (bucket.count >= limit) {
      return {
        ok: false,
        limit,
        remaining: 0,
        retryAfter: Math.max(1, bucket.resetAt - now),
      };
    }
    bucket.count += 1;
    await env.ASK_DOCS_RATE_LIMIT.put(key, JSON.stringify(bucket), {
      expirationTtl: Math.max(60, bucket.resetAt - now),
    });
    return {
      ok: true,
      limit,
      remaining: Math.max(0, limit - bucket.count),
      retryAfter: 0,
    };
  }

  const bucket = memoryBuckets.get(key);
  if (!bucket || bucket.resetAt <= now) {
    memoryBuckets.set(key, { count: 1, resetAt: now + windowSeconds });
    return { ok: true, limit, remaining: limit - 1, retryAfter: 0 };
  }
  if (bucket.count >= limit) {
    return { ok: false, limit, remaining: 0, retryAfter: Math.max(1, bucket.resetAt - now) };
  }
  bucket.count += 1;
  return { ok: true, limit, remaining: Math.max(0, limit - bucket.count), retryAfter: 0 };
}

function normalizeSources(value) {
  if (!Array.isArray(value)) return [];
  return value
    .slice(0, 8)
    .map((source) => ({
      title: normalizeText(source?.title || '').slice(0, 160),
      url: normalizePath(source?.url || '').slice(0, 240),
      excerpt: normalizeText(source?.excerpt || '').slice(0, 1800),
    }))
    .filter((source) => source.title && source.url && source.excerpt);
}

function normalizeText(value) {
  return String(value).replace(/\s+/g, ' ').trim();
}

function normalizePath(value) {
  const text = String(value).trim();
  if (text.startsWith('/')) return text;
  try {
    const url = new URL(text);
    return `${url.pathname}${url.search}${url.hash}`;
  } catch {
    return '/';
  }
}

function parsePositiveInt(value, fallback) {
  const number = Number.parseInt(value, 10);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function json(body, status, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      ...headers,
    },
  });
}
