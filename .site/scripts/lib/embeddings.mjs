import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const projectRoot = path.resolve(import.meta.dirname, '..', '..');
const envPath = path.join(projectRoot, '.env');
const cachePath = path.join(projectRoot, 'kb', '.embed-cache.json');
const model = '@cf/baai/bge-m3';
const batchSize = 32;

loadDotEnv();

export function isEmbeddingEnabled() {
  return Boolean(accountId() && apiToken());
}

export async function embedDocuments(items) {
  const cache = await readCache();
  const next = {};
  const output = new Map();
  const stale = [];

  for (const item of items) {
    const key = String(item.key || '').trim();
    const text = normalizeText(item.text);
    if (!key || !text) continue;

    const hash = hashText(text);
    const cached = cache[key];
    if (cached?.hash === hash && cached?.model === model && isVector(cached.vector)) {
      next[key] = cached;
      output.set(key, cached.vector);
      continue;
    }

    stale.push({ key, text, hash });
  }

  for (const batch of chunk(stale, batchSize)) {
    const vectors = await withRetry(() => cloudflareEmbed(batch.map((item) => item.text)));
    if (vectors.length !== batch.length) {
      throw new Error(`Workers AI returned ${vectors.length} vectors for ${batch.length} inputs`);
    }

    batch.forEach((item, index) => {
      const vector = normalizeVector(vectors[index]);
      next[item.key] = { model, hash: item.hash, vector };
      output.set(item.key, vector);
    });
  }

  await writeCache(next);
  console.log(`[embeddings] ${output.size} docs (${stale.length} re-embedded, ${output.size - stale.length} cached)`);
  return output;
}

async function cloudflareEmbed(texts) {
  const response = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${accountId()}/ai/run/${model}`,
    {
      method: 'POST',
      headers: {
        authorization: `Bearer ${apiToken()}`,
        'content-type': 'application/json',
      },
      body: JSON.stringify({ text: texts }),
    },
  );

  if (!response.ok) {
    throw new Error(`Workers AI ${response.status}: ${await response.text()}`);
  }

  const payload = await response.json();
  if (payload.success === false) {
    throw new Error(`Workers AI: ${JSON.stringify(payload.errors ?? [])}`);
  }

  const vectors = payload.result?.data ?? payload.result?.embeddings ?? payload.data;
  if (!Array.isArray(vectors) || !vectors.every(Array.isArray)) {
    throw new Error('Workers AI response did not include embedding vectors');
  }
  return vectors;
}

async function readCache() {
  try {
    return JSON.parse(await readFile(cachePath, 'utf8'));
  } catch {
    return {};
  }
}

async function writeCache(cache) {
  await mkdir(path.dirname(cachePath), { recursive: true });
  const ordered = Object.fromEntries(Object.keys(cache).sort().map((key) => [key, cache[key]]));
  await writeFile(cachePath, `${JSON.stringify(ordered, null, 2)}\n`);
}

function loadDotEnv() {
  let source = '';
  try {
    source = readFileSync(envPath, 'utf8');
  } catch {
    return;
  }

  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const separator = trimmed.indexOf('=');
    if (separator <= 0) continue;

    const key = trimmed.slice(0, separator).trim();
    const rawValue = trimmed.slice(separator + 1).trim();
    if (!key || process.env[key] !== undefined) continue;

    process.env[key] = unquote(rawValue);
  }
}

function unquote(value) {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function accountId() {
  return process.env.CF_ACCOUNT_ID || process.env.CLOUDFLARE_ACCOUNT_ID || '';
}

function apiToken() {
  return process.env.CF_AI_TOKEN || process.env.CLOUDFLARE_API_TOKEN || '';
}

function hashText(text) {
  return createHash('sha256').update(`${model}\n${text}`).digest('hex');
}

function normalizeText(text) {
  return String(text || '').replace(/\s+/g, ' ').trim();
}

function normalizeVector(vector) {
  const values = vector.map((value) => Number(value));
  if (!isVector(values)) throw new Error('Embedding vector contained non-numeric values');

  const norm = Math.hypot(...values) || 1;
  return values.map((value) => Math.round((value / norm) * 1_000_000) / 1_000_000);
}

function isVector(value) {
  return Array.isArray(value) && value.length > 0 && value.every((item) => Number.isFinite(Number(item)));
}

function chunk(items, size) {
  return Array.from({ length: Math.ceil(items.length / size) }, (_, index) =>
    items.slice(index * size, index * size + size),
  );
}

async function withRetry(fn, tries = 3) {
  let lastError;
  for (let index = 0; index < tries; index += 1) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (index < tries - 1) {
        await new Promise((resolve) => setTimeout(resolve, 800 * (index + 1)));
      }
    }
  }
  throw lastError;
}
