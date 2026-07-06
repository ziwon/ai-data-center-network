import { getCollection } from 'astro:content';
import { generateOpenGraphImage } from 'astro-og-canvas';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

export const prerender = true;

const siteTitle = 'AI Data Center Systems';
let ogBackgroundPathPromise;

export async function getStaticPaths() {
  const docs = await getCollection('docs');

  return docs.map((entry) => {
    const slug = ogImageSlugFromEntry(entry);
    const routeSlug = routeSlugFromEntry(entry);

    return {
      params: { slug },
      props: {
        title: entry.data.title,
        description: ogImageDescription(entry.data.description, routeSlug),
      },
    };
  });
}

export async function GET({ props }) {
  const ogBackgroundPath = await getOgBackgroundPath();

  const image = await generateOpenGraphImage({
    title: props.title,
    description: props.description,
    bgGradient: [
      [23, 23, 23],
      [35, 35, 35],
      [5, 5, 5],
    ],
    bgImage: {
      path: ogBackgroundPath,
      fit: 'cover',
      position: 'center',
    },
    padding: 72,
    font: {
      title: {
        color: [248, 244, 237],
        size: 68,
        weight: 'Bold',
        lineHeight: 0.98,
      },
      description: {
        color: [216, 209, 199],
        size: 30,
        lineHeight: 1.25,
      },
    },
  });

  return new Response(image, {
    headers: {
      'Content-Type': 'image/png',
      'Cache-Control': 'public, max-age=31536000, immutable',
    },
  });
}

function ogImageSlugFromEntry(entry) {
  return routeSlugFromEntry(entry) || 'index';
}

function routeSlugFromEntry(entry) {
  return String(entry.data.slug || entry.id.replace(/(^|\/)index\.md$/, '').replace(/\.md$/, ''))
    .replace(/^\/+|\/+$/g, '');
}

function ogImageDescription(description, routeSlug) {
  if (description && isAscii(description)) return description;
  const section = routeSlug.split('/')[0] || siteTitle;
  return `${titleCase(section)} · ${siteTitle}`;
}

function isAscii(value) {
  return /^[\x00-\x7F]*$/.test(value);
}

function titleCase(value) {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => formatTitlePart(part))
    .join(' ');
}

function formatTitlePart(part) {
  const upper = part.toUpperCase();
  if (['AI', 'GPU', 'LLM', 'CME'].includes(upper)) return upper;
  return part.charAt(0).toUpperCase() + part.slice(1);
}

function getOgBackgroundPath() {
  ogBackgroundPathPromise ||= createOgBackground();
  return ogBackgroundPathPromise;
}

async function createOgBackground() {
  const outPath = path.join(process.cwd(), 'node_modules', '.astro-og-canvas', 'adcs-og-cross.png');
  await mkdir(path.dirname(outPath), { recursive: true });
  await sharp(Buffer.from(homeCrossBackgroundSvg())).png().toFile(outPath);
  return outPath;
}

function homeCrossBackgroundSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <radialGradient id="blueGlow" cx="18%" cy="12%" r="38%">
      <stop offset="0" stop-color="#1884ff" stop-opacity="0.18"/>
      <stop offset="1" stop-color="#1884ff" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="cyanGlow" cx="84%" cy="18%" r="42%">
      <stop offset="0" stop-color="#13e9ff" stop-opacity="0.16"/>
      <stop offset="1" stop-color="#13e9ff" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="cyanLine" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0" stop-color="#13e9ff" stop-opacity="0"/>
      <stop offset="0.38" stop-color="#13e9ff" stop-opacity="0.22"/>
      <stop offset="0.58" stop-color="#13e9ff" stop-opacity="0.16"/>
      <stop offset="1" stop-color="#13e9ff" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="blueLine" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0" stop-color="#1884ff" stop-opacity="0"/>
      <stop offset="0.28" stop-color="#1884ff" stop-opacity="0.24"/>
      <stop offset="0.64" stop-color="#1884ff" stop-opacity="0.16"/>
      <stop offset="1" stop-color="#1884ff" stop-opacity="0"/>
    </linearGradient>
    <filter id="lineBlur" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="7"/>
    </filter>
    <filter id="softBlur" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="18"/>
    </filter>
    <linearGradient id="textShade" x1="0%" y1="0%" x2="82%" y2="100%">
      <stop offset="0" stop-color="#020817" stop-opacity="0.42"/>
      <stop offset="0.55" stop-color="#020817" stop-opacity="0.18"/>
      <stop offset="1" stop-color="#020817" stop-opacity="0.55"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="#020817"/>
  <rect width="1200" height="630" fill="url(#blueGlow)"/>
  <rect width="1200" height="630" fill="url(#cyanGlow)"/>
  <line x1="845" y1="-135" x2="455" y2="760" stroke="url(#cyanLine)" stroke-width="18" stroke-linecap="round" filter="url(#lineBlur)"/>
  <line x1="-150" y1="185" x2="1350" y2="520" stroke="url(#blueLine)" stroke-width="16" stroke-linecap="round" filter="url(#lineBlur)"/>
  <line x1="845" y1="-135" x2="455" y2="760" stroke="#13e9ff" stroke-opacity="0.07" stroke-width="42" stroke-linecap="round" filter="url(#softBlur)"/>
  <line x1="-150" y1="185" x2="1350" y2="520" stroke="#1884ff" stroke-opacity="0.08" stroke-width="40" stroke-linecap="round" filter="url(#softBlur)"/>
  <rect width="1200" height="630" fill="#020817" opacity="0.38"/>
  <rect width="1200" height="630" fill="url(#textShade)"/>
  <rect x="0" y="0" width="14" height="630" fill="#D9392E"/>
</svg>`;
}
