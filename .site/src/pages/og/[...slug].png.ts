import { getCollection } from 'astro:content';
import { generateOpenGraphImage } from 'astro-og-canvas';

export const prerender = true;

const siteTitle = 'AI Data Center Systems';

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
  const image = await generateOpenGraphImage({
    title: props.title,
    description: props.description,
    bgGradient: [
      [23, 23, 23],
      [35, 35, 35],
      [5, 5, 5],
    ],
    border: {
      color: [217, 57, 46],
      width: 14,
      side: 'inline-start',
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
