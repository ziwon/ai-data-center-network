import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import starlightGitHubAlerts from 'starlight-github-alerts';
import starlightImageZoom from 'starlight-image-zoom';
import starlightSidebarTopics from 'starlight-sidebar-topics';

const googleAnalyticsId = getGoogleTagId('PUBLIC_GA_MEASUREMENT_ID', /^G-[A-Z0-9]+$/);
const googleAdSenseClient =
  getGoogleTagId('PUBLIC_GOOGLE_ADSENSE_CLIENT', /^ca-pub-\d+$/) ?? 'ca-pub-8128231647578658';

function getGoogleTagId(name, pattern) {
  const value = process.env[name]?.trim();
  return value && pattern.test(value) ? value : undefined;
}

function googleAnalyticsHeadEntries(measurementId) {
  if (!measurementId) {
    return [];
  }

  return [
    {
      tag: 'script',
      attrs: {
        async: true,
        src: `https://www.googletagmanager.com/gtag/js?id=${measurementId}`,
      },
    },
    {
      tag: 'script',
      content: `
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', '${measurementId}');
      `,
    },
  ];
}

function googleAdSenseHeadEntries(client) {
  if (!client) {
    return [];
  }

  return [
    {
      tag: 'script',
      attrs: {
        async: true,
        crossorigin: 'anonymous',
        src: `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${client}`,
      },
    },
  ];
}

export default defineConfig({
  site: 'https://adcs.restack.tech',
  trailingSlash: 'always',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    optimizeDeps: {
      include: ['d3-drag', 'd3-force', 'd3-selection', 'd3-zoom'],
    },
  },
  integrations: [
    mermaid(),
    starlight({
      plugins: [
        starlightGitHubAlerts(),
        starlightImageZoom(),
        starlightSidebarTopics(
          [
            {
              label: 'AI Data Center Network',
              link: '/ai-data-center-network/',
              icon: 'random',
              items: [{ autogenerate: { directory: 'ai-data-center-network', collapsed: true } }],
            },
            {
              label: 'Efficient LLM Inference Systems',
              link: '/efficient-llm-inference-systems/',
              icon: 'rocket',
              items: [{ autogenerate: { directory: 'efficient-llm-inference-systems', collapsed: true } }],
            },
            {
              label: 'GPU Systems',
              link: '/gpu/',
              icon: 'laptop',
              items: [{ autogenerate: { directory: 'gpu', collapsed: true } }],
            },
            {
              label: 'Deep Learning for Network Engineers',
              link: '/deep-learning-for-network-engineers/',
              icon: 'open-book',
              items: [{ autogenerate: { directory: 'deep-learning-for-network-engineers', collapsed: true } }],
            },
            {
              label: 'AI Systems Performance Engineering',
              link: '/ai-system-performance-engineering/',
              icon: 'setting',
              items: [{ autogenerate: { directory: 'ai-system-performance-engineering', collapsed: true } }],
            },
            {
              label: 'CME295 Lecture Notes',
              link: '/cme295/',
              icon: 'pencil',
              items: [{ autogenerate: { directory: 'cme295', collapsed: true } }],
            },
            {
              label: 'Training',
              link: '/training/',
              icon: 'analytics',
              items: [{ autogenerate: { directory: 'training' } }],
            },
            {
              label: 'Storage',
              link: '/storage/',
              icon: 'database',
              items: [{ autogenerate: { directory: 'storage', collapsed: true } }],
            },
          ],
        ),
      ],
      title: 'AI Data Center Systems',
      description:
        'AI data center networking, LLM inference, training, storage, and systems performance engineering study notes.',
      favicon: '/favicon.svg',
      components: {
        PageFrame: './src/components/PageFrame.astro',
        PageSidebar: './src/components/PageSidebar.astro',
      },
      head: [
        ...googleAnalyticsHeadEntries(googleAnalyticsId),
        ...googleAdSenseHeadEntries(googleAdSenseClient),
        {
          tag: 'meta',
          attrs: { property: 'og:image', content: 'https://adcs.restack.tech/og.png' },
        },
        {
          tag: 'meta',
          attrs: { property: 'og:image:width', content: '1200' },
        },
        {
          tag: 'meta',
          attrs: { property: 'og:image:height', content: '630' },
        },
        {
          tag: 'meta',
          attrs: { name: 'twitter:image', content: 'https://adcs.restack.tech/og.png' },
        },
      ],
      customCss: ['./src/styles/custom.css'],
      lastUpdated: true,
      editLink: {
        baseUrl: 'https://github.com/ziwon/ai-data-center-systems/edit/main/',
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/ziwon/ai-data-center-systems',
        },
      ],
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 2,
      },
    }),
  ],
});
