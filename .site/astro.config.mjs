import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import starlightGitHubAlerts from 'starlight-github-alerts';
import starlightImageZoom from 'starlight-image-zoom';
import starlightSidebarTopics from 'starlight-sidebar-topics';

export default defineConfig({
  site: 'https://adcs.restack.tech',
  trailingSlash: 'always',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
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
              items: [{ autogenerate: { directory: 'ai-data-center-network' } }],
            },
            {
              label: 'Efficient LLM Inference Systems',
              link: '/efficient-llm-inference-systems/',
              icon: 'rocket',
              items: [{ autogenerate: { directory: 'efficient-llm-inference-systems' } }],
            },
            {
              label: 'Deep Learning for Network Engineers',
              link: '/deep-learning-for-network-engineers/',
              icon: 'open-book',
              items: [{ autogenerate: { directory: 'deep-learning-for-network-engineers' } }],
            },
            {
              label: 'AI Systems Performance Engineering',
              link: '/ai-system-performance-engineering/',
              icon: 'setting',
              items: [{ autogenerate: { directory: 'ai-system-performance-engineering' } }],
            },
            {
              label: 'CME295 Lecture Notes',
              link: '/cme295/',
              icon: 'pencil',
              items: [{ autogenerate: { directory: 'cme295' } }],
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
              items: [{ autogenerate: { directory: 'storage' } }],
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
      },
      head: [
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
