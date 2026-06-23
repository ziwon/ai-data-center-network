import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import starlightGitHubAlerts from 'starlight-github-alerts';
import starlightImageZoom from 'starlight-image-zoom';

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
      plugins: [starlightGitHubAlerts(), starlightImageZoom()],
      title: 'AI Data Center Systems',
      description:
        'AI data center networking, LLM inference, training, storage, and systems performance engineering study notes.',
      favicon: '/favicon.svg',
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
        maxHeadingLevel: 4,
      },
      sidebar: [
        { label: 'Overview', slug: '' },
        {
          label: 'AI Data Center Network',
          items: [{ autogenerate: { directory: 'ai-data-center-network' } }],
        },
        {
          label: 'Efficient LLM Inference Systems',
          items: [{ autogenerate: { directory: 'efficient-llm-inference-systems' } }],
        },
        {
          label: 'Deep Learning for Network Engineers',
          items: [{ autogenerate: { directory: 'deep-learning-for-network-engineers' } }],
        },
        {
          label: 'AI Systems Performance Engineering',
          items: [{ autogenerate: { directory: 'ai-system-performance-engineering' } }],
        },
        {
          label: 'CME295 Lecture Notes',
          items: [{ autogenerate: { directory: 'cme295' } }],
        },
        {
          label: 'Training',
          items: [{ autogenerate: { directory: 'training' } }],
        },
        {
          label: 'Storage',
          items: [{ autogenerate: { directory: 'storage' } }],
        },
      ],
    }),
  ],
});
