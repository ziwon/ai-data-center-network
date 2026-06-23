import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

const projectRoot = process.cwd();
const sourceRoot = path.resolve(projectRoot, '..');
const docsOut = path.join(projectRoot, 'src', 'content', 'docs');
const publicOut = path.join(projectRoot, 'public');
const siteUrl = 'https://adcs.restack.tech';

const docRoots = [
  '.',
  'ai-data-center-network',
  'efficient-llm-inference-systems',
  'deep-learning-for-network-engineers',
  'ai-system-performance-engineering',
  'cme295',
  'training',
  'storage',
  'talks',
];

const assetExtensions = new Set([
  '.gif',
  '.jpeg',
  '.jpg',
  '.pdf',
  '.png',
  '.svg',
  '.webp',
]);

const ignoredDirs = new Set([
  '.git',
  '.github',
  'node_modules',
  'public',
  'scripts',
  '.site',
  'src',
  'dist',
]);

const generatedPublicRoots = [
  'fabric.svg',
  'ai-data-center-network',
  'efficient-llm-inference-systems',
  'deep-learning-for-network-engineers',
  'ai-system-performance-engineering',
  'cme295',
  'training',
  'storage',
  'talks',
];

const pages = [];

await cleanGeneratedOutput();
await walk(sourceRoot);
await writeLlmsFiles();

async function cleanGeneratedOutput() {
  await rm(docsOut, { recursive: true, force: true });
  await mkdir(docsOut, { recursive: true });

  await mkdir(publicOut, { recursive: true });
  await Promise.all(
    generatedPublicRoots.map((entry) =>
      rm(path.join(publicOut, entry), { recursive: true, force: true }),
    ),
  );
  await rm(path.join(publicOut, 'llms.txt'), { force: true });
  await rm(path.join(publicOut, 'llms-full.txt'), { force: true });
}

async function walk(currentDir) {
  const entries = await readdir(currentDir, { withFileTypes: true });

  for (const entry of entries) {
    const absolute = path.join(currentDir, entry.name);
    const relative = toPosix(path.relative(sourceRoot, absolute));

    if (entry.isDirectory()) {
      if (ignoredDirs.has(entry.name)) continue;
      if (relative === 'ai-data-center-network/ib-packets') continue;
      await walk(absolute);
      continue;
    }

    if (!entry.isFile()) continue;

    const ext = path.extname(entry.name).toLowerCase();
    if (ext === '.md' && shouldPublishMarkdown(relative)) {
      await publishMarkdown(absolute, relative);
      continue;
    }

    if (assetExtensions.has(ext) && shouldPublishAsset(relative)) {
      await publishAsset(absolute, relative);
    }
  }
}

function shouldPublishMarkdown(relative) {
  if (relative === 'AGENTS.md') return false;
  if (relative.startsWith('.')) return false;
  return docRoots.some((docRoot) => docRoot === '.' || relative.startsWith(`${docRoot}/`));
}

function shouldPublishAsset(relative) {
  if (relative.startsWith('.')) return false;
  return generatedPublicRoots.some((docRoot) => relative === docRoot || relative.startsWith(`${docRoot}/`));
}

async function publishMarkdown(absolute, relative) {
  const source = await readFile(absolute, 'utf8');
  const isSiteHome = relative === 'README.md';
  const title = extractTitle(source, relative);
  const body = isSiteHome ? siteHomeBody() : rewriteMarkdown(stripLeadingTitle(source), relative);
  const outRelative = markdownOutputPath(relative);
  const outAbsolute = path.join(docsOut, outRelative);
  const sourceStats = await stat(absolute);

  await mkdir(path.dirname(outAbsolute), { recursive: true });
  await writeFile(
    outAbsolute,
    [
      '---',
      `title: ${JSON.stringify(title)}`,
      `description: ${JSON.stringify(
        isSiteHome
          ? 'A study wiki for AI data center networking, LLM inference, distributed training, storage, and systems performance engineering.'
          : descriptionFrom(source, title),
      )}`,
      ...(isSiteHome ? ['template: splash'] : []),
      `lastUpdated: ${sourceStats.mtime.toISOString()}`,
      '---',
      '',
      body.trimStart(),
      '',
    ].join('\n'),
  );

  pages.push({
    title,
    sourcePath: relative,
    route: routeFromOutput(outRelative),
    body: body.trim(),
  });
}

function siteHomeBody() {
  return String.raw`
<section class="adcs-home-hero">
  <div class="adcs-home-hero-copy">
    <p class="adcs-home-kicker">Study wiki</p>
    <h2>AI infrastructure notes for the places where models meet machines.</h2>
    <p>
      Networking, inference, training, storage, and performance engineering notes organized as
      connected study tracks.
    </p>
    <div class="adcs-home-actions">
      <a href="/ai-data-center-network/">Start with AI fabric</a>
      <a href="/efficient-llm-inference-systems/">Explore inference</a>
    </div>
  </div>
  <a class="adcs-home-hero-visual" href="/ai-data-center-network/" aria-label="Open AI Data Center Network track">
    <img src="/fabric.svg" alt="Animated AI performance fabric" />
  </a>
</section>

<section class="adcs-track-grid" aria-label="Study tracks">
  <a class="adcs-track-card" href="/ai-data-center-network/">
    <span>AI Data Center Network</span>
    <small>RDMA, InfiniBand, RoCE, Clos fabrics, telemetry, and congestion control.</small>
  </a>
  <a class="adcs-track-card" href="/efficient-llm-inference-systems/">
    <span>Efficient LLM Inference Systems</span>
    <small>KV cache, batching, quantization, GPU profiling, and serving trade-offs.</small>
  </a>
  <a class="adcs-track-card" href="/ai-system-performance-engineering/">
    <span>AI Systems Performance Engineering</span>
    <small>GPU hardware, OS and container tuning, CUDA, PyTorch, and distributed communication.</small>
  </a>
  <a class="adcs-track-card" href="/deep-learning-for-network-engineers/">
    <span>Deep Learning for Network Engineers</span>
    <small>Training fundamentals, parallelism, collectives, RDMA, RoCE, and NCCL.</small>
  </a>
  <a class="adcs-track-card" href="/cme295/">
    <span>CME295 Lecture Notes</span>
    <small>Transformer and LLM lecture notes with practical systems annotations.</small>
  </a>
  <a class="adcs-track-card" href="/training/">
    <span>Training</span>
    <small>MLPerf Training, distributed training workloads, LLMs, MoE, and LoRA.</small>
  </a>
  <a class="adcs-track-card" href="/storage/">
    <span>Storage</span>
    <small>AI workload storage, ZFS, MLPerf Storage, and checkpoint data paths.</small>
  </a>
</section>

## Labs and Talks

<section class="adcs-link-band" aria-label="Labs and talks">
  <a href="/ai-data-center-network/clos-ebgp-lab/">Clos Fabric Lab Series</a>
  <a href="/ai-data-center-network/ib-packet-analysis/">InfiniBand Packet Analysis</a>
  <a href="/ai-data-center-network/rdma-examples/">RDMA Read/Write Examples</a>
  <a href="/talks/sr-iov-with-dgx-b200/making-dgx-b200-rdma-ready.pdf">Making DGX B200 RDMA-ready</a>
</section>
`;
}

async function publishAsset(absolute, relative) {
  const outAbsolute = path.join(publicOut, relative);
  await mkdir(path.dirname(outAbsolute), { recursive: true });
  await copyFile(absolute, outAbsolute);
}

function markdownOutputPath(relative) {
  if (relative === 'README.md') return 'index.md';
  if (relative.endsWith('/README.md')) {
    return `${relative.slice(0, -'/README.md'.length)}/index.md`;
  }
  return relative;
}

function routeFromOutput(outRelative) {
  if (outRelative === 'index.md') return '/';
  return `/${outRelative.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '/')}`.toLowerCase();
}

function extractTitle(markdown, relative) {
  const firstLine = markdown.split(/\r?\n/, 1)[0] ?? '';
  const heading = firstLine.match(/^#\s+(.+?)\s*$/);
  if (heading) return heading[1].replace(/\s+#+$/, '').trim();

  const basename = path.basename(relative, '.md');
  if (basename.toLowerCase() === 'readme') {
    return titleCase(path.basename(path.dirname(relative)));
  }
  return titleCase(basename);
}

function descriptionFrom(markdown, title) {
  const plain = stripLeadingTitle(markdown)
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\[[^\]]+\]\([^)]+\)/g, ' ')
    .replace(/[#>*_`|[\]-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return plain.slice(0, 160) || title;
}

function stripLeadingTitle(markdown) {
  return markdown.replace(/^#\s+.+(?:\r?\n)+/, '');
}

function rewriteMarkdown(markdown, sourceRelative) {
  return markdown
    .replace(/(!\[[^\]]*\]\()([^)\s]+)([^)]*)(\))/g, (_, open, href, rest = '', close) => {
      if (shouldLeaveHref(href)) return `${open}${href}${rest}${close}`;
      if (isMarkdownHref(href)) return `${open}${href}${rest}${close}`;
      return `${open}${publicHref(href, sourceRelative)}${rest}${close}`;
    })
    .replace(/(\]\()([^)\s]+?)(README\.md)(#[^)]+)?(\))/g, (_, open, prefix, _readme, hash = '', close) => {
      if (isExternalLink(prefix)) return `${open}${prefix}README.md${hash}${close}`;
      return `${open}${prefix}${hash}${close}`;
    })
    .replace(/(\]\()([^)\s]+?\.md)(#[^)]+)?(\))/g, (_, open, href, hash = '', close) => {
      if (isExternalLink(href)) return `${open}${href}${hash}${close}`;
      if (href === 'AGENTS.md' || href.endsWith('/AGENTS.md')) {
        return `${open}https://github.com/ziwon/ai-data-center-systems/blob/main/${resolveSourceHref(
          href,
          sourceRelative,
        )}${hash}${close}`;
      }
      return `${open}${href.replace(/\.md$/, '/').toLowerCase()}${hash}${close}`;
    })
    .replace(/(<img\b[^>]*\bsrc=["'])([^"']+)(["'][^>]*>)/g, (_, open, src, close) => {
      if (shouldLeaveHref(src)) return `${open}${src}${close}`;
      return `${open}${publicHref(src, sourceRelative)}${close}`;
    });
}

function isExternalLink(href) {
  return /^(?:[a-z][a-z0-9+.-]*:|#)/i.test(href);
}

function shouldLeaveHref(href) {
  return isExternalLink(href) || href.startsWith('/');
}

function isMarkdownHref(href) {
  return href.endsWith('.md') || href.includes('.md#');
}

function publicHref(href, sourceRelative) {
  return `/${resolveSourceHref(href, sourceRelative)}`;
}

function resolveSourceHref(href, sourceRelative) {
  const sourceDir = sourceRelative === 'README.md' ? '' : path.posix.dirname(sourceRelative);
  return path.posix.normalize(path.posix.join(sourceDir, href));
}

function titleCase(value) {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

async function writeLlmsFiles() {
  pages.sort((a, b) => a.route.localeCompare(b.route));

  const llmsLines = [
    '# AI Data Center Systems',
    '',
    'AI data center networking, LLM inference, training, storage, and systems performance engineering study notes.',
    '',
    '## Docs',
    '',
    ...pages.map((page) => `- [${page.title}](${siteUrl}${page.route}) - ${page.sourcePath}`),
    '',
  ];

  const fullLines = [
    '# AI Data Center Systems',
    '',
    'This file concatenates the source Markdown used to build the public documentation site.',
    '',
    ...pages.flatMap((page) => [
      `## ${page.title}`,
      '',
      `Source: ${page.sourcePath}`,
      `URL: ${siteUrl}${page.route}`,
      '',
      page.body,
      '',
    ]),
  ];

  await writeFile(path.join(publicOut, 'llms.txt'), llmsLines.join('\n'));
  await writeFile(path.join(publicOut, 'llms-full.txt'), fullLines.join('\n'));
}

function toPosix(value) {
  return value.split(path.sep).join('/');
}
