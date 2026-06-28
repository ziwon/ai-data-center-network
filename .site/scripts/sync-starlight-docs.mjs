import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { embedDocuments, isEmbeddingEnabled } from './lib/embeddings.mjs';

const projectRoot = process.cwd();
const sourceRoot = path.resolve(projectRoot, '..');
const docsOut = path.join(projectRoot, 'src', 'content', 'docs');
const publicOut = path.join(projectRoot, 'public');
const conceptsPath = path.join(projectRoot, 'kb', 'concepts.json');
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
  '.mypy_cache',
  '.pytest_cache',
  '.venv',
  'node_modules',
  'public',
  'refs',
  'scripts',
  '.site',
  '__pycache__',
  'src',
  'dist',
  'venv',
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
  'refs',
];

const pages = [];

const stopWords = new Set([
  'about',
  'after',
  'all',
  'also',
  'and',
  'are',
  'because',
  'before',
  'between',
  'but',
  'can',
  'chapter',
  'check',
  'content',
  'contents',
  'could',
  'data',
  'does',
  'each',
  'example',
  'for',
  'from',
  'have',
  'into',
  'learning',
  'more',
  'most',
  'note',
  'practical',
  'questions',
  'results',
  'only',
  'over',
  'section',
  'should',
  'summary',
  'such',
  'system',
  'table',
  'than',
  'that',
  'the',
  'their',
  'there',
  'these',
  'this',
  'through',
  'using',
  'was',
  'when',
  'where',
  'which',
  'while',
  'will',
  'with',
  'were',
  'you',
  'your',
  '있는',
  '한다',
  '위해',
  '대한',
  '그리고',
  '또는',
  '에서',
  '으로',
]);

const weakConceptAliases = new Set([
  'benchmark',
  'gdr',
  'recovery',
  'verbs',
]);

const weakSingleHitAliases = new Set([
  ...weakConceptAliases,
  'attention',
  'checkpoint',
  'checkpointing',
  'collectives',
  'remote memory',
  'tail latency',
]);

await cleanGeneratedOutput();
await walk(sourceRoot);
await writeLlmsFiles();
await writeKnowledgeGraph();

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
  await rm(path.join(publicOut, 'kb', 'dcs-kb-graph.json'), { force: true });
  await rm(path.join(publicOut, 'kb', 'pages'), { recursive: true, force: true });
}

async function walk(currentDir) {
  const entries = await readdir(currentDir, { withFileTypes: true });

  for (const entry of entries) {
    const absolute = path.join(currentDir, entry.name);
    const relative = toPosix(path.relative(sourceRoot, absolute));

    if (entry.isDirectory()) {
      if (shouldIgnoreDirectory(entry.name)) continue;
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
      ...(isSiteHome ? [] : [`slug: ${JSON.stringify(slugFromRoute(routeFromOutput(outRelative)))}`]),
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
  if (relative.endsWith('.md')) {
    return `${relative.slice(0, -'.md'.length)}/index.md`;
  }
  return relative;
}

function routeFromOutput(outRelative) {
  if (outRelative === 'index.md') return '/';
  return `/${outRelative.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '/')}`.toLowerCase();
}

function slugFromRoute(route) {
  return route.replace(/^\//, '').replace(/\/$/, '');
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

async function writeKnowledgeGraph() {
  const concepts = await readConceptCatalog();
  const routeByKey = new Map(pages.map((page) => [normalizeRoute(page.route), page.route]));
  const nodeTerms = new Map();
  const documentFrequency = new Map();

  for (const page of pages) {
    const terms = termFrequency(page);
    nodeTerms.set(page.route, terms);
    for (const term of terms.keys()) {
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    }
  }

  const nodes = pages.map((page) => {
    const keywords = topKeywords(nodeTerms.get(page.route), documentFrequency, pages.length, 10);
    return {
      id: page.route,
      title: page.title,
      route: page.route,
      group: groupFromRoute(page.route),
      keywords: keywords.map((keyword) => keyword.term),
      excerpt: descriptionFrom(page.body, page.title),
    };
  });

  const nodeByRoute = new Map(nodes.map((node) => [node.route, node]));
  const conceptNodes = concepts.map((concept) => ({
    id: conceptNodeId(concept.id),
    kind: 'concept',
    title: concept.label,
    label: concept.label,
    group: concept.group,
    description: concept.description,
    keywords: concept.aliases.slice(0, 8),
  }));
  const conceptById = new Map(conceptNodes.map((concept) => [concept.id, concept]));
  const pageConcepts = new Map(
    pages.map((page) => [page.route, matchConcepts(page, concepts)]),
  );
  const idfByTerm = new Map(
    [...documentFrequency].map(([term, count]) => [
      term,
      Math.log((pages.length + 1) / (count + 1)) + 1,
    ]),
  );
  const edgeScores = new Map();

  for (const page of pages) {
    for (const targetRoute of extractInternalLinks(page, routeByKey)) {
      if (targetRoute === page.route) continue;
      addEdge(edgeScores, page.route, targetRoute, 'link', 1, []);
    }
  }

  let vectors = null;
  if (isEmbeddingEnabled()) {
    try {
      vectors = await embedDocuments(
        pages.map((page) => ({ key: page.route, text: embeddingText(page) })),
      );
    } catch (error) {
      console.warn(`[embeddings] disabled; falling back to keyword edges: ${error?.message ?? error}`);
      vectors = null;
    }
  }

  if (vectors) {
    addSemanticEdges(edgeScores, pages, vectors, nodeByRoute);
  } else {
    addKeywordEdges(edgeScores, pages, nodeTerms, idfByTerm, nodeByRoute);
  }

  for (const [route, matches] of pageConcepts) {
    for (const match of matches) {
      const weight = Math.min(0.95, 0.78 + Math.min(match.count, 6) * 0.025);
      addEdge(edgeScores, route, conceptNodeId(match.id), 'mentions', weight, match.aliases.slice(0, 4));
    }
  }

  for (const page of pages) {
    const siblings = pages
      .filter((candidate) => candidate.route !== page.route && groupFromRoute(candidate.route) === groupFromRoute(page.route))
      .slice(0, 3);
    for (const sibling of siblings) {
      addEdge(edgeScores, page.route, sibling.route, 'section', 0.18, []);
    }
  }

  const edges = [...edgeScores.values()]
    .filter((edge) => edge.source !== edge.target)
    .sort(compareEdges);

  await mkdir(path.join(publicOut, 'kb'), { recursive: true });
  await writeGlobalGraph(nodes, conceptNodes, edges);
  await writePageGraphs(nodes, conceptNodes, edges, nodeByRoute, conceptById);
}

async function writeGlobalGraph(nodes, conceptNodes, edges) {
  const graphEdges = capGlobalEdges(edges);
  const degree = new Map();
  const weightedDegree = new Map();

  for (const edge of graphEdges) {
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
    weightedDegree.set(edge.source, (weightedDegree.get(edge.source) ?? 0) + edge.weight);
    weightedDegree.set(edge.target, (weightedDegree.get(edge.target) ?? 0) + edge.weight);
  }

  const globalNodes = [
    ...nodes.map((node) => ({
      id: node.id,
      kind: 'doc',
      title: node.title,
      route: node.route,
      group: node.group,
      degree: degree.get(node.id) ?? 0,
      weightedDegree: roundWeight(weightedDegree.get(node.id) ?? 0),
    })),
    ...conceptNodes.map((node) => ({
      id: node.id,
      kind: 'concept',
      title: node.title,
      group: node.group,
      description: node.description,
      degree: degree.get(node.id) ?? 0,
      weightedDegree: roundWeight(weightedDegree.get(node.id) ?? 0),
    })),
  ];

  const globalGraph = {
    version: 1,
    generatedAt: new Date().toISOString(),
    nodes: globalNodes,
    edgeTypes: [...new Set(graphEdges.map((edge) => edge.type))].sort(),
    edges: compactGlobalEdges(globalNodes, graphEdges),
  };

  await writeFile(
    path.join(publicOut, 'kb', 'dcs-kb-graph.json'),
    `${JSON.stringify(globalGraph)}\n`,
  );
}

function compactGlobalEdges(nodes, edges) {
  const nodeIndex = new Map(nodes.map((node, index) => [node.id, index]));
  const edgeTypeIndex = new Map([...new Set(edges.map((edge) => edge.type))].sort().map((type, index) => [type, index]));
  return edges
    .map((edge) => ({
      s: nodeIndex.get(edge.source),
      t: nodeIndex.get(edge.target),
      k: edgeTypeIndex.get(edge.type),
      w: edge.weight,
    }))
    .filter((edge) => edge.s !== undefined && edge.t !== undefined && edge.k !== undefined);
}

function capGlobalEdges(edges) {
  const mentions = edges.filter((edge) => edge.type.includes('mentions'));
  const links = edges.filter((edge) => !edge.type.includes('mentions') && edge.type.includes('link'));
  const rest = edges.filter((edge) => !edge.type.includes('mentions') && !edge.type.includes('link'));
  const cappedRest = capEdgesByNode(rest, 7);
  const unique = new Map();

  for (const edge of [...mentions, ...links, ...cappedRest].sort(compareEdges)) {
    unique.set(`${edge.source} ${edge.target}`, edge);
  }
  return [...unique.values()].sort(compareEdges);
}

function capEdgesByNode(edges, limit) {
  const selected = new Map();
  const byNode = new Map();
  for (const edge of edges.sort(compareEdges)) {
    if (!byNode.has(edge.source)) byNode.set(edge.source, []);
    if (!byNode.has(edge.target)) byNode.set(edge.target, []);
    byNode.get(edge.source).push(edge);
    byNode.get(edge.target).push(edge);
  }

  for (const [, nodeEdges] of byNode) {
    for (const edge of nodeEdges.slice(0, limit)) {
      selected.set(`${edge.source} ${edge.target}`, edge);
    }
  }
  return [...selected.values()].sort(compareEdges);
}

async function readConceptCatalog() {
  const source = await readFile(conceptsPath, 'utf8').catch(() => '[]');
  const parsed = JSON.parse(source);
  return parsed
    .map((concept) => ({
      id: String(concept.id || '').trim(),
      label: String(concept.label || concept.id || '').trim(),
      group: String(concept.group || 'concept').trim(),
      description: String(concept.description || '').trim(),
      aliases: [
        String(concept.label || concept.id || '').trim(),
        ...(concept.aliases ?? []).map((alias) => String(alias).trim()),
      ].filter((alias, index, aliases) => alias && aliases.indexOf(alias) === index && !weakConceptAliases.has(alias.toLowerCase())),
    }))
    .filter((concept) => concept.id && concept.label && concept.aliases.length > 0);
}

function matchConcepts(page, concepts) {
  const text = searchableText(page);
  return concepts
    .map((concept) => {
      const hits = concept.aliases
        .map((alias) => ({ alias, count: aliasMatchCount(text, alias) }))
        .filter((hit) => hit.count > 0);
      if (hits.length === 0) return null;

      const count = hits.reduce((sum, hit) => sum + hit.count, 0);
      const hasDistinctiveAlias = hits.some((hit) => isDistinctiveConceptAlias(hit.alias));
      if (count < 2 && !hasDistinctiveAlias) return null;

      return {
        ...concept,
        aliases: hits.sort((a, b) => b.count - a.count || a.alias.localeCompare(b.alias)).map((hit) => hit.alias),
        count,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.count - a.count || b.aliases.length - a.aliases.length || a.label.localeCompare(b.label))
    .slice(0, 12);
}

function searchableText(page) {
  return [
    page.title,
    page.body.replace(/```[\s\S]*?```/g, ' '),
  ]
    .join('\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\[[^\]]+\]\([^)]+\)/g, ' ')
    .replace(/\s+/g, ' ')
    .toLowerCase();
}

function aliasMatchCount(text, alias) {
  const normalized = alias.toLowerCase().trim();
  if (!normalized) return 0;
  const escaped = normalized.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  if (/^[a-z0-9][a-z0-9 .+/#-]*$/i.test(normalized)) {
    return [...text.matchAll(new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, 'gi'))].length;
  }
  let count = 0;
  let index = text.indexOf(normalized);
  while (index !== -1) {
    count += 1;
    index = text.indexOf(normalized, index + normalized.length);
  }
  return count;
}

function isDistinctiveConceptAlias(alias) {
  const normalized = alias.trim();
  const lower = normalized.toLowerCase();
  if (!normalized || weakSingleHitAliases.has(lower)) return false;
  return (
    /[\s/+.-]/.test(normalized) ||
    /^[A-Z0-9]{3,}$/.test(normalized) ||
    /[A-Z][a-z]+[A-Z]/.test(normalized)
  );
}

function termFrequency(page) {
  const text = [
    page.title,
    page.title,
    page.body.replace(/```[\s\S]*?```/g, ' '),
  ]
    .join('\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\[[^\]]+\]\([^)]+\)/g, ' ')
    .replace(/[^\p{L}\p{N}\s./+-]/gu, ' ');
  const terms = new Map();
  for (const raw of text.match(/[\p{L}\p{N}][\p{L}\p{N}./+-]{1,}/gu) ?? []) {
    const term = normalizeTerm(raw);
    if (term.length < 3 && !/^[가-힣]{2,}$/u.test(term)) continue;
    if (!term || stopWords.has(term)) continue;
    terms.set(term, (terms.get(term) ?? 0) + 1);
  }
  return terms;
}

function normalizeTerm(term) {
  const normalized = term
    .toLowerCase()
    .replace(/^[-./+]+|[-./+]+$/g, '')
    .trim();
  const withoutLongSuffix = normalized.replace(
    /(?:에서|으로|에게|보다|부터|까지|처럼|마다|이다|이며|이고|하고|되는|된다|한다|하여|해서|하면|들과|들의)$/u,
    '',
  );
  if (withoutLongSuffix.length > 3) {
    return withoutLongSuffix.replace(/(?:은|는|이|가|을|를|의|에|로|도|만|와|과)$/u, '');
  }
  return withoutLongSuffix;
}

function topKeywords(terms, documentFrequency, documentCount, limit) {
  return [...terms.entries()]
    .map(([term, count]) => ({
      term,
      score: count * (Math.log((documentCount + 1) / ((documentFrequency.get(term) ?? 0) + 1)) + 1),
    }))
    .filter((keyword) => keyword.term.length > 2)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function addSemanticEdges(edgeScores, pages, vectors, nodeByRoute) {
  const selectedPairs = new Set();
  for (const page of pages) {
    const sourceVector = vectors.get(page.route);
    if (!sourceVector) continue;

    pages
      .filter((target) => target.route !== page.route && vectors.has(target.route))
      .map((target) => ({
        target: target.route,
        score: dotProduct(sourceVector, vectors.get(target.route)),
        shared: sharedKeywordsBetween(page.route, target.route, nodeByRoute),
      }))
      .filter((candidate) => candidate.score >= 0.55)
      .sort((a, b) => b.score - a.score || a.target.localeCompare(b.target))
      .slice(0, 6)
      .forEach((candidate) => {
        const key = edgeKey(page.route, candidate.target);
        if (selectedPairs.has(key)) return;
        selectedPairs.add(key);
        addEdge(
          edgeScores,
          page.route,
          candidate.target,
          'semantic',
          Math.min(0.9, Math.max(0.3, candidate.score)),
          candidate.shared.slice(0, 4),
        );
      });
  }
}

function addKeywordEdges(edgeScores, pages, nodeTerms, idfByTerm, nodeByRoute) {
  const selectedPairs = new Set();
  for (const page of pages) {
    const candidates = [];
    for (const target of pages) {
      if (target.route === page.route) continue;
      const score = relatedness(nodeTerms.get(page.route), nodeTerms.get(target.route), idfByTerm);
      if (score <= 0) continue;
      candidates.push({
        target: target.route,
        score,
        shared: sharedKeywordsBetween(page.route, target.route, nodeByRoute),
      });
    }

    candidates
      .sort((a, b) => b.score - a.score || a.target.localeCompare(b.target))
      .slice(0, 5)
      .forEach((candidate) => {
        const key = edgeKey(page.route, candidate.target);
        if (selectedPairs.has(key)) return;
        selectedPairs.add(key);
        addEdge(
          edgeScores,
          page.route,
          candidate.target,
          'keyword',
          Math.min(0.82, Math.max(0.22, candidate.score)),
          candidate.shared.slice(0, 4),
        );
      });
  }
}

function sharedKeywordsBetween(sourceRoute, targetRoute, nodeByRoute) {
  const sourceKeywords = new Set(nodeByRoute.get(sourceRoute)?.keywords ?? []);
  return (nodeByRoute.get(targetRoute)?.keywords ?? []).filter((term) => sourceKeywords.has(term));
}

function edgeKey(source, target) {
  return source.localeCompare(target) <= 0 ? `${source} ${target}` : `${target} ${source}`;
}

function dotProduct(source, target) {
  if (!source || !target || source.length !== target.length) return 0;
  return source.reduce((sum, value, index) => sum + value * target[index], 0);
}

function embeddingText(page) {
  const headings = [...page.body.matchAll(/^#{1,3}\s+(.+)$/gm)]
    .map((match) => match[1])
    .slice(0, 16);
  return [
    page.title,
    ...headings,
    descriptionFrom(page.body, page.title),
  ]
    .join('\n')
    .slice(0, 6000);
}

function relatedness(sourceTerms, targetTerms, idfByTerm) {
  if (!sourceTerms || !targetTerms) return 0;
  let sharedScore = 0;
  let sourceScore = 0;
  let targetScore = 0;

  for (const [term, count] of sourceTerms) {
    const idf = idfByTerm.get(term) ?? 1;
    sourceScore += Math.min(count, 8) * idf;
    if (targetTerms.has(term)) {
      sharedScore += Math.min(count, targetTerms.get(term), 8) * idf;
    }
  }
  for (const [term, count] of targetTerms) {
    const idf = idfByTerm.get(term) ?? 1;
    targetScore += Math.min(count, 8) * idf;
  }

  const denominator = Math.sqrt(sourceScore * targetScore);
  return denominator === 0 ? 0 : sharedScore / denominator;
}

function extractInternalLinks(page, routeByKey) {
  const links = new Set();
  const patterns = [
    /\[[^\]]*\]\(([^)\s]+)(?:\s+["'][^"']+["'])?\)/g,
    /<a\b[^>]*\bhref=["']([^"']+)["'][^>]*>/g,
  ];

  for (const pattern of patterns) {
    for (const match of page.body.matchAll(pattern)) {
      const route = routeFromHref(match[1], page.route, routeByKey);
      if (route) links.add(route);
    }
  }
  return links;
}

function routeFromHref(href, sourceRoute, routeByKey) {
  if (!href || /^(?:[a-z][a-z0-9+.-]*:|#|mailto:)/i.test(href)) return '';
  const [cleanHref] = href.split(/[?#]/, 1);
  if (!cleanHref || assetExtensions.has(path.extname(cleanHref).toLowerCase())) return '';

  const baseRoute = sourceRoute.endsWith('/') ? sourceRoute : `${sourceRoute}/`;
  const rawRoute = cleanHref.startsWith('/')
    ? cleanHref
    : `/${path.posix.normalize(path.posix.join(baseRoute, cleanHref))}`;
  const normalized = normalizeRoute(rawRoute.replace(/README\.md$/i, '').replace(/\.md$/i, '/'));
  return routeByKey.get(normalized) ?? '';
}

function addEdge(edgeScores, source, target, type, weight, terms) {
  const [a, b] = source.localeCompare(target) <= 0 ? [source, target] : [target, source];
  const key = `${a} ${b}`;
  const previous = edgeScores.get(key);
  if (!previous) {
    edgeScores.set(key, { source: a, target: b, type, weight: roundWeight(weight), terms });
    return;
  }

  previous.weight = roundWeight(Math.min(1, previous.weight + weight * 0.65));
  if (!previous.type.includes(type)) previous.type = `${previous.type}+${type}`;
  previous.terms = [...new Set([...(previous.terms ?? []), ...terms])].slice(0, 6);
}

function normalizeRoute(route) {
  if (!route) return '/';
  const normalized = `/${route}`.replace(/\/+/g, '/').toLowerCase();
  return normalized.endsWith('/') ? normalized : `${normalized}/`;
}

function groupFromRoute(route) {
  const [first] = route.replace(/^\/|\/$/g, '').split('/');
  return first || 'home';
}

async function writePageGraphs(nodes, conceptNodes, edges, nodeByRoute, conceptById) {
  const pageOut = path.join(publicOut, 'kb', 'pages');
  await mkdir(pageOut, { recursive: true });

  await Promise.all(nodes.map(async (current) => {
    const localEdges = edges
      .filter((edge) => edge.source === current.id || edge.target === current.id)
      .sort(compareEdges)
      .slice(0, 10);
    const localIds = new Set([current.id]);
    for (const edge of localEdges) {
      localIds.add(edge.source === current.id ? edge.target : edge.source);
    }

    const currentConceptIds = edges
      .filter((edge) => edge.type.includes('mentions') && edge.source === current.id)
      .sort(compareEdges)
      .slice(0, 8)
      .map((edge) => edge.target);
    for (const conceptId of currentConceptIds) {
      localIds.add(conceptId);
    }

    for (const conceptId of currentConceptIds) {
      const conceptDocs = edges
        .filter((edge) => edge.type.includes('mentions') && edge.target === conceptId && edge.source !== current.id)
        .sort(compareEdges)
        .slice(0, 4);
      for (const edge of conceptDocs) {
        localIds.add(edge.source);
      }
    }

    const localNodes = [...localIds]
      .map((id) => nodeByRoute.get(id) ?? conceptById.get(id))
      .filter(Boolean);
    const localGraph = {
      version: 1,
      current,
      nodes: localNodes,
      edges: edges.filter((edge) => localIds.has(edge.source) && localIds.has(edge.target)),
    };

    await writeFile(
      path.join(pageOut, `${routeGraphKey(current.route)}.json`),
      `${JSON.stringify(localGraph, null, 2)}\n`,
    );
  }));
}

function conceptNodeId(id) {
  return `concept:${id}`;
}

function compareEdges(a, b) {
  return b.weight - a.weight || a.source.localeCompare(b.source) || a.target.localeCompare(b.target);
}

function routeGraphKey(route) {
  const normalized = normalizeRoute(route);
  if (normalized === '/') return 'index';
  return normalized
    .replace(/^\/|\/$/g, '')
    .split('/')
    .map((segment) => encodeURIComponent(segment).replace(/%/g, '~'))
    .join('__')
    .toLowerCase();
}

function roundWeight(value) {
  return Math.round(value * 1000) / 1000;
}

function shouldIgnoreDirectory(name) {
  return ignoredDirs.has(name) || /^\.?venv(?:$|[-_.])/.test(name);
}

function toPosix(value) {
  return value.split(path.sep).join('/');
}
