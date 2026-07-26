import { access, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';

const distRoot = path.join(process.cwd(), 'dist');
const errors = [];

const canonicalRoutes = [
  '/',
  '/network/',
  '/network/chap01/',
  '/gpu/',
  '/training/',
  '/inference/',
  '/inference/week01/',
  '/mlops/',
  '/storage/',
  '/systems-performance/',
  '/systems-performance/chap01/',
  '/courses/',
  '/courses/cme295/',
  '/courses/cme295/lec-03/',
  '/courses/deep-learning-for-network-engineers/',
  '/courses/deep-learning-for-network-engineers/week04/',
];

const legacyMappings = [
  { from: '/ai-data-center-network', to: '/network' },
  { from: '/efficient-llm-inference-systems', to: '/inference' },
  { from: '/ai-system-performance-engineering', to: '/systems-performance' },
  { from: '/cme295', to: '/courses/cme295' },
  {
    from: '/deep-learning-for-network-engineers',
    to: '/courses/deep-learning-for-network-engineers',
  },
];
const legacyPrefixes = legacyMappings.map(({ from }) => from);
let legacyRedirectCount = 0;

for (const route of canonicalRoutes) {
  const htmlPath = route === '/'
    ? path.join(distRoot, 'index.html')
    : path.join(distRoot, route.slice(1), 'index.html');
  await expectFile(htmlPath, `canonical route ${route}`);
}

for (const { from, to } of legacyMappings) {
  await expectFile(
    path.join(distRoot, from.slice(1), 'index.html'),
    `legacy fallback route ${from}/`,
  );

  const redirectFiles = await collectFiles(
    path.join(distRoot, from.slice(1)),
    (file) => file.endsWith('.html'),
  );
  legacyRedirectCount += redirectFiles.length;

  for (const redirectFile of redirectFiles) {
    const relativeDirectory = path
      .relative(path.join(distRoot, from.slice(1)), path.dirname(redirectFile))
      .split(path.sep)
      .filter((segment) => segment && segment !== '.')
      .join('/');
    const targetRoute = `${to}${relativeDirectory ? `/${relativeDirectory}` : ''}/`;
    const redirectHtml = await readFile(redirectFile, 'utf8');
    if (!redirectHtml.includes(`content="0;url=${targetRoute}"`)) {
      errors.push(`${from}/${relativeDirectory} does not redirect to ${targetRoute}.`);
    }
    if (!redirectHtml.includes('<meta name="robots" content="noindex">')) {
      errors.push(`${from}/${relativeDirectory} is missing the noindex directive.`);
    }
    await expectFile(
      path.join(distRoot, targetRoute.slice(1), 'index.html'),
      `redirect target ${targetRoute}`,
    );
  }
}

await expectFile(path.join(distRoot, 'CNAME'), 'custom-domain CNAME');
await expectFile(path.join(distRoot, 'llms.txt'), 'llms.txt');
await expectFile(path.join(distRoot, 'llms-full.txt'), 'llms-full.txt');
await expectFile(path.join(distRoot, 'kb', 'dcs-kb-graph.json'), 'global knowledge graph');

const sitemapFiles = (await readdir(distRoot)).filter((name) => /^sitemap-\d+\.xml$/.test(name));
if (sitemapFiles.length === 0) {
  errors.push('No sitemap-N.xml file was generated.');
} else {
  const sitemap = (
    await Promise.all(sitemapFiles.map((name) => readFile(path.join(distRoot, name), 'utf8')))
  ).join('\n');
  for (const route of canonicalRoutes) {
    const url = `https://adcs.restack.tech${route}`;
    if (!sitemap.includes(`<loc>${url}</loc>`)) {
      errors.push(`Sitemap is missing canonical URL ${url}.`);
    }
  }
  assertNoLegacyRoutes(sitemap, 'sitemap');
}

const llms = await readText(path.join(distRoot, 'llms.txt'));
assertNoLegacyRoutes(llms, 'llms.txt');
for (const route of canonicalRoutes.filter((route) => route !== '/')) {
  if (!llms.includes(`https://adcs.restack.tech${route}`)) {
    errors.push(`llms.txt is missing canonical URL https://adcs.restack.tech${route}.`);
  }
}

const graphText = await readText(path.join(distRoot, 'kb', 'dcs-kb-graph.json'));
assertNoLegacyRoutes(graphText, 'global knowledge graph');
if (graphText) {
  try {
    const graph = JSON.parse(graphText);
    const graphRoutes = new Set(
      (graph.nodes ?? []).map((node) => node.route).filter(Boolean),
    );
    for (const route of canonicalRoutes) {
      if (!graphRoutes.has(route)) errors.push(`Knowledge graph is missing route ${route}.`);
    }
  } catch (error) {
    errors.push(`Knowledge graph is not valid JSON: ${error.message}`);
  }
}

for (const route of canonicalRoutes.filter((route) => route !== '/')) {
  const htmlPath = path.join(distRoot, route.slice(1), 'index.html');
  const html = await readText(htmlPath);
  if (html && !html.includes(`<link rel="canonical" href="https://adcs.restack.tech${route}"`)) {
    errors.push(`Canonical tag is missing or incorrect for ${route}.`);
  }
}

const htmlFiles = await collectFiles(distRoot, (file) => file.endsWith('.html'));
for (const file of htmlFiles) {
  const relative = `/${path.relative(distRoot, file).split(path.sep).join('/')}`;
  if (legacyPrefixes.some((prefix) => relative.startsWith(`${prefix}/`))) continue;
  const html = await readFile(file, 'utf8');
  for (const prefix of legacyPrefixes) {
    if (html.includes(`href="${prefix}`) || html.includes(`src="${prefix}`)) {
      errors.push(`Canonical page ${relative} still references legacy prefix ${prefix}.`);
    }
  }
}

if (errors.length > 0) {
  console.error(`Build validation failed with ${errors.length} error(s):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exitCode = 1;
} else {
  console.log(
    `Validated ${canonicalRoutes.length} canonical routes, ${legacyRedirectCount} legacy fallbacks, ${htmlFiles.length} HTML files, sitemap, LLM indexes, and knowledge graph.`,
  );
}

async function expectFile(filePath, label) {
  try {
    await access(filePath);
  } catch {
    errors.push(`Missing ${label}: ${path.relative(distRoot, filePath)}`);
  }
}

async function readText(filePath) {
  try {
    return await readFile(filePath, 'utf8');
  } catch {
    return '';
  }
}

function assertNoLegacyRoutes(text, label) {
  for (const prefix of legacyPrefixes) {
    if (text.includes(`https://adcs.restack.tech${prefix}`)) {
      errors.push(`${label} still contains legacy URL prefix ${prefix}.`);
    }
  }
}

async function collectFiles(currentDir, predicate) {
  const files = [];
  for (const entry of await readdir(currentDir, { withFileTypes: true })) {
    const absolute = path.join(currentDir, entry.name);
    if (entry.isDirectory()) files.push(...await collectFiles(absolute, predicate));
    else if (entry.isFile() && predicate(absolute)) files.push(absolute);
  }
  return files;
}
