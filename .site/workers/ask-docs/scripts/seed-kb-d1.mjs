import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

const workerRoot = path.resolve(import.meta.dirname, '..');
const siteRoot = path.resolve(workerRoot, '..', '..');
const graphRoot = path.join(siteRoot, 'public', 'kb', 'pages');
const globalGraphPath = path.join(siteRoot, 'public', 'kb', 'dcs-kb-graph.json');
const conceptsPath = path.join(siteRoot, 'kb', 'concepts.json');
const database = 'adcs-qa-logs';
const remote = process.argv.includes('--remote');

const files = (await readdir(graphRoot)).filter((file) => file.endsWith('.json')).sort();
if (files.length === 0) {
  throw new Error(`No page graph JSON files found in ${graphRoot}`);
}

const rows = [];
for (const file of files) {
  const graphKey = file.slice(0, -'.json'.length);
  const graphJson = await readFile(path.join(graphRoot, file), 'utf8');
  const graph = JSON.parse(graphJson);
  rows.push({
    route: graph.current.route,
    graphKey,
    graphJson: JSON.stringify(graph),
    updatedAt: Math.floor(Date.now() / 1000),
  });
}

const concepts = JSON.parse(await readFile(conceptsPath, 'utf8')).map((concept) => ({
  id: String(concept.id || '').trim(),
  label: String(concept.label || concept.id || '').trim(),
  group: String(concept.group || 'concept').trim(),
  description: String(concept.description || '').trim(),
  aliases: JSON.stringify(concept.aliases ?? []),
  updatedAt: Math.floor(Date.now() / 1000),
})).filter((concept) => concept.id && concept.label);
const globalGraphJson = await readFile(globalGraphPath, 'utf8');
const updatedAt = Math.floor(Date.now() / 1000);

const sql = [
  [
    'INSERT INTO kb_global_graph (id, graph_json, updated_at)',
    `VALUES ('main', ${sqlString(globalGraphJson)}, ${updatedAt})`,
    'ON CONFLICT(id) DO UPDATE SET',
    'graph_json = excluded.graph_json,',
    'updated_at = excluded.updated_at;',
  ].join(' '),
  ...concepts.map((concept) => [
    'INSERT INTO kb_concepts (id, label, concept_group, description, aliases, updated_at)',
    `VALUES (${sqlString(concept.id)}, ${sqlString(concept.label)}, ${sqlString(concept.group)}, ${sqlString(concept.description)}, ${sqlString(concept.aliases)}, ${concept.updatedAt})`,
    'ON CONFLICT(id) DO UPDATE SET',
    'label = excluded.label,',
    'concept_group = excluded.concept_group,',
    'description = excluded.description,',
    'aliases = excluded.aliases,',
    'updated_at = excluded.updated_at;',
  ].join(' ')),
  ...rows.map((row) => [
    'INSERT INTO kb_page_graphs (route, graph_key, graph_json, updated_at)',
    `VALUES (${sqlString(row.route)}, ${sqlString(row.graphKey)}, ${sqlString(row.graphJson)}, ${row.updatedAt})`,
    'ON CONFLICT(route) DO UPDATE SET',
    'graph_key = excluded.graph_key,',
    'graph_json = excluded.graph_json,',
    'updated_at = excluded.updated_at;',
  ].join(' ')),
  `DELETE FROM kb_page_graphs WHERE route NOT IN (${rows.map((row) => sqlString(row.route)).join(', ')});`,
  `DELETE FROM kb_concepts WHERE id NOT IN (${concepts.map((concept) => sqlString(concept.id)).join(', ')});`,
  '',
].join('\n');

const tempDir = await mkdtemp(path.join(tmpdir(), 'adcs-kb-seed-'));
const sqlFile = path.join(tempDir, 'kb-seed.sql');
await writeFile(sqlFile, sql);

try {
  await runWrangler([
    'd1',
    'execute',
    database,
    ...(remote ? ['--remote'] : ['--local']),
    '--file',
    sqlFile,
  ]);
  console.log(`Seeded ${rows.length} page graphs, ${concepts.length} concepts, and 1 global graph into ${database}${remote ? ' (remote)' : ' (local)'}.`);
} finally {
  await rm(tempDir, { recursive: true, force: true });
}

function sqlString(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function runWrangler(args) {
  return new Promise((resolve, reject) => {
    const child = spawn('npx', ['wrangler', ...args], {
      cwd: workerRoot,
      stdio: 'inherit',
      shell: process.platform === 'win32',
    });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`wrangler exited with code ${code}`));
    });
  });
}
