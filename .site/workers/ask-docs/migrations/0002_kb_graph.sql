CREATE TABLE IF NOT EXISTS kb_page_graphs (
  route TEXT PRIMARY KEY,
  graph_key TEXT NOT NULL UNIQUE,
  graph_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_concepts (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  concept_group TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  aliases TEXT NOT NULL DEFAULT '[]',
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kb_page_graphs_graph_key
  ON kb_page_graphs (graph_key);

CREATE INDEX IF NOT EXISTS idx_kb_concepts_group
  ON kb_concepts (concept_group);
