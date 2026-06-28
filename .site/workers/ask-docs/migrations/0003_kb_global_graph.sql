CREATE TABLE IF NOT EXISTS kb_global_graph (
  id TEXT PRIMARY KEY,
  graph_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
