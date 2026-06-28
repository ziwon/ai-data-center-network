CREATE TABLE IF NOT EXISTS qa_logs (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  ts       INTEGER NOT NULL,
  ip_hash  TEXT    NOT NULL,
  question TEXT    NOT NULL,
  answer   TEXT    NOT NULL,
  sources  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qa_logs_ts ON qa_logs (ts);
