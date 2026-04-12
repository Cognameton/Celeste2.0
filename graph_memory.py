from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Optional


class GraphMemory:
    def __init__(self, path: str, *, enabled: bool = True):
        self.path = path
        self.enabled = bool(enabled)
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    canonical_key TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src_id INTEGER NOT NULL,
                    relation TEXT NOT NULL,
                    dst_id INTEGER NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(src_id, relation, dst_id),
                    FOREIGN KEY(src_id) REFERENCES nodes(id),
                    FOREIGN KEY(dst_id) REFERENCES nodes(id)
                );
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER,
                    edge_id INTEGER,
                    source_kind TEXT NOT NULL,
                    source_ref TEXT,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(node_id) REFERENCES nodes(id),
                    FOREIGN KEY(edge_id) REFERENCES edges(id)
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
                CREATE INDEX IF NOT EXISTS idx_edges_src_relation ON edges(src_id, relation);
                CREATE INDEX IF NOT EXISTS idx_edges_dst_relation ON edges(dst_id, relation);
                CREATE INDEX IF NOT EXISTS idx_observations_node ON observations(node_id);
                CREATE INDEX IF NOT EXISTS idx_observations_edge ON observations(edge_id);
                """
            )

    def _normalize_key(self, node_type: str, key: str) -> str:
        clean_type = (node_type or "entity").strip().lower()
        clean_key = (key or "").strip().lower()
        return f"{clean_type}:{clean_key}"

    def _json(self, payload: Optional[dict[str, Any]]) -> str:
        return json.dumps(payload or {}, sort_keys=True, ensure_ascii=False)

    def upsert_node(
        self,
        node_type: str,
        key: str,
        *,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None
        canonical_key = self._normalize_key(node_type, key)
        node_name = (name or key or canonical_key).strip()
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM nodes WHERE canonical_key = ?",
                (canonical_key,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE nodes
                    SET type = ?, name = ?, metadata_json = ?, updated_at = ?
                    WHERE canonical_key = ?
                    """,
                    (node_type, node_name, self._json(metadata), now, canonical_key),
                )
                return int(existing["id"])
            cur = conn.execute(
                """
                INSERT INTO nodes(type, name, canonical_key, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (node_type, node_name, canonical_key, self._json(metadata), now, now),
            )
            return int(cur.lastrowid)

    def connect(
        self,
        src_type: str,
        src_key: str,
        relation: str,
        dst_type: str,
        dst_key: str,
        *,
        src_name: Optional[str] = None,
        dst_name: Optional[str] = None,
        src_metadata: Optional[dict[str, Any]] = None,
        dst_metadata: Optional[dict[str, Any]] = None,
        edge_metadata: Optional[dict[str, Any]] = None,
        weight: float = 1.0,
        evidence: Optional[str] = None,
        source_kind: str = "runtime",
        source_ref: Optional[str] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None
        src_id = self.upsert_node(src_type, src_key, name=src_name, metadata=src_metadata)
        dst_id = self.upsert_node(dst_type, dst_key, name=dst_name, metadata=dst_metadata)
        if src_id is None or dst_id is None:
            return None
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM edges WHERE src_id = ? AND relation = ? AND dst_id = ?",
                (src_id, relation, dst_id),
            ).fetchone()
            if existing:
                edge_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE edges
                    SET weight = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (float(weight), self._json(edge_metadata), now, edge_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO edges(src_id, relation, dst_id, weight, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (src_id, relation, dst_id, float(weight), self._json(edge_metadata), now, now),
                )
                edge_id = int(cur.lastrowid)
            if evidence:
                conn.execute(
                    """
                    INSERT INTO observations(node_id, edge_id, source_kind, source_ref, text, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (None, edge_id, source_kind, source_ref, evidence.strip(), self._json(edge_metadata), now),
                )
            return edge_id

    def observe_node(
        self,
        node_type: str,
        key: str,
        *,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        text: str,
        source_kind: str = "runtime",
        source_ref: Optional[str] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None
        node_id = self.upsert_node(node_type, key, name=name, metadata=metadata)
        if node_id is None:
            return None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO observations(node_id, edge_id, source_kind, source_ref, text, metadata_json, created_at)
                VALUES (?, NULL, ?, ?, ?, ?, ?)
                """,
                (node_id, source_kind, source_ref, text.strip(), self._json(metadata), time.time()),
            )
            return int(cur.lastrowid)

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        tokens = [token.casefold() for token in (query or "").split() if token.strip()]
        if not tokens:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    e.id AS edge_id,
                    src.type AS src_type,
                    src.name AS src_name,
                    e.relation AS relation,
                    dst.type AS dst_type,
                    dst.name AS dst_name,
                    e.weight AS weight,
                    e.metadata_json AS edge_metadata_json,
                    (
                        SELECT o.text
                        FROM observations o
                        WHERE o.edge_id = e.id
                        ORDER BY o.created_at DESC
                        LIMIT 1
                    ) AS evidence_text
                FROM edges e
                JOIN nodes src ON src.id = e.src_id
                JOIN nodes dst ON dst.id = e.dst_id
                ORDER BY e.updated_at DESC
                """
            ).fetchall()
        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            text = f"{row['src_name']} [{row['src_type']}] -> {row['relation']} -> {row['dst_name']} [{row['dst_type']}]"
            evidence = str(row["evidence_text"] or "").strip()
            haystack = f"{text} {evidence}".casefold()
            score = 0.0
            for token in tokens:
                if token in haystack:
                    score += 1.0
            if score <= 0.0:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            if evidence:
                text = f"{text} | Evidence: {evidence}"
            ranked.append(
                {
                    "text": text,
                    "score": score,
                    "meta": {
                        "memory_channel": "graph",
                        "edge_id": int(row["edge_id"]),
                        "weight": float(row["weight"] or 1.0),
                    },
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[: max(1, top_k)]
