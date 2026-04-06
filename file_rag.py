from __future__ import annotations

import gzip
import gc
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from typing import Any, Callable, Optional

from joblib import dump, load
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config_types import AgentConfig


TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".log", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".xml", ".csv", ".sh",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".java", ".kt", ".cs", ".go", ".rs", ".php",
    ".sql", ".bat", ".ps1", ".pdf",
}
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".idea", ".vscode", "dist", "build"}
FILE_RAG_SCHEMA_VERSION = 2
FILE_RAG_INDEX_KIND_TFIDF = "tfidf"
FILE_RAG_INDEX_KIND_SEMANTIC = "semantic"
FILE_RAG_SEMANTIC_BATCH_SIZE = 64
FILE_RAG_RRF_K = 60.0
FILE_RAG_MIN_LEXICAL_SCORE = 0.07
FILE_RAG_MIN_SEMANTIC_SCORE = 0.24
FILE_RAG_BROAD_MIN_LEXICAL_SCORE = 0.11
FILE_RAG_BROAD_MIN_SEMANTIC_SCORE = 0.30
FILE_RAG_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "bank", "be", "by", "can", "cofc", "cover",
    "describe", "describes", "did", "do", "does", "for", "from", "give", "had",
    "has", "have", "hereafter", "how", "if", "in", "index", "information", "into",
    "is", "it", "its", "library", "me", "most", "number", "of", "on", "or", "over",
    "provide", "published", "reply", "say", "source", "sources", "talk", "tell",
    "that", "the", "their", "this", "through", "to", "topic", "was", "what",
    "when", "where", "which", "with", "you", "your",
}


class FileRAG:
    def __init__(self, cfg: AgentConfig, shared_embedder: Any = None, shared_device: str = "cpu"):
        self.cfg = cfg
        self.catalog_path = os.path.join(cfg.data_dir, "file_rag_index.json")
        self.deep_meta_path = os.path.join(cfg.data_dir, "file_rag_deep_meta.json")
        self.deep_chunks_path = os.path.join(cfg.data_dir, "file_rag_deep_chunks.json.gz")
        self.deep_index_path = os.path.join(cfg.data_dir, "file_rag_deep_index.joblib")
        self.deep_embeddings_path = os.path.join(cfg.data_dir, "file_rag_deep_embeddings.npy")

        self.directories: list[str] = []
        self.files: list[dict[str, Any]] = []
        self._catalog_tfidf = None
        self._catalog_matrix = None

        self._deep_loaded = False
        self._deep_dirs: list[str] = []
        self._deep_chunks: list[dict[str, Any]] = []
        self._deep_vectorizer = None
        self._deep_matrix = None
        self._deep_embedding_matrix = None
        self._deep_embedding_dim: int | None = None

        shared_device_name = str(shared_device or "cpu").strip().lower()
        self._shared_device = shared_device_name or "cpu"
        self._share_embedder = bool(getattr(self.cfg, "file_rag_share_embedder", False))
        self.device = self._resolve_embedding_device()
        if self._share_embedder and shared_embedder is not None and self._shared_device == self.device:
            self.embedder = shared_embedder
            self._embedder_mode = "shared"
        else:
            self.embedder = None
            self._embedder_mode = f"dedicated-{self.device}"
            if shared_embedder is not None and self._shared_device == "cuda" and self.device == "cuda":
                logging.info("File RAG will use a dedicated CUDA embedder instead of the shared memory embedder.")

        os.makedirs(cfg.data_dir, exist_ok=True)

        requested_dirs = self._normalize_dirs(getattr(cfg, "file_rag_dirs", []) or [])
        if getattr(cfg, "file_rag_enabled", True) and requested_dirs:
            if not self._load_catalog(requested_dirs):
                self.rebuild(requested_dirs)

    def _catalog_signature(self, files: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for item in sorted(files, key=lambda row: row.get("rel_path", "")):
            record = "|".join(
                [
                    item.get("rel_path", ""),
                    item.get("basename", ""),
                    str(item.get("size", 0)),
                    str(item.get("mtime_ns", 0)),
                ]
            )
            digest.update(record.encode("utf-8", errors="ignore"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _doc_id(self, item: dict[str, Any]) -> str:
        path = str(item.get("path", ""))
        return hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _file_signature(self, item: dict[str, Any]) -> str:
        record = "|".join(
            [
                str(item.get("path", "")),
                str(item.get("size", 0)),
                str(item.get("mtime_ns", 0)),
            ]
        )
        return hashlib.sha1(record.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _chunk_id(self, item: dict[str, Any], chunk_index: int, text: str) -> str:
        record = "|".join(
            [
                self._doc_id(item),
                self._file_signature(item),
                str(chunk_index),
                hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16],
            ]
        )
        return hashlib.sha1(record.encode("utf-8", errors="ignore")).hexdigest()[:20]

    def _build_chunk_record(self, item: dict[str, Any], chunk_index: int, text: str) -> dict[str, Any]:
        return {
            "index_version": FILE_RAG_SCHEMA_VERSION,
            "index_kind": FILE_RAG_INDEX_KIND_TFIDF,
            "doc_id": self._doc_id(item),
            "chunk_id": self._chunk_id(item, chunk_index, text),
            "file_signature": self._file_signature(item),
            "path": item["path"],
            "rel_path": item["rel_path"],
            "display_name": item["display_name"],
            "basename": item.get("basename", os.path.basename(item["path"])),
            "size": int(item.get("size", 0)),
            "mtime_ns": int(item.get("mtime_ns", 0)),
            "chunk_index": int(chunk_index),
            "text": text,
        }

    def _semantic_enabled(self) -> bool:
        return bool(getattr(self.cfg, "file_rag_use_embeddings", False))

    def _embedding_model_id(self) -> str:
        model = str(getattr(self.cfg, "embedding_model", "") or "").strip()
        if model and os.path.exists(model):
            return os.path.abspath(model)
        return model

    def _cuda_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _resolve_embedding_device(self) -> str:
        requested = str(getattr(self.cfg, "file_rag_embedding_device", "auto") or "auto").strip().lower()
        if requested == "cuda":
            return "cuda" if self._cuda_available() else "cpu"
        if requested == "cpu":
            return "cpu"
        return "cuda" if self._cuda_available() else "cpu"

    def _semantic_meta(self, *, enabled: bool, dim: int | None = None) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "model": self._embedding_model_id(),
            "dim": int(dim) if dim is not None else None,
        }

    def _release_embedder(self) -> None:
        if self._embedder_mode == "shared" or self.embedder is None:
            return
        self.embedder = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_embedder(self) -> bool:
        if not self._semantic_enabled():
            return False
        if self.embedder is not None:
            return True
        from sentence_transformers import SentenceTransformer

        try:
            logging.info(
                "Loading File RAG embedder from %s on device=%s (%s)",
                self.cfg.embedding_model,
                self.device,
                self._embedder_mode,
            )
            self.embedder = SentenceTransformer(self.cfg.embedding_model, device=self.device)
            logging.info("File RAG embedder ready (device=%s).", self.device)
            return True
        except Exception:
            logging.exception("Failed to load File RAG embedder on device=%s.", self.device)
            self.embedder = None
            if self.device == "cuda":
                logging.warning("Retrying File RAG embedder on CPU after CUDA load failure.")
                self.device = "cpu"
                self._embedder_mode = "dedicated-cpu"
                try:
                    self.embedder = SentenceTransformer(self.cfg.embedding_model, device=self.device)
                    logging.info("File RAG embedder ready after CPU fallback.")
                    return True
                except Exception:
                    logging.exception("CPU fallback also failed for File RAG embedder.")
                    self.embedder = None
            return False

    def _encode_texts(self, texts: list[str], batch_size: int = FILE_RAG_SEMANTIC_BATCH_SIZE) -> np.ndarray:
        if not texts:
            dim = self._deep_embedding_dim or 0
            return np.empty((0, dim), dtype=np.float32)
        if not self._ensure_embedder():
            raise RuntimeError("File RAG embedder unavailable")
        vec = self.embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        arr = np.asarray(vec, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return arr / norms

    def _hit_key(self, hit: dict[str, Any]) -> str:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if chunk_id:
            return chunk_id
        return f"{hit.get('path', '')}:{hit.get('chunk_index', -1)}"

    def _chunk_hit(self, chunk: dict[str, Any], score: float, source: str) -> dict[str, Any]:
        return {
            "path": chunk["path"],
            "rel_path": chunk["rel_path"],
            "display_name": chunk["display_name"],
            "doc_id": chunk.get("doc_id"),
            "chunk_id": chunk.get("chunk_id"),
            "file_signature": chunk.get("file_signature"),
            "index_version": chunk.get("index_version"),
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "score": float(score),
            "retrieval_source": source,
        }

    def _merge_ranked_hits(
        self,
        lexical_hits: list[dict[str, Any]],
        semantic_hits: list[dict[str, Any]],
        top_k: int,
        *,
        query: str = "",
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        def add_hits(hits: list[dict[str, Any]], label: str) -> None:
            for rank, hit in enumerate(hits, start=1):
                key = self._hit_key(hit)
                entry = merged.get(key)
                if entry is None:
                    entry = dict(hit)
                    entry["rrf_score"] = 0.0
                    entry["lexical_score"] = None
                    entry["semantic_score"] = None
                    entry["retrieval_sources"] = []
                    merged[key] = entry
                entry["rrf_score"] += 1.0 / (FILE_RAG_RRF_K + rank)
                if label == FILE_RAG_INDEX_KIND_TFIDF:
                    entry["lexical_score"] = float(hit.get("score", 0.0))
                else:
                    entry["semantic_score"] = float(hit.get("score", 0.0))
                if label not in entry["retrieval_sources"]:
                    entry["retrieval_sources"].append(label)

        def finalize_scores(items: dict[str, dict[str, Any]]) -> None:
            for entry in items.values():
                entry["score"] = max(
                    float(entry.get("semantic_score") or -1.0),
                    float(entry.get("lexical_score") or -1.0),
                    float(entry.get("rrf_score", 0.0)),
                )

        add_hits(lexical_hits, FILE_RAG_INDEX_KIND_TFIDF)
        add_hits(semantic_hits, FILE_RAG_INDEX_KIND_SEMANTIC)
        finalize_scores(merged)
        broad_query = self._is_broad_library_query(query)
        lexical_threshold = (
            FILE_RAG_BROAD_MIN_LEXICAL_SCORE if broad_query else FILE_RAG_MIN_LEXICAL_SCORE
        )
        semantic_threshold = (
            FILE_RAG_BROAD_MIN_SEMANTIC_SCORE if broad_query else FILE_RAG_MIN_SEMANTIC_SCORE
        )

        ranked = sorted(
            (
                item
                for item in merged.values()
                if self._passes_merge_threshold(
                    item,
                    broad_query=broad_query,
                    lexical_threshold=lexical_threshold,
                    semantic_threshold=semantic_threshold,
                )
            ),
            key=lambda item: (
                len(item.get("retrieval_sources", [])),
                float(item.get("rrf_score", 0.0)),
                float(item.get("semantic_score") or -1.0),
                float(item.get("lexical_score") or -1.0),
            ),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        per_path_counts: dict[str, int] = {}
        per_doc_counts: dict[str, int] = {}
        per_doc_limit = 1 if broad_query else 2
        for item in ranked:
            path = str(item.get("path", ""))
            doc_key = str(item.get("doc_id") or path)
            count = per_path_counts.get(path, 0)
            if count >= 2:
                continue
            doc_count = per_doc_counts.get(doc_key, 0)
            if doc_count >= per_doc_limit:
                continue
            per_path_counts[path] = count + 1
            per_doc_counts[doc_key] = doc_count + 1
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected

    def _query_terms(self, query: str) -> list[str]:
        return [
            token
            for token in self._normalize_text(query).split()
            if len(token) > 2 and token not in FILE_RAG_QUERY_STOPWORDS
        ]

    def _is_broad_library_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        broad_cues = (
            "using the indexed",
            "from the indexed",
            "using the library",
            "from the library",
            "indexed library",
            "indexed documents",
            "indexed files",
            "talk to me about",
            "what can you tell me about",
            "what does",
            "summarize",
            "summary",
            "explain",
            "search the indexed library",
        )
        if any(cue in q for cue in broad_cues):
            return True
        return len(self._query_terms(query)) >= 3

    def _passes_merge_threshold(
        self,
        hit: dict[str, Any],
        *,
        broad_query: bool,
        lexical_threshold: float,
        semantic_threshold: float,
    ) -> bool:
        lexical_score = float(hit.get("lexical_score") or 0.0)
        semantic_score = float(hit.get("semantic_score") or 0.0)
        retrieval_sources = set(hit.get("retrieval_sources") or [])
        has_strong_lexical = lexical_score >= lexical_threshold
        has_strong_semantic = semantic_score >= semantic_threshold
        if has_strong_semantic:
            return True
        if has_strong_lexical and semantic_score >= semantic_threshold - 0.04:
            return True
        if not broad_query and has_strong_lexical and FILE_RAG_INDEX_KIND_SEMANTIC not in retrieval_sources:
            return lexical_score >= lexical_threshold + 0.03
        if FILE_RAG_INDEX_KIND_SEMANTIC in retrieval_sources and semantic_score >= semantic_threshold - 0.02:
            return True
        return False

    def _current_signature(self) -> str:
        return self._catalog_signature(self.files)

    def _normalize_dirs(self, directories: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for path in directories:
            full = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
            if not os.path.isdir(full) or full in seen:
                continue
            seen.add(full)
            normalized.append(full)
        return normalized

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()

    def _normalize_query_for_retrieval(self, query: str) -> str:
        text = (query or "").strip()
        if not text:
            return text

        patterns = [
            r"^\s*(hello|hi|hey)\s+[a-z0-9_-]+[,.!\s]*",
            r"\b(using|from)\s+the\s+library(?:\s+you\s+have\s+access\s+to)?\b",
            r"\busing\s+the\s+indexed\s+(documents|files)\b",
            r"\bfrom\s+the\s+indexed\s+(documents|files)\b",
            r"\byou\s+have\s+access\s+to\b",
            r"^\s*(talk to me about|tell me about|explain|give me an overview of)\s+",
            r"\bblend\s+what\s+you\s+find\s+with\s+your\s+general\s+knowledge\b",
            r"\bwith\s+citation[s]?\b",
            r"\bcite\s+(your|the)?\s*sources\b",
            r"\busing\s+only\s+the\s+indexed\s+library\b",
            r"\bgive\s+me\s+a\s+(practical|concise|fuller)\s+summary\b",
            r"\bsummarize\s+the\s+findings\b",
        ]
        for pattern in patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" \t\r\n.,!?;:-")
        return text or (query or "").strip()

    def _query_candidates(self, query: str) -> list[str]:
        candidates: list[str] = []
        raw = (query or "").strip()
        if not raw:
            return candidates

        patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r"\b([A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})\b",
            r"\b([A-Z][A-Za-z0-9_.-]{2,})\b",
        ]
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, raw):
                norm = self._normalize_text(match)
                if (
                    norm
                    and norm not in seen
                    and norm not in FILE_RAG_QUERY_STOPWORDS
                    and len(norm) > 2
                ):
                    seen.add(norm)
                    candidates.append(norm)

        full_norm = self._normalize_text(raw)
        if full_norm and full_norm not in seen and full_norm not in FILE_RAG_QUERY_STOPWORDS:
            candidates.append(full_norm)
        return candidates

    def _iter_files(self, directories: list[str]):
        for root in directories:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    if self._should_index(path):
                        yield path

    def _should_index(self, path: str) -> bool:
        suffix = os.path.splitext(path)[1].lower()
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size == 0:
            return False
        if suffix == ".pdf":
            return size <= 50_000_000
        if size > 10_000_000:
            return False
        return suffix in TEXT_SUFFIXES or suffix == ""

    def _build_catalog_tfidf(self) -> None:
        if not self.files:
            self._catalog_tfidf = None
            self._catalog_matrix = None
            return
        texts = [f"{item['display_name']} {item['rel_path']}" for item in self.files]
        self._catalog_tfidf = TfidfVectorizer(max_features=4096)
        self._catalog_matrix = self._catalog_tfidf.fit_transform(texts)

    def _save_catalog(self) -> None:
        payload = {
            "directories": self.directories,
            "files": self.files,
            "signature": self._current_signature(),
        }
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_catalog(self, requested_dirs: list[str]) -> bool:
        if not os.path.isfile(self.catalog_path):
            return False
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False

        saved_dirs = self._normalize_dirs(payload.get("directories", []))
        if saved_dirs != requested_dirs:
            return False

        files = payload.get("files", [])
        if not isinstance(files, list):
            return False
        self.directories = requested_dirs
        self.files = [f for f in files if isinstance(f, dict) and f.get("path") and f.get("rel_path")]
        self._build_catalog_tfidf()
        return bool(self.files)

    def _invalidate_deep_index(self) -> None:
        self._deep_loaded = False
        self._deep_dirs = []
        self._deep_chunks = []
        self._deep_vectorizer = None
        self._deep_matrix = None
        self._deep_embedding_matrix = None
        self._deep_embedding_dim = None
        for path in (self.deep_meta_path, self.deep_chunks_path, self.deep_index_path, self.deep_embeddings_path):
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass

    def _deep_meta(self) -> dict[str, Any] | None:
        if not os.path.isfile(self.deep_meta_path):
            return None
        try:
            with open(self.deep_meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _deep_matches_catalog(self, directories: list[str], signature: str) -> bool:
        meta = self._deep_meta()
        if not meta:
            return False
        saved_dirs = self._normalize_dirs(meta.get("directories", []))
        saved_signature = str(meta.get("signature", ""))
        saved_version = int(meta.get("index_version", 0))
        saved_kind = str(meta.get("index_kind", ""))
        semantic_meta = meta.get("semantic", {}) if isinstance(meta.get("semantic", {}), dict) else {}
        semantic_enabled = bool(semantic_meta.get("enabled", False))
        return (
            saved_dirs == directories
            and saved_signature == signature
            and saved_version == FILE_RAG_SCHEMA_VERSION
            and saved_kind == FILE_RAG_INDEX_KIND_TFIDF
            and semantic_enabled == self._semantic_enabled()
            and (
                not semantic_enabled
                or str(semantic_meta.get("model", "")) == self._embedding_model_id()
            )
        )

    def deep_index_available(self) -> bool:
        base_ready = (
            os.path.isfile(self.deep_meta_path)
            and os.path.isfile(self.deep_chunks_path)
            and os.path.isfile(self.deep_index_path)
            and self._deep_matches_catalog(self.directories, self._current_signature())
        )
        if not base_ready:
            return False
        if self._semantic_enabled():
            return os.path.isfile(self.deep_embeddings_path)
        return True

    def deep_index_stats(self) -> dict[str, Any]:
        meta = self._deep_meta() or {}
        if not meta:
            return {
                "catalog_documents": len(self.files),
                "deep_documents": 0,
                "chunks_indexed": 0,
                "deep_index_ready": False,
            }
        return {
            "catalog_documents": len(self.files),
            "deep_documents": int(meta.get("files_indexed", 0) or 0),
            "chunks_indexed": int(meta.get("chunks_indexed", 0) or 0),
            "deep_index_ready": self.deep_index_available(),
        }

    def rebuild(self, directories: list[str] | None = None) -> dict[str, Any]:
        target_dirs = self._normalize_dirs(self.directories if directories is None else directories)
        common_root = os.path.commonpath(target_dirs) if target_dirs else ""
        new_files: list[dict[str, Any]] = []
        files_indexed = 0
        for path in self._iter_files(target_dirs):
            rel_path = os.path.relpath(path, start=common_root) if common_root else path
            base = os.path.basename(path)
            display_name = os.path.splitext(base)[0].replace("_", " ").replace("-", " ").strip() or base
            try:
                stat = os.stat(path)
                size = int(stat.st_size)
                mtime_ns = int(stat.st_mtime_ns)
            except OSError:
                size = 0
                mtime_ns = 0
            new_files.append(
                {
                    "path": path,
                    "rel_path": rel_path,
                    "basename": base,
                    "display_name": display_name,
                    "size": size,
                    "mtime_ns": mtime_ns,
                }
            )
            files_indexed += 1

        new_signature = self._catalog_signature(new_files)
        if not self._deep_matches_catalog(target_dirs, new_signature):
            self._invalidate_deep_index()

        self.directories = target_dirs
        self.files = new_files
        self._save_catalog()
        self._build_catalog_tfidf()
        return {
            "directories": list(self.directories),
            "files_indexed": files_indexed,
            "chunks_indexed": 0,
            "deep_index_ready": self.deep_index_available(),
        }

    def document_count(self) -> int:
        return len(self.files)

    def list_documents(self, limit: Optional[int] = None) -> list[str]:
        names = sorted((item["rel_path"] for item in self.files), key=str.casefold)
        if limit is None:
            return names
        return names[: max(0, limit)]

    def search_files(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        if not getattr(self.cfg, "file_rag_enabled", True) or not self.files:
            return []

        top_k = max(1, top_k or getattr(self.cfg, "file_rag_top_k", 4))
        q_norm = self._normalize_text(query)
        q_candidates = self._query_candidates(query)
        ranked: list[tuple[float, dict[str, Any]]] = []

        for item in self.files:
            basename_target = self._normalize_text(item["basename"])
            display_target = self._normalize_text(item["display_name"])
            rel_target = self._normalize_text(item["rel_path"])
            target = self._normalize_text(f"{item['display_name']} {item['rel_path']}")
            score = 0.0
            for candidate in q_candidates:
                if candidate and (candidate == basename_target or candidate == display_target):
                    score = max(score, 10.0)
                elif candidate and (candidate in basename_target or candidate in display_target):
                    score = max(score, 8.0)
                elif candidate and candidate in rel_target:
                    score = max(score, 6.0)
            if q_norm and q_norm in target:
                score = max(score, 3.0)
            elif q_norm:
                q_tokens = [token for token in q_norm.split() if len(token) > 2]
                if q_tokens:
                    hits = sum(1 for token in q_tokens if token in target)
                    score = max(score, hits / len(q_tokens))
            ranked.append((score, item))

        ranked.sort(key=lambda row: row[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, item in ranked:
            if score <= 0 and out:
                break
            out.append(
                {
                    "path": item["path"],
                    "rel_path": item["rel_path"],
                    "display_name": item["display_name"],
                    "basename": item["basename"],
                    "size": int(item.get("size", 0)),
                    "mtime_ns": int(item.get("mtime_ns", 0)),
                    "doc_id": self._doc_id(item),
                    "file_signature": self._file_signature(item),
                    "index_version": FILE_RAG_SCHEMA_VERSION,
                    "score": float(score),
                }
            )
            if len(out) >= top_k:
                break

        if out:
            return out

        if self._catalog_tfidf is None or self._catalog_matrix is None:
            self._build_catalog_tfidf()
        if self._catalog_tfidf is None or self._catalog_matrix is None:
            return []

        q_vec = self._catalog_tfidf.transform([query])
        sims = cosine_similarity(q_vec, self._catalog_matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [
            {
                "path": self.files[int(i)]["path"],
                "rel_path": self.files[int(i)]["rel_path"],
                "display_name": self.files[int(i)]["display_name"],
                "basename": self.files[int(i)]["basename"],
                "size": int(self.files[int(i)].get("size", 0)),
                "mtime_ns": int(self.files[int(i)].get("mtime_ns", 0)),
                "doc_id": self._doc_id(self.files[int(i)]),
                "file_signature": self._file_signature(self.files[int(i)]),
                "index_version": FILE_RAG_SCHEMA_VERSION,
                "score": float(sims[i]),
            }
            for i in idxs
            if sims[i] > 0
        ]

    def _read_pdf_text(self, path: str) -> str:
        try:
            res = subprocess.run(
                ["/usr/bin/pdftotext", "-layout", path, "-"],
                capture_output=True,
                timeout=120,
                check=True,
            )
        except Exception:
            return ""
        return res.stdout.decode("utf-8", errors="ignore").strip()

    def _read_file_text(self, path: str) -> str:
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".pdf":
            return self._read_pdf_text(path)
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            return ""
        if b"\x00" in raw:
            return ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = raw.decode(encoding)
                printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
                if not text or printable / max(1, len(text)) < 0.92:
                    continue
                return text
            except Exception:
                continue
        return ""

    def _chunk_text(self, text: str, *, chunk_chars: int = 1400, overlap: int = 220) -> list[str]:
        clean = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not clean:
            return []
        if len(clean) <= chunk_chars:
            return [clean]
        chunks: list[str] = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + chunk_chars)
            if end < len(clean):
                split = clean.rfind("\n", start + chunk_chars // 2, end)
                if split == -1:
                    split = clean.rfind(" ", start + chunk_chars // 2, end)
                if split > start:
                    end = split
            chunk = clean[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(clean):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _emit_deep_progress(
        self,
        progress_cb: Optional[Callable[..., None]],
        message: str,
        percent: int,
    ) -> None:
        if progress_cb is None:
            return
        bounded = max(0, min(100, int(percent)))
        try:
            progress_cb(message, bounded)
        except TypeError:
            progress_cb(message)

    def _run_with_stage_heartbeat(
        self,
        work: Callable[[], Any],
        *,
        stage_name: str,
        progress_cb: Optional[Callable[..., None]],
        progress_message: str,
        percent: int,
        heartbeat_seconds: float = 15.0,
    ) -> Any:
        stop_event = threading.Event()
        start_time = time.time()

        def heartbeat() -> None:
            while not stop_event.wait(heartbeat_seconds):
                elapsed = int(time.time() - start_time)
                logging.info("%s still running (%ss elapsed).", stage_name, elapsed)
                self._emit_deep_progress(progress_cb, f"{progress_message} ({elapsed}s elapsed)", percent)

        thread = threading.Thread(target=heartbeat, name=f"celeste-{stage_name}-heartbeat", daemon=True)
        thread.start()
        try:
            return work()
        finally:
            stop_event.set()
            thread.join(timeout=0.2)

    def build_deep_index(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        if not self.directories:
            self._invalidate_deep_index()
            return {"directories": [], "files_indexed": 0, "chunks_indexed": 0, "deep_index_ready": False}

        chunks: list[dict[str, Any]] = []
        files_indexed = 0
        total_files = len(self.files)
        stale_tmp_path = self.deep_embeddings_path + ".tmp.npy"
        try:
            if os.path.exists(stale_tmp_path):
                os.unlink(stale_tmp_path)
                logging.info("Removed stale deep-index temp embedding file: %s", stale_tmp_path)
        except OSError:
            logging.warning("Could not remove stale deep-index temp embedding file: %s", stale_tmp_path)
        logging.info("Deep index build starting for %s files across %s directories.", total_files, len(self.directories))
        for idx, item in enumerate(self.files, start=1):
            if progress_cb and (idx == 1 or idx % 25 == 0 or idx == total_files):
                percent = max(1, round((idx / max(total_files, 1)) * 55))
                self._emit_deep_progress(
                    progress_cb,
                    f"Deep indexing file {idx}/{total_files}: {item['rel_path']}",
                    percent,
                )
            text = self._read_file_text(item["path"])
            if not text.strip():
                continue
            for chunk_index, chunk in enumerate(self._chunk_text(text)):
                chunks.append(self._build_chunk_record(item, chunk_index, chunk))
            files_indexed += 1

        logging.info("Deep index chunking complete: %s readable files -> %s chunks.", files_indexed, len(chunks))

        self._emit_deep_progress(
            progress_cb,
            f"Building deep TF-IDF index from {len(chunks)} chunks...",
            60,
        )

        if not chunks:
            self._invalidate_deep_index()
            return {
                "directories": list(self.directories),
                "files_indexed": files_indexed,
                "chunks_indexed": 0,
                "deep_index_ready": False,
            }

        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.92)
        logging.info("Deep index TF-IDF fit_transform starting for %s chunks.", len(chunks))
        matrix = self._run_with_stage_heartbeat(
            lambda: vectorizer.fit_transform([chunk["text"] for chunk in chunks]),
            stage_name="deep-tfidf",
            progress_cb=progress_cb,
            progress_message="Building deep TF-IDF index",
            percent=60,
        )
        logging.info("Deep index TF-IDF fit_transform complete.")
        self._emit_deep_progress(progress_cb, "Deep TF-IDF index complete. Preparing semantic index...", 62)

        semantic_enabled = self._semantic_enabled() and self._ensure_embedder()
        semantic_dim: int | None = None
        if self._semantic_enabled() and not semantic_enabled:
            logging.warning("Semantic retrieval requested, but File RAG embedder was unavailable. Continuing with TF-IDF only.")
            self._emit_deep_progress(
                progress_cb,
                "Semantic retrieval is enabled, but the embedding model could not be loaded. Building TF-IDF only.",
                70,
            )
        if semantic_enabled:
            semantic_dim = int(self.embedder.get_sentence_embedding_dimension())
            logging.info(
                "Deep semantic index starting with device=%s, dim=%s, batch_size=%s.",
                self.device,
                semantic_dim,
                FILE_RAG_SEMANTIC_BATCH_SIZE,
            )
            tmp_embeddings_path = self.deep_embeddings_path + ".tmp.npy"
            embeddings = np.lib.format.open_memmap(
                tmp_embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(chunks), semantic_dim),
            )
            total_batches = max(1, math.ceil(len(chunks) / FILE_RAG_SEMANTIC_BATCH_SIZE))
            for batch_index, start in enumerate(range(0, len(chunks), FILE_RAG_SEMANTIC_BATCH_SIZE), start=1):
                end = min(len(chunks), start + FILE_RAG_SEMANTIC_BATCH_SIZE)
                if batch_index == 1 or batch_index % 25 == 0 or end == len(chunks):
                    logging.info(
                        "Deep semantic batch %s/%s (%s:%s).",
                        batch_index,
                        total_batches,
                        start,
                        end,
                    )
                if progress_cb and (batch_index == 1 or batch_index % 25 == 0 or end == len(chunks)):
                    percent = 65 + round((end / max(len(chunks), 1)) * 30)
                    self._emit_deep_progress(
                        progress_cb,
                        f"Encoding semantic chunk embeddings {batch_index}/{total_batches}...",
                        percent,
                    )
                texts = [chunk["text"] for chunk in chunks[start:end]]
                batch_percent = 65 + round((end / max(len(chunks), 1)) * 30)
                embeddings[start:end] = self._run_with_stage_heartbeat(
                    lambda texts=texts: self._encode_texts(
                        texts,
                        batch_size=min(FILE_RAG_SEMANTIC_BATCH_SIZE, len(texts)),
                    ),
                    stage_name=f"deep-semantic-batch-{batch_index}",
                    progress_cb=progress_cb,
                    progress_message=f"Encoding semantic chunk embeddings {batch_index}/{total_batches}",
                    percent=batch_percent,
                    heartbeat_seconds=20.0,
                )
            embeddings.flush()
            del embeddings
            os.replace(tmp_embeddings_path, self.deep_embeddings_path)
            logging.info("Deep semantic index build complete.")
        else:
            try:
                if os.path.exists(self.deep_embeddings_path):
                    os.unlink(self.deep_embeddings_path)
            except OSError:
                pass

        self._emit_deep_progress(progress_cb, "Finalizing deep index files...", 97)
        logging.info("Deep index finalizing on-disk artifacts.")

        with gzip.open(self.deep_chunks_path, "wt", encoding="utf-8") as f:
            json.dump(chunks, f)
        dump({"vectorizer": vectorizer, "matrix": matrix}, self.deep_index_path)
        with open(self.deep_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "index_version": FILE_RAG_SCHEMA_VERSION,
                    "index_kind": FILE_RAG_INDEX_KIND_TFIDF,
                    "semantic": self._semantic_meta(enabled=semantic_enabled, dim=semantic_dim),
                    "directories": self.directories,
                    "files_indexed": files_indexed,
                    "chunks_indexed": len(chunks),
                    "signature": self._current_signature(),
                },
                f,
                indent=2,
            )

        self._deep_loaded = True
        self._deep_dirs = list(self.directories)
        self._deep_chunks = chunks
        self._deep_vectorizer = vectorizer
        self._deep_matrix = matrix
        self._deep_embedding_dim = semantic_dim
        self._deep_embedding_matrix = (
            np.load(self.deep_embeddings_path, mmap_mode="r") if semantic_enabled and os.path.isfile(self.deep_embeddings_path) else None
        )
        self._release_embedder()

        self._emit_deep_progress(progress_cb, "Deep index build complete.", 100)
        logging.info("Deep index build finished successfully.")

        return {
            "directories": list(self.directories),
            "files_indexed": files_indexed,
            "chunks_indexed": len(chunks),
            "deep_index_ready": True,
            "semantic_index_ready": bool(semantic_enabled),
        }

    def _load_deep_index(self) -> bool:
        if self._deep_loaded:
            return True
        if not self.deep_index_available():
            return False
        try:
            meta = self._deep_meta()
            if not meta:
                return False
            dirs = self._normalize_dirs(meta.get("directories", []))
            if dirs != self.directories:
                return False
            if str(meta.get("signature", "")) != self._current_signature():
                return False
            if int(meta.get("index_version", 0)) != FILE_RAG_SCHEMA_VERSION:
                return False
            if str(meta.get("index_kind", "")) != FILE_RAG_INDEX_KIND_TFIDF:
                return False
            semantic_meta = meta.get("semantic", {}) if isinstance(meta.get("semantic", {}), dict) else {}
            with gzip.open(self.deep_chunks_path, "rt", encoding="utf-8") as f:
                self._deep_chunks = json.load(f)
            if self._deep_chunks and any(
                int(chunk.get("index_version", 0)) != FILE_RAG_SCHEMA_VERSION
                for chunk in self._deep_chunks[:5]
            ):
                return False
            bundle = load(self.deep_index_path)
            self._deep_vectorizer = bundle["vectorizer"]
            self._deep_matrix = bundle["matrix"]
            self._deep_embedding_dim = int(semantic_meta.get("dim")) if semantic_meta.get("dim") is not None else None
            if bool(semantic_meta.get("enabled", False)):
                if not os.path.isfile(self.deep_embeddings_path):
                    return False
                self._deep_embedding_matrix = np.load(self.deep_embeddings_path, mmap_mode="r")
            else:
                self._deep_embedding_matrix = None
            self._deep_dirs = dirs
            self._deep_loaded = True
            return True
        except Exception:
            return False

    def _search_deep_lexical(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._deep_vectorizer is None or self._deep_matrix is None or not self._deep_chunks:
            return []
        q_vec = self._deep_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._deep_matrix)[0]
        idxs = sims.argsort()[::-1][: max(8, top_k * 2)]
        out: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        min_score = (
            FILE_RAG_BROAD_MIN_LEXICAL_SCORE
            if self._is_broad_library_query(query)
            else FILE_RAG_MIN_LEXICAL_SCORE
        )
        for i in idxs:
            score = float(sims[i])
            if score < min_score:
                continue
            chunk = self._deep_chunks[int(i)]
            path = chunk["path"]
            if path in seen_paths and len(out) >= top_k:
                continue
            out.append(self._chunk_hit(chunk, score, FILE_RAG_INDEX_KIND_TFIDF))
            seen_paths.add(path)
            if len(out) >= top_k:
                break
        return out

    def _search_deep_semantic(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._deep_embedding_matrix is None or not self._deep_chunks:
            return []
        query_vec = self._encode_texts([query], batch_size=1)[0]
        sims = np.asarray(self._deep_embedding_matrix @ query_vec, dtype=np.float32)
        idxs = sims.argsort()[::-1][: max(12, top_k * 4)]
        out: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        min_score = (
            FILE_RAG_BROAD_MIN_SEMANTIC_SCORE
            if self._is_broad_library_query(query)
            else FILE_RAG_MIN_SEMANTIC_SCORE
        )
        for i in idxs:
            score = float(sims[int(i)])
            if score < min_score:
                continue
            chunk = self._deep_chunks[int(i)]
            path = chunk["path"]
            if path in seen_paths and len(out) >= top_k:
                continue
            out.append(self._chunk_hit(chunk, score, FILE_RAG_INDEX_KIND_SEMANTIC))
            seen_paths.add(path)
            if len(out) >= top_k:
                break
        return out

    def search_deep(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        if not self._load_deep_index():
            return []
        top_k = max(1, top_k or getattr(self.cfg, "file_rag_top_k", 4))
        effective_query = self._normalize_query_for_retrieval(query)
        lexical_hits = self._search_deep_lexical(effective_query, top_k=max(top_k, 4))
        if self._semantic_enabled() and self._deep_embedding_matrix is not None:
            semantic_hits = self._search_deep_semantic(effective_query, top_k=max(top_k, 4))
            return self._merge_ranked_hits(lexical_hits, semantic_hits, top_k=top_k, query=effective_query)
        return lexical_hits[:top_k]

    def search_document_titles(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        top_k = max(1, top_k or getattr(self.cfg, "file_rag_top_k", 4))
        effective_query = self._normalize_query_for_retrieval(query)
        ranked: dict[str, dict[str, Any]] = {}

        for hit in self.search_deep(effective_query, top_k=max(8, top_k * 3)):
            path = hit["path"]
            current = ranked.get(path)
            score = float(hit.get("score", 0.0)) + 2.0
            if current is None or score > current["score"]:
                ranked[path] = {
                    "path": path,
                    "rel_path": hit["rel_path"],
                    "display_name": hit["display_name"],
                    "score": score,
                    "source": "deep",
                }

        for hit in self.search_files(effective_query, top_k=max(8, top_k * 3)):
            path = hit["path"]
            current = ranked.get(path)
            score = float(hit.get("score", 0.0))
            if current is None or score > current["score"]:
                ranked[path] = {
                    "path": path,
                    "rel_path": hit["rel_path"],
                    "display_name": hit["display_name"],
                    "score": score,
                    "source": "catalog",
                }

        return sorted(ranked.values(), key=lambda item: item["score"], reverse=True)[:top_k]

    def _should_open_file(self, query: str, matches: list[dict[str, Any]]) -> bool:
        if not matches:
            return False
        top_score = float(matches[0].get("score", 0.0))
        second_score = float(matches[1].get("score", 0.0)) if len(matches) > 1 else -1.0
        q = query.lower()
        content_cues = (
            "summar", "what does", "what is in", "contents", "chapter", "quote",
            "find in", "search in", "tell me about", "analyze", "explain", "overview",
        )
        library_cues = (
            "using the library",
            "from the library",
            "library you have access to",
            "indexed documents",
            "indexed files",
            "documents",
            "files",
            "library",
        )
        if any(cue in q for cue in library_cues):
            return top_score >= 10.0 and any(cue in q for cue in content_cues)
        if top_score >= 10.0:
            return True
        if top_score >= 8.0:
            return second_score < 8.0 and (top_score - second_score) >= 1.0
        return any(cue in q for cue in content_cues) and top_score >= 3.0 and (top_score - second_score) >= 1.0

    def get_context(self, query: str, top_k: Optional[int] = None) -> dict[str, Any]:
        effective_query = self._normalize_query_for_retrieval(query)
        matches = self.search_files(effective_query, top_k=top_k)
        result: dict[str, Any] = {
            "matches": matches,
            "opened_file": None,
            "snippets": [],
            "snippet_records": [],
            "library_snippets": [],
            "deep_index_ready": self.deep_index_available(),
        }

        if not self._should_open_file(query, matches):
            result["library_snippets"] = self.search_deep(effective_query, top_k=top_k)
            return result

        target = matches[0]
        text = self._read_file_text(target["path"])
        if not text:
            return result
        chunks = self._chunk_text(text)
        if not chunks:
            return result

        chunk_records = [self._build_chunk_record(target, idx, chunk) for idx, chunk in enumerate(chunks)]
        vec = TfidfVectorizer(max_features=4096).fit(chunks + [effective_query])
        X = vec.transform(chunks)
        qv = vec.transform([effective_query])
        sims = cosine_similarity(qv, X)[0]
        idxs = sims.argsort()[::-1][: max(1, top_k or getattr(self.cfg, "file_rag_top_k", 4))]
        snippet_records = [
            {**chunk_records[int(i)], "score": float(sims[i])}
            for i in idxs
            if sims[i] > 0
        ]
        if not snippet_records:
            snippet_records = [{**record, "score": 0.0} for record in chunk_records[:2]]

        result["opened_file"] = target
        result["snippet_records"] = snippet_records
        result["snippets"] = [record["text"] for record in snippet_records]
        return result
