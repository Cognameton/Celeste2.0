# memory.py — Celeste vector memory
# - GPU-first SentenceTransformers embeddings (fallback to CPU)
# - Explicit Chroma usage (no Chroma embedding wrapper)
# - Only index selected kinds to avoid prompt bloat
# - JSON log fallback + TF-IDF retrieval when vector DB unavailable

import os
import json
import uuid
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config_types import AgentConfig

# Keep tokenizer threads quiet
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Only these kinds are stored in the vector DB
INDEXABLE_KINDS = {"note", "user"}


class MemoryPipeline:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

        # Paths
        os.makedirs(self.cfg.data_dir, exist_ok=True)
        self.json_path = os.path.join(self.cfg.data_dir, "memory.json")
        if getattr(self.cfg, "use_chroma", False):
            os.makedirs(getattr(self.cfg, "persist_dir", self.cfg.data_dir), exist_ok=True)

        # State
        self.use_chroma = False
        self.client = None
        self.collection = None
        self.embedder = None
        self.dim: Optional[int] = None
        self.device = "cpu"

        # Ensure JSON store exists
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump([], f)

        # Vector store + embedder
        if getattr(self.cfg, "use_chroma", False):
            try:
                # Detect CUDA for embeddings (Torch 2.6+cu124 in your venv)
                try:
                    import torch as _torch  # noqa: F401
                    self.device = "cuda" if _torch.cuda.is_available() else "cpu"
                except Exception:
                    self.device = "cpu"

                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(self.cfg.embedding_model, device=self.device)
                self.dim = int(self.embedder.get_sentence_embedding_dimension())

                import chromadb
                self.client = chromadb.PersistentClient(path=self.cfg.persist_dir)
                self.collection = self.client.get_or_create_collection(name="agent_mem")
                self.use_chroma = True
                print(f"[memory] vector store ready (device={self.device}, dim={self.dim})")
            except Exception as e:
                print(f"[memory] Chroma unavailable or embedder load failed ({e}). Falling back to JSON+TFIDF.")

    # ---- Helpers ----
    def _append_json(self, text: str, kind: str, metadata: Dict[str, Any]):
        """Append to the JSON log (always)."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
        data.append({"text": text, "kind": kind, "meta": metadata})
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode with SentenceTransformers on GPU if available, then L2-normalize.
        Returns float32 numpy on CPU (what Chroma expects).
        """
        # encode() runs on self.device internally; result is a CPU numpy array
        vec = self.embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalize ourselves for explicit control
            show_progress_bar=False,
        )
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return (vec / norms).astype(np.float32)

    # ---- Public API ----
    def add(self, text: str, kind: str = "note", metadata: Optional[Dict[str, Any]] = None):
        """
        Persist an item to the JSON log, and (if enabled) index in Chroma when kind is indexable.
        """
        metadata = metadata or {}

        # Always log to JSON for transparency/audit
        self._append_json(text, kind, metadata)

        # Index selected kinds in vector DB
        if self.use_chroma and self.embedder is not None and self.collection is not None:
            if kind in INDEXABLE_KINDS:
                try:
                    _id = str(uuid.uuid4())
                    vec = self._encode([text])
                    self.collection.add(
                        ids=[_id],
                        documents=[text],
                        metadatas=[{"kind": kind, **metadata}],
                        embeddings=vec.tolist(),
                    )
                except Exception as e:
                    print(f"[memory] Chroma add failed ({e}). Continuing with JSON only.")

    def search(self, query: str, top_k: int = 5, kinds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search vector DB first (filtered by kinds if provided), otherwise fall back to JSON+TFIDF.
        Returns a list of {text, score, meta}.
        """
        kinds = kinds or list(INDEXABLE_KINDS)

        # Vector path
        if self.use_chroma and self.embedder is not None and self.collection is not None:
            try:
                qv = self._encode([query]).tolist()
                where = {"kind": {"$in": kinds}} if kinds else None
                res = self.collection.query(query_embeddings=qv, n_results=top_k, where=where)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                out: List[Dict[str, Any]] = []
                for d, m, dist in zip(docs, metas, dists):
                    score = (1.0 - dist) if dist is not None else 0.0
                    out.append({"text": d, "score": float(score), "meta": m or {}})
                return out
            except Exception as e:
                print(f"[memory] Chroma query failed ({e}). Falling back to JSON+TFIDF.")

        # JSON fallback path (TF-IDF cosine)
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []

        # Filter by kind to avoid assistant/meta bloat
        if kinds:
            data = [d for d in data if d.get("kind") in kinds]

        if not data:
            return []

        texts = [d["text"] for d in data]
        vec = TfidfVectorizer(max_features=4096).fit(texts + [query])
        X = vec.transform(texts)
        q = vec.transform([query])
        sims = cosine_similarity(q, X)[0]
        idxs = sims.argsort()[::-1][:top_k]
        out: List[Dict[str, Any]] = []
        for i in idxs:
            out.append({"text": texts[i], "score": float(sims[i]), "meta": data[i].get("meta", {})})
        return out
