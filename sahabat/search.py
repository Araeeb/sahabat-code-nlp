"""Semantic search over the project catalogue (Lab 5).

Two stages: FAISS retrieval with multilingual sentence embeddings, then an
optional cross-encoder re-rank. Embeddings are L2-normalised on BOTH the index
and the query side, so inner product = cosine similarity.
The index is saved with a manifest that pins the model, data and preprocessing.
"""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from . import config
from .preprocessing import PREPROC_VERSION, preprocess


class ProjectSearch:
    def __init__(self, embedder=None):
        from sentence_transformers import SentenceTransformer
        self.embedder = embedder or SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = None
        self.catalog = None
        self._reranker = None

    def _encode(self, texts):
        return np.asarray(self.embedder.encode(list(texts), normalize_embeddings=True), dtype="float32")

    def build(self, catalog_df):
        import faiss
        self.catalog = catalog_df.reset_index(drop=True)
        vectors = self._encode(self.catalog["clean_text"])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self

    def save(self, folder=config.INDEX_DIR):
        import faiss
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "projects.faiss"))
        self.catalog.to_csv(folder / "catalog.csv", index=False)
        digest = hashlib.sha256("\n".join(self.catalog["clean_text"]).encode()).hexdigest()[:16]
        manifest = {"embedding_model": config.EMBEDDING_MODEL, "preproc_version": PREPROC_VERSION,
                    "size": int(self.index.ntotal), "catalog_sha256": digest,
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    @classmethod
    def load(cls, folder=config.INDEX_DIR):
        import faiss
        import pandas as pd
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["preproc_version"] != PREPROC_VERSION:
            raise RuntimeError("Index built with another preprocessing version. Rebuild it with train.py.")
        obj = cls()
        obj.index = faiss.read_index(str(folder / "projects.faiss"))
        obj.catalog = pd.read_csv(folder / "catalog.csv")
        return obj

    def search(self, query, k=5, rerank=False):
        clean = preprocess(query)
        scores, ids = self.index.search(self._encode([clean]), k)
        hits = self.catalog.iloc[ids[0]].copy()
        hits["score"] = scores[0]
        if rerank:
            from sentence_transformers import CrossEncoder
            self._reranker = self._reranker or CrossEncoder(config.RERANKER_MODEL)
            hits["score"] = self._reranker.predict([[clean, t] for t in hits["clean_text"]])
            hits = hits.sort_values("score", ascending=False)
        return hits.reset_index(drop=True)
