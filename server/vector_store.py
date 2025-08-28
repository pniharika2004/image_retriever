import os
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # type: ignore
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # type: ignore


class VectorStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.store_dir = os.path.join(base_dir, "server", "vectorstore")
        os.makedirs(self.store_dir, exist_ok=True)
        self.index_path = os.path.join(self.store_dir, "index.faiss")
        self.meta_path = os.path.join(self.store_dir, "meta.jsonl")
        self.index: Any = None
        self.dim: Optional[int] = None
        self.next_id: int = 1
        self.meta_by_id: Dict[int, Dict[str, Any]] = {}

    def _save_meta(self) -> None:
        tmp = os.path.join(self.store_dir, "meta.tmp.jsonl")
        with open(tmp, "w", encoding="utf-8") as f:
            for _id, meta in self.meta_by_id.items():
                rec = {"id": _id, **meta}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, self.meta_path)

    def _load_meta(self) -> None:
        self.meta_by_id.clear()
        if not os.path.exists(self.meta_path):
            return
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = int(rec.pop("id"))
                self.meta_by_id[rid] = rec
        if self.meta_by_id:
            self.next_id = max(self.meta_by_id.keys()) + 1

    def load_or_init(self) -> None:
        if faiss is None or np is None:
            return
        self._load_meta()
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            try:
                # If not ID mapped, wrap it
                if not isinstance(self.index, faiss.IndexIDMap2):
                    idmap = faiss.IndexIDMap2(self.index)
                    self.index = idmap
            except Exception:
                pass
            # Try to infer dim
            try:
                self.dim = self.index.d
            except Exception:
                self.dim = None
        else:
            self.index = None
            self.dim = None

    def save(self) -> None:
        if faiss is None or self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        self._save_meta()

    def _ensure_index(self, dim: int) -> None:
        if self.index is not None:
            return
        # Cosine similarity via inner product on normalized vectors
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(base)
        self.dim = dim

    def _normalize_rows(self, x: "np.ndarray") -> "np.ndarray":
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def _embed(self, texts: List[str]) -> Optional["np.ndarray"]:
        if ollama is None or np is None:
            return None
        model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        vecs: List[List[float]] = []
        for t in texts:
            try:
                resp = ollama.embeddings(model=model, prompt=t)
                e = (resp or {}).get("embedding")
                if isinstance(e, list):
                    vecs.append(e)
            except Exception:
                return None
        if not vecs:
            return None
        arr = np.array(vecs, dtype="float32")
        arr = self._normalize_rows(arr)
        return arr

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == n:
                break
            start = max(end - overlap, start + 1)
        return chunks

    def index_document(self, doc_id: str, pages: List[Dict[str, Any]]) -> int:
        if faiss is None or np is None:
            return 0
        added = 0
        for page in pages:
            page_num = int(page.get("page_number", 0))
            text = page.get("text", "")
            chunks = self._chunk_text(text)
            if not chunks:
                continue
            emb = self._embed(chunks)
            if emb is None:
                return added
            self._ensure_index(emb.shape[1])
            ids = []
            for _ in range(emb.shape[0]):
                ids.append(self.next_id)
                self.next_id += 1
            id_arr = np.array(ids, dtype="int64")
            self.index.add_with_ids(emb, id_arr)
            for cid, chunk_text in zip(ids, chunks):
                self.meta_by_id[cid] = {
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "text": chunk_text[:1200],
                }
            added += len(ids)
        if added:
            self.save()
        return added

    def search(self, query: str, top_k: int = 6, restrict_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if faiss is None or np is None or self.index is None or not query.strip():
            return []
        q = self._embed([query])
        if q is None:
            return []
        D, I = self.index.search(q, top_k * 4)
        ids = I[0]
        results: List[Dict[str, Any]] = []
        seen: set[Tuple[str, int]] = set()
        for cid in ids:
            if int(cid) <= 0:
                continue
            meta = self.meta_by_id.get(int(cid))
            if not meta:
                continue
            if restrict_doc_id and meta.get("doc_id") != restrict_doc_id:
                continue
            key = (str(meta.get("doc_id")), int(meta.get("page_number", 0)))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "doc_id": meta.get("doc_id"),
                "page_number": int(meta.get("page_number", 0)),
                "text": meta.get("text", ""),
            })
            if len(results) >= top_k:
                break
        return results

    def has_doc(self, doc_id: str) -> bool:
        for meta in self.meta_by_id.values():
            if meta.get("doc_id") == doc_id:
                return True
        return False


_store: Optional[VectorStore] = None


def init_vector_store(base_dir: str) -> None:
    global _store
    vs = VectorStore(base_dir)
    vs.load_or_init()
    _store = vs


def index_doc_pages(doc_id: str, pages: List[Dict[str, Any]]) -> int:
    if _store is None:
        return 0
    return _store.index_document(doc_id, pages)


def vector_search(query: str, top_k: int = 6, restrict_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if _store is None:
        return []
    return _store.search(query, top_k=top_k, restrict_doc_id=restrict_doc_id)


def is_vector_store_ready() -> bool:
    return _store is not None and _store.index is not None


def has_doc(doc_id: str) -> bool:
    if _store is None:
        return False
    return _store.has_doc(doc_id)


