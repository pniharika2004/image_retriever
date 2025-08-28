import os
import re
import json
import math
import time
import secrets
import hashlib
from typing import Dict, List, Any, Set

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .db import (
    init_db,
    insert_document,
    insert_pages_bulk,
    get_documents as db_get_documents,
    get_document as db_get_document,
    search_pages as db_search_pages,
    is_fts_enabled,
    get_pages_for_doc,
    get_page_text,
)
from .vector_store import (
    init_vector_store,
    index_doc_pages,
    vector_search,
    is_vector_store_ready,
    has_doc as vec_has_doc,
)

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # Will fallback if not available/running

try:
    from pypdf import PdfReader  # modern PyPDF2
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install pypdf (pip install pypdf)") from e


app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
PDFS_DIR = os.path.join(PUBLIC_DIR, "pdfs")
os.makedirs(PDFS_DIR, exist_ok=True)


# In-memory doc store
Document = Dict[str, Any]
documents: Dict[str, Document] = {}
# Initialize SQLite database on startup
@app.on_event("startup")
def _startup() -> None:
    init_db()
    init_vector_store(BASE_DIR)
    # Index any existing PDFs in the public folder if not present in DB
    try:
        for name in os.listdir(PDFS_DIR):
            if not name.lower().endswith(".pdf"):
                continue
            doc_id = os.path.splitext(name)[0]
            # Skip if already present
            if db_get_document(doc_id):
                continue
            pdf_path = os.path.join(PDFS_DIR, name)
            extracted = extract_per_page_text(pdf_path)
            pdf_url = f"/pdfs/{name}"
            documents[doc_id] = {
                "pdf_url": pdf_url,
                "pages": extracted["pages"],
                "idf_by_token": extracted["idf_by_token"],
                "num_pages": extracted["num_pages"],
                "filename": name,
            }
            insert_document(doc_id, name, pdf_url, extracted["num_pages"])
            insert_pages_bulk(doc_id, extracted["pages"])
            # Index into vector store
            try:
                index_doc_pages(doc_id, extracted["pages"])
            except Exception:
                pass
        # Ensure vector index for any docs that are in DB but missing in vectors
        try:
            for r in db_get_documents():
                did = r.get("doc_id")
                if did and not vec_has_doc(did):
                    pages = get_pages_for_doc(did)
                    if pages:
                        index_doc_pages(did, pages)
        except Exception:
            pass
    except Exception:
        # Best-effort indexing; continue startup
        pass



STOPWORDS = {
    'the','a','an','and','or','but','if','then','else','of','in','on','at','to','for','from','by','with','as','is','are','was','were','be','been','it','its','that','this','these','those','we','you','they','he','she','his','her','their','our','your','i'
}


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in cleaned.split() if t and t not in STOPWORDS]


def compute_idf(pages_tokens: List[List[str]]) -> Dict[str, float]:
    num_pages = len(pages_tokens)
    doc_freq: Dict[str, int] = {}
    for tokens in pages_tokens:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    idf: Dict[str, float] = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((num_pages + 1) / (df + 1)) + 1
    return idf


def score_pages(pages: List[Dict[str, Any]], idf_by_token: Dict[str, float], query: str) -> List[Dict[str, Any]]:
    q_tokens = tokenize(query)
    q_set = set(q_tokens)
    scores = []
    for page in pages:
        tf: Dict[str, int] = {}
        for tok in page["tokens"]:
            tf[tok] = tf.get(tok, 0) + 1
        score = 0.0
        for tok in q_set:
            tfv = tf.get(tok, 0)
            if tfv > 0:
                score += (1 + math.log(tfv)) * idf_by_token.get(tok, 1.0)
        scores.append({"page_number": page["page_number"], "score": score})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def build_extractive_summary(pages: List[Dict[str, Any]], page_numbers: List[int], query: str, max_sentences: int = 3) -> str:
    # Collect sentences from the selected pages
    texts: List[str] = []
    for n in page_numbers:
        page = next((p for p in pages if p["page_number"] == n), None)
        if page and page.get("text"):
            texts.append(page["text"]) 
    combined = "\n".join(texts)
    if not combined.strip():
        return ""

    # Split into sentences (very simple splitter)
    sentences = re.split(r"(?<=[\.!?])\s+", combined)
    sentences = [s.strip() for s in sentences if s and len(s.strip()) > 0]
    if not sentences:
        return ""

    # Score sentences by query token overlap (unique tokens)
    q_tokens = set(tokenize(query)) if query else set()
    scored: List[tuple[float, str]] = []
    for s in sentences:
        s_tokens = set(tokenize(s))
        overlap = len(s_tokens & q_tokens) if q_tokens else 0
        # Prefer informative sentences by length (cap length influence)
        length_bonus = min(len(s) / 200.0, 1.0)
        score = overlap * 2.0 + length_bonus
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _score, s in scored[:max_sentences]]
    # If no query overlap produced results, fallback to the first few sentences
    if not any(token in tokenize(" ".join(top_sentences)) for token in q_tokens) and q_tokens:
        top_sentences = sentences[:max_sentences]

    summary = " ".join(top_sentences)
    return summary.strip()


def extract_per_page_text(pdf_path: str) -> Dict[str, Any]:
    reader = PdfReader(pdf_path)
    pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()
        tokens = tokenize(text)
        pages.append({"page_number": idx, "text": text, "tokens": tokens})
    idf_by_token = compute_idf([p["tokens"] for p in pages])
    return {"pages": pages, "idf_by_token": idf_by_token, "num_pages": len(pages)}


def generate_doc_id() -> str:
    return f"{int(time.time()*1000):x}-{secrets.token_hex(4)}"


# API
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    doc_id = generate_doc_id()
    filename = f"{doc_id}.pdf"
    save_path = os.path.join(PDFS_DIR, filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        extracted = extract_per_page_text(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    pdf_url = f"/pdfs/{filename}"
    # In-memory cache for fast per-doc scoring (legacy)
    documents[doc_id] = {
        "pdf_url": pdf_url,
        "pages": extracted["pages"],
        "idf_by_token": extracted["idf_by_token"],
        "num_pages": extracted["num_pages"],
        "filename": filename,
    }

    # Persist to DB for cross-document search
    try:
        insert_document(doc_id, filename, pdf_url, extracted["num_pages"])
        insert_pages_bulk(doc_id, extracted["pages"])
        # Index into vector store
        try:
            index_doc_pages(doc_id, extracted["pages"])
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save to DB: {e}")
    return {"docId": doc_id, "pdfUrl": pdf_url, "numPages": extracted["num_pages"]}


@app.get("/api/docs")
async def list_docs():
    try:
        rows = db_get_documents()
        items = [
            {"docId": r["doc_id"], "pdfUrl": r["pdf_url"], "numPages": r["num_pages"]}
            for r in rows
        ]
        return {"documents": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@app.post("/api/chat")
async def chat(payload: Dict[str, Any]):
    message = (payload or {}).get("message")
    doc_id = (payload or {}).get("docId")
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    # Gather candidate pages either within a document or across all documents
    candidates: List[Dict[str, Any]] = []  # each: {doc_id, page_number, text}
    doc_meta: Dict[str, Any] | None = None
    used_db = False
    backend: str | None = None
    scope: str = "global" if not doc_id else "single_doc"
    search_backend: str | None = None
    if doc_id:
        # Try DB-backed search first
        # Prefer vector store if available
        if is_vector_store_ready():
            try:
                candidates = vector_search(message, top_k=6, restrict_doc_id=doc_id)
                if candidates:
                    used_db = False
                    backend = "vector_faiss"
                    search_backend = backend
            except Exception:
                candidates = []
        if not candidates:
            try:
                candidates = db_search_pages(message, limit=5, restrict_doc_id=doc_id)
                used_db = True
                backend = "fts5" if is_fts_enabled() else "like"
                search_backend = backend
            except Exception:
                candidates = []
        # Fallback to in-memory scoring if DB had no results
        doc = documents.get(doc_id)
        if not candidates and doc:
            ranked = score_pages(doc["pages"], doc["idf_by_token"], message)
            top = [r["page_number"] for r in ranked[:3]] or [1]
            for n in top:
                page = next((p for p in doc["pages"] if p["page_number"] == n), None)
                if page:
                    candidates.append({"doc_id": doc_id, "page_number": n, "text": page.get("text", "")})
        # Load doc metadata for image URLs
        doc_meta = db_get_document(doc_id) or ({"doc_id": doc_id, **documents.get(doc_id, {})} if documents.get(doc_id) else None)
        if not doc_meta:
            raise HTTPException(status_code=404, detail="Unknown docId")
    else:
        # Global search across all documents in DB
        if is_vector_store_ready():
            try:
                candidates = vector_search(message, top_k=6)
                if candidates:
                    used_db = False
                    backend = "vector_faiss"
                    search_backend = backend
            except Exception:
                candidates = []
        if not candidates:
            try:
                candidates = db_search_pages(message, limit=6)
                used_db = True
                backend = "fts5" if is_fts_enabled() else "like"
                search_backend = backend
            except Exception:
                candidates = []
        # If still no candidates, continue and handle with a friendly fallback

    # Build context and image mapping
    doc_to_pages: Dict[str, List[int]] = {}
    context_blocks: List[str] = []
    for row in candidates:
        did = str(row.get("doc_id"))
        pn = int(row.get("page_number", 0))
        if pn <= 0:
            continue
        if did not in doc_to_pages:
            doc_to_pages[did] = []
        if pn not in doc_to_pages[did]:
            doc_to_pages[did].append(pn)
        snippet = (row.get("text") or "")[:4000]
        context_blocks.append(f"Doc {did} - Page {pn}:\n{snippet}")
    if not context_blocks:
        # Soft fallback response when nothing indexed or matched
        return JSONResponse({
            "answer": "I couldn't find relevant content yet. Try uploading a PDF or asking a broader question.",
            "needs_image": False,
            "related_pages": [],
            "pdf_url": None,
            "images": [],
            "search_info": {
                "used_db": used_db,
                "backend": backend,
                "scope": scope,
                "candidate_count": 0,
            },
        })
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided document context. "
        "If an image/figure/diagram/table is directly relevant, set needs_image to true and include related page numbers. "
        "Respond in STRICT JSON only with keys: answer (string), needs_image (boolean), related_pages (array of integers). "
        "Do not include any extra commentary."
    )
    user_prompt = f"Question: {message}\n\nDocument context:\n{context}"

    answer = ""
    needs_image = False

    # For backward compatibility when a single doc is targeted
    first_doc_id = next(iter(doc_to_pages.keys())) if doc_to_pages else (doc_id or "")
    related_pages: List[int] = doc_to_pages.get(first_doc_id, []) if first_doc_id else []

    try:
        if ollama is None:
            raise RuntimeError("ollama not available")
        response = ollama.chat(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.2},
        )
        content = (response.get("message", {}) or {}).get("content", "").strip()
        json_text = content
        m = re.search(r"\{[\s\S]*\}", json_text)
        if m:
            json_text = m.group(0)
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("answer"), str):
                answer = parsed["answer"]
            if isinstance(parsed.get("needs_image"), bool):
                needs_image = parsed["needs_image"]
            # related_pages from model only applies within the first doc for compatibility
            if isinstance(parsed.get("related_pages"), list) and parsed["related_pages"] and first_doc_id:
                rp = []
                # Need doc_meta for num_pages when single-doc
                if doc_meta is None and first_doc_id:
                    doc_meta = db_get_document(first_doc_id)
                max_pages = (doc_meta or {}).get("num_pages", 0)
                for n in parsed["related_pages"]:
                    try:
                        num = int(n)
                        if 1 <= num <= (max_pages or 10_000):
                            rp.append(num)
                    except Exception:
                        continue
                if rp:
                    related_pages = rp[:3]
    except Exception:
        # Fallback to extractive snippet over the candidate pages
        # Construct lightweight pages for summary
        pages_for_summary = [
            {"page_number": int(r.get("page_number", 0)), "text": r.get("text", "")}
            for r in candidates
        ]
        summary = build_extractive_summary(pages_for_summary, [p["page_number"] for p in pages_for_summary if p.get("page_number")][:3], message, max_sentences=3)
        answer = summary or "No clear answer found in the provided pages."
        needs_image = True if doc_to_pages else False

    # Ensure we always provide a non-empty answer
    if not (isinstance(answer, str) and answer.strip()):
        pages_for_summary = [
            {"page_number": int(r.get("page_number", 0)), "text": r.get("text", "")}
            for r in candidates
        ]
        summary = build_extractive_summary(pages_for_summary, [p["page_number"] for p in pages_for_summary if p.get("page_number")][:3], message, max_sentences=3)
        answer = summary or "No clear answer found in the provided pages."

    # Build images payload with all related pages per document (deduped above)
    images: List[Dict[str, Any]] = []
    seen_signatures: Set[str] = set()
    for did, pages in doc_to_pages.items():
        meta = db_get_document(did)
        # If missing in DB, try in-memory cache
        if not meta and documents.get(did):
            meta = {"doc_id": did, "pdf_url": documents[did]["pdf_url"]}
        if not meta:
            continue
        deduped_pages: List[int] = []
        for pn in pages:
            text = get_page_text(did, pn)
            if text is None and documents.get(did):
                # fallback to in-memory cache
                page = next((p for p in documents[did]["pages"] if p["page_number"] == pn), None)
                text = (page or {}).get("text", "")
            norm = re.sub(r"\s+", " ", (text or ""))[:5000].strip().lower()
            sig = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            deduped_pages.append(pn)
        if deduped_pages:
            images.append({"docId": did, "pdf_url": meta.get("pdf_url"), "pages": deduped_pages})

    # Backward compatibility for existing frontend
    pdf_url_out = None
    if doc_id and doc_meta:
        pdf_url_out = doc_meta.get("pdf_url")
    elif images:
        pdf_url_out = images[0].get("pdf_url")

    return JSONResponse({
        "answer": answer,
        "needs_image": needs_image,
        "related_pages": related_pages,
        "pdf_url": pdf_url_out,
        "images": images,
        "search_info": {
            "used_db": used_db,
            "backend": search_backend or backend,
            "scope": scope,
            "candidate_count": len(candidates),
        },
    })


# Static files
# Mount API first, then static at '/'
app.mount("/pdfs", StaticFiles(directory=PDFS_DIR), name="pdfs")
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")


