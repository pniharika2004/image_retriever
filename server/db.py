import os
import sqlite3
from typing import List, Dict, Any, Optional, Tuple


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "server", "app.db")


HAS_FTS5 = True


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    global HAS_FTS5
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                pdf_url TEXT NOT NULL,
                num_pages INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_doc_page ON pages(doc_id, page_number)")

        # Try to create FTS5 table; if unavailable, fall back later
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                    doc_id, 
                    page_number, 
                    text, 
                    tokenize = 'porter'
                )
                """
            )
            HAS_FTS5 = True
        except sqlite3.OperationalError:
            HAS_FTS5 = False

        conn.commit()
    finally:
        conn.close()


def insert_document(doc_id: str, filename: str, pdf_url: str, num_pages: int) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO documents(doc_id, filename, pdf_url, num_pages)
            VALUES (?, ?, ?, ?)
            """,
            (doc_id, filename, pdf_url, num_pages),
        )
        conn.commit()
    finally:
        conn.close()


def insert_pages_bulk(doc_id: str, pages: List[Dict[str, Any]]) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO pages(doc_id, page_number, text) VALUES (?, ?, ?)",
            [(doc_id, p["page_number"], p.get("text", "")) for p in pages],
        )
        if HAS_FTS5:
            # Insert into FTS as well
            # We need the rowids that were just inserted; simplest is to re-select by doc_id
            cur.execute("SELECT doc_id, page_number, text FROM pages WHERE doc_id = ?", (doc_id,))
            rows = cur.fetchall()
            cur.execute("DELETE FROM pages_fts WHERE doc_id = ?", (doc_id,))
            cur.executemany(
                "INSERT INTO pages_fts(doc_id, page_number, text) VALUES (?, ?, ?)",
                [(r["doc_id"], r["page_number"], r["text"]) for r in rows],
            )
        conn.commit()
    finally:
        conn.close()


def get_documents() -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT doc_id, pdf_url, num_pages FROM documents ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT doc_id, filename, pdf_url, num_pages FROM documents WHERE doc_id = ?", (doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_page_text(doc_id: str, page_number: int) -> Optional[str]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT text FROM pages WHERE doc_id = ? AND page_number = ?", (doc_id, page_number))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_pages_for_doc(doc_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT page_number, text FROM pages WHERE doc_id = ? ORDER BY page_number", (doc_id,))
        rows = cur.fetchall()
        return [{"page_number": int(r["page_number"]), "text": r["text"]} for r in rows]
    finally:
        conn.close()


def search_pages(query: str, limit: int = 5, restrict_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        if HAS_FTS5:
            if restrict_doc_id:
                cur.execute(
                    """
                    SELECT doc_id, page_number, text
                    FROM pages_fts
                    WHERE pages_fts MATCH ? AND doc_id = ?
                    ORDER BY bm25(pages_fts) LIMIT ?
                    """,
                    (query, restrict_doc_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT doc_id, page_number, text
                    FROM pages_fts
                    WHERE pages_fts MATCH ?
                    ORDER BY bm25(pages_fts) LIMIT ?
                    """,
                    (query, limit),
                )
            return [dict(row) for row in cur.fetchall()]
        else:
            # Fallback naive search using LIKE on tokens
            tokens = [t for t in query.split() if t]
            if not tokens:
                return []
            where = []
            params: List[Any] = []
            for t in tokens:
                where.append("text LIKE ?")
                params.append(f"%{t}%")
            where_clause = " OR ".join(where)
            if restrict_doc_id:
                sql = f"SELECT doc_id, page_number, text FROM pages WHERE doc_id = ? AND ({where_clause}) LIMIT ?"
                params = [restrict_doc_id] + params + [limit]
            else:
                sql = f"SELECT doc_id, page_number, text FROM pages WHERE {where_clause} LIMIT ?"
                params = params + [limit]
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def is_fts_enabled() -> bool:
    return HAS_FTS5


