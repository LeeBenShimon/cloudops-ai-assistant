from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import (
    Flask,
    g,
    jsonify,
    render_template,
    request,
    session
)

from rag_example import (
    DATA_FOLDER,
    TOP_K,
    ask_gemini,
    create_faiss_index,
    embed_texts_with_huggingface,
    load_documents,
    retrieve,
    setup_nltk,
)


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
DATABASE_PATH = INSTANCE_DIR / "chat_memory.sqlite3"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "oz-flask-intro-dev-key")
app.config["SESSION_COOKIE_NAME"] = "oz_flask_chat"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

RAG_LOCK = threading.Lock()
RAG_STATE = {
    "ready": False,
    "chunks": None,
    "index": None,
}
ALLOWED_EXTENSIONS = {"txt", "pdf"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exception: Exception | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def initialize_storage() -> None:
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE_PATH)
    try:
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)")
        db.commit()
    finally:
        db.close()


def get_session_id() -> str:
    session_id = session.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["chat_session_id"] = session_id
    return session_id


def ensure_session_row(session_id: str, title: str = "New conversation") -> None:
    db = get_db()
    now = utc_now()
    db.execute(
        """
        INSERT INTO sessions (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO NOTHING
        """,
        (session_id, title, now, now),
    )
    db.commit()


def update_session_title(session_id: str, title: str) -> None:
    db = get_db()
    db.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, utc_now(), session_id),
    )
    db.commit()


def get_session_meta(session_id: str):
    db = get_db()
    return db.execute(
        "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()


def save_message(session_id: str, role: str, content: str) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, utc_now()),
    )
    db.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (utc_now(), session_id),
    )
    db.commit()


def fetch_messages(session_id: str, limit: int = 12):
    db = get_db()
    rows = db.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return list(reversed(rows))


def format_history(messages) -> str:
    if not messages:
        return "No prior conversation."

    lines = []
    for message in messages:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def derive_session_title(question: str) -> str:
    summary = " ".join(question.split())
    if len(summary) > 48:
        summary = summary[:48].rstrip() + "..."
    return summary or "New conversation"


def load_rag_resources():
    if RAG_STATE["ready"]:
        return RAG_STATE

    with RAG_LOCK:
        if RAG_STATE["ready"]:
            return RAG_STATE

        setup_nltk()
        chunks = load_documents(DATA_FOLDER)
        embeddings = embed_texts_with_huggingface([
            chunk["text"] for chunk in chunks
        ])
        index = create_faiss_index(embeddings)

        RAG_STATE.update(
            {
                "ready": True,
                "chunks": chunks,
                "index": index,
            }
        )

    return RAG_STATE

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    session_id = get_session_id()
    ensure_session_row(session_id)
    session_meta = get_session_meta(session_id)
    messages = fetch_messages(session_id)

    document_count = sum(1 for _ in Path(DATA_FOLDER).glob("*.txt"))

    return render_template(
        "index.html",
        session_id=session_id,
        session_title=session_meta["title"] if session_meta else "New conversation",
        messages=messages,
        message_count=len(messages),
        document_count=document_count,
        retrieval_k=TOP_K,
    )


@app.route("/about")
def about():
    session_id = get_session_id()
    ensure_session_row(session_id)
    session_meta = get_session_meta(session_id)
    document_count = sum(1 for _ in Path(DATA_FOLDER).glob("*.txt"))

    return render_template(
        "about.html",
        session_id=session_id,
        session_title=session_meta["title"] if session_meta else "New conversation",
        document_count=document_count,
        retrieval_k=TOP_K,
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()

    if not question:
        return jsonify({"error": "Message is required."}), 400

    session_id = get_session_id()
    ensure_session_row(session_id)

    save_message(session_id, "user", question)

    try:
        rag_state = load_rag_resources()
        history = fetch_messages(session_id, limit=10)
        history_text = format_history(history[:-1])

        retrieved_chunks = retrieve(
            query=question,
            index=rag_state["index"],
            chunks=rag_state["chunks"],
            k=TOP_K,
    )

        # NO RELEVANT DOCUMENTS
        if not retrieved_chunks:

            answer = "No relevant information found in the knowledge base."

        else:

            context = "\n\n".join([
                chunk["text"] for chunk in retrieved_chunks
            ])

            answer = ask_gemini(
                context,
                question,
                history=history_text
            )
    except Exception as exc:
        return jsonify({"error": f"Unable to answer right now: {exc}"}), 500

    save_message(session_id, "assistant", answer)

    session_meta = get_session_meta(session_id)
    if session_meta and session_meta["title"] == "New conversation":
        update_session_title(session_id, derive_session_title(question))
        session_meta = get_session_meta(session_id)

    return jsonify(
        {
            "answer": answer,
            "retrieved": retrieved_chunks,
            "session_id": session_id,
            "session_title": session_meta["title"] if session_meta else "New conversation",
        }
    )

@app.route("/api/upload", methods=["POST"])
def api_upload():

    if "file" not in request.files:

        return jsonify({
            "error": "No file uploaded."
        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "error": "No file selected."
        }), 400

    if not allowed_file(file.filename):

        return jsonify({
            "error": "Only PDF and TXT files are allowed."
        }), 400

    save_path = os.path.join(DATA_FOLDER, file.filename)

    file.save(save_path)

    # RESET RAG CACHE
    with RAG_LOCK:

        RAG_STATE["ready"] = False
        RAG_STATE["chunks"] = None
        RAG_STATE["index"] = None

    return jsonify({
        "success": True,
        "filename": file.filename
    })

@app.route("/api/files", methods=["GET"])
def api_files():

    files = []

    if os.path.exists(DATA_FOLDER):

        for file_name in os.listdir(DATA_FOLDER):

            if allowed_file(file_name):

                files.append(file_name)

    return jsonify({
        "files": files
    })

@app.route("/api/reset", methods=["POST"])
def api_reset():
    old_session_id = get_session_id()
    new_session_id = str(uuid.uuid4())

    session["chat_session_id"] = new_session_id
    ensure_session_row(new_session_id)

    db = get_db()
    db.execute("DELETE FROM messages WHERE session_id = ?", (old_session_id,))
    db.execute("DELETE FROM sessions WHERE id = ?", (old_session_id,))
    db.commit()

    return jsonify(
        {
            "session_id": new_session_id,
            "session_title": "New conversation",
        }
    )


initialize_storage()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)