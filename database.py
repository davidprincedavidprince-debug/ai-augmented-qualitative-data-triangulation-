"""
SQLite storage for the Programmatic Dissonance app.
Quantitative path is deliberately AI-free (plain writes). Qualitative path
stores an embedding per entry. Dissonance review is a separate table so the
human verification step has its own persistent record, distinct from the
raw similarity computation.
"""

import sqlite3
import json
import datetime

DB_PATH = "dissonance.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Indicators: the bridge between quant and qual. target_description is
    # the plain-language statement of what success looks like, and is the
    # ONLY thing that gets embedded on the indicator side.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        target_description TEXT,
        target_description_embedding TEXT,
        target_value REAL,
        unit TEXT,
        created_at TEXT
    )""")

    # Quantitative entries: no AI involved at all, plain structured numbers.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quant_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        indicator_id INTEGER,
        value REAL,
        source TEXT,           -- e.g. "field visit month 3", "ODK form batch 2"
        entry_method TEXT,     -- "excel_upload" | "manual_entry"
        created_at TEXT,
        FOREIGN KEY (indicator_id) REFERENCES indicators(id)
    )""")

    # Qualitative entries: beneficiary narratives, embedded on arrival.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS qual_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        embedding TEXT,
        source TEXT,
        created_at TEXT
    )""")

    # Dissonance reviews: one row per (indicator, qual_entry) pair the human
    # has actually looked at. Nothing here is written until a human acts.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dissonance_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        indicator_id INTEGER,
        qual_entry_id INTEGER,
        similarity_score REAL,
        status TEXT,            -- "confirmed_dissonance" | "dismissed_as_noise" | "needs_more_info"
        human_note TEXT,
        reviewed_at TEXT,
        FOREIGN KEY (indicator_id) REFERENCES indicators(id),
        FOREIGN KEY (qual_entry_id) REFERENCES qual_entries(id)
    )""")

    conn.commit()
    conn.close()


def reset_db():
    conn = get_connection()
    cur = conn.cursor()
    for table in ["dissonance_reviews", "qual_entries", "quant_entries", "indicators"]:
        cur.execute(f"DELETE FROM {table}")
        cur.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,))
    conn.commit()
    conn.close()


def _now():
    return datetime.datetime.utcnow().isoformat()


# ---------- indicators ----------

def insert_indicator(name, target_description, target_description_embedding, target_value, unit):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO indicators (name, target_description, target_description_embedding, "
        "target_value, unit, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (name, target_description, json.dumps(target_description_embedding), target_value, unit, _now()),
    )
    conn.commit()
    iid = cur.lastrowid
    conn.close()
    return iid


def list_indicators():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM indicators ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_indicator(indicator_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM indicators WHERE id = ?", (indicator_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------- quantitative entries (no AI) ----------

def insert_quant_entry(indicator_id, value, source, entry_method):
    conn = get_connection()
    conn.execute(
        "INSERT INTO quant_entries (indicator_id, value, source, entry_method, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (indicator_id, value, source, entry_method, _now()),
    )
    conn.commit()
    conn.close()


def get_quant_entries(indicator_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM quant_entries WHERE indicator_id = ? ORDER BY id DESC", (indicator_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- qualitative entries ----------

def insert_qual_entry(text, embedding, source):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO qual_entries (text, embedding, source, created_at) VALUES (?, ?, ?, ?)",
        (text, json.dumps(embedding), source, _now()),
    )
    conn.commit()
    qid = cur.lastrowid
    conn.close()
    return qid


def list_qual_entries():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM qual_entries ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- dissonance reviews (human verification) ----------

def upsert_review(indicator_id, qual_entry_id, similarity_score, status, human_note):
    """One review row per (indicator, qual_entry) pair — replaces any prior
    review of the same pair so re-reviewing overwrites rather than duplicates."""
    conn = get_connection()
    cur = conn.cursor()
    existing = cur.execute(
        "SELECT id FROM dissonance_reviews WHERE indicator_id = ? AND qual_entry_id = ?",
        (indicator_id, qual_entry_id),
    ).fetchone()
    if existing:
        cur.execute(
            "UPDATE dissonance_reviews SET similarity_score = ?, status = ?, human_note = ?, "
            "reviewed_at = ? WHERE id = ?",
            (similarity_score, status, human_note, _now(), existing["id"]),
        )
    else:
        cur.execute(
            "INSERT INTO dissonance_reviews (indicator_id, qual_entry_id, similarity_score, "
            "status, human_note, reviewed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (indicator_id, qual_entry_id, similarity_score, status, human_note, _now()),
        )
    conn.commit()
    conn.close()


def get_review(indicator_id, qual_entry_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM dissonance_reviews WHERE indicator_id = ? AND qual_entry_id = ?",
        (indicator_id, qual_entry_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_reviews_for_indicator(indicator_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM dissonance_reviews WHERE indicator_id = ?", (indicator_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
