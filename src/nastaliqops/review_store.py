"""The human in the loop store.

SQLite for phase 1, Postgres later by swapping the connection string. Every
prediction is logged, every correction a reviewer makes becomes a new
labelled row, and that table is what the retraining job reads from.
"""

import json
import sqlite3
import time

from nastaliqops.config import Paths


DB_PATH = Paths.root / "data" / "review.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    image_path TEXT NOT NULL,
    model_variant TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    final_text TEXT NOT NULL,
    confidence REAL NOT NULL,
    route TEXT NOT NULL,
    llm_calls INTEGER NOT NULL,
    needs_review INTEGER NOT NULL,
    reviewed INTEGER NOT NULL DEFAULT 0,
    corrected_text TEXT,
    reviewer_note TEXT
);
"""


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row
    connection.execute(SCHEMA)
    connection.commit()
    return connection


def insert_prediction(record):
    connection = connect()
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO predictions
        (created_at, image_path, model_variant, raw_text, final_text,
         confidence, route, llm_calls, needs_review)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            time.time(),
            record["image_path"],
            record["model_variant"],
            record["raw_text"],
            record["final_text"],
            record["confidence"],
            record["route"],
            record["llm_calls"],
            1 if record["needs_review"] else 0,
        ),
    )
    connection.commit()
    new_id = cursor.lastrowid
    connection.close()
    return new_id


def pending_reviews(limit=50):
    connection = connect()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT * FROM predictions
        WHERE needs_review = 1 AND reviewed = 0
        ORDER BY confidence ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    connection.close()

    output = []
    for row in rows:
        item = {}
        for key in row.keys():
            item[key] = row[key]
        output.append(item)
    return output


def submit_correction(prediction_id, corrected_text, note=""):
    connection = connect()
    cursor = connection.cursor()
    cursor.execute(
        """
        UPDATE predictions
        SET reviewed = 1, corrected_text = ?, reviewer_note = ?
        WHERE id = ?
        """,
        (corrected_text, note, prediction_id),
    )
    connection.commit()
    changed = cursor.rowcount
    connection.close()
    return changed > 0


def stats():
    connection = connect()
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) AS total FROM predictions")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS waiting FROM predictions WHERE needs_review = 1 AND reviewed = 0")
    waiting = cursor.fetchone()["waiting"]

    cursor.execute("SELECT COUNT(*) AS corrected FROM predictions WHERE reviewed = 1")
    corrected = cursor.fetchone()["corrected"]

    cursor.execute("SELECT AVG(confidence) AS mean_confidence FROM predictions")
    mean_confidence = cursor.fetchone()["mean_confidence"]

    cursor.execute("SELECT SUM(llm_calls) AS llm_calls FROM predictions")
    llm_calls = cursor.fetchone()["llm_calls"]

    connection.close()

    result = {}
    result["total_predictions"] = total
    result["awaiting_review"] = waiting
    result["corrections_collected"] = corrected
    if mean_confidence is None:
        result["mean_confidence"] = 0.0
    else:
        result["mean_confidence"] = mean_confidence
    if llm_calls is None:
        result["llm_calls"] = 0
    else:
        result["llm_calls"] = llm_calls
    return result


def export_corrections(out_path):
    """Dump reviewer corrections as a jsonl file the next training run can use."""
    connection = connect()
    cursor = connection.cursor()
    cursor.execute("SELECT image_path, corrected_text FROM predictions WHERE reviewed = 1")
    rows = cursor.fetchall()
    connection.close()

    handle = open(out_path, "w", encoding="utf-8")
    count = 0
    for row in rows:
        record = {}
        record["file_name"] = row["image_path"]
        record["text"] = row["corrected_text"]
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        count = count + 1
    handle.close()
    return count
