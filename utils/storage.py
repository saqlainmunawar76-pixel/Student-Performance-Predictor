"""
utils/storage.py
Lightweight local JSON storage for prediction history — powers the
Dashboard, History, and stats on the SaaS-style app.
"""

import json
import os
import uuid
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions.json")


def _load():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r") as f:
        return json.load(f)


def _save(records):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(records, f, indent=2)


def add_prediction(profile: dict, outcome: str, probability: float):
    records = _load()
    records.insert(0, {
        "id": str(uuid.uuid4())[:8],
        "profile": profile,
        "outcome": outcome,
        "probability": round(probability * 100, 1),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _save(records)


def get_history():
    return _load()


def clear_history():
    _save([])


def get_stats():
    records = _load()
    total = len(records)
    passes = sum(1 for r in records if r["outcome"] == "Pass")
    avg_prob = round(sum(r["probability"] for r in records) / total, 1) if total else 0
    return {
        "total_predictions": total,
        "pass_count": passes,
        "fail_count": total - passes,
        "avg_probability": avg_prob,
    }
