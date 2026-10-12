"""Transactional operational state. Immutable artifacts remain the authority."""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
import psutil

STATES = {"planned", "blocked", "ready", "running", "interrupted", "failed", "unsuccessful", "complete", "reviewed"}


class Ledger:
    def __init__(self, path, matrix_id):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS identity(matrix_id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS cases(case_id TEXT PRIMARY KEY, status TEXT NOT NULL, completion TEXT, reason TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, case_id TEXT REFERENCES cases(case_id),
                occurred REAL NOT NULL, status TEXT NOT NULL, details TEXT NOT NULL);
        """)
        found = self.db.execute("SELECT matrix_id FROM identity").fetchall()
        if found and found != [(matrix_id,)]:
            self.db.close()
            raise ValueError("Ledger belongs to another matrix")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO identity VALUES (?)", (matrix_id,))

    def register(self, cases):
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO cases VALUES (?, 'planned', NULL, NULL)", ((k,) for k in cases))

    def transition(self, key, status, completion=None, reason=None, **details):
        if status not in STATES:
            raise ValueError("Invalid scheduler state")
        if reason is not None and not isinstance(reason, str):
            reason = json.dumps(reason, allow_nan=False)
        with self.db:
            changed = self.db.execute("UPDATE cases SET status=?,completion=?,reason=? WHERE case_id=?",
                                      (status, completion, reason, key)).rowcount
            if changed != 1:
                raise ValueError("Unregistered experiment case")
            self.db.execute("INSERT INTO events(case_id,occurred,status,details) VALUES (?,?,?,?)",
                            (key, time.time(), status, json.dumps({"completion": completion, "reason": reason, **details}, allow_nan=False)))

    def rows(self):
        return {k: {"status": s, "completion": p, "reason": r} for k, s, p, r in self.db.execute("SELECT * FROM cases ORDER BY case_id")}

    def close(self): self.db.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


@contextmanager
def scheduler_lock(path, recover=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        owner = json.loads(path.read_text(encoding="utf-8"))
        def alive(pid, created):
            try:
                return psutil.Process(pid).create_time() == created
            except psutil.NoSuchProcess:
                return False
        if alive(owner["pid"], owner["created"]) or (owner.get("child") and alive(**owner["child"])):
            raise ValueError("An experiment scheduler or its worker is still active")
        if not recover:
            raise ValueError("Stale scheduler ownership requires explicit recovery")
        path.unlink()
    owner = {"pid": os.getpid(), "created": psutil.Process().create_time(), "child": None}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    def bind(child):
        try:
            owner["child"] = {"pid": child.pid, "created": psutil.Process(child.pid).create_time()} if child else None
        except psutil.NoSuchProcess:
            owner["child"] = None
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(owner), encoding="utf-8")
        os.replace(temporary, path)
    try:
        yield bind
    finally:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if current and current["pid"] == owner["pid"] and current["created"] == owner["created"]:
            path.unlink()
