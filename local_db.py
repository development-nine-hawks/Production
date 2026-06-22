"""
local_db.py — Offline-first local database fallback.

Mimics the pymongo API surface used by PhoneCDP so all app code works
transparently when MongoDB Atlas is unreachable (no network).

Storage: one JSON file per collection in config.DATA_DIR/localdb/.
Thread safety: a per-collection file lock (threading.Lock) protects
concurrent reads/writes within a single process.

Supported pymongo methods:
  Collection.find(filter, projection)  → list of dicts
  Collection.find_one(filter, projection) → dict | None
  Collection.insert_one(doc)
  Collection.find_one_and_update(filter, update, upsert, return_document)
  Collection.count_documents(filter)
  Collection.delete_one(filter)
  Collection.update_one(filter, update)
  Collection.create_index(...)  → no-op (no indexes needed locally)
"""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from datetime import datetime
from typing import Any

import config


# ── helpers ───────────────────────────────────────────────────────────────────

def _localdb_dir() -> str:
    d = os.path.join(config.DATA_DIR, "localdb")
    os.makedirs(d, exist_ok=True)
    return d


def _match(doc: dict, filt: dict) -> bool:
    """Very small subset of MongoDB query semantics."""
    for k, v in filt.items():
        if k not in doc:
            return False
        if isinstance(v, dict):
            dv = doc[k]
            if "$in" in v and dv not in v["$in"]:
                return False
            if "$gt" in v and not (dv > v["$gt"]):
                return False
            if "$lt" in v and not (dv < v["$lt"]):
                return False
        else:
            if doc[k] != v:
                return False
    return True


def _apply_projection(doc: dict, projection: dict | None) -> dict:
    if not projection:
        return doc
    exclude_id = projection.get("_id") == 0
    include_keys = {k for k, v in projection.items() if v and k != "_id"}
    if include_keys:
        result = {k: doc[k] for k in include_keys if k in doc}
    else:
        result = {k: v for k, v in doc.items() if k != "_id"}
    if exclude_id:
        result.pop("_id", None)
    return result


def _serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


def _deserialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__datetime__" in obj:
            return datetime.fromisoformat(obj["__datetime__"])
        return {k: _deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize(i) for i in obj]
    return obj


# ── Collection ────────────────────────────────────────────────────────────────

class LocalCollection:
    def __init__(self, name: str):
        self._name = name
        self._path = os.path.join(_localdb_dir(), f"{name}.json")
        self._lock = threading.Lock()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _read(self) -> list[dict]:
        if not os.path.isfile(self._path):
            return []
        with open(self._path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return _deserialize(raw)

    def _write(self, docs: list[dict]) -> None:
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_serialize(docs), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)

    # ── pymongo-compatible API ────────────────────────────────────────────────

    def create_index(self, *args, **kwargs):
        """No-op — local storage doesn't need indexes."""
        pass

    def find(self, filt: dict | None = None, projection: dict | None = None):
        filt = filt or {}
        with self._lock:
            docs = self._read()
        results = [_apply_projection(deepcopy(d), projection)
                   for d in docs if _match(d, filt)]
        return _LocalCursor(results)

    def find_one(self, filt: dict | None = None, projection: dict | None = None):
        filt = filt or {}
        with self._lock:
            docs = self._read()
        for d in docs:
            if _match(d, filt):
                return _apply_projection(deepcopy(d), projection)
        return None

    def insert_one(self, doc: dict):
        with self._lock:
            docs = self._read()
            docs.append(deepcopy(doc))
            self._write(docs)

    def find_one_and_update(self, filt: dict, update: dict, *,
                            upsert: bool = False, return_document=None):
        """Supports $inc and $set operators."""
        from pymongo import ReturnDocument
        after = (return_document == ReturnDocument.AFTER)
        with self._lock:
            docs = self._read()
            for i, d in enumerate(docs):
                if _match(d, filt):
                    before = deepcopy(d)
                    _apply_update(docs[i], update)
                    self._write(docs)
                    return deepcopy(docs[i]) if after else before
            if upsert:
                new_doc = dict(filt)
                _apply_update(new_doc, update)
                docs.append(new_doc)
                self._write(docs)
                return deepcopy(new_doc) if after else None
        return None

    def delete_one(self, filt: dict):
        with self._lock:
            docs = self._read()
            for i, d in enumerate(docs):
                if _match(d, filt):
                    docs.pop(i)
                    self._write(docs)
                    return
    
    def update_one(self, filt: dict, update: dict, *, upsert: bool = False):
        with self._lock:
            docs = self._read()
            for i, d in enumerate(docs):
                if _match(d, filt):
                    _apply_update(docs[i], update)
                    self._write(docs)
                    return
            if upsert:
                new_doc = dict(filt)
                _apply_update(new_doc, update)
                docs.append(new_doc)
                self._write(docs)

    def count_documents(self, filt: dict | None = None) -> int:
        filt = filt or {}
        with self._lock:
            docs = self._read()
        return sum(1 for d in docs if _match(d, filt))


def _apply_update(doc: dict, update: dict) -> None:
    if "$inc" in update:
        for k, v in update["$inc"].items():
            doc[k] = doc.get(k, 0) + v
    if "$set" in update:
        for k, v in update["$set"].items():
            doc[k] = v


# ── Cursor ────────────────────────────────────────────────────────────────────

class _LocalCursor:
    def __init__(self, docs: list[dict]):
        self._docs = docs
        self._sort_key = None
        self._sort_dir = 1
        self._limit_n = None
        self._skip_n = 0

    def sort(self, key: str | list, direction: int = 1):
        if isinstance(key, list):
            # pymongo-style list of (key, direction) tuples
            self._sort_key, self._sort_dir = key[0]
        else:
            self._sort_key, self._sort_dir = key, direction
        return self

    def limit(self, n: int):
        self._limit_n = n
        return self

    def skip(self, n: int):
        self._skip_n = n
        return self

    def __iter__(self):
        docs = self._docs
        if self._sort_key:
            docs = sorted(docs,
                          key=lambda d: d.get(self._sort_key) or 0,
                          reverse=(self._sort_dir == -1))
        docs = docs[self._skip_n:]
        if self._limit_n is not None:
            docs = docs[:self._limit_n]
        return iter(docs)

    def __len__(self):
        return len(list(self.__iter__()))


# ── Database ──────────────────────────────────────────────────────────────────

class LocalDatabase:
    """Mimics a pymongo Database object with collection attribute access."""

    def __init__(self):
        self._collections: dict[str, LocalCollection] = {}

    def __getattr__(self, name: str) -> LocalCollection:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._collections:
            self._collections[name] = LocalCollection(name)
        return self._collections[name]

    def command(self, cmd: str):
        """Stub for admin.command('ping') used in init_db health check."""
        return {"ok": 1}
