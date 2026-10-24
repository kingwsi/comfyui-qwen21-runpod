"""Persist independent generation jobs and uploaded asset references."""
from contextlib import contextmanager
import json
import sqlite3
import time
import uuid
from pathlib import Path


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'studio.sqlite3'
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs
                  (id TEXT PRIMARY KEY, owner TEXT NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, updated);
                CREATE TABLE IF NOT EXISTS assets
                  (id TEXT PRIMARY KEY, owner TEXT NOT NULL, path TEXT NOT NULL, input_name TEXT);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, owner, data):
        data['updated'] = time.time()
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,data=excluded.data WHERE jobs.owner=excluded.owner',
                       (data['id'], owner, data['updated'], json.dumps(data, ensure_ascii=False)))

    def get(self, owner, jid):
        with self.connect() as db:
            row = db.execute('SELECT data FROM jobs WHERE id=? AND owner=?', (jid, owner)).fetchone()
        if row is None:
            raise KeyError(jid)
        return json.loads(row[0])

    def latest(self, owner):
        with self.connect() as db:
            row = db.execute('SELECT id FROM jobs WHERE owner=? ORDER BY updated DESC LIMIT 1', (owner,)).fetchone()
        return row[0] if row else None

    def add_asset(self, owner, path, input_name=None):
        aid = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO assets VALUES (?,?,?,?)', (aid, owner, str(path), input_name))
        return aid

    def asset(self, owner, aid):
        with self.connect() as db:
            row = db.execute('SELECT path,input_name FROM assets WHERE id=? AND owner=?', (aid, owner)).fetchone()
        if row is None:
            raise KeyError(aid)
        return dict(path=row[0], input_name=row[1])
