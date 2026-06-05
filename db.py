"""
SQLite persistence layer for JARVIS.
Single shared connection guarded by a lock (FastAPI runs sync endpoints in a
threadpool, so check_same_thread=False + a lock is the safe minimal setup).
Holds: watchlist (Priority 1), positions (Priority 6).
"""
import os
import sqlite3
import threading

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(_DATA_DIR, "jarvis.db")

_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row

with _lock:
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol   TEXT NOT NULL UNIQUE,
            name     TEXT,
            sector   TEXT,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP,
            sort_order INTEGER DEFAULT 0,
            notes    TEXT
        )
        """
    )
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol     TEXT NOT NULL,
            shares     REAL NOT NULL,
            avg_cost   REAL NOT NULL,
            entry_date TEXT NOT NULL,
            notes      TEXT,
            added_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _conn.commit()


def get_config(key: str, default: str = "") -> str:
    with _lock:
        row = _conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_config(key: str, value: str) -> None:
    with _lock:
        _conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?,?)", (key, value))
        _conn.commit()


def list_watchlist():
    """Return all watchlist rows, newest first within the same sort_order."""
    with _lock:
        rows = _conn.execute(
            "SELECT symbol, name, sector, added_at, notes "
            "FROM watchlist ORDER BY sort_order ASC, added_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def add_watchlist(symbol, name=None, sector=None, notes=None):
    """Insert a symbol (idempotent — INSERT OR IGNORE on the UNIQUE symbol)."""
    symbol = symbol.strip().upper()
    if not symbol:
        return list_watchlist()
    with _lock:
        _conn.execute(
            "INSERT OR IGNORE INTO watchlist (symbol, name, sector, notes) "
            "VALUES (?, ?, ?, ?)",
            (symbol, name, sector, notes),
        )
        _conn.commit()
    return list_watchlist()


def remove_watchlist(symbol):
    """Delete a symbol from the watchlist."""
    symbol = symbol.strip().upper()
    with _lock:
        _conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        _conn.commit()
    return list_watchlist()


def in_watchlist(symbol):
    symbol = symbol.strip().upper()
    with _lock:
        row = _conn.execute(
            "SELECT 1 FROM watchlist WHERE symbol = ?", (symbol,)
        ).fetchone()
    return row is not None


# ── Positions ────────────────────────────────────────────────────────────────

def list_positions():
    with _lock:
        rows = _conn.execute(
            "SELECT id, symbol, shares, avg_cost, entry_date, notes, added_at "
            "FROM positions ORDER BY added_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def add_position(symbol: str, shares: float, avg_cost: float, entry_date: str, notes: str | None = None) -> int:
    symbol = symbol.strip().upper()
    with _lock:
        cur = _conn.execute(
            "INSERT INTO positions (symbol, shares, avg_cost, entry_date, notes) VALUES (?, ?, ?, ?, ?)",
            (symbol, shares, avg_cost, entry_date, notes),
        )
        _conn.commit()
    return cur.lastrowid


def update_position(id_: int, shares: float, avg_cost: float, entry_date: str, notes: str | None = None):
    with _lock:
        _conn.execute(
            "UPDATE positions SET shares=?, avg_cost=?, entry_date=?, notes=? WHERE id=?",
            (shares, avg_cost, entry_date, notes, id_),
        )
        _conn.commit()


def delete_position(id_: int):
    with _lock:
        _conn.execute("DELETE FROM positions WHERE id=?", (id_,))
        _conn.commit()
