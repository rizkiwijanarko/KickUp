"""
SQLite Evidence Cache — Persists mined raw evidence with TTL to prevent redundant scraping.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Sequence

from src.mining.provider import RawEvidence
from src.models.common import DataSource

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DB_PATH = ".cache/ventureforge.db"
DEFAULT_CACHE_TTL_SECONDS = 86400.0  # 24 hours


class SQLiteEvidenceCache:
    """SQLite-backed cache for RawEvidence keyed by domain with TTL expiration."""

    def __init__(self, db_path: str = DEFAULT_CACHE_DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(db_file), check_same_thread=False)

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mined_evidence_cache (
                        domain TEXT NOT NULL,
                        source TEXT NOT NULL,
                        raw_json TEXT NOT NULL,
                        mined_at REAL NOT NULL,
                        PRIMARY KEY (domain, source)
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"[SQLiteEvidenceCache] Failed to initialize cache table: {e}")

    def get(
        self,
        domain: str,
        max_age_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> list[RawEvidence] | None:
        """Retrieve cached evidence for a domain if fresher than max_age_seconds."""
        domain_key = domain.strip().lower()
        now = time.time()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT raw_json, mined_at FROM mined_evidence_cache
                    WHERE domain = ?
                    """,
                    (domain_key,),
                )
                rows = cursor.fetchall()
                if not rows:
                    return None

                results: list[RawEvidence] = []
                for raw_json, mined_at in rows:
                    if now - mined_at > max_age_seconds:
                        logger.info(f"[SQLiteEvidenceCache] Cache expired for domain '{domain}'.")
                        return None

                    items_data = json.loads(raw_json)
                    for item in items_data:
                        source_val = item.get("source", "web")
                        try:
                            source_enum = DataSource(source_val)
                        except ValueError:
                            source_enum = DataSource.WEB

                        results.append(
                            RawEvidence(
                                text=item.get("text", ""),
                                url=item.get("url", ""),
                                source=source_enum,
                                title=item.get("title", ""),
                                author=item.get("author", ""),
                                score=item.get("score", 0),
                                metadata=item.get("metadata", {}),
                            )
                        )

                if results:
                    logger.info(
                        f"[SQLiteEvidenceCache] Cache HIT for domain='{domain}' "
                        f"({len(results)} items loaded from SQLite)."
                    )
                    return results

        except Exception as e:
            logger.warning(f"[SQLiteEvidenceCache] Error reading cache for domain '{domain}': {e}")

        return None

    def set(self, domain: str, evidence: Sequence[RawEvidence]) -> None:
        """Save evidence items grouped by source for a domain with current timestamp."""
        if not evidence:
            return

        domain_key = domain.strip().lower()
        now = time.time()

        # Group by source
        by_source: dict[str, list[dict]] = {}
        for item in evidence:
            src_name = item.source.value if hasattr(item.source, "value") else str(item.source)
            item_dict = {
                "text": item.text,
                "url": item.url,
                "source": src_name,
                "title": item.title,
                "author": item.author,
                "score": item.score,
                "metadata": item.metadata,
            }
            by_source.setdefault(src_name, []).append(item_dict)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for src_name, items_list in by_source.items():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO mined_evidence_cache (domain, source, raw_json, mined_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (domain_key, src_name, json.dumps(items_list), now),
                    )
                conn.commit()
                logger.info(
                    f"[SQLiteEvidenceCache] Saved {len(evidence)} evidence items for domain='{domain}'."
                )
        except Exception as e:
            logger.warning(f"[SQLiteEvidenceCache] Error writing cache for domain '{domain}': {e}")

    def clear(self, domain: str | None = None) -> None:
        """Clear cache entries for a specific domain or entire cache if domain is None."""
        try:
            with self._get_connection() as conn:
                if domain is None:
                    conn.execute("DELETE FROM mined_evidence_cache")
                else:
                    conn.execute(
                        "DELETE FROM mined_evidence_cache WHERE domain = ?",
                        (domain.strip().lower(),),
                    )
                conn.commit()
        except Exception as e:
            logger.warning(f"[SQLiteEvidenceCache] Error clearing cache: {e}")
