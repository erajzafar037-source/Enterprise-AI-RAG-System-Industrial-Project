"""Best-effort Redis response cache with index-version isolation."""
import hashlib
import json

from redis import Redis

from .config import settings
from .metrics import CACHE_HITS, CACHE_MISSES


class Cache:
    def __init__(self):
        self.client = Redis.from_url(
            settings.redis_url, decode_responses=True, socket_connect_timeout=1
        )

    def key(self, question: str, top_k: int = 4, index_version: str = "local") -> str:
        value = f"{index_version}:{top_k}:{question.strip().lower()}"
        return "rag:v2:" + hashlib.sha256(value.encode()).hexdigest()

    def get(self, question: str, top_k: int = 4, index_version: str = "local"):
        try:
            value = self.client.get(self.key(question, top_k, index_version))
            if value:
                CACHE_HITS.inc()
                return json.loads(value)
        except Exception:
            # Redis is an optimization, not a dependency for answering questions.
            pass
        CACHE_MISSES.inc()
        return None

    def set(
        self, question: str, value: dict, ttl: int = 900,
        top_k: int = 4, index_version: str = "local",
    ) -> None:
        try:
            self.client.setex(
                self.key(question, top_k, index_version), ttl, json.dumps(value)
            )
        except Exception:
            pass
