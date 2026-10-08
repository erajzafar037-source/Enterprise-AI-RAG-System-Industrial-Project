import hashlib,json
from redis import Redis
from .config import settings
from .metrics import CACHE_HITS,CACHE_MISSES
class Cache:
    def __init__(self): self.client=Redis.from_url(settings.redis_url,decode_responses=True,socket_connect_timeout=1)
    def key(self,question): return "rag:v1:"+hashlib.sha256(question.strip().lower().encode()).hexdigest()
    def get(self,question):
        try:
            value=self.client.get(self.key(question))
            if value: CACHE_HITS.inc(); return json.loads(value)
            CACHE_MISSES.inc()
        except Exception: CACHE_MISSES.inc()
        return None
    def set(self,question,value,ttl=900):
        try: self.client.setex(self.key(question),ttl,json.dumps(value))
        except Exception: pass
