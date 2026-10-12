"""
ДЗ 2.4 — кеширование ответов LLM.

Уровень 1 — LLMCache (in-memory):
  get(model, messages, temperature) -> ответ | None
  ключ — SHA-256 от model + temperature + messages (JSON с sort_keys,
  поэтому порядок полей в словарях на ключ не влияет);
  TTL (по умолчанию 1 час) — просроченные записи не возвращаются и удаляются;
  статистика hits / misses, stats() -> hit rate в процентах.

Уровень 2 — RedisLLMCache: тот же интерфейс, хранение в Redis через redis-py,
сериализация в JSON, TTL — средствами Redis (SETEX). Счётчики hits/misses
тоже в Redis (INCR), поэтому статистика общая для всех процессов.

Что кешировать: детерминированные ответы (temperature=0) на одинаковые
запросы. При temperature > 0 кеш «замораживает» один из случайных вариантов —
для FAQ-бота это приемлемо, для генерации идей — нет.
"""

import hashlib
import json
import time

import redis


def make_key(model: str, messages: list[dict], temperature: float) -> str:
    payload = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hit_rate(hits: int, misses: int) -> float:
    total = hits + misses
    return round(hits / total * 100, 1) if total else 0.0


class LLMCache:
    def __init__(self, ttl_seconds: float = 3600, clock=time.monotonic):
        self.ttl = ttl_seconds
        self._clock = clock
        self._store: dict[str, tuple[float, dict]] = {}  # key -> (expires_at, value)
        self.hits = 0
        self.misses = 0

    def get(self, model: str, messages: list[dict], temperature: float) -> dict | None:
        key = make_key(model, messages, temperature)
        entry = self._store.get(key)
        if entry is not None and entry[0] > self._clock():
            self.hits += 1
            return entry[1]
        if entry is not None:  # протухла — удаляем, чтобы не копить мусор
            del self._store[key]
        self.misses += 1
        return None

    def set(self, model: str, messages: list[dict], temperature: float, value: dict) -> None:
        self._store[make_key(model, messages, temperature)] = (self._clock() + self.ttl, value)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        return {
            "backend": "memory",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": _hit_rate(self.hits, self.misses),
            "size": len(self._store),
        }


class RedisLLMCache:
    def __init__(self, url: str = "redis://localhost:6379/0", ttl_seconds: int = 3600, prefix: str = "llmcache:"):
        self.ttl = int(ttl_seconds)
        self.prefix = prefix
        self.redis = redis.Redis.from_url(url, decode_responses=True)

    def get(self, model: str, messages: list[dict], temperature: float) -> dict | None:
        raw = self.redis.get(self.prefix + make_key(model, messages, temperature))
        if raw is None:
            self.redis.incr(self.prefix + "stats:misses")
            return None
        self.redis.incr(self.prefix + "stats:hits")
        return json.loads(raw)

    def set(self, model: str, messages: list[dict], temperature: float, value: dict) -> None:
        key = self.prefix + make_key(model, messages, temperature)
        self.redis.setex(key, self.ttl, json.dumps(value, ensure_ascii=False))

    def clear(self) -> None:
        for key in self.redis.scan_iter(self.prefix + "*"):
            self.redis.delete(key)

    def stats(self) -> dict:
        hits = int(self.redis.get(self.prefix + "stats:hits") or 0)
        misses = int(self.redis.get(self.prefix + "stats:misses") or 0)
        size = sum(1 for k in self.redis.scan_iter(self.prefix + "*") if ":stats:" not in k)
        return {"backend": "redis", "hits": hits, "misses": misses, "hit_rate_pct": _hit_rate(hits, misses), "size": size}


def make_cache(backend: str, ttl_seconds: float, redis_url: str):
    if backend == "redis":
        return RedisLLMCache(redis_url, int(ttl_seconds))
    if backend == "memory":
        return LLMCache(ttl_seconds)
    raise ValueError(f"CACHE_BACKEND должен быть memory или redis, получено {backend!r}")
