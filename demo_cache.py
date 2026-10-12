"""
ДЗ 2.4 — демонстрация LLMCache и RedisLLMCache.

Для каждого бэкенда:
  1. 3 разных вопроса — промахи, ответ берётся у LLM;
  2. те же 3 вопроса ещё раз — попадания, ответ из кеша (сравниваем время);
  3. тот же вопрос с другой temperature — другой ключ, промах;
  4. TTL: кеш с TTL 2 с, ждём 3 с — запись просрочена, промах;
  5. stats() с hit rate.

Запуск (нужен Redis: docker run -d --name m2-redis -p 127.0.0.1:6379:6379 redis:7.4-alpine):
    python demo_cache.py | tee logs/2.4_run.log
"""

import json
import os
import time

from llm_cache import LLMCache, RedisLLMCache
from llm_config import default_provider

provider = default_provider()
client = provider.client(timeout=120)

QUESTIONS = [
    "Что такое финансовая подушка безопасности? Одно предложение.",
    "Как вести учёт расходов? Одно предложение.",
    "Что такое кешбэк? Одно предложение.",
]


def cached_chat(cache, question: str, temperature: float = 0) -> tuple[str, bool, float]:
    messages = [{"role": "user", "content": question}]
    start = time.perf_counter()
    cached = cache.get(provider.model, messages, temperature)
    if cached is not None:
        return cached["content"], True, time.perf_counter() - start
    response = client.chat.completions.create(
        model=provider.model, messages=messages, temperature=temperature, max_tokens=80
    )
    content = response.choices[0].message.content or ""
    cache.set(provider.model, messages, temperature, {"content": content})
    return content, False, time.perf_counter() - start


def run(cache, title: str) -> dict:
    print(f"\n######## {title} ########")
    for round_no in (1, 2):
        print(f"--- проход {round_no} ---")
        for q in QUESTIONS:
            answer, hit, elapsed = cached_chat(cache, q)
            print(f"  [{'HIT ' if hit else 'MISS'}] {elapsed * 1000:8.1f} мс | {q[:40]}… → {answer.strip()[:60]}")

    answer, hit, elapsed = cached_chat(cache, QUESTIONS[0], temperature=0.7)
    print(f"  [{'HIT ' if hit else 'MISS'}] {elapsed * 1000:8.1f} мс | тот же вопрос, temperature=0.7 → другой ключ")

    stats = cache.stats()
    print("stats():", json.dumps(stats, ensure_ascii=False))
    return stats


def ttl_demo(cache_cls, **kwargs) -> None:
    cache = cache_cls(ttl_seconds=2, **kwargs)
    cached_chat(cache, QUESTIONS[1])
    _, hit_now, _ = cached_chat(cache, QUESTIONS[1])
    time.sleep(3)
    _, hit_later, _ = cached_chat(cache, QUESTIONS[1])
    print(f"TTL=2 с ({cache_cls.__name__}): сразу → {'HIT' if hit_now else 'MISS'}, через 3 с → {'HIT' if hit_later else 'MISS (просрочено)'}")


def main() -> None:
    print(f"Провайдер: {provider.name}, модель: {provider.model}")
    results = {"memory": run(LLMCache(), "Уровень 1: LLMCache (in-memory)")}

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_cache = RedisLLMCache(redis_url, prefix="demo2.4:")
    redis_cache.clear()  # демо с чистого листа
    results["redis"] = run(redis_cache, "Уровень 2: RedisLLMCache")

    print("\n######## TTL ########")
    ttl_demo(LLMCache)
    ttl_demo(RedisLLMCache, url=redis_url, prefix="demo2.4-ttl:")

    with open("logs/2.4_stats.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
