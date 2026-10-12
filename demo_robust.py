"""
ДЗ 2.3 — демонстрация retry и fallback у RobustLLMClient.

Сбои внедряются через FaultTransport (подменный httpx-транспорт): он отвечает
заданными кодами 429/500/503/401 или таймаутом, а когда сценарий сбоев
закончился — пропускает запрос в настоящую сеть. Так сценарии
воспроизводимы и не зависят от того, лежит ли сейчас провайдер на самом деле.

Цепочка как в задании: OpenAI → OpenRouter → «Сервис временно недоступен».
«OpenAI» в сценариях всегда отвечает внедрённой ошибкой, поэтому реальный
ключ OpenAI для демо не нужен. Настоящие ответы даёт OpenRouter
(если нет его ключа — локальный Ollama).

Запуск:
    python demo_robust.py
Лог retry-последовательности: logs/robust_client.log
"""

import json
import logging
import os
from collections import deque

import httpx2 as httpx  # openai SDK 3.x работает поверх httpx2 (API совместим с httpx)

from llm_config import Provider, get_provider
from robust_client import RateLimiter, RobustLLMClient, setup_logging

log = logging.getLogger("demo")

MESSAGES = [{"role": "user", "content": "Ответь одним словом: какой цвет у неба днём?"}]
PARAMS = {"max_tokens": 10, "temperature": 0}


class FaultTransport(httpx.BaseTransport):
    """Первые запросы — по сценарию faults, затем (или всегда, если always задан) — сбой/реальная сеть."""

    def __init__(self, faults: list | None = None, always: int | str | None = None):
        self.faults = deque(faults or [])
        self.always = always
        self.inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        fault = self.faults.popleft() if self.faults else self.always
        if fault is None:
            return self.inner.handle_request(request)
        if fault == "timeout":
            raise httpx.ReadTimeout("injected timeout", request=request)
        return httpx.Response(fault, json={"error": {"message": f"injected HTTP {fault}"}}, request=request)


def fake_openai() -> Provider:
    return Provider("openai", "https://api.openai.com/v1", os.getenv("OPENAI_API_KEY") or "sk-demo-not-used", "gpt-4o-mini")


def real_provider() -> Provider:
    provider = get_provider("openrouter")
    return provider if provider.configured else get_provider("ollama")


def client(chain: list[Provider], transports: dict, **kwargs) -> RobustLLMClient:
    http_clients = {name: httpx.Client(transport=t) for name, t in transports.items()}
    return RobustLLMClient(providers=chain, http_clients=http_clients, **kwargs)


def scenario(title: str) -> None:
    log.info("=" * 20 + " %s " + "=" * 20, title)


def main() -> None:
    setup_logging()
    real = real_provider()
    log.info("Реальный провайдер в демо: %s (%s)", real.name, real.model)

    scenario("1. Временные сбои: 429, 503 → успех на 3-й попытке (задержки ~1 с, ~2 с)")
    c = client([real], {real.name: FaultTransport(faults=[429, 503])})
    r = c.chat(MESSAGES, **PARAMS)
    log.info("ответ=%r provider=%s attempts=%d", r.content, r.provider, r.attempts)

    scenario("2. Таймаут → retry → успех")
    c = client([real], {real.name: FaultTransport(faults=["timeout"])})
    r = c.chat(MESSAGES, **PARAMS)
    log.info("ответ=%r provider=%s attempts=%d", r.content, r.provider, r.attempts)

    scenario("3. OpenAI лежит (500): полная серия 1→2→4→8→16 с → fallback на OpenRouter")
    c = client([fake_openai(), real], {"openai": FaultTransport(always=500)})
    r = c.chat(MESSAGES, **PARAMS)
    log.info("ответ=%r provider=%s attempts=%d", r.content, r.provider, r.attempts)

    scenario("4. 401 у OpenAI — не ретраим, сразу fallback")
    c = client([fake_openai(), real], {"openai": FaultTransport(always=401)})
    r = c.chat(MESSAGES, **PARAMS)
    log.info("ответ=%r provider=%s attempts=%d", r.content, r.provider, r.attempts)

    scenario("5. Circuit breaker: 3 неудачных запроса подряд → OpenAI пропускается 60 с (задержки ускорены)")
    c = client([fake_openai(), real], {"openai": FaultTransport(always=503)}, base_delay=0.05, max_retries=2)
    for i in range(1, 5):
        r = c.chat(MESSAGES, **PARAMS)
        log.info("запрос %d: provider=%s attempts=%d", i, r.provider, r.attempts)

    scenario("6. Все провайдеры лежат → сообщение-заглушка (задержки ускорены)")
    c = client(
        [fake_openai(), real],
        {"openai": FaultTransport(always=503), real.name: FaultTransport(always=503)},
        base_delay=0.05,
    )
    r = c.chat(MESSAGES, **PARAMS)
    log.info("ответ=%r degraded=%s attempts=%d", r.content, r.degraded, r.attempts)

    scenario("7. Rate limiter: 3 запроса в минуту, 5 вызовов (виртуальное время, без ожидания)")
    now = [0.0]
    limiter = RateLimiter(3, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    for i in range(1, 6):
        limiter.acquire()
        log.info("вызов %d пропущен в t=%.1f с", i, now[0])

    scenario("8. Учёт usage и стоимости: 3 реальных запроса без сбоев")
    usage = client_usage_demo(real)
    print(json.dumps(usage, ensure_ascii=False, indent=2))
    with open("logs/2.3_usage.json", "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)


def client_usage_demo(real: Provider) -> dict:
    """3 реальных запроса без сбоев — показывает накопление токенов и $."""
    c = RobustLLMClient(providers=[real])
    for q in ("Сколько будет 2+2?", "Назови столицу Франции.", "Какой сегодня день недели, если вчера был вторник?"):
        c.chat([{"role": "user", "content": q}], max_tokens=20, temperature=0)
    return c.usage_report()


if __name__ == "__main__":
    main()
