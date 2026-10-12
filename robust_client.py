"""
ДЗ 2.3 — надёжный клиент LLM API.

RobustLLMClient — обёртка над OpenAI SDK:
  - перехват 429, 5xx, таймаутов и обрывов соединения;
  - retry: экспоненциальная задержка 1 → 2 → 4 → 8 → 16 с + jitter, до 5 повторов
    на провайдера (своя реализация — так в логе видна каждая попытка);
    заголовок Retry-After от сервера имеет приоритет;
  - circuit breaker оборачивает вызов провайдера целиком (вместе с ретраями)
    и считает неудачные ЗАПРОСЫ, а не попытки — иначе он открывался бы на
    3-й попытке и ретраи 8 с / 16 с никогда бы не случились;
  - fallback-цепочка: OpenAI → OpenRouter → «Сервис временно недоступен»
    (провайдер без ключа пропускается);
  - логирование: timestamp, провайдер, код ошибки, номер попытки, задержка;
  - учёт usage: prompt/completion токены и стоимость в $ по каждому провайдеру;
  - бонус: circuit breaker (3 ошибки подряд → провайдер пропускается 60 с);
  - бонус: rate limiter на стороне клиента (N запросов в минуту).

Ошибки 400/401/403/404 не ретраятся: повтор не поможет, сразу идём к
следующему провайдеру.
"""

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import openai

from llm_config import Provider, cost_usd, get_provider

FALLBACK_MESSAGE = "Сервис временно недоступен, попробуйте позже."

RETRYABLE = (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError)

log = logging.getLogger("robust_client")


@dataclass
class LLMResult:
    content: str
    provider: str | None
    model: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    attempts: int = 0
    degraded: bool = False  # True — ответ-заглушка, ни один провайдер не ответил


@dataclass
class CircuitBreaker:
    """Closed → (threshold ошибок подряд) → Open на cooldown секунд → снова пробуем."""

    threshold: int = 3
    cooldown_sec: float = 60.0
    failures: int = 0
    opened_at: float | None = None

    def allow(self, now: float) -> bool:
        if self.opened_at is None:
            return True
        if now - self.opened_at >= self.cooldown_sec:
            self.opened_at = None  # half-open: даём одну попытку
            self.failures = self.threshold - 1  # новая ошибка сразу снова откроет
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self, now: float) -> bool:
        """Возвращает True, если breaker только что открылся."""
        self.failures += 1
        if self.failures >= self.threshold and self.opened_at is None:
            self.opened_at = now
            return True
        return False


class RateLimiter:
    """Скользящее окно: не больше max_per_minute запросов за последние 60 с."""

    def __init__(self, max_per_minute: int, clock=time.monotonic, sleep=time.sleep):
        self.max_per_minute = max_per_minute
        self._calls: deque[float] = deque()
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                if len(self._calls) < self.max_per_minute:
                    self._calls.append(now)
                    return
                wait = 60 - (now - self._calls[0])
                log.info("rate_limiter wait=%.2fs (лимит %d/мин)", wait, self.max_per_minute)
                self._sleep(wait)


@dataclass
class UsageStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class RobustLLMClient:
    def __init__(
        self,
        providers: list[Provider] | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 16.0,
        timeout: float = 30.0,
        cb_threshold: int = 3,
        cb_cooldown_sec: float = 60.0,
        rate_limit_per_minute: int | None = None,
        http_clients: dict | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        # По умолчанию — цепочка из задания: OpenAI → OpenRouter.
        chain = providers or [get_provider("openai"), get_provider("openrouter")]
        self.providers = [p for p in chain if p.configured]
        for p in chain:
            if not p.configured:
                log.info("provider=%s пропущен: нет ключа в .env", p.name)

        http_clients = http_clients or {}
        # max_retries=0: ретраи делаем сами, иначе SDK молча повторит ещё 2 раза.
        self._clients = {
            p.name: p.client(timeout=timeout, max_retries=0, http_client=http_clients.get(p.name))
            for p in self.providers
        }
        self.breakers = {p.name: CircuitBreaker(cb_threshold, cb_cooldown_sec) for p in self.providers}
        self.usage = {p.name: UsageStats() for p in self.providers}
        self.max_retries, self.base_delay, self.max_delay = max_retries, base_delay, max_delay
        self.rate_limiter = RateLimiter(rate_limit_per_minute, clock, sleep) if rate_limit_per_minute else None
        self._sleep, self._clock = sleep, clock

    def backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """attempt=1 → ~1 с, 2 → ~2 с, 3 → ~4 с, 4 → ~8 с, 5+ → ~16 с (+ jitter до 25 %)."""
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        delay = min(self.base_delay * 2 ** (attempt - 1), self.max_delay)
        return delay + random.uniform(0, delay * 0.25)

    def chat(self, messages: list[dict], **params) -> LLMResult:
        if self.rate_limiter:
            self.rate_limiter.acquire()

        total_attempts = 0
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allow(self._clock()):
                log.warning("provider=%s circuit=OPEN — пропускаем провайдера", provider.name)
                continue

            max_attempts = 1 + self.max_retries
            for attempt in range(1, max_attempts + 1):
                total_attempts += 1
                try:
                    response = self._clients[provider.name].chat.completions.create(
                        model=provider.model, messages=messages, **params
                    )
                except RETRYABLE as e:
                    code = getattr(e, "status_code", None) or type(e).__name__
                    if attempt == max_attempts:
                        log.error("provider=%s attempt=%d error=%s → повторы исчерпаны", provider.name, attempt, code)
                        break
                    delay = self.backoff_delay(attempt, _retry_after(e))
                    log.warning("provider=%s attempt=%d error=%s retry_in=%.2fs", provider.name, attempt, code, delay)
                    self._sleep(delay)
                    continue
                except openai.APIStatusError as e:  # 4xx — повтор бессмыслен
                    log.error("provider=%s attempt=%d error=%s (не ретраим)", provider.name, attempt, e.status_code)
                    break

                breaker.record_success()
                return self._record(provider, response, total_attempts)

            if breaker.record_failure(self._clock()):
                log.error("provider=%s %d неудачных запросов подряд → circuit OPEN на %.0f с",
                          provider.name, breaker.threshold, breaker.cooldown_sec)
            log.warning("provider=%s недоступен → fallback на следующего", provider.name)

        log.error("все провайдеры недоступны → ответ-заглушка")
        return LLMResult(content=FALLBACK_MESSAGE, provider=None, model=None, attempts=total_attempts, degraded=True)

    def _record(self, provider: Provider, response, attempts: int) -> LLMResult:
        usage = response.usage
        pt, ct = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
        cost = cost_usd(provider.model, pt, ct)
        stats = self.usage[provider.name]
        stats.requests += 1
        stats.prompt_tokens += pt
        stats.completion_tokens += ct
        stats.cost_usd += cost
        log.info("provider=%s model=%s ok attempts=%d tokens=%d/%d cost=$%.6f",
                 provider.name, provider.model, attempts, pt, ct, cost)
        return LLMResult(
            content=response.choices[0].message.content or "",
            provider=provider.name,
            model=provider.model,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=cost,
            attempts=attempts,
        )

    def usage_report(self) -> dict:
        report = {name: vars(s).copy() for name, s in self.usage.items()}
        report["total"] = {
            "requests": sum(s.requests for s in self.usage.values()),
            "prompt_tokens": sum(s.prompt_tokens for s in self.usage.values()),
            "completion_tokens": sum(s.completion_tokens for s in self.usage.values()),
            "cost_usd": round(sum(s.cost_usd for s in self.usage.values()), 6),
        }
        return report


def _retry_after(e: Exception) -> float | None:
    response = getattr(e, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value else None
    except ValueError:
        return None


def setup_logging(path: str = "logs/robust_client.log") -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
