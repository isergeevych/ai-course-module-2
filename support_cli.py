"""
ДЗ 2.5 — ИИ-помощник техподдержки «Финпомощника» (CLI).

Собирает блоки 2.1–2.4:
  - system prompt по РРФО + 4 few-shot примера (2.2, support_prompt.py);
  - retry + fallback (2.3, RobustLLMClient): OpenAI → OpenRouter → локальный
    Ollama → «Сервис временно недоступен»; провайдеры без ключа пропускаются;
  - кеш (2.4): CACHE_BACKEND=memory|redis, hit rate в /stats;
  - история: последние 10 сообщений идут в контекст;
  - команды: /clear, /stats, /help, /quit.
Бонусы: классификация FAQ / Техническая проблема / Жалоба (модель ставит
метку по правилам промпта), эскалация «Передаю вопрос специалисту»
(+ запись в logs/escalations.jsonl), лог в файл logs/support_cli.log.

Ключ кеша — вся отправляемая переписка (system + история + вопрос) при
temperature=0. Поэтому из кеша приходит только ответ на тот же вопрос в
том же контексте: одинаковый первый вопрос в разных сессиях (после /clear)
— попадание; «а как это сделать?» в другом контексте — промах, а не чужой ответ.

Запуск:
    python support_cli.py
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone

from llm_cache import make_cache
from llm_config import get_provider
from robust_client import RobustLLMClient
from support_prompt import build_system_prompt

HISTORY_LIMIT = 10
TEMPERATURE = 0
CACHE_MODEL_KEY = "support-cli/v1"  # логическое имя: ответ может прийти от любого провайдера цепочки
CATEGORY_RE = re.compile(r"\[Категория:\s*(FAQ|Техническая проблема|Жалоба)\]\s*$")
ESCALATION_PHRASE = "передаю вопрос специалисту"

HELP = "Команды: /clear — очистить историю, /stats — статистика, /help — помощь, /quit — выход."

log = logging.getLogger("support_cli")


def setup_logging() -> None:
    handler = logging.FileHandler("logs/support_cli.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


class SupportAssistant:
    def __init__(self):
        self.system_prompt = build_system_prompt(with_examples=True)
        self.client = RobustLLMClient(
            providers=[get_provider("openai"), get_provider("openrouter"), get_provider("ollama")],
            max_retries=2,  # интерактивный режим: 1 + 2 + 4 с максимум, пользователь не ждёт по 30 с
        )
        self.cache = make_cache(
            os.getenv("CACHE_BACKEND", "memory"),
            float(os.getenv("CACHE_TTL_SECONDS", "3600")),
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        )
        self.history: deque[dict] = deque(maxlen=HISTORY_LIMIT)
        self.categories: Counter = Counter()
        self.escalations = 0
        self.turns = 0

    def ask(self, question: str) -> tuple[str, str | None, bool, str]:
        """Возвращает (ответ, категория, из_кеша, источник)."""
        messages = [{"role": "system", "content": self.system_prompt}, *self.history,
                    {"role": "user", "content": question}]
        start = time.perf_counter()

        cached = self.cache.get(CACHE_MODEL_KEY, messages, TEMPERATURE)
        if cached is not None:
            answer, source, from_cache = cached["content"], f"кеш ({cached['provider']})", True
        else:
            result = self.client.chat(messages, temperature=TEMPERATURE, max_tokens=400)
            answer, source, from_cache = result.content, result.provider or "заглушка", False
            if not result.degraded:  # заглушку «сервис недоступен» не кешируем
                self.cache.set(CACHE_MODEL_KEY, messages, TEMPERATURE, {"content": answer, "provider": result.provider})

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self.turns += 1

        match = CATEGORY_RE.search(answer.strip())
        category = match.group(1) if match else None
        if category:
            self.categories[category] += 1
        if ESCALATION_PHRASE in answer.lower():
            self._escalate(question, category)

        log.info(
            "turn=%d source=%s cache_hit=%s category=%s latency_ms=%.0f question_sha=%s",
            self.turns, source, from_cache, category, (time.perf_counter() - start) * 1000,
            hashlib.sha256(question.encode()).hexdigest()[:12],
        )
        return answer, category, from_cache, source

    def _escalate(self, question: str, category: str | None) -> None:
        self.escalations += 1
        ticket = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "question": question[:200],
            "history_len": len(self.history),
        }
        with open("logs/escalations.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        self.history.clear()

    def stats(self) -> dict:
        return {
            "сообщений_обработано": self.turns,
            "в_истории": len(self.history),
            "кеш": self.cache.stats(),
            "категории": dict(self.categories),
            "эскалаций": self.escalations,
            "usage": self.client.usage_report(),
        }


def main() -> None:
    setup_logging()
    assistant = SupportAssistant()
    interactive = sys.stdin.isatty()
    print("Помощник техподдержки «Финпомощник». " + HELP)

    while True:
        try:
            line = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break
        if not interactive:
            print(line)  # при вводе из файла показываем реплику, чтобы лог сессии читался
        if not line:
            continue

        if line == "/quit":
            print("До свидания!")
            break
        if line == "/clear":
            assistant.clear()
            print("История очищена.")
            continue
        if line == "/stats":
            print(json.dumps(assistant.stats(), ensure_ascii=False, indent=2))
            continue
        if line == "/help":
            print(HELP)
            continue

        answer, category, from_cache, source = assistant.ask(line)
        body = CATEGORY_RE.sub("", answer.strip()).strip()
        print(f"Помощник: {body}")
        print(f"  └ категория: {category or 'не определена'} | источник: {source}{' | из кеша' if from_cache else ''}")
        if ESCALATION_PHRASE in answer.lower():
            print("  🔔 Эскалация: вопрос записан в очередь специалиста (logs/escalations.jsonl)")


if __name__ == "__main__":
    main()
