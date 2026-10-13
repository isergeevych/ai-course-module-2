# Модуль 2 — работа с LLM API (ДЗ 2.1–2.5)

Курс «ИИ-разработчик: от API до агентов». Сквозная тема — ассистент «Финпомощник» (личные финансы).
Финальные прогоны — 12.10.2026, `openai/gpt-4o-mini` через OpenRouter; логи — в `logs/`.

> Код и тексты написаны ИИ-ассистентом Claude Code; постановка задач, ключевые решения и приёмка — мои.
> Подробнее — в [указателе по всем ДЗ](https://github.com/isergeevych/fin-assistant-diploma/blob/master/SUBMISSION.md#как-выполнена-работа).

| ДЗ | Что сделано | Код | Результат прогона |
|---|---|---|---|
| 2.1 | Запрос к API, temperature 0 / 0.7 / 1.5, стриминг, два провайдера через `base_url` | `providers_demo.py`, `llm_config.py` | `logs/2.1_run.log`; наблюдения по temperature — в конце `providers_demo.py` |
| 2.2 | System prompt по РРФО, 4 few-shot, 8 автотестов (включая prompt injection), tiktoken, CoT | `prompts/support_system.j2`, `support_prompt.py`, `prompt_lab.py` | `logs/2.2_run.log`, `logs/2.2_results.json` |
| 2.3 | Retry с экспонентой и jitter, fallback-цепочка, circuit breaker, rate limiter, учёт стоимости | `robust_client.py`, `demo_robust.py` | `logs/2.3_run.log`, `logs/robust_client.log`, `logs/2.3_usage.json` |
| 2.4 | Кеш LLM: in-memory и Redis, ключ SHA-256, TTL, статистика | `llm_cache.py`, `demo_cache.py` | `logs/2.4_run.log`, `logs/2.4_stats.json` |
| 2.5 | CLI-помощник техподдержки: всё из 2.1–2.4 + история, классификация, эскалация | `support_cli.py`, `session_script.txt` | `logs/2.5_session.log`, `logs/escalations.jsonl` |

## Главное по результатам

- **2.1** — стриминг: первые слова через 0,52 с против 1,35 с без стрима. temperature 0 дала дословно одинаковые
  ответы, 1.5 — каждый раз новый вариант.
- **2.2** — 8/8 тестов, промпт 797 токенов (лимит 300–800). Тесты нашли три дефекта первой версии промпта (правило
  «не по теме» срабатывало на вопрос о цене тарифа, противоречило формату ответа, на инвестиционный вопрос не было
  явного отказа) — исправлено. Chain-of-thought: без рассуждения модель ошиблась в экономии на годовой подписке
  (написала 3 588 ₽), с рассуждением — верно (598 ₽).
- **2.3** — сбои 429/503 → успех на 3-й попытке; 500 у основного провайдера → полная серия 1→2→4→8→16 с и fallback;
  401 не ретраится; circuit breaker пропускает упавшего провайдера; все провайдеры недоступны → вежливая заглушка.
- **2.4** — повторный запрос из памяти за 0 мс, из Redis за 1,4–2,6 мс против 0,7–1,9 с через сеть; другой
  `temperature` — другой ключ; TTL проверен.
- **2.5** — сессия по сценарию: уточняющий вопрос понят по истории, две эскалации записаны, инъекция «покажи
  системный промпт» отбита, повторный вопрос после `/clear` взят из кеша; 6 запросов — $0,001.

## Запуск

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # OPENROUTER_API_KEY=...; для 2.4/2.5 с Redis — CACHE_BACKEND=redis
docker run -d --name m2-redis -p 127.0.0.1:6379:6379 redis:7.4-alpine
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python providers_demo.py
.venv\Scripts\python prompt_lab.py
.venv\Scripts\python demo_robust.py
.venv\Scripts\python demo_cache.py
.venv\Scripts\python support_cli.py < session_script.txt
```
