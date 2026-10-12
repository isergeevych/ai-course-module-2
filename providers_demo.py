"""
ДЗ 2.1 — API провайдеров.

Демонстрирует:
  1. basic       — запрос с system + user сообщениями;
  2. temperature — один и тот же запрос при temperature 0 / 0.7 / 1.5;
  3. stream      — стриминг (stream=True) + замер TTFT и полного времени
                   в сравнении с обычным запросом (бонус);
  4. providers   — один запрос к двум провайдерам: переключение только
                   через base_url + model (OpenRouter и локальный Ollama;
                   OpenAI напрямую — если задан OPENAI_API_KEY).

Провайдер по умолчанию выбирается через LLM_PROVIDER в .env (бонус).

Запуск:
    python providers_demo.py            # все разделы
    python providers_demo.py temperature

Наблюдения по temperature — в конце файла (TEMPERATURE_OBSERVATIONS),
записаны по реальному прогону, лог: logs/2.1_run.log.
"""

import sys
import time

from llm_config import default_provider, get_provider

SYSTEM = "Ты — помощник приложения для учёта личных финансов «Финпомощник». Отвечай по-русски, кратко."


def basic() -> None:
    provider = default_provider()
    client = provider.client()
    print(f"\n### 1. Базовый запрос ({provider.name}: {provider.model})")
    response = client.chat.completions.create(
        model=provider.model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Дай три совета, как начать вести бюджет."},
        ],
        max_tokens=300,
    )
    print(response.choices[0].message.content)
    print(f"[usage] {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")


def temperature() -> None:
    """Креативный запрос — на нём разница temperature видна сильнее всего.
    Каждую температуру прогоняем дважды, чтобы увидеть (не)повторяемость."""
    provider = default_provider()
    client = provider.client()
    prompt = "Придумай один короткий слоган для приложения учёта расходов. Только слоган."
    print(f"\n### 2. Эксперимент с temperature ({provider.name}: {provider.model})")
    for t in (0, 0.7, 1.5):
        for run in (1, 2):
            response = client.chat.completions.create(
                model=provider.model,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=t,
                max_tokens=60,
            )
            print(f"  t={t:<3} прогон {run}: {response.choices[0].message.content.strip()}")


def stream() -> None:
    provider = default_provider()
    client = provider.client()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Объясни в одном абзаце, что такое финансовая подушка безопасности."},
    ]
    print(f"\n### 3. Стриминг vs обычный запрос ({provider.name}: {provider.model})")

    # Прогрев: у локальной модели первый вызов включает загрузку весов в память,
    # у облака — установку TLS-соединения. Без прогрева сравнение нечестное.
    client.chat.completions.create(model=provider.model, messages=[{"role": "user", "content": "ok"}], max_tokens=1)

    start = time.perf_counter()
    response = client.chat.completions.create(model=provider.model, messages=messages, max_tokens=300)
    full_plain = time.perf_counter() - start
    print(f"Обычный запрос: ответ целиком через {full_plain:.2f} с ({len(response.choices[0].message.content)} симв.)")

    print("Стрим: ", end="", flush=True)
    start = time.perf_counter()
    ttft = None
    chunks = client.chat.completions.create(model=provider.model, messages=messages, max_tokens=300, stream=True)
    for chunk in chunks:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start
            print(delta, end="", flush=True)
    full_stream = time.perf_counter() - start
    print(f"\nСтрим: TTFT {ttft:.2f} с, полное время {full_stream:.2f} с")
    print(f"Вывод: в стриме первые слова видны через {ttft:.2f} с, без стрима — только через {full_plain:.2f} с.")


def providers() -> None:
    print("\n### 4. Два провайдера: меняются только base_url и model")
    question = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Что выгоднее: подписка 299 ₽ в месяц или 2990 ₽ в год? Ответь одним предложением."},
    ]
    for name in ("openrouter", "openai", "ollama"):
        provider = get_provider(name)
        if not provider.configured:
            print(f"- {name}: пропущен (нет ключа в .env)")
            continue
        start = time.perf_counter()
        try:
            response = provider.client(timeout=120).chat.completions.create(
                model=provider.model, messages=question, max_tokens=120
            )
        except Exception as e:  # демо: показываем ошибку провайдера и идём дальше
            print(f"- {name}: ошибка {type(e).__name__}: {e}")
            continue
        elapsed = time.perf_counter() - start
        print(f"- {name} ({provider.base_url}, {provider.model}) за {elapsed:.2f} с:")
        print(f"    {response.choices[0].message.content.strip()}")


SECTIONS = {"basic": basic, "temperature": temperature, "stream": stream, "providers": providers}


if __name__ == "__main__":
    selected = sys.argv[1:] or list(SECTIONS)
    for name in selected:
        SECTIONS[name]()


TEMPERATURE_OBSERVATIONS = """
(заполняется по реальному прогону — см. logs/2.1_run.log)
"""
