"""
Общая конфигурация LLM-провайдеров для ДЗ 2.1–2.5.

Все ключи и модели читаются из .env (см. .env.example). Провайдеры —
OpenAI-совместимые API, поэтому переключение = другой base_url + model:

- openrouter — облако, один ключ на модели разных вендоров;
- openai     — OpenAI напрямую (опционально, если есть ключ);
- ollama     — локальная модель на этой машине, ключ не нужен.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Цены $ за 1M токенов (input, output) — сверены с openrouter.ai/api/v1/models
# на 2026-10-12. Локальная модель бесплатна. Неизвестная модель -> стоимость 0
# с пометкой в логе, а не падение.
PRICES_PER_1M = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-mini": (0.75, 4.50),
    "deepseek/deepseek-chat": (0.32, 0.89),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key: str | None
    model: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or self.name == "ollama"

    def client(self, **kwargs) -> OpenAI:
        # Ollama ключ не проверяет, но SDK требует непустую строку.
        return OpenAI(base_url=self.base_url, api_key=self.api_key or "ollama", **kwargs)


def get_provider(name: str) -> Provider:
    if name == "openrouter":
        return Provider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY") or None,
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )
    if name == "openai":
        return Provider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY") or None,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    if name == "ollama":
        return Provider(
            name="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=None,
            model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        )
    raise ValueError(f"Неизвестный провайдер {name!r}: ожидается openrouter / openai / ollama")


def default_provider() -> Provider:
    """Провайдер из LLM_PROVIDER; если у него нет ключа — локальный Ollama."""
    provider = get_provider(os.getenv("LLM_PROVIDER", "openrouter"))
    return provider if provider.configured else get_provider("ollama")


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICES_PER_1M.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
