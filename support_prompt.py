"""
Сборка system prompt техподдержки из Jinja2-шаблона prompts/support_system.j2
(ДЗ 2.2, переиспользуется в 2.5).

Шаблон рендерится в SandboxedEnvironment и с StrictUndefined: пользовательский
ввод никогда не попадает в текст шаблона — только в отдельное user-сообщение,
поэтому template injection ({{ ... }} от пользователя) исключён.
"""

from pathlib import Path

from jinja2 import FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

PRODUCT_NAME = "Финпомощник"

FEW_SHOT = [
    {
        "question": "Как поменять категорию у покупки?",
        "answer": "Откройте «Профиль → Категории» и выберите нужную категорию.\n[Категория: FAQ]",
    },
    {
        "question": "Не загружается выписка из Т-Банка, пишет «ошибка формата».",
        "answer": "Проверьте, что файл в формате CSV или PDF и не больше 10 МБ. "
        "Если не поможет — обновите приложение.\n[Категория: Техническая проблема]",
    },
    {
        "question": "Я отменил подписку, а деньги всё равно списали! Это обман!",
        "answer": "Понимаю ваше недовольство. Отмена оформляется в магазине приложений, "
        "а за уже начавшийся период деньги не возвращаются. Передаю вопрос специалисту.\n[Категория: Жалоба]",
    },
    {
        "question": "Посоветуй, какие акции сейчас купить?",
        "answer": "Я не даю инвестиционных рекомендаций, но помогу с настройками приложения.\n[Категория: FAQ]",
    },
]

_env = SandboxedEnvironment(
    loader=FileSystemLoader(Path(__file__).parent / "prompts"),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_system_prompt(with_examples: bool = True, chain_of_thought: bool = False) -> str:
    template = _env.get_template("support_system.j2")
    return template.render(
        product_name=PRODUCT_NAME,
        examples=FEW_SHOT if with_examples else [],
        chain_of_thought=chain_of_thought,
    ).strip()
