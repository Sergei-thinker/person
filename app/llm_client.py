from __future__ import annotations

from typing import Any, List, Tuple, Optional

from .config import Config
from .logger import get_logger
from .persona_prompt import build_messages


HistoryItem = Tuple[str, str]


class LLMClient:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._openai_client = None
        self._gemini_model: Optional[object] = None
        self._log = get_logger(__name__)
        if (api_key := (config.OPENAI_API_KEY or "").strip()):
            try:
                from openai import AsyncOpenAI  # type: ignore

                self._openai_client = AsyncOpenAI(api_key=api_key)
            except Exception:
                self._openai_client = None
        if (gkey := (config.GEMINI_API_KEY or "").strip()):
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=gkey)
                self._gemini_model = genai.GenerativeModel(self._config.GEMINI_MODEL)
            except Exception:
                self._gemini_model = None

    async def generate_persona(self, history: List[HistoryItem]) -> str:
        if self._gemini_model is not None:
            self._log.info("LLMClient: using Gemini model=%s", self._config.GEMINI_MODEL)
            raw = await self._generate_via_gemini(history)
            return self._clean_response(raw)
        if self._openai_client:
            self._log.info("LLMClient: using OpenAI model=%s", self._config.OPENAI_MODEL)
            raw = await self._generate_via_openai(history)
            return self._clean_response(raw)
        self._log.warning("LLMClient: no LLM configured, returning demo persona")
        return self._clean_response(self._demo_persona())

    async def _generate_via_openai(self, history: List[HistoryItem]) -> str:
        assert self._openai_client is not None
        messages = build_messages(history)
        # Use Responses API
        try:
            response = await self._openai_client.responses.create(  # type: ignore[attr-defined]
                model=self._config.OPENAI_MODEL,
                temperature=0.6,
                max_output_tokens=4000,
                input={"messages": messages},
            )
            # responses API returns output_text helper in SDK >=1.40
            text = getattr(response, "output_text", None)
            if callable(text):
                return text()
            # Fallback: parse first text item
            content = getattr(response, "output", None)
            texts: list[str] = []
            if isinstance(content, list):
                for item in content:
                    item_content: Any
                    if isinstance(item, dict):
                        item_content = item.get("content")
                    else:
                        item_content = getattr(item, "content", None)
                    if isinstance(item_content, list):
                        for part in item_content:
                            if isinstance(part, dict):
                                text_part = part.get("text") or part.get("content")
                                if text_part:
                                    texts.append(str(text_part))
                            else:
                                texts.append(str(part))
                    elif isinstance(item_content, str):
                        texts.append(item_content)
            combined = "\n".join(t.strip() for t in texts if t and t.strip())
            if combined:
                return combined
            return ""
        except Exception as e:
            self._log.error("OpenAI generation failed: %s", e)
            return self._demo_persona()

    async def _generate_via_gemini(self, history: List[HistoryItem]) -> str:
        # google-generativeai SDK синхронный; выполним в отдельном потоке
        assert self._gemini_model is not None
        messages = build_messages(history)

        def _compose() -> str:
            # Gemini ожидает один промпт; склеим роли в последовательный текст
            sys = ""
            user_parts: List[str] = []
            for m in messages:
                if m.get("role") == "system":
                    sys = m.get("content", "")
                elif m.get("role") in {"user", "assistant"}:
                    user_parts.append(f"{m['role']}: {m['content']}")
            prompt = sys + "\n\n" + "\n".join(user_parts)
            try:
                kwargs = {}
                if self._config.GEMINI_MAX_TOKENS:
                    kwargs["generation_config"] = {"max_output_tokens": self._config.GEMINI_MAX_TOKENS}
                result = self._gemini_model.generate_content(prompt, **kwargs)  # type: ignore[union-attr]
                # Разные версии SDK: result.text или candidates[0].content.parts[0].text
                text = getattr(result, "text", None)
                if isinstance(text, str) and text.strip():
                    return text
                candidates = getattr(result, "candidates", None)
                if isinstance(candidates, list) and candidates:
                    first = candidates[0]
                    content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
                    if content is not None:
                        parts = getattr(content, "parts", None) or (content.get("parts") if isinstance(content, dict) else None)
                        if isinstance(parts, list) and parts:
                            part0 = parts[0]
                            text_inner = getattr(part0, "text", None) or (part0.get("text") if isinstance(part0, dict) else None)
                            if isinstance(text_inner, str):
                                return text_inner
                return ""
            except Exception as e:
                self._log.error("Gemini generate_content failed: %s", e)
                return ""

        import asyncio

        text = await asyncio.to_thread(_compose)
        return text or self._demo_persona()

    def _clean_response(self, text: str) -> str:
        """Удаляет символы # и * из ответа LLM"""
        if not text:
            return text
        # Убираем звездочки (markdown bold/italic)
        cleaned = text.replace("*", "")
        # Убираем решетки (markdown заголовки)
        lines = []
        for line in cleaned.split("\n"):
            # Удаляем # только в начале строки (заголовки)
            stripped = line.lstrip()
            if stripped.startswith("#"):
                # Убираем все # и пробелы после них
                clean_line = stripped.lstrip("# ").strip()
                if clean_line:  # Если после удаления # осталось содержимое
                    lines.append(clean_line)
            else:
                lines.append(line)
        return "\n".join(lines)

    def _demo_persona(self) -> str:
        return (
            "Анна, руководитель маркетинга\n\n"
            "Роль/должность: Head of Marketing\n"
            "Отрасль/сегмент: B2B SaaS\n"
            "Размер компании/уровень: средняя компания, 50–200 сотрудников\n"
            "Регион: Россия\n\n"
            "Базовый профиль:\n"
            "Возраст/стаж: 32 года, 8 лет в маркетинге\n"
            "Ключевые задачи по работе: лидогенерация, воронка, бренд\n"
            "Контекст использования продукта: ищет повышение конверсии и аналитику\n"
            "Каналы информации/медиа: Телеграм, vc.ru, профильные чаты\n\n"
            "Психографический портрет:\n"
            "Ценности и отношение к рискам: прагматизм, осторожные эксперименты\n"
            "Стиль принятия решений: гипотезы → тест → метрики\n"
            "Триггеры внимания: кейсы с цифрами, отзывы из РФ\n"
            "Возражения и опасения: долгое внедрение, скрытые расходы\n\n"
            "Проблематика и боли:\n"
            "1) Низкая конверсия MQL → SQL\n"
            "2) Слабая атрибуция каналов\n"
            "3) Нехватка ресурсов на контент\n\n"
            "Мотивация и триггеры:\n"
            "Ключевая цель: рост SQL и демо-заявок\n"
            "Что станет моментом действия: быстрый пилот с ростом CTR\n"
            "Социальное доказательство (какое работает): кейсы с рынком РФ\n\n"
            "Путь к покупке (этапы):\n"
            "1) Осознание проблемы → падение лидов\n"
            "2) Исследование решений → сравнение инструментов\n"
            "3) Сравнение поставщиков → запрос демо\n"
            "4) Пробный период/демо → пилот на 2 недели\n"
            "5) Покупка/внедрение → интеграция и обучение\n\n"
            "Сообщения и офферы:\n"
            "Главный месседж (одно предложение): Увеличьте SQL на 20% за месяц без лишних затрат.\n"
            "Альтернативные месседжи (2–3 шт.): Повышаем CTR в рекламе; Чёткая атрибуция; Кейсы с рынка РФ.\n"
            "Оффер/лид-магнит: бесплатный аудит воронки и 10 гипотез\n"
            "Каналы и форматы (топ-3): Telegram, поисковая реклама, вебинары\n\n"
            "Контент-идеи (3–5 шт.):\n"
            "Чек-лист атрибуции; кейсы с цифрами; вебинар по гипотезам.\n\n"
            "Метрики успеха:\n"
            "Первичные (конверсия из визита в лид, CTR и т. п.): рост CTR, CR\n"
            "Вторичные (демо-запросы, удержание и т. п.): число демо, удержание\n"
        )

