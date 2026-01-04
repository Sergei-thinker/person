from __future__ import annotations

from typing import Any, Union

from pydantic import field_validator, AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    BOT_TOKEN: str
    CHANNEL_ID: Union[str, int]
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    # Поддерживаем оба имени переменной окружения: GEMINI_API_KEY и GOOGLE_GENERATIVE_AI_API_KEY
    GEMINI_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
    )
    GEMINI_MODEL: str = "gemini-2.5-pro"
    GEMINI_MAX_TOKENS: int | None = None
    FAST_TIMEOUT_GEMINI: int | None = None
    FAST_TIMEOUT_TOTAL: int | None = None
    CHANNEL_URL: str | None = None
    DEBUG: bool = False

    @field_validator("CHANNEL_ID")
    @classmethod
    def validate_channel_id(cls, value: Any) -> Union[str, int]:
        # Accept @username or integer ID
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("@") and len(value) > 1:
                return value
            # try to parse numeric string to int
            try:
                return int(value)
            except ValueError as exc:  # noqa: F841 - clarity
                raise ValueError("CHANNEL_ID должен быть @username или числовым ID")
        raise ValueError("CHANNEL_ID должен быть @username или числовым ID")


def load_config() -> Config:
    return Config()  # type: ignore[call-arg]




