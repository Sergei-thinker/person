from __future__ import annotations

import asyncio
import signal
from contextlib import suppress

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, MenuButtonCommands

from .config import load_config
from .handlers import setup_handlers
from .llm_client import LLMClient
from .logger import get_logger
from .subscription import SubscriptionCache


logger = get_logger(__name__)


async def main() -> None:
    config = load_config()
    if config.DEBUG:
        import logging

        logger.setLevel(logging.DEBUG)

    try:
        import uvloop  # type: ignore

        uvloop.install()
    except Exception:
        pass

    bot = Bot(token=config.BOT_TOKEN)
    dp = Dispatcher()

    # TTL=None => бесконечно до рестарта процесса
    cache = SubscriptionCache(ttl_seconds=None)
    llm = LLMClient(config)
    dp.include_router(setup_handlers(config, cache, llm))

    stop_event = asyncio.Event()

    def _cancel(*_: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(s, _cancel)

    # Ensure we are in long-polling mode (disable webhook if was set elsewhere)
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass

    # Установим команды бота (синее меню)
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Начать"),
                BotCommand(command="gen", description="Сгенерировать новую персону"),
                BotCommand(command="channel", description="Канал “В эпоху AI”"),
                BotCommand(command="creator_app", description="Платформа Креатора"),
                BotCommand(command="creator_site", description="Сайт Креатора"),
            ]
        )
        # Явно укажем, что в синем меню показываются команды
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    except Exception:
        pass

    logger.info("Starting polling")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())

