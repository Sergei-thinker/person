from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


CHANNEL_URL = "https://t.me/create_products"
CREATOR_PLATFORM_URL = "https://app.create-products.com/"
CREATOR_SITE_URL = "https://create-products.com/"


def main_menu() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(text="Сгенерировать новую персону", callback_data="gen")],
        [InlineKeyboardButton(text="Канал \"В эпоху AI\"", url=CHANNEL_URL)],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def subscribe_menu() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(text="Подписаться на канал", url=CHANNEL_URL)],
        [InlineKeyboardButton(text="Проверить подписку", callback_data="check_sub")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


