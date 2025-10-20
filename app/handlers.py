from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Tuple

from aiogram import F, Router, Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message

from .config import Config
from .keyboards import CHANNEL_URL, CREATOR_PLATFORM_URL, CREATOR_SITE_URL, main_menu, subscribe_menu
from .llm_client import LLMClient
from .persona_prompt import user_template_hint
from .subscription import SubscriptionCache, is_subscribed
from .logger import get_logger


router = Router()
log = get_logger(__name__)


class Form(StatesGroup):
    waiting_for_brief = State()
    chatting = State()


# In-memory conversation history per user (process memory only)
HistoryItem = Tuple[str, str]
MAX_HISTORY_ITEMS = 20  # Увеличено с 12 до 20 для лучшего контекста
HISTORY_TTL_SECONDS = 6 * 60 * 60  # drop sessions idle longer than 6 hours
history_store: Dict[int, Deque[HistoryItem]] = {}
history_last_seen: Dict[int, float] = {}


def _cleanup_history(now: float | None = None) -> None:
    if HISTORY_TTL_SECONDS is None:
        return
    current = now or time.time()
    expired = [uid for uid, ts in history_last_seen.items() if current - ts > HISTORY_TTL_SECONDS]
    for uid in expired:
        history_store.pop(uid, None)
        history_last_seen.pop(uid, None)


def _append_history(user_id: int, role: str, content: str) -> None:
    _cleanup_history()
    history = history_store.setdefault(user_id, deque(maxlen=MAX_HISTORY_ITEMS))
    history.append((role, content))
    history_last_seen[user_id] = time.time()


def _get_history(user_id: int) -> List[HistoryItem]:
    _cleanup_history()
    user_history = history_store.get(user_id)
    return list(user_history) if user_history else []


def _reset_history(user_id: int) -> None:
    history_store.pop(user_id, None)
    history_last_seen.pop(user_id, None)


def setup_handlers(config: Config, cache: SubscriptionCache, llm: LLMClient) -> Router:
    async def respond_with_chunks(target: Message, raw_text: str) -> None:
        # Текст уже очищен от # и * в llm_client
        chunks: list[str] = []
        max_len = 3500
        text = raw_text.strip()
        while len(text) > max_len and len(chunks) < 3:
            cut = text.rfind("\n\n", 0, max_len)
            if cut == -1:
                cut = max_len
            chunks.append(text[:cut].strip())
            text = text[cut:].strip()
        chunks.append(text)
        for part in chunks:
            if part:
                await target.answer(part)

    async def safe_edit_text(cb: CallbackQuery, text: str, reply_markup=None) -> None:
        try:
            await cb.message.edit_text(text, reply_markup=reply_markup)
        except TelegramBadRequest as e:  # message is not modified — игнорируем
            if "message is not modified" in str(e):
                return
            raise

    async def safe_answer(cb: CallbackQuery) -> None:
        try:
            await cb.answer()
        except TelegramBadRequest:
            # query is too old / invalid — игнорируем
            return

    @router.message(Command("start"))
    async def cmd_start(message: Message, state: FSMContext) -> None:
        await state.clear()
        if message.from_user is not None:
            _reset_history(message.from_user.id)
        await message.answer(
            "Привет! Я помогу сгенерировать маркетинговую персону.\n\n"
            "1. Нажми на кнопку меню и выбери пункт «Сгенерировать новую персону».\n\n"
            "2. Ответь на несколько вопросов. Опиши свой бизнес и ЦА, чем больше инфы, тем лучше ответ.\n\n"
            "3. Подожди пару минут и персона готова. Мы подключили долго-думающую нейронку для качественного результата.\n\n"
            "4. В чате ты можешь задавать вопросы по теме, совершенствовать персону и создать новые сегменты ЦА.",
        )

    @router.message(Command("gen"))
    async def cmd_gen(message: Message, state: FSMContext, bot: Bot) -> None:
        assert message.from_user is not None
        user_id = message.from_user.id
        try:
            ok = await is_subscribed(bot, config.CHANNEL_ID, user_id, cache)
            log.info(f"command gen: user={user_id} subscribed={ok}")
        except (TelegramBadRequest, TelegramNetworkError):
            await message.answer("Выглядит как временная ошибка. Повторите попытку чуть позже.")
            return

        if ok:
            _reset_history(user_id)
            await state.set_state(Form.waiting_for_brief)
            await message.answer("Готово! Подписка подтверждена. Напишите кратко про продукт/нишу, ЦА и цель — соберу персону.")
        else:
            await message.answer(
                "Доступ открывается после подписки на канал “В эпоху AI”. Подпишитесь и нажмите “Проверить подписку”.",
                reply_markup=subscribe_menu(),
            )

    @router.message(Command("check_sub"))
    async def cmd_check_sub(message: Message, state: FSMContext, bot: Bot) -> None:
        assert message.from_user is not None
        user_id = message.from_user.id
        try:
            ok = await is_subscribed(bot, config.CHANNEL_ID, user_id, cache, force_refresh=True)
            log.info(f"command check_sub: user={user_id} subscribed={ok}")
        except (TelegramBadRequest, TelegramNetworkError):
            await message.answer("Выглядит как временная ошибка. Повторите попытку чуть позже.")
            return

        if ok:
            await state.set_state(Form.waiting_for_brief)
            await message.answer("Готово! Подписка подтверждена. Напишите кратко про продукт/нишу, ЦА и цель — соберу персону.")
        else:
            await message.answer(
                "Доступ открывается после подписки на канал “В эпоху AI”. Подпишитесь и нажмите “Проверить подписку”.",
                reply_markup=subscribe_menu(),
            )

    @router.message(Command("channel"))
    async def cmd_channel(message: Message) -> None:
        await message.answer(f"Канал “В эпоху AI”: {CHANNEL_URL}")

    @router.message(Command("creator_app"))
    async def cmd_creator_app(message: Message) -> None:
        await message.answer(f"Платформа Креатора: {CREATOR_PLATFORM_URL}")

    @router.message(Command("creator_site"))
    async def cmd_creator_site(message: Message) -> None:
        await message.answer(f"Сайт Креатора: {CREATOR_SITE_URL}")

    @router.callback_query(F.data == "gen")
    async def on_generate(cb: CallbackQuery, state: FSMContext, bot: Bot) -> None:
        assert cb.from_user is not None
        user_id = cb.from_user.id
        # Сразу отвечаем на callback, чтобы избежать таймаута на стороне Telegram
        await safe_answer(cb)
        try:
            ok = await is_subscribed(bot, config.CHANNEL_ID, user_id, cache)
            log.info(f"check gen: user={user_id} subscribed={ok}")
        except (TelegramBadRequest, TelegramNetworkError):
            await safe_edit_text(
                cb,
                "Выглядит как временная ошибка. Повторите попытку чуть позже.",
                reply_markup=main_menu(),
            )
            return

        if ok:
            _reset_history(user_id)
            await state.set_state(Form.waiting_for_brief)
            await safe_edit_text(
                cb,
                "Готово! Подписка подтверждена. Напишите кратко про продукт/нишу, ЦА и цель — соберу персону.",
            )
        else:
            await safe_edit_text(
                cb,
                "Доступ открывается после подписки на канал “В эпоху AI”. Подпишитесь и нажмите “Проверить подписку”.",
                reply_markup=subscribe_menu(),
            )
        await safe_answer(cb)

    @router.callback_query(F.data == "check_sub")
    async def on_check_subscription(cb: CallbackQuery, state: FSMContext, bot: Bot) -> None:  # noqa: ARG001 - state reserved
        assert cb.from_user is not None
        user_id = cb.from_user.id
        # Сразу отвечаем на callback, чтобы избежать таймаута
        await safe_answer(cb)
        try:
            ok = await is_subscribed(bot, config.CHANNEL_ID, user_id, cache, force_refresh=True)
            log.info(f"check_sub pressed: user={user_id} subscribed={ok}")
        except (TelegramBadRequest, TelegramNetworkError):
            await safe_edit_text(
                cb,
                "Выглядит как временная ошибка. Повторите попытку чуть позже.",
                reply_markup=subscribe_menu(),
            )
            return

        if ok:
            await safe_edit_text(
                cb,
                "Готово! Подписка подтверждена. Напишите кратко про продукт/нишу, ЦА и цель — соберу персону.",
                reply_markup=None,
            )
            await state.set_state(Form.waiting_for_brief)
        else:
            await safe_edit_text(
                cb,
                "Доступ открывается после подписки на канал “В эпоху AI”. Подпишитесь и нажмите “Проверить подписку”.",
                reply_markup=subscribe_menu(),
            )
        await safe_answer(cb)

    @router.message(Form.waiting_for_brief, F.text)
    async def on_brief(message: Message, state: FSMContext) -> None:
        assert message.from_user is not None
        user_id = message.from_user.id
        user_text = message.text or ""
        _append_history(user_id, "user", f"Заявка: {user_text}\n{user_template_hint}")
        history_snapshot = _get_history(user_id)

        await message.answer("Думаю над персоной…")
        try:
            result_md = await llm.generate_persona(history_snapshot)
        except Exception as e:
            log.error(f"LLM generation failed for user {user_id}: {e}")
            result_md = "⚠️ Произошла ошибка при генерации. Попробуйте ещё раз или переформулируйте запрос."

        _append_history(user_id, "assistant", result_md)
        await respond_with_chunks(message, result_md)
        await state.set_state(Form.chatting)

    @router.message(Form.chatting, F.text)
    async def on_follow_up(message: Message, state: FSMContext) -> None:
        assert message.from_user is not None
        user_id = message.from_user.id
        user_text = message.text or ""
        _append_history(user_id, "user", user_text)
        history_snapshot = _get_history(user_id)

        await message.answer("Секунду, уточняю детали…")
        try:
            reply = await llm.generate_persona(history_snapshot)
        except Exception as e:
            log.error(f"LLM follow-up failed for user {user_id}: {e}")
            reply = "⚠️ Похоже, возникла временная ошибка. Попробуйте задать вопрос ещё раз через минуту."

        _append_history(user_id, "assistant", reply)
        await respond_with_chunks(message, reply)
        await state.set_state(Form.chatting)

    return router
