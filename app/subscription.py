from __future__ import annotations

import asyncio
import time
from typing import Dict, Tuple, Optional

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramForbiddenError


CacheValue = Tuple[bool, float]


class SubscriptionCache:
    def __init__(self, ttl_seconds: Optional[int] = 6 * 60 * 60) -> None:
        # None означает: без истечения срока в рамках жизни процесса
        self._ttl: Optional[int] = ttl_seconds
        self._cache: Dict[int, CacheValue] = {}
        self._last_check_ts: Dict[int, float] = {}

    def get(self, user_id: int) -> bool | None:
        self._cleanup()
        if user_id in self._cache:
            ok, expires_at = self._cache[user_id]
            if self._ttl is None:
                return ok
            if time.time() < expires_at:
                return ok
            else:
                self._cache.pop(user_id, None)
        return None

    def set(self, user_id: int, value: bool) -> None:
        expires = float("inf") if self._ttl is None else (time.time() + self._ttl)
        self._cache[user_id] = (value, expires)

    def delay_allowed(self, user_id: int, min_interval: float = 1.0) -> float:
        now = time.time()
        last = self._last_check_ts.get(user_id, 0.0)
        remaining = min_interval - (now - last)
        if remaining > 0:
            return remaining
        self._last_check_ts[user_id] = now
        return 0.0

    def mark_checked(self, user_id: int) -> None:
        self._last_check_ts[user_id] = time.time()

    def _cleanup(self) -> None:
        now = time.time()
        if self._ttl is None:
            return
        expired = [uid for uid, (_, exp) in self._cache.items() if exp <= now]
        for uid in expired:
            self._cache.pop(uid, None)


async def is_subscribed(
    bot: Bot,
    channel_id: int | str,
    user_id: int,
    cache: SubscriptionCache,
    force_refresh: bool = False,
) -> bool:
    if not force_refresh:
        cached = cache.get(user_id)
        if cached is not None:
            return cached

    # simple rate-limit per user on repeated checks
    wait_sec = cache.delay_allowed(user_id, min_interval=1.0)
    if wait_sec > 0:
        await asyncio.sleep(wait_sec)

    try:
        member = await bot.get_chat_member(chat_id=channel_id, user_id=user_id)
    except (TelegramBadRequest, TelegramNetworkError, TelegramForbiddenError):
        # Consider as not subscribed on transient errors; caller will show UX message
        cache.set(user_id, False)
        return False

    status = getattr(member, "status", None)
    is_member_flag = getattr(member, "is_member", None)

    allowed_statuses = {"creator", "administrator", "member"}
    ok = False
    if status in allowed_statuses:
        ok = True
    elif status == "restricted":
        ok = bool(is_member_flag)

    cache.set(user_id, ok)
    cache.mark_checked(user_id)
    return ok


