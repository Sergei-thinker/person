from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.subscription import SubscriptionCache, is_subscribed


@dataclass
class FakeMember:
    status: str
    is_member: bool | None = None


class FakeBot:
    def __init__(self, member: FakeMember) -> None:
        self._member = member

    async def get_chat_member(self, chat_id: int | str, user_id: int):  # noqa: ARG002
        await asyncio.sleep(0)  # simulate awaitable
        return self._member


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,is_member,expected",
    [
        ("member", None, True),
        ("administrator", None, True),
        ("creator", None, True),
        ("left", None, False),
        ("kicked", None, False),
        ("restricted", True, True),
        ("restricted", False, False),
        ("restricted", None, False),
    ],
)
async def test_is_subscribed_statuses(status: str, is_member: bool | None, expected: bool):
    bot = FakeBot(FakeMember(status=status, is_member=is_member))
    cache = SubscriptionCache(ttl_seconds=60)
    ok = await is_subscribed(bot, "@create_products", 123, cache)
    assert ok is expected


@pytest.mark.asyncio
async def test_cache_hits():
    # First call returns not-subscribed; second should hit cache even if underlying member changes
    bot1 = FakeBot(FakeMember(status="left"))
    cache = SubscriptionCache(ttl_seconds=3600)
    ok1 = await is_subscribed(bot1, "@create_products", 123, cache)
    assert ok1 is False

    # Change underlying status to member, but cache should still return cached False
    bot2 = FakeBot(FakeMember(status="member"))
    ok2 = await is_subscribed(bot2, "@create_products", 123, cache)
    assert ok2 is False




