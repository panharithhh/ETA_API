"""
Shared test fixtures.

The data layer is MongoDB (Motor). Tests run against an in-memory mongomock
database via mongomock_motor — no real MongoDB server needed, no network calls.
"""

import pytest_asyncio
from mongomock_motor import AsyncMongoMockClient


@pytest_asyncio.fixture
async def db():
    """Fresh in-memory Mongo database per test."""
    client = AsyncMongoMockClient()
    yield client["chonchoun_test"]
