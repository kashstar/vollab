import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from vollab.ingestion.tradier_client import TradierClient

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def load_fixture() -> Callable[[str], dict[str, Any]]:
    def _load(name: str) -> dict[str, Any]:
        return json.loads((DATA_DIR / name).read_text())

    return _load


@pytest.fixture
def make_tradier_client() -> Callable[..., TradierClient]:
    def _make(
        handler: Callable[[httpx.Request], httpx.Response],
        **kwargs: Any,
    ) -> TradierClient:
        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(transport=transport, base_url="https://api.tradier.com/v1/")
        kwargs.setdefault("sleep", lambda _seconds: None)
        return TradierClient(http_client, **kwargs)

    return _make
