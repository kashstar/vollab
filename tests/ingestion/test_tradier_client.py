from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from vollab.ingestion.models import OptionType
from vollab.ingestion.tradier_client import TradierError


def _route(
    handlers: dict[str, Callable[[httpx.Request], httpx.Response]],
) -> Callable[[httpx.Request], httpx.Response]:
    def _handler(request: httpx.Request) -> httpx.Response:
        for fragment, fn in handlers.items():
            if fragment in str(request.url.path):
                return fn(request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    return _handler


def test_get_expirations_multi(make_tradier_client: Any, load_fixture: Any) -> None:
    body = load_fixture("tradier_expirations_multi.json")
    client = make_tradier_client(lambda request: httpx.Response(200, json=body))
    assert client.get_expirations("SPY") == [
        date(2024, 1, 19),
        date(2024, 2, 16),
        date(2024, 3, 15),
    ]


def test_get_expirations_single_scalar_collapse(
    make_tradier_client: Any, load_fixture: Any
) -> None:
    body = load_fixture("tradier_expirations_single.json")
    client = make_tradier_client(lambda request: httpx.Response(200, json=body))
    assert client.get_expirations("SPY") == [date(2024, 1, 19)]


def test_get_expirations_empty(make_tradier_client: Any, load_fixture: Any) -> None:
    body = load_fixture("tradier_expirations_empty.json")
    client = make_tradier_client(lambda request: httpx.Response(200, json=body))
    assert client.get_expirations("SPY") == []


def test_get_chain_builds_quotes_with_underlying_price(
    make_tradier_client: Any, load_fixture: Any
) -> None:
    handler = _route(
        {
            "options/chains": lambda r: httpx.Response(
                200, json=load_fixture("tradier_chain_multi.json")
            ),
            "quotes": lambda r: httpx.Response(
                200, json=load_fixture("tradier_quote_underlying.json")
            ),
        }
    )
    client = make_tradier_client(handler)
    quotes = client.get_chain("SPY", date(2099, 1, 19))

    assert len(quotes) == 2
    call, put = quotes
    assert call.option_type is OptionType.CALL
    assert put.option_type is OptionType.PUT
    assert call.underlying_price == 470.12
    assert put.last is None
    assert call.snapshot_ts == put.snapshot_ts


def test_get_chain_single_option_scalar_collapse(
    make_tradier_client: Any, load_fixture: Any
) -> None:
    handler = _route(
        {
            "options/chains": lambda r: httpx.Response(
                200, json=load_fixture("tradier_chain_single.json")
            ),
            "quotes": lambda r: httpx.Response(
                200, json=load_fixture("tradier_quote_underlying.json")
            ),
        }
    )
    client = make_tradier_client(handler)
    quotes = client.get_chain("SPY", date(2099, 1, 19))
    assert len(quotes) == 1


def test_error_response_raises_tradier_error(make_tradier_client: Any) -> None:
    client = make_tradier_client(lambda request: httpx.Response(401, text="unauthorized"))
    with pytest.raises(TradierError):
        client.get_expirations("SPY")


def test_malformed_option_payload_raises_validation_error(
    make_tradier_client: Any, load_fixture: Any
) -> None:
    malformed_chain = {
        "options": {
            "option": [
                {
                    "underlying": "SPY",
                    "option_type": "call",
                    "bid": 1.0,
                    "ask": 1.1,
                    "last": 1.0,
                    "volume": 1,
                    "open_interest": 1,
                    "expiration_date": "2099-01-19",
                    # "strike" missing on purpose
                }
            ]
        }
    }
    handler = _route(
        {
            "options/chains": lambda r: httpx.Response(200, json=malformed_chain),
            "quotes": lambda r: httpx.Response(
                200, json=load_fixture("tradier_quote_underlying.json")
            ),
        }
    )
    client = make_tradier_client(handler)
    with pytest.raises((ValidationError, KeyError)):
        client.get_chain("SPY", date(2099, 1, 19))
