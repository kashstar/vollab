from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import pytest

from vollab.ingestion.errors import TradierError
from vollab.ingestion.models import OptionType


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
    assert put.underlying_price == 470.12
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


def test_401_raises_tradier_error_no_retry(make_tradier_client: Any) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="unauthorized")

    client = make_tradier_client(handler)
    with pytest.raises(TradierError) as exc_info:
        client.get_expirations("SPY")

    assert calls == 1
    assert exc_info.value.status_code == 401


def test_500_retries_then_raises_after_max_retries(make_tradier_client: Any) -> None:
    calls = 0
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="server error")

    client = make_tradier_client(
        handler,
        max_retries=3,
        min_request_interval_seconds=0,
        sleep=lambda seconds: sleep_calls.append(seconds),
    )
    with pytest.raises(TradierError):
        client.get_expirations("SPY")

    assert calls == 4
    assert sleep_calls == [pytest.approx(0.5), pytest.approx(1.0), pytest.approx(2.0)]


def test_transient_network_error_then_success(make_tradier_client: Any, load_fixture: Any) -> None:
    calls = 0
    body = load_fixture("tradier_expirations_multi.json")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(200, json=body)

    client = make_tradier_client(
        handler,
        min_request_interval_seconds=0,
        sleep=lambda seconds: None,
    )
    dates = client.get_expirations("SPY")

    assert calls == 3
    assert dates == [date(2024, 1, 19), date(2024, 2, 16), date(2024, 3, 15)]


def test_malformed_json_response_raises_without_retry(make_tradier_client: Any) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text="not json")

    client = make_tradier_client(handler)
    with pytest.raises(TradierError):
        client.get_expirations("SPY")

    assert calls == 1


def test_malformed_option_payload_wrapped_as_tradier_error(
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
    with pytest.raises(TradierError):
        client.get_chain("SPY", date(2099, 1, 19))


def test_pacing_enforces_min_interval(make_tradier_client: Any, load_fixture: Any) -> None:
    clock_values = iter([0.0, 0.2, 0.5])
    sleep_calls: list[float] = []

    def fake_clock() -> float:
        return next(clock_values)

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
    client = make_tradier_client(
        handler,
        clock=fake_clock,
        sleep=lambda seconds: sleep_calls.append(seconds),
        min_request_interval_seconds=0.5,
    )
    client.get_chain("SPY", date(2099, 1, 19))

    assert sleep_calls == [pytest.approx(0.3)]


def test_request_path_preserves_v1_prefix(make_tradier_client: Any, load_fixture: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/markets/options/expirations"
        return httpx.Response(200, json=load_fixture("tradier_expirations_empty.json"))

    client = make_tradier_client(handler)
    client.get_expirations("SPY")
