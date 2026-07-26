from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from vollab.ingestion.models import OptionQuote, OptionType

SNAPSHOT_TS = datetime(2026, 1, 15, 14, 30, tzinfo=UTC)


def _make_quote(**overrides: Any) -> OptionQuote:
    defaults: dict[str, Any] = {
        "source": "tradier",
        "underlying": "SPY",
        "snapshot_ts": SNAPSHOT_TS,
        "expiry": date(2026, 1, 16),
        "strike": 470.0,
        "option_type": OptionType.CALL,
        "bid": 1.0,
        "ask": 1.1,
        "last": 1.05,
        "volume": 10,
        "open_interest": 100,
        "underlying_price": 469.5,
    }
    defaults.update(overrides)
    return OptionQuote(**defaults)


def test_valid_quote_constructs() -> None:
    quote = _make_quote()
    assert quote.strike == 470.0
    assert quote.option_type is OptionType.CALL


def test_strike_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _make_quote(strike=0)
    with pytest.raises(ValidationError):
        _make_quote(strike=-5)


def test_bid_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        _make_quote(bid=-0.01)


def test_bid_zero_is_allowed() -> None:
    quote = _make_quote(bid=0.0)
    assert quote.bid == 0.0


def test_expiry_must_be_after_snapshot_date() -> None:
    with pytest.raises(ValidationError):
        _make_quote(expiry=SNAPSHOT_TS.date())
    with pytest.raises(ValidationError):
        _make_quote(expiry=SNAPSHOT_TS.date() - timedelta(days=1))


def test_expiry_after_snapshot_passes() -> None:
    quote = _make_quote(expiry=SNAPSHOT_TS.date() + timedelta(days=1))
    assert quote.expiry > SNAPSHOT_TS.date()


def test_snapshot_ts_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        _make_quote(snapshot_ts=datetime(2026, 1, 15, 14, 30))


def test_snapshot_ts_non_utc_normalized_to_utc() -> None:
    eastern = timezone(timedelta(hours=-5))
    non_utc = datetime(2026, 1, 15, 9, 30, tzinfo=eastern)
    quote = _make_quote(snapshot_ts=non_utc, expiry=date(2026, 1, 16))
    assert quote.snapshot_ts.tzinfo == UTC
    assert quote.snapshot_ts == SNAPSHOT_TS


def test_quote_is_frozen() -> None:
    quote = _make_quote()
    with pytest.raises(ValidationError):
        quote.strike = 999.0
