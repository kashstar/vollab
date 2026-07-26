from datetime import UTC, date, datetime
from typing import Any

import httpx

from vollab.ingestion.models import OptionQuote, OptionType

PRODUCTION_BASE_URL = "https://www.deribit.com/api/v2/"
TEST_BASE_URL = "https://test.deribit.com/api/v2/"


class DeribitError(Exception):
    """Raised when a Deribit API request fails."""


class DeribitClient:
    """Fetches option chains from Deribit's public market data API.

    No account or API key needed — this only touches public endpoints.
    """

    def __init__(self, *, testnet: bool = False, client: httpx.Client | None = None) -> None:
        base_url = TEST_BASE_URL if testnet else PRODUCTION_BASE_URL
        self._client = client or httpx.Client(base_url=base_url, timeout=30.0)

    def close(self) -> None:
        """Close the underlying http session."""
        self._client.close()

    def get_expirations(self, currency: str) -> list[date]:
        """Return available option expiration dates for currency (e.g. "BTC").

        Excludes today's date even if a not-yet-expired same-day contract
        exists (Deribit runs daily expiries) — OptionQuote requires expiry
        to be strictly after the snapshot date, so today's expiry is not
        fetchable through get_chain anyway.
        """
        instruments = self._get(
            "public/get_instruments",
            {"currency": currency, "kind": "option", "expired": "false"},
        )
        today = datetime.now(UTC).date()
        dates = {self._expiration_date(i["expiration_timestamp"]) for i in instruments}
        return sorted(d for d in dates if d > today)

    def get_chain(self, currency: str, expiration: date) -> list[OptionQuote]:
        """Return the option chain for currency at expiration.

        All quotes returned share one snapshot_ts. Deribit quotes option
        premiums in the underlying crypto, not USD — bid/ask/last are
        converted to USD here via each instrument's index price.
        """
        snapshot_ts = datetime.now(UTC)
        instruments = self._get(
            "public/get_instruments",
            {"currency": currency, "kind": "option", "expired": "false"},
        )
        wanted = {
            i["instrument_name"]: i
            for i in instruments
            if self._expiration_date(i["expiration_timestamp"]) == expiration
        }
        summaries = self._get(
            "public/get_book_summary_by_currency",
            {"currency": currency, "kind": "option"},
        )

        quotes = []
        for row in summaries:
            instrument = wanted.get(row["instrument_name"])
            if instrument is None:
                continue
            underlying_price = row["underlying_price"]
            last = row.get("last")
            quotes.append(
                OptionQuote(
                    source="deribit",
                    underlying=currency,
                    snapshot_ts=snapshot_ts,
                    expiry=expiration,
                    strike=instrument["strike"],
                    option_type=OptionType(instrument["option_type"]),
                    bid=(row.get("bid_price") or 0.0) * underlying_price,
                    ask=(row.get("ask_price") or 0.0) * underlying_price,
                    last=last * underlying_price if last is not None else None,
                    volume=row.get("volume", 0.0),
                    open_interest=row.get("open_interest", 0.0),
                    underlying_price=underlying_price,
                )
            )
        return quotes

    @staticmethod
    def _expiration_date(expiration_timestamp_ms: int) -> date:
        return datetime.fromtimestamp(expiration_timestamp_ms / 1000, tz=UTC).date()

    def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        response = self._client.get(path, params=params)
        if response.status_code != 200:
            raise DeribitError(
                f"Deribit request to {path} failed: "
                f"HTTP {response.status_code} {response.text[:200]}"
            )
        body: dict[str, Any] = response.json()
        if "error" in body:
            raise DeribitError(f"Deribit request to {path} failed: {body['error']}")
        result: list[dict[str, Any]] = body["result"]
        return result
