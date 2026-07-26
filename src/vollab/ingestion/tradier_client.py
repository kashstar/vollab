from datetime import UTC, date, datetime
from typing import Any

import httpx
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from vollab.ingestion.models import OptionQuote, OptionType


class Settings(BaseSettings):
    """Environment-derived configuration for Tradier ingestion.

    Reads VOLLAB_TRADIER_* environment variables (and an optional .env
    file). Instantiating Settings() with no VOLLAB_TRADIER_TOKEN set raises
    a pydantic ValidationError immediately.
    """

    model_config = SettingsConfigDict(
        env_prefix="VOLLAB_TRADIER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    token: SecretStr
    sandbox: bool = False

    @property
    def base_url(self) -> str:
        """Tradier API base URL. Trailing slash required for httpx path joins."""
        host = "sandbox.tradier.com" if self.sandbox else "api.tradier.com"
        return f"https://{host}/v1/"


class TradierError(Exception):
    """Raised when a Tradier API request fails."""


def _as_list(value: object) -> list[Any]:
    """Normalize Tradier's list-collapsing quirk into a plain list.

    Tradier collapses a single-item JSON array to a bare scalar, and
    returns null for an empty collection.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class TradierClient:
    """Fetches option chains from the Tradier Brokerage API."""

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.token.get_secret_value()}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def close(self) -> None:
        """Close the underlying http session."""
        self._client.close()

    def get_expirations(self, symbol: str) -> list[date]:
        """Return available option expiration dates for symbol."""
        body = self._get("markets/options/expirations", {"symbol": symbol})
        raw_dates = _as_list((body.get("expirations") or {}).get("date"))
        return [date.fromisoformat(raw_date) for raw_date in raw_dates]

    def get_chain(self, symbol: str, expiration: date) -> list[OptionQuote]:
        """Return the option chain for symbol at expiration.

        All quotes returned share one snapshot_ts and underlying_price.
        """
        snapshot_ts = datetime.now(UTC)
        chain_body = self._get(
            "markets/options/chains",
            {"symbol": symbol, "expiration": expiration.isoformat()},
        )
        raw_options = _as_list((chain_body.get("options") or {}).get("option"))
        underlying_price = self._get_underlying_price(symbol)
        return [
            OptionQuote(
                source="tradier",
                underlying=raw["underlying"],
                snapshot_ts=snapshot_ts,
                expiry=date.fromisoformat(raw["expiration_date"]),
                strike=raw["strike"],
                option_type=OptionType(raw["option_type"]),
                bid=raw["bid"],
                ask=raw["ask"],
                last=raw.get("last"),
                volume=raw["volume"],
                open_interest=raw["open_interest"],
                underlying_price=underlying_price,
            )
            for raw in raw_options
        ]

    def _get_underlying_price(self, symbol: str) -> float:
        body = self._get("markets/quotes", {"symbols": symbol})
        raw_quotes = _as_list((body.get("quotes") or {}).get("quote"))
        if not raw_quotes:
            raise TradierError(f"No underlying quote returned for {symbol}")
        return float(raw_quotes[0]["last"])

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        response = self._client.get(path, params=params)
        if response.status_code != 200:
            raise TradierError(
                f"Tradier request to {path} failed: "
                f"HTTP {response.status_code} {response.text[:200]}"
            )
        body: dict[str, Any] = response.json()
        return body
