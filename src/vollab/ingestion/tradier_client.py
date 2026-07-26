import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import ValidationError

from vollab.ingestion._tradier_parsing import extract_collection
from vollab.ingestion.errors import TradierError
from vollab.ingestion.market_data_client import MarketDataClient
from vollab.ingestion.models import OptionQuote, OptionType
from vollab.ingestion.settings import Settings


class TradierClient(MarketDataClient):
    """MarketDataClient implementation backed by the Tradier Brokerage API."""

    def __init__(
        self,
        http_client: httpx.Client,
        *,
        min_request_interval_seconds: float = 0.5,
        max_retries: int = 3,
        base_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a TradierClient around a fully preconfigured httpx.Client.

        http_client must already be configured with Tradier's base_url
        (trailing slash), Bearer auth + Accept headers, and a timeout.
        TradierClient never constructs this itself from raw settings — see
        `from_settings`.
        """
        self._http_client = http_client
        self._min_request_interval_seconds = min_request_interval_seconds
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build a TradierClient wired up from environment-derived Settings.

        This is the only place Settings-driven httpx.Client construction
        happens; the primary constructor stays pure-DI for testability.
        """
        http_client = httpx.Client(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.token.get_secret_value()}",
                "Accept": "application/json",
            },
            timeout=settings.timeout_seconds,
        )
        return cls(
            http_client,
            min_request_interval_seconds=settings.min_request_interval_seconds,
            max_retries=settings.max_retries,
        )

    def close(self) -> None:
        """Close the underlying httpx.Client's connection pool."""
        self._http_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def get_expirations(self, symbol: str) -> list[date]:
        """Fetch available option expiration dates for symbol."""
        body = self._request("markets/options/expirations", params={"symbol": symbol})
        raw_dates = extract_collection(body, "expirations", "date")
        return [date.fromisoformat(raw_date) for raw_date in raw_dates]

    def get_chain(self, symbol: str, expiration: date) -> list[OptionQuote]:
        """Fetch the full option chain for symbol at expiration.

        All quotes returned share one snapshot_ts and underlying_price. A
        single malformed contract fails the whole call.
        """
        snapshot_ts = datetime.now(UTC)
        chain_body = self._request(
            "markets/options/chains",
            params={"symbol": symbol, "expiration": expiration.isoformat()},
        )
        raw_options = extract_collection(chain_body, "options", "option")
        underlying_price = self._get_underlying_price(symbol)
        return [
            self._parse_quote(
                raw_option, snapshot_ts=snapshot_ts, underlying_price=underlying_price
            )
            for raw_option in raw_options
        ]

    def _get_underlying_price(self, symbol: str) -> float:
        body = self._request("markets/quotes", params={"symbols": symbol})
        raw_quotes = extract_collection(body, "quotes", "quote")
        if not raw_quotes:
            raise TradierError(
                f"Tradier returned no quote for underlying {symbol}",
                endpoint="markets/quotes",
            )
        try:
            return float(raw_quotes[0]["last"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TradierError(
                f"Malformed quote payload for {symbol}: {exc}",
                endpoint="markets/quotes",
            ) from exc

    def _parse_quote(
        self,
        raw: dict[str, Any],
        *,
        snapshot_ts: datetime,
        underlying_price: float,
    ) -> OptionQuote:
        try:
            return OptionQuote(
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
        except (KeyError, ValueError, ValidationError) as exc:
            raise TradierError(
                f"Malformed option payload from Tradier: {exc}",
                endpoint="markets/options/chains",
            ) from exc

    def _request(self, path: str, *, params: Mapping[str, str]) -> dict[str, Any]:
        """Execute a single GET with pacing, retry, and error mapping."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._pace()
            try:
                response = self._http_client.get(path, params=params)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    self._sleep(self._backoff_seconds(attempt))
                    continue
                raise TradierError(
                    f"Tradier request to {path} failed after {attempt + 1} attempts: {exc}",
                    endpoint=path,
                ) from exc

            if response.status_code in (401, 403):
                raise TradierError(
                    f"Tradier authentication failed for {path} "
                    f"(HTTP {response.status_code}): {response.text[:500]}. "
                    "Check VOLLAB_TRADIER_TOKEN.",
                    status_code=response.status_code,
                    endpoint=path,
                )

            if response.status_code == 429 or response.status_code >= 500:
                last_exc = TradierError(
                    f"Tradier request to {path} returned HTTP {response.status_code}: "
                    f"{response.text[:500]}",
                    status_code=response.status_code,
                    endpoint=path,
                )
                if attempt < self._max_retries:
                    self._sleep(self._backoff_seconds(attempt))
                    continue
                raise last_exc

            if response.status_code >= 400:
                raise TradierError(
                    f"Tradier rejected request to {path} "
                    f"(HTTP {response.status_code}): {response.text[:500]}",
                    status_code=response.status_code,
                    endpoint=path,
                )

            try:
                body: dict[str, Any] = response.json()
            except ValueError as exc:
                raise TradierError(
                    f"Tradier response for {path} was not valid JSON", endpoint=path
                ) from exc
            return body

        assert last_exc is not None  # loop always returns or raises above
        raise TradierError(f"Tradier request to {path} failed", endpoint=path) from last_exc

    def _pace(self) -> None:
        """Sleep, if needed, to enforce min_request_interval_seconds."""
        now = self._clock()
        if self._last_request_at is not None:
            wait = self._min_request_interval_seconds - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_request_at = now

    def _backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff, capped, for zero-indexed retry attempt."""
        return min(self._base_backoff_seconds * (2.0**attempt), self._max_backoff_seconds)
