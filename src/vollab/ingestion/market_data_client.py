from abc import ABC, abstractmethod
from datetime import date

from vollab.ingestion.models import OptionQuote


class MarketDataClient(ABC):
    """Interface for a market-data provider that lists option chains."""

    @abstractmethod
    def get_expirations(self, symbol: str) -> list[date]:
        """Return available option expiration dates for symbol."""
        raise NotImplementedError

    @abstractmethod
    def get_chain(self, symbol: str, expiration: date) -> list[OptionQuote]:
        """Return the full option chain for symbol at expiration."""
        raise NotImplementedError
