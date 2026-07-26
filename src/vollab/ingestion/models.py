from datetime import UTC, date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OptionType(StrEnum):
    """Option right."""

    CALL = "call"
    PUT = "put"


class OptionQuote(BaseModel):
    """A single validated options quote snapshot from a market data source."""

    model_config = ConfigDict(frozen=True)

    source: str
    underlying: str
    snapshot_ts: datetime
    expiry: date
    strike: float = Field(gt=0)
    option_type: OptionType
    bid: float = Field(ge=0)
    ask: float
    last: float | None = None
    volume: float
    open_interest: float
    underlying_price: float

    @field_validator("snapshot_ts")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("snapshot_ts must be timezone-aware (UTC)")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_expiry_after_snapshot(self) -> "OptionQuote":
        if self.expiry <= self.snapshot_ts.date():
            raise ValueError("expiry must be after snapshot_ts date")
        return self
