from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ForwardEstimate(BaseModel):
    """Forward price and discount factor recovered from put-call parity
    for one expiry, via a linear fit of (C - P) against strike.
    """

    model_config = ConfigDict(frozen=True)

    expiry: date
    forward: float = Field(gt=0)
    discount_factor: float = Field(gt=0)
    num_pairs: int = Field(ge=4)
    r_squared: float = Field(ge=0, le=1)


class ArbitrageViolation(BaseModel):
    """One detected no-arbitrage violation in a fitted vol surface.

    kind "butterfly": a single slice implies a negative probability
    density at log_moneyness. kind "calendar": expiry's total variance at
    log_moneyness is less than other_expiry's (an earlier expiry), which
    can't happen since variance only accumulates over time.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["butterfly", "calendar"]
    expiry: date
    other_expiry: date | None = None
    log_moneyness: float
    detail: str
