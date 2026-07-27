from datetime import date

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
