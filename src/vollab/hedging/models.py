from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vollab.ingestion.models import OptionType


@dataclass
class PathBatch:
    """A batch of simulated (spot, variance) paths.

    Shape (num_paths, num_steps + 1) for both arrays: index 0 is today,
    index num_steps is expiry. dt is the time step size in years, so
    num_steps * dt == the horizon the paths were simulated over.
    """

    spot: np.ndarray
    variance: np.ndarray
    dt: float

    @property
    def num_paths(self) -> int:
        return int(self.spot.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.spot.shape[1]) - 1


@dataclass
class HedgeState:
    """What a HedgingStrategy sees at one point in time, on one path.

    time_to_expiry is what remains, not the option's original tenor.
    prev_position is the strategy's own position after the last step
    (0.0 on the first step), included so a strategy can react to its own
    recent decisions.
    """

    spot: float
    variance: float
    strike: float
    option_type: OptionType
    time_to_expiry: float
    prev_position: float


@dataclass
class HedgePosition:
    """A hedging strategy's target position at one point in time.

    underlying: units of the underlying to hold.
    hedge_option: units of a second, fixed option to hold, for strategies
        that need a second instrument to neutralize gamma as well as
        delta (DeltaGammaHedger). 0.0 for strategies that only use the
        underlying.
    """

    underlying: float
    hedge_option: float = 0.0


class CostModel(BaseModel):
    """Proportional transaction cost: cost = spread_bps/10000 * price * |trade size|.

    Applied to both legs (underlying, hedge option) whenever a strategy's
    position changes between steps.
    """

    model_config = ConfigDict(frozen=True)

    spread_bps: float = Field(ge=0)

    def cost(self, price: float, trade_size: float) -> float:
        """Dollar cost of trading trade_size units at price."""
        return self.spread_bps / 10_000 * price * abs(trade_size)


class HedgePnLReport(BaseModel):
    """Summary of one HedgeBacktester run across many simulated paths.

    P&L is terminal P&L per path (option payoff owed, minus hedge P&L,
    minus total transaction costs), from the hedger's point of view as
    the seller of one option who is hedging their exposure.
    """

    model_config = ConfigDict(frozen=True)

    num_paths: int = Field(gt=0)
    mean_pnl: float
    std_pnl: float = Field(ge=0)
    cvar_50: float
    cvar_95: float
    worst_decile_mean: float
    total_transaction_costs: float = Field(ge=0)
