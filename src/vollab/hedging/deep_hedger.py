import math

import torch
from torch import nn

from vollab.hedging.hedging_strategy import HedgingStrategy
from vollab.hedging.models import HedgePosition, HedgeState

DEFAULT_HIDDEN_WIDTH = 32
NUM_FEATURES = 4


class DeepHedger(nn.Module, HedgingStrategy):
    """A small feedforward network that maps hedging state directly to a
    target underlying position, trained (not derived analytically like
    DeltaHedger) to minimize a risk measure of terminal hedging P&L.

    One shared network is applied identically at every time step (rather
    than a separate network per step), taking log-moneyness, remaining
    time to expiry, current vol (sqrt of variance), and the strategy's
    own previous position as input -- the same information a HedgeState
    carries, just packaged as a tensor. Underlying-only, like Buehler et
    al. (2019)'s base case: no second hedge-option leg (unlike
    DeltaGammaHedger), so results are directly comparable to DeltaHedger.
    """

    def __init__(self, hidden_width: int = DEFAULT_HIDDEN_WIDTH) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(NUM_FEATURES, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (..., NUM_FEATURES) tensor of (log_moneyness,
        time_to_expiry, vol, prev_position). Returns (..., 1) target
        underlying position.
        """
        output: torch.Tensor = self._net(features)
        return output

    @staticmethod
    def features_from_state(state: HedgeState) -> torch.Tensor:
        """Build the network's input tensor from a single HedgeState."""
        log_moneyness = math.log(state.spot / state.strike)
        vol = math.sqrt(max(state.variance, 0.0))
        return torch.tensor(
            [log_moneyness, state.time_to_expiry, vol, state.prev_position],
            dtype=torch.float32,
        )

    def position(self, state: HedgeState) -> HedgePosition:
        """Satisfies HedgingStrategy, for evaluation via HedgeBacktester.
        Runs the trained network in inference mode (no gradient tracking).
        """
        with torch.no_grad():
            features = self.features_from_state(state)
            output = self.forward(features)
        return HedgePosition(underlying=float(output.item()))
