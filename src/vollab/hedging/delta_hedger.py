from vollab.hedging.greeks import DEFAULT_BUMP_PCT
from vollab.hedging.greeks import delta as compute_delta
from vollab.hedging.hedging_strategy import HedgingStrategy
from vollab.hedging.models import HedgePosition, HedgeState
from vollab.pricing.cos_pricer import COSPricer
from vollab.pricing.models import HestonParams


class DeltaHedger(HedgingStrategy):
    """Classic delta hedging: hold delta shares of the underlying to
    offset a short option position.

    Delta is computed via COSPricer at each step, re-priced using the
    path's own current variance (state.variance) as v0 rather than the
    original calibration's v0 -- a real hedger reprices with the vol
    they observe right now, not the vol from when they first sold the
    option.
    """

    def __init__(
        self, pricer: COSPricer, params: HestonParams, bump_pct: float = DEFAULT_BUMP_PCT
    ) -> None:
        self._pricer = pricer
        self._params = params
        self._bump_pct = bump_pct

    def position(self, state: HedgeState) -> HedgePosition:
        params_now = self._params.model_copy(update={"v0": state.variance})
        position_delta = compute_delta(
            self._pricer,
            state.spot,
            state.strike,
            state.option_type,
            state.time_to_expiry,
            params_now,
            self._bump_pct,
        )
        return HedgePosition(underlying=position_delta)
