from vollab.hedging.greeks import DEFAULT_BUMP_PCT, delta_gamma
from vollab.hedging.hedging_strategy import HedgingStrategy
from vollab.hedging.models import HedgePosition, HedgeState
from vollab.ingestion.models import OptionType
from vollab.pricing.cos_pricer import COSPricer
from vollab.pricing.models import HestonParams

MIN_HEDGE_GAMMA = 1e-8
MAX_HEDGE_RATIO = 10.0


class DeltaGammaHedger(HedgingStrategy):
    """Neutralizes both delta and gamma of a short option position, using
    the underlying plus one fixed second option (same assumed expiry as
    the option being hedged, a different strike) to do it.

    A single instrument (the underlying) has zero gamma, so it can only
    neutralize delta; the fixed hedge option supplies the gamma exposure
    needed to also neutralize gamma. Solves the 2x2 system at each step:

        n_hedge * gamma_hedge = gamma_target
        n_underlying + n_hedge * delta_hedge = delta_target

    Known limitation, found via a real backtest and worth understanding
    rather than just guarding against: a fixed out-of-the-money hedge
    option's gamma collapses toward zero as expiry approaches, while an
    at-the-money target option's gamma grows sharply in the same window.
    That's real options behavior, not a bug, but it means the exact
    solution to the 2x2 system can demand an economically absurd hedge
    position very close to expiry (measured: from ~1x to ~15x the
    underlying's own notional, over a 30-day option's final days). Rather
    than solve exactly and hold that position, MAX_HEDGE_RATIO caps it:
    beyond that ratio, this falls back to plain delta hedging for the
    step, the same way MIN_HEDGE_GAMMA already does for the near-zero-
    gamma case.
    """

    def __init__(
        self,
        pricer: COSPricer,
        params: HestonParams,
        hedge_strike: float,
        hedge_option_type: OptionType,
        bump_pct: float = DEFAULT_BUMP_PCT,
    ) -> None:
        self._pricer = pricer
        self._params = params
        self._hedge_strike = hedge_strike
        self._hedge_option_type = hedge_option_type
        self._bump_pct = bump_pct

    def position(self, state: HedgeState) -> HedgePosition:
        params_now = self._params.model_copy(update={"v0": state.variance})

        target_delta, target_gamma = delta_gamma(
            self._pricer,
            state.spot,
            state.strike,
            state.option_type,
            state.time_to_expiry,
            params_now,
            self._bump_pct,
        )
        hedge_delta, hedge_gamma = delta_gamma(
            self._pricer,
            state.spot,
            self._hedge_strike,
            self._hedge_option_type,
            state.time_to_expiry,
            params_now,
            self._bump_pct,
        )

        if abs(hedge_gamma) < MIN_HEDGE_GAMMA:
            # The hedge option has negligible gamma here (e.g. deep
            # in/out-of-the-money near expiry). Dividing by it would
            # produce an enormous, unstable position for essentially no
            # real gamma benefit, so fall back to plain delta hedging
            # for this step instead.
            return HedgePosition(underlying=target_delta, hedge_option=0.0)

        n_hedge = target_gamma / hedge_gamma
        if abs(n_hedge) > MAX_HEDGE_RATIO:
            return HedgePosition(underlying=target_delta, hedge_option=0.0)

        n_underlying = target_delta - n_hedge * hedge_delta
        return HedgePosition(underlying=n_underlying, hedge_option=n_hedge)
