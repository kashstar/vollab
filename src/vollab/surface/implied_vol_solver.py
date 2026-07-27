from math import log, sqrt

from scipy.optimize import brentq
from scipy.stats import norm

from vollab.ingestion.models import OptionQuote, OptionType
from vollab.surface.models import ForwardEstimate

MIN_VOL = 0.001
MAX_VOL = 5.0


class ImpliedVolSolver:
    """Solves for the Black-76 implied volatility of a single quote.

    Black-76 prices European options on a forward rather than on spot
    directly, which fits naturally here since ForwardEstimator already
    recovers the forward and discount factor for each expiry.
    """

    def solve(self, quote: OptionQuote, forward_estimate: ForwardEstimate) -> float:
        """Return the implied volatility that reproduces quote's market price.

        The market price used is the mid of bid/ask. Time to expiry comes
        from quote.expiry and quote.snapshot_ts, using an Act/365
        convention (days between the two, divided by 365).

        Raises:
            ValueError: if no volatility between MIN_VOL and MAX_VOL
                reproduces the market price. This usually means the quote
                violates no-arbitrage bounds (e.g. priced below intrinsic
                value), not that the solver failed.
        """
        market_price = (quote.bid + quote.ask) / 2
        days_to_expiry = (quote.expiry - quote.snapshot_ts.date()).days
        time_to_expiry = days_to_expiry / 365.0

        def price_error(vol: float) -> float:
            model_price = self._black76_price(
                forward=forward_estimate.forward,
                strike=quote.strike,
                time_to_expiry=time_to_expiry,
                discount_factor=forward_estimate.discount_factor,
                vol=vol,
                option_type=quote.option_type,
            )
            return model_price - market_price

        try:
            return brentq(price_error, MIN_VOL, MAX_VOL)
        except ValueError as exc:
            raise ValueError(
                f"No implied vol between {MIN_VOL:.1%} and {MAX_VOL:.0%} "
                f"reproduces the market price {market_price:.2f} for strike "
                f"{quote.strike} ({quote.option_type.value}); the quote may "
                "violate no-arbitrage bounds."
            ) from exc

    def _black76_price(
        self,
        forward: float,
        strike: float,
        time_to_expiry: float,
        discount_factor: float,
        vol: float,
        option_type: OptionType,
    ) -> float:
        """Black-76 price of a European option on a forward.

        d1 = (ln(F/K) + 0.5*vol^2*T) / (vol*sqrt(T))
        d2 = d1 - vol*sqrt(T)
        call = discount_factor * (F*N(d1) - K*N(d2))
        put  = discount_factor * (K*N(-d2) - F*N(-d1))
        """
        d1 = (log(forward / strike) + 0.5 * vol**2 * time_to_expiry) / (
            vol * sqrt(time_to_expiry)
        )
        d2 = d1 - vol * sqrt(time_to_expiry)

        if option_type is OptionType.CALL:
            undiscounted = forward * norm.cdf(d1) - strike * norm.cdf(d2)
        else:
            undiscounted = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)

        return discount_factor * undiscounted
