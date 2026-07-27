from collections.abc import Sequence
from datetime import date

from scipy.stats import linregress

from vollab.ingestion.models import OptionQuote, OptionType
from vollab.surface.models import ForwardEstimate

MIN_USABLE_PAIRS = 4
MIN_DAYS_TO_EXPIRY = 4


class ForwardEstimator:
    """Recovers the forward price and discount factor for one expiry from
    put-call parity: fitting C - P = a + b*K across strikes.
    """

    def estimate(
        self,
        quotes: Sequence[OptionQuote],
        expiry: date,
        max_moneyness: float = 0.02,
    ) -> ForwardEstimate:
        """Fit put-call parity across strikes for a single expiry.

        Put-call parity says C - P = e^(-rT) * (F - K), which rearranges to
        a straight line in K: (C - P) = a + b*K, where:
            b (the slope)     = -e^(-rT)   -> discount_factor = -b
            a (the intercept)  = e^(-rT)*F  -> forward = a / discount_factor

        max_moneyness restricts the fit to strikes within this fraction of
        the underlying's spot price (default 2%). Deep in/out-of-the-money
        options tend to have wide, unreliable bid/ask spreads even when
        technically two-sided, which otherwise adds noise to the fit.

        Raises:
            ValueError: if expiry is under MIN_DAYS_TO_EXPIRY days out, or
                if fewer than MIN_USABLE_PAIRS strikes have both a call
                and a put with a real two-sided market within
                max_moneyness of spot.
        """
        # Step 1: find the spot price and snapshot time from any quote at
        # this expiry. Both are needed before we can filter or fit anything.
        spot = None
        snapshot_ts = None
        for quote in quotes:
            if quote.expiry == expiry:
                spot = quote.underlying_price
                snapshot_ts = quote.snapshot_ts
                break

        if snapshot_ts is not None:
            days_to_expiry = (expiry - snapshot_ts.date()).days
            if days_to_expiry < MIN_DAYS_TO_EXPIRY:
                raise ValueError(
                    f"{expiry} is only {days_to_expiry} day(s) out; a "
                    f"put-call parity fit needs at least "
                    f"{MIN_DAYS_TO_EXPIRY} days of time value to be "
                    "reliable, short-dated options are dominated by "
                    "intrinsic value and the fit is too noisy."
                )

        # Step 2: pair up calls and puts at the same strike.
        strikes, differences = self._paired_strike_differences(
            quotes, expiry, spot, max_moneyness
        )

        if len(strikes) < MIN_USABLE_PAIRS:
            raise ValueError(
                f"Only {len(strikes)} usable strike pairs for {expiry}, "
                f"need at least {MIN_USABLE_PAIRS}."
            )

        # Step 3: fit a straight line through (strikes, differences), then
        # convert the slope and intercept into forward and discount_factor.
        fit = linregress(strikes, differences)

        discount_factor = -fit.slope
        forward = fit.intercept / discount_factor
        r_squared = fit.rvalue**2

        return ForwardEstimate(
            expiry=expiry,
            forward=forward,
            discount_factor=discount_factor,
            num_pairs=len(strikes),
            r_squared=r_squared,
        )

    def _paired_strike_differences(
        self,
        quotes: Sequence[OptionQuote],
        expiry: date,
        spot: float | None,
        max_moneyness: float,
    ) -> tuple[list[float], list[float]]:
        """Pair up calls and puts at the same strike for one expiry.

        Only strikes where both legs have a real two-sided market (bid > 0
        and ask > 0), and are within max_moneyness of spot, are included.
        A mid-price computed from a one-sided or empty market isn't a real
        price, and deep in/out-of-the-money strikes tend to have wide,
        unreliable spreads even when technically two-sided; both just add
        noise to the fit.

        Returns:
            (strikes, differences) where differences[i] is the mid-price
            (call - put) at strikes[i]. Same length, same order.
        """
        if spot is None:
            return [], []

        # First, split the quotes into two lookups keyed by strike: one
        # for calls, one for puts. That makes it easy to find the matching
        # put for each call in the next step.
        calls: dict[float, OptionQuote] = {}
        puts: dict[float, OptionQuote] = {}
        for quote in quotes:
            if quote.expiry != expiry:
                continue
            if quote.option_type is OptionType.CALL:
                calls[quote.strike] = quote
            else:
                puts[quote.strike] = quote

        # Now walk the calls and look up the matching put at each strike.
        strikes: list[float] = []
        differences: list[float] = []
        for strike, call in calls.items():
            put = puts.get(strike)
            if put is None:
                continue

            has_real_market = call.bid > 0 and call.ask > 0 and put.bid > 0 and put.ask > 0
            if not has_real_market:
                continue

            moneyness = abs(strike - spot) / spot
            if moneyness > max_moneyness:
                continue

            call_mid = (call.bid + call.ask) / 2
            put_mid = (put.bid + put.ask) / 2
            strikes.append(strike)
            differences.append(call_mid - put_mid)

        return strikes, differences
