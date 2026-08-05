from math import log, sqrt

from scipy.stats import norm

from vollab.ingestion.models import OptionType


def black76_price(
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
