from vollab.ingestion.models import OptionType
from vollab.pricing.cos_pricer import COSPricer
from vollab.pricing.models import HestonParams, OptionContract

DEFAULT_BUMP_PCT = 0.001


def price_at(
    pricer: COSPricer,
    spot: float,
    strike: float,
    option_type: OptionType,
    time_to_expiry: float,
    params: HestonParams,
) -> float:
    """COSPricer's price for one option, treating spot as the forward
    (this project's hedging tier assumes zero rates throughout, so
    forward == spot and discount_factor == 1.0).
    """
    contract = OptionContract(
        strike=strike,
        option_type=option_type,
        forward=spot,
        discount_factor=1.0,
        time_to_expiry=time_to_expiry,
    )
    return pricer.price(contract, params).price


def delta(
    pricer: COSPricer,
    spot: float,
    strike: float,
    option_type: OptionType,
    time_to_expiry: float,
    params: HestonParams,
    bump_pct: float = DEFAULT_BUMP_PCT,
) -> float:
    """Delta only, via a 2-point central difference (no mid price) --
    for strategies (DeltaHedger) that don't need gamma, this is a third
    fewer COSPricer calls than delta_gamma, which matters when this runs
    once per path per step across a backtest.
    """
    h = spot * bump_pct
    price_up = price_at(pricer, spot + h, strike, option_type, time_to_expiry, params)
    price_down = price_at(pricer, spot - h, strike, option_type, time_to_expiry, params)
    return (price_up - price_down) / (2 * h)


def delta_gamma(
    pricer: COSPricer,
    spot: float,
    strike: float,
    option_type: OptionType,
    time_to_expiry: float,
    params: HestonParams,
    bump_pct: float = DEFAULT_BUMP_PCT,
) -> tuple[float, float]:
    """Delta and gamma via central finite differences on COSPricer.

    bump_pct is the bump size as a fraction of spot. Delta and gamma
    share the same three price evaluations (up/mid/down), computed once
    here rather than twice across separate delta() and gamma() calls.
    """
    h = spot * bump_pct
    price_up = price_at(pricer, spot + h, strike, option_type, time_to_expiry, params)
    price_mid = price_at(pricer, spot, strike, option_type, time_to_expiry, params)
    price_down = price_at(pricer, spot - h, strike, option_type, time_to_expiry, params)

    delta = (price_up - price_down) / (2 * h)
    gamma = (price_up - 2 * price_mid + price_down) / h**2
    return delta, gamma
