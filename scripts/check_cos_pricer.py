"""Check COSPricer against a simple, hand-checkable case.

Run with: python scripts/check_cos_pricer.py

This will raise NotImplementedError until you fill in the COS summation
in COSPricer.price(). Once you have, this prices an at-the-money call and
put and checks two things by hand:

1. Put-call parity: call - put should equal discount_factor*(forward - strike).
2. As xi (vol of vol) shrinks toward 0 with v0 == theta, Heston should
   collapse toward a case with almost no extra randomness in volatility --
   worth comparing informally against a Black-76 price at vol=sqrt(theta)
   if you want an independent sanity check beyond parity.
"""

from vollab.ingestion.models import OptionType
from vollab.pricing import COSPricer, HestonParams, OptionContract

pricer = COSPricer()
params = HestonParams(kappa=2.0, theta=0.09, xi=0.3, rho=-0.6, v0=0.08)

forward = 65000.0
strike = 65000.0
discount_factor = 0.999
time_to_expiry = 0.25

call_contract = OptionContract(
    strike=strike,
    option_type=OptionType.CALL,
    forward=forward,
    discount_factor=discount_factor,
    time_to_expiry=time_to_expiry,
)
put_contract = OptionContract(
    strike=strike,
    option_type=OptionType.PUT,
    forward=forward,
    discount_factor=discount_factor,
    time_to_expiry=time_to_expiry,
)

call_result = pricer.price(call_contract, params)
put_result = pricer.price(put_contract, params)

print(f"Call price: {call_result.price:.4f}")
print(f"Put price:  {put_result.price:.4f}")

parity_lhs = call_result.price - put_result.price
parity_rhs = discount_factor * (forward - strike)
print(f"\nPut-call parity check: call - put = {parity_lhs:.4f}")
print(f"                  discount*(F-K)   = {parity_rhs:.4f}")
print(f"                  difference       = {parity_lhs - parity_rhs:+.6f}")
