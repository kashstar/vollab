"""Check ForwardEstimator against real, live Deribit BTC chains.

Run with: python scripts/check_forward_estimator.py

Tries each available expiry in order and skips any ForwardEstimator
refuses (too few days to expiry, or too few usable strike pairs), since
that's expected for very short-dated expiries. Prints the first one that
succeeds, plus a sanity check against the real spot price (the forward
should be reasonably close to spot, and the discount factor should be
close to 1).
"""

from vollab.ingestion import DeribitClient
from vollab.surface import ForwardEstimator

client = DeribitClient()
estimator = ForwardEstimator()

for expiry in client.get_expirations("BTC"):
    quotes = client.get_chain("BTC", expiry)
    spot = quotes[0].underlying_price

    try:
        estimate = estimator.estimate(quotes, expiry)
    except ValueError as exc:
        print(f"{expiry}: skipped ({exc})")
        continue

    print(f"\nExpiry: {expiry} ({len(quotes)} contracts)")
    print(f"Spot (index price): {spot:.2f}")
    print(f"Estimated forward:    {estimate.forward:.2f}")
    print(f"Discount factor:      {estimate.discount_factor:.6f}")
    print(f"Strike pairs used:    {estimate.num_pairs}")
    print(f"Fit quality (R^2):    {estimate.r_squared:.4f}")
    print(
        f"Forward vs spot difference: {estimate.forward - spot:+.2f} "
        f"({(estimate.forward / spot - 1) * 100:+.3f}%)"
    )
    break

client.close()
