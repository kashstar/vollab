"""Check SVICalibrator against a real, live Deribit BTC chain.

Run with: python scripts/check_svi_calibrator.py

Finds the first expiry ForwardEstimator accepts, computes implied vols for
every usable quote, fits an SVI slice through them, then prints the fitted
parameters and how closely the curve's own implied vol matches the real
market vol at each strike. A good fit should track the market smile
closely near the money, where weights (tighter spreads) are highest.
"""

from math import log

from vollab.ingestion import DeribitClient
from vollab.surface import ForwardEstimator, ImpliedVolSolver, SVICalibrator

client = DeribitClient()
estimator = ForwardEstimator()
solver = ImpliedVolSolver()
calibrator = SVICalibrator()

for expiry in client.get_expirations("BTC"):
    quotes = client.get_chain("BTC", expiry)

    try:
        forward_estimate = estimator.estimate(quotes, expiry)
    except ValueError as exc:
        print(f"{expiry}: skipped ({exc})")
        continue

    time_to_expiry = (expiry - quotes[0].snapshot_ts.date()).days / 365.0

    usable_quotes = []
    vols = []
    for quote in quotes:
        if quote.bid <= 0 or quote.ask <= 0:
            continue
        try:
            vol = solver.solve(quote, forward_estimate)
        except ValueError:
            continue
        usable_quotes.append(quote)
        vols.append(vol)

    slice_ = calibrator.calibrate(usable_quotes, vols, forward_estimate, time_to_expiry)

    print(f"\nExpiry: {expiry}  forward={forward_estimate.forward:.2f}")
    print(
        f"SVI params: a={slice_.a:.6f}  b={slice_.b:.6f}  rho={slice_.rho:+.4f}  "
        f"m={slice_.m:+.4f}  sigma={slice_.sigma:.4f}\n"
    )
    print(f"{'strike':>10} {'type':>5} {'market_vol':>11} {'svi_vol':>9} {'diff':>8}")

    rows = sorted(zip(usable_quotes, vols, strict=True), key=lambda pair: pair[0].strike)
    for quote, market_vol in rows:
        k = log(quote.strike / forward_estimate.forward)
        svi_vol = slice_.implied_vol(k, time_to_expiry)
        diff = svi_vol - market_vol
        print(
            f"{quote.strike:>10.0f} {quote.option_type.value:>5} "
            f"{market_vol:>10.1%} {svi_vol:>8.1%} {diff:>+7.1%}"
        )

    break

client.close()
