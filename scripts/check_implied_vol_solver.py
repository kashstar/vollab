"""Check ImpliedVolSolver against a real, live Deribit BTC chain.

Run with: python scripts/check_implied_vol_solver.py

Finds the first expiry ForwardEstimator accepts, then prints the implied
volatility for every usable quote in that chain, sorted by strike. Real
smiles are smooth, not erratic; a solver bug tends to show up here as
implausible or wildly jumping numbers.
"""

from vollab.ingestion import DeribitClient
from vollab.surface import ForwardEstimator, ImpliedVolSolver

client = DeribitClient()
estimator = ForwardEstimator()
solver = ImpliedVolSolver()

for expiry in client.get_expirations("BTC"):
    quotes = client.get_chain("BTC", expiry)

    try:
        forward_estimate = estimator.estimate(quotes, expiry)
    except ValueError as exc:
        print(f"{expiry}: skipped ({exc})")
        continue

    print(
        f"\nExpiry: {expiry}  forward={forward_estimate.forward:.2f}  "
        f"discount_factor={forward_estimate.discount_factor:.4f}\n"
    )

    rows = []
    for quote in quotes:
        if quote.bid <= 0 or quote.ask <= 0:
            continue
        try:
            vol = solver.solve(quote, forward_estimate)
        except ValueError as exc:
            print(f"  strike={quote.strike:>10.0f} skipped ({exc})")
            continue
        rows.append((quote.strike, quote.option_type.value, vol))

    for strike, option_type, vol in sorted(rows):
        print(f"  strike={strike:>10.0f} {option_type:<4} implied_vol={vol:.1%}")

    break

client.close()
