"""Run CrossChecker across a strike/expiry grid.

Run with: python scripts/check_cross_checker.py

This is the pricing tier's actual test suite: if COSPricer and
MonteCarloPricer, two completely independent implementations, agree
everywhere on this grid, that's strong evidence both are correct.
"""

from vollab.ingestion.models import OptionType
from vollab.pricing import COSPricer, CrossChecker, HestonParams, MonteCarloPricer, OptionContract

cos_pricer = COSPricer()
mc_pricer = MonteCarloPricer(num_paths=50_000, num_steps=100, seed=7)
checker = CrossChecker(cos_pricer, mc_pricer, max_std_errors=3.0)

params = HestonParams(kappa=2.0, theta=0.09, xi=0.3, rho=-0.6, v0=0.08)

forward = 65000.0
discount_factor = 0.999
strikes = [50000.0, 57500.0, 65000.0, 72500.0, 80000.0]
expiries = [0.1, 0.5]

contracts = [
    OptionContract(
        strike=strike,
        option_type=option_type,
        forward=forward,
        discount_factor=discount_factor,
        time_to_expiry=time_to_expiry,
    )
    for time_to_expiry in expiries
    for strike in strikes
    for option_type in [OptionType.CALL, OptionType.PUT]
]

results = checker.check(contracts, params)

print(f"{'T':>5} {'strike':>8} {'type':>5} {'COS':>10} {'MC':>10} {'diff/se':>9}  status")
for r in results:
    status = "PASS" if r.passed else "FAIL"
    print(
        f"{r.time_to_expiry:>5.2f} {r.strike:>8.0f} {r.option_type.value:>5} "
        f"{r.cos_price:>10.2f} {r.mc_price:>10.2f} {r.diff_in_std_errors:>+9.2f}  {status}"
    )

num_passed = sum(1 for r in results if r.passed)
print(f"\n{num_passed}/{len(results)} passed (max allowed: 3.0 standard errors)")
