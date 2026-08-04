"""Check MonteCarloPricer against COSPricer.

Run with: python scripts/check_monte_carlo_pricer.py

Two completely different pricing methods -- an infinite-series Fourier
expansion (COS) and a brute-force simulation (Monte Carlo) -- should
agree, within Monte Carlo's own reported standard error, if both are
implemented correctly. This is the informal preview of what
CrossChecker will formalize.
"""

from vollab.ingestion.models import OptionType
from vollab.pricing import COSPricer, HestonParams, MonteCarloPricer, OptionContract

cos_pricer = COSPricer()
mc_pricer = MonteCarloPricer(num_paths=50_000, num_steps=100, seed=7)

params = HestonParams(kappa=2.0, theta=0.09, xi=0.3, rho=-0.6, v0=0.08)

forward = 65000.0
discount_factor = 0.999
time_to_expiry = 0.25

print(f"{'strike':>8} {'type':>5} {'COS':>10} {'MC':>10} {'MC stderr':>10} {'diff/stderr':>12}")

for strike in [55000.0, 60000.0, 65000.0, 70000.0, 80000.0]:
    for option_type in [OptionType.CALL, OptionType.PUT]:
        contract = OptionContract(
            strike=strike,
            option_type=option_type,
            forward=forward,
            discount_factor=discount_factor,
            time_to_expiry=time_to_expiry,
        )
        cos_result = cos_pricer.price(contract, params)
        mc_result = mc_pricer.price(contract, params)

        diff = mc_result.price - cos_result.price
        assert mc_result.standard_error is not None
        diff_in_stderrs = diff / mc_result.standard_error

        print(
            f"{strike:>8.0f} {option_type.value:>5} {cos_result.price:>10.2f} "
            f"{mc_result.price:>10.2f} {mc_result.standard_error:>10.3f} "
            f"{diff_in_stderrs:>+12.2f}"
        )
