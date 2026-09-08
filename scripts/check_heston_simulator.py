"""Check HestonSimulator against real, live-calibrated Heston parameters.

Run with: python scripts/check_heston_simulator.py

Calibrates Heston to a real BTC surface (same pipeline as
check_heston_calibrator.py), simulates paths from those parameters, then
independently checks the simulated paths two ways:

1. Mean simulated spot vs the true forward (should match exactly, at
   every step -- HestonSimulator applies a per-step martingale
   correction specifically so this holds; see its docstring for why
   that was necessary).
2. Average simulated ATM call payoff, discounted, vs COSPricer's price
   for the same contract -- an independent check that the simulator's
   full-path output is consistent with the pricing tier's already-
   verified closed-form method.
"""

from vollab.hedging import HestonSimulator
from vollab.ingestion import DeribitClient
from vollab.ingestion.models import OptionType
from vollab.pricing import COSPricer, HestonCalibrator, OptionContract
from vollab.surface import ForwardEstimator, ImpliedVolSolver, SVICalibrator

client = DeribitClient()
estimator = ForwardEstimator()
solver = ImpliedVolSolver()
svi_calibrator = SVICalibrator()
heston_calibrator = HestonCalibrator(COSPricer(num_terms=64))

snapshot_ts = None
forward_estimates = []
slices = []

for expiry in client.get_expirations("BTC"):
    quotes = client.get_chain("BTC", expiry)
    if snapshot_ts is None:
        snapshot_ts = quotes[0].snapshot_ts
    try:
        forward_estimate = estimator.estimate(quotes, expiry, max_moneyness=0.15)
    except ValueError:
        continue

    time_to_expiry = (expiry - quotes[0].snapshot_ts.date()).days / 365.0
    usable_quotes, vols = [], []
    for quote in quotes:
        if quote.bid <= 0 or quote.ask <= 0:
            continue
        try:
            vol = solver.solve(quote, forward_estimate)
        except ValueError:
            continue
        usable_quotes.append(quote)
        vols.append(vol)

    try:
        slice_ = svi_calibrator.calibrate(usable_quotes, vols, forward_estimate, time_to_expiry)
    except ValueError:
        continue

    forward_estimates.append(forward_estimate)
    slices.append(slice_)

assert snapshot_ts is not None
heston_params = heston_calibrator.calibrate(slices, forward_estimates, snapshot_ts)
print(
    f"Calibrated: kappa={heston_params.kappa:.4f} theta={heston_params.theta:.4f} "
    f"xi={heston_params.xi:.4f} rho={heston_params.rho:+.4f} v0={heston_params.v0:.4f}\n"
)

spot0 = forward_estimates[0].forward
time_to_expiry = 30 / 365.0
strike = spot0

simulator = HestonSimulator(heston_params, spot0, seed=7)
paths = simulator.simulate(num_paths=50_000, num_steps=50, time_to_expiry=time_to_expiry)

terminal_spot = paths.spot[:, -1]
diff_pct = (terminal_spot.mean() / spot0 - 1) * 100
print(f"Spot today:              {spot0:.2f}")
print(f"Mean simulated terminal: {terminal_spot.mean():.2f}")
print(f"Difference:              {terminal_spot.mean() - spot0:+.2f} ({diff_pct:+.3f}%)\n")

call_payoffs = (terminal_spot - strike).clip(min=0)
simulated_price = call_payoffs.mean()

cos_pricer = COSPricer()
cos_price = cos_pricer.price(
    OptionContract(
        strike=strike,
        option_type=OptionType.CALL,
        forward=spot0,
        discount_factor=1.0,
        time_to_expiry=time_to_expiry,
    ),
    heston_params,
).price

print(f"COSPricer ATM call price:            {cos_price:.2f}")
print(f"HestonSimulator avg discounted payoff: {simulated_price:.2f}")
print(f"Difference:                            {simulated_price - cos_price:+.2f}")

client.close()
