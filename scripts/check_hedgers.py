"""Check DeltaHedger and DeltaGammaHedger against real, live-calibrated
Heston parameters.

Run with: python scripts/check_hedgers.py

Two independent sanity checks, not just "it runs":

1. DeltaHedger: an at-the-money call's delta should be close to 0.5, and
   an at-the-money put's close to -0.5 (textbook Black-Scholes/Black-76
   behavior that Heston should still roughly reproduce near the money).
2. DeltaGammaHedger: after solving for (n_underlying, n_hedge), the
   resulting portfolio's net delta and net gamma (computed independently
   from the same greeks) should both be ~0 -- proof the 2x2 system was
   actually solved correctly, not just that it produced *some* numbers.
"""

from vollab.hedging import DeltaGammaHedger, DeltaHedger, HedgeState
from vollab.hedging.greeks import delta_gamma
from vollab.ingestion import DeribitClient
from vollab.ingestion.models import OptionType
from vollab.pricing import COSPricer, HestonCalibrator
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
        fe = estimator.estimate(quotes, expiry, max_moneyness=0.15)
    except ValueError:
        continue
    tte = (expiry - quotes[0].snapshot_ts.date()).days / 365.0
    uq, vols = [], []
    for q in quotes:
        if q.bid <= 0 or q.ask <= 0:
            continue
        try:
            v = solver.solve(q, fe)
        except ValueError:
            continue
        uq.append(q)
        vols.append(v)
    try:
        s = svi_calibrator.calibrate(uq, vols, fe, tte)
    except ValueError:
        continue
    forward_estimates.append(fe)
    slices.append(s)

assert snapshot_ts is not None
heston_params = heston_calibrator.calibrate(slices, forward_estimates, snapshot_ts)
spot0 = forward_estimates[0].forward
print(f"Spot: {spot0:.2f}")

pricer = COSPricer()

print("\n--- DeltaHedger: ATM delta sanity check ---")
for option_type in [OptionType.CALL, OptionType.PUT]:
    hedger = DeltaHedger(pricer, heston_params)
    state = HedgeState(
        spot=spot0,
        variance=heston_params.v0,
        strike=spot0,
        option_type=option_type,
        time_to_expiry=30 / 365,
        prev_position=0.0,
    )
    position = hedger.position(state)
    print(f"{option_type.value:>4} ATM delta: {position.underlying:+.4f} (expect near +-0.5)")

print("\n--- DeltaGammaHedger: net delta/gamma should be ~0 ---")
hedge_strike = spot0 * 1.05
dg_hedger = DeltaGammaHedger(pricer, heston_params, hedge_strike, OptionType.CALL)
state = HedgeState(
    spot=spot0,
    variance=heston_params.v0,
    strike=spot0 * 0.95,
    option_type=OptionType.PUT,
    time_to_expiry=30 / 365,
    prev_position=0.0,
)
position = dg_hedger.position(state)
print(f"n_underlying={position.underlying:+.4f}  n_hedge_option={position.hedge_option:+.4f}")

target_delta, target_gamma = delta_gamma(
    pricer, state.spot, state.strike, state.option_type, state.time_to_expiry, heston_params
)
hedge_delta, hedge_gamma = delta_gamma(
    pricer, state.spot, hedge_strike, OptionType.CALL, state.time_to_expiry, heston_params
)
net_delta = position.underlying + position.hedge_option * hedge_delta - target_delta
net_gamma = position.hedge_option * hedge_gamma - target_gamma
print(f"net delta (portfolio - target): {net_delta:+.6f} (expect ~0)")
print(f"net gamma (portfolio - target): {net_gamma:+.6f} (expect ~0)")

client.close()
