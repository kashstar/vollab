"""The headline experiment: delta vs delta-gamma hedging at 0/5/10bps.

Run with: python scripts/run_headline_experiment.py

Calibrates Heston to a real, live BTC surface, simulates a batch of
30-day price paths from those parameters, then backtests DeltaHedger and
DeltaGammaHedger over the same paths at three transaction-cost levels.
DeepHedger is left out here: its loss/objective is still an open
exercise (see DeepHedgerTrainer.loss), so there's no trained policy to
include yet. Once that's implemented, scripts/check_deep_hedger.py shows
how to train one; slotting a trained DeepHedger into this same
comparison is a small addition once that piece exists.

Results (mean P&L, std, CVaR95, total costs) print as a table and are
also saved to docs/headline_experiment.json for the README.
"""

import json
import time
from pathlib import Path

from vollab.hedging import (
    CostModel,
    DeltaGammaHedger,
    DeltaHedger,
    HedgeBacktester,
    HestonSimulator,
)
from vollab.ingestion import DeribitClient
from vollab.ingestion.models import OptionType
from vollab.pricing import COSPricer, HestonCalibrator
from vollab.surface import ForwardEstimator, ImpliedVolSolver, SVICalibrator

NUM_PATHS = 3000
NUM_STEPS = 20
TIME_TO_EXPIRY = 30 / 365
SPREAD_LEVELS_BPS = [0.0, 5.0, 10.0]

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
client.close()

assert snapshot_ts is not None
heston_params = heston_calibrator.calibrate(slices, forward_estimates, snapshot_ts)
spot0 = forward_estimates[0].forward
print(
    f"Calibrated Heston: kappa={heston_params.kappa:.4f} theta={heston_params.theta:.4f} "
    f"xi={heston_params.xi:.4f} rho={heston_params.rho:+.4f} v0={heston_params.v0:.4f}"
)
print(f"Spot: {spot0:.2f}\n")

pricer = COSPricer(num_terms=64)
simulator = HestonSimulator(heston_params, spot0, seed=42)
paths = simulator.simulate(NUM_PATHS, NUM_STEPS, TIME_TO_EXPIRY)

hedge_strike = spot0 * 1.05
results: dict[str, dict[str, dict[str, float]]] = {"delta": {}, "delta_gamma": {}}

print(f"{'strategy':>12} {'spread':>7} {'mean_pnl':>10} {'std':>10} {'cvar_95':>10} {'costs':>10}")

for spread_bps in SPREAD_LEVELS_BPS:
    cost_model = CostModel(spread_bps=spread_bps)

    t0 = time.time()
    delta_strategy = DeltaHedger(pricer, heston_params)
    delta_backtester = HedgeBacktester(
        delta_strategy, pricer, heston_params, cost_model, strike=spot0, option_type=OptionType.CALL
    )
    delta_report = delta_backtester.run(paths)
    delta_elapsed = time.time() - t0

    t0 = time.time()
    dg_strategy = DeltaGammaHedger(pricer, heston_params, hedge_strike, OptionType.CALL)
    dg_backtester = HedgeBacktester(
        dg_strategy,
        pricer,
        heston_params,
        cost_model,
        strike=spot0,
        option_type=OptionType.CALL,
        hedge_strike=hedge_strike,
        hedge_option_type=OptionType.CALL,
    )
    dg_report = dg_backtester.run(paths)
    dg_elapsed = time.time() - t0

    key = f"{spread_bps:.0f}bps"
    results["delta"][key] = delta_report.model_dump()
    results["delta_gamma"][key] = dg_report.model_dump()

    print(
        f"{'delta':>12} {spread_bps:>6.0f}b {delta_report.mean_pnl:>10.2f} "
        f"{delta_report.std_pnl:>10.2f} {delta_report.cvar_95:>10.2f} "
        f"{delta_report.total_transaction_costs:>10.2f}  ({delta_elapsed:.1f}s)"
    )
    print(
        f"{'delta_gamma':>12} {spread_bps:>6.0f}b {dg_report.mean_pnl:>10.2f} "
        f"{dg_report.std_pnl:>10.2f} {dg_report.cvar_95:>10.2f} "
        f"{dg_report.total_transaction_costs:>10.2f}  ({dg_elapsed:.1f}s)"
    )

Path("docs").mkdir(exist_ok=True)
with open("docs/headline_experiment.json", "w") as f:
    json.dump(
        {
            "spot0": spot0,
            "num_paths": NUM_PATHS,
            "num_steps": NUM_STEPS,
            "time_to_expiry_days": 30,
            "heston_params": heston_params.model_dump(),
            "results": results,
        },
        f,
        indent=2,
    )
print("\nSaved docs/headline_experiment.json")
