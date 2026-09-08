"""Check HedgeBacktester against real, live-calibrated Heston parameters.

Run with: python scripts/check_hedge_backtester.py

Small-scale, fast correctness check (not the full headline experiment,
see scripts/run_headline_experiment.py for that). The frictionless
(0bps) sanity check: a strategy that continuously delta-hedges a fairly-
priced option should have mean P&L close to zero, since delta hedging
theoretically replicates the option's payoff. With only a handful of
discrete rehedges rather than continuous hedging, some P&L spread from
discretization error is expected -- the mean should still land near
zero.
"""

from vollab.hedging import CostModel, DeltaHedger, HedgeBacktester, HestonSimulator
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
client.close()

assert snapshot_ts is not None
heston_params = heston_calibrator.calibrate(slices, forward_estimates, snapshot_ts)
spot0 = forward_estimates[0].forward
print(f"Spot: {spot0:.2f}\n")

pricer = COSPricer(num_terms=64)
simulator = HestonSimulator(heston_params, spot0, seed=11)
paths = simulator.simulate(num_paths=200, num_steps=10, time_to_expiry=30 / 365)

for spread_bps in [0.0, 5.0, 10.0]:
    strategy = DeltaHedger(pricer, heston_params)
    backtester = HedgeBacktester(
        strategy,
        pricer,
        heston_params,
        CostModel(spread_bps=spread_bps),
        strike=spot0,
        option_type=OptionType.CALL,
    )
    report = backtester.run(paths)
    print(
        f"spread={spread_bps:>4.0f}bps  mean_pnl={report.mean_pnl:>+9.2f}  "
        f"std={report.std_pnl:>8.2f}  cvar95={report.cvar_95:>+9.2f}  "
        f"costs={report.total_transaction_costs:>9.2f}"
    )
