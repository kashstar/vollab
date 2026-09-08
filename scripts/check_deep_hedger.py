"""Check DeepHedger training against real, live-calibrated Heston params.

Run with: python scripts/check_deep_hedger.py

This will raise NotImplementedError until you fill in
DeepHedgerTrainer.loss(). Once you have, this trains for a handful of
epochs and prints the loss curve (should generally trend down), then
runs the SPEC.md-described frictionless sanity check: with zero
transaction costs, a well-trained policy should land close to
DeltaHedger's analytical delta at a few sample states, since in that
frictionless limit the risk-minimizing hedge is delta hedging.
"""

from vollab.hedging import (
    CostModel,
    DeepHedger,
    DeepHedgerTrainer,
    DeltaHedger,
    HedgeState,
    HestonSimulator,
)
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
network = DeepHedger()
trainer = DeepHedgerTrainer(
    network,
    simulator,
    pricer,
    heston_params,
    strike=spot0,
    option_type=OptionType.CALL,
    cost_model=CostModel(spread_bps=0.0),
)

print("Training (frictionless, 0bps)...")
loss_history = trainer.train(num_epochs=200, num_paths=1000, num_steps=15, time_to_expiry=30 / 365)
print(f"Loss: epoch 0 = {loss_history[0]:.4f}, epoch -1 = {loss_history[-1]:.4f}")

print("\n--- Frictionless sanity check: DeepHedger vs DeltaHedger ---")
delta_hedger = DeltaHedger(pricer, heston_params)
for spot, tte in [(spot0 * 0.95, 20 / 365), (spot0, 15 / 365), (spot0 * 1.05, 5 / 365)]:
    state = HedgeState(
        spot=spot,
        variance=heston_params.v0,
        strike=spot0,
        option_type=OptionType.CALL,
        time_to_expiry=tte,
        prev_position=0.0,
    )
    deep_position = network.position(state).underlying
    delta_position = delta_hedger.position(state).underlying
    print(
        f"spot={spot:>10.2f} tte={tte*365:>4.0f}d  "
        f"deep={deep_position:>+7.4f}  delta={delta_position:>+7.4f}  "
        f"diff={deep_position - delta_position:>+7.4f}"
    )
