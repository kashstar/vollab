"""Generate the P&L distribution plot used in README.md.

Run with: python scripts/plot_hedge_pnl.py

Re-runs the same calibration and backtest as run_headline_experiment.py
(same seed) at 5bps, and plots the raw terminal P&L distributions for
DeltaHedger vs DeltaGammaHedger side by side, so the variance reduction
gamma hedging buys is visible directly, not just in summary statistics.
Saves to docs/hedge_pnl.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt

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

DELTA_COLOR = "#2a78d6"
DELTA_GAMMA_COLOR = "#1baf7a"

NUM_PATHS = 3000
NUM_STEPS = 20
TIME_TO_EXPIRY = 30 / 365
SPREAD_BPS = 5.0

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

pricer = COSPricer(num_terms=64)
simulator = HestonSimulator(heston_params, spot0, seed=42)
paths = simulator.simulate(NUM_PATHS, NUM_STEPS, TIME_TO_EXPIRY)
cost_model = CostModel(spread_bps=SPREAD_BPS)
hedge_strike = spot0 * 1.05

delta_strategy = DeltaHedger(pricer, heston_params)
delta_backtester = HedgeBacktester(
    delta_strategy, pricer, heston_params, cost_model, strike=spot0, option_type=OptionType.CALL
)
delta_pnl, _ = delta_backtester.simulate_pnl(paths)

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
dg_pnl, _ = dg_backtester.simulate_pnl(paths)

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
bins = 40
ax.hist(delta_pnl, bins=bins, alpha=0.55, color=DELTA_COLOR, label="Delta hedge", density=True)
ax.hist(
    dg_pnl, bins=bins, alpha=0.55, color=DELTA_GAMMA_COLOR, label="Delta-gamma hedge", density=True
)
ax.axvline(delta_pnl.mean(), color=DELTA_COLOR, linestyle="--", linewidth=1.5)
ax.axvline(dg_pnl.mean(), color=DELTA_GAMMA_COLOR, linestyle="--", linewidth=1.5)

ax.set_xlabel("Terminal P&L ($)")
ax.set_ylabel("Density")
ax.set_title(
    f"Hedging P&L distribution, 30-day BTC call, {SPREAD_BPS:.0f}bps costs "
    "(live-calibrated Heston)"
)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

Path("docs").mkdir(exist_ok=True)
fig.savefig("docs/hedge_pnl.png")
print(
    f"Saved docs/hedge_pnl.png\n"
    f"delta:       mean={delta_pnl.mean():.2f}  std={delta_pnl.std(ddof=1):.2f}\n"
    f"delta_gamma: mean={dg_pnl.mean():.2f}  std={dg_pnl.std(ddof=1):.2f}"
)
