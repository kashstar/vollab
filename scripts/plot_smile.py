"""Generate the SVI smile plot used in README.md.

Run with: python scripts/plot_smile.py

Pulls a real, live BTC chain, fits an SVI slice through it, and plots
market implied vols against the fitted curve. Saves to docs/smile.png,
which is committed to the repo since GitHub renders README images from
files in the repo, not from a script run at README-view time.
"""

from math import exp, log
from pathlib import Path

import matplotlib.pyplot as plt

from vollab.ingestion import DeribitClient
from vollab.ingestion.models import OptionType
from vollab.surface import ForwardEstimator, ImpliedVolSolver, SVICalibrator

CALL_COLOR = "#2a78d6"
PUT_COLOR = "#eb6834"
FIT_COLOR = "#1baf7a"
FORWARD_COLOR = "#898781"

client = DeribitClient()
estimator = ForwardEstimator()
solver = ImpliedVolSolver()
calibrator = SVICalibrator()

MIN_DAYS_FOR_PLOT = 20  # avoid the near-dated expiries with documented SVI instability

expiry = None
forward_estimate = None
quotes = None
for candidate_expiry in client.get_expirations("BTC"):
    candidate_quotes = client.get_chain("BTC", candidate_expiry)
    days_out = (candidate_expiry - candidate_quotes[0].snapshot_ts.date()).days
    if days_out < MIN_DAYS_FOR_PLOT:
        continue
    try:
        candidate_forward = estimator.estimate(
            candidate_quotes, candidate_expiry, max_moneyness=0.15
        )
    except ValueError:
        continue
    expiry = candidate_expiry
    forward_estimate = candidate_forward
    quotes = candidate_quotes
    break

assert expiry is not None
assert forward_estimate is not None
assert quotes is not None

time_to_expiry = (expiry - quotes[0].snapshot_ts.date()).days / 365.0

# Restrict to a representative near-the-money band. The full raw chain
# spans strikes with genuine call/put implied-vol disagreement at the far
# wings (real market microstructure noise, not a bug) -- fine to include
# in a calibration's vega weighting, but a wide, noisy illustration isn't
# what "here's what a volatility smile looks like" should show.
PLOT_MONEYNESS_RANGE = 0.2

call_strikes: list[float] = []
call_vols: list[float] = []
put_strikes: list[float] = []
put_vols: list[float] = []
usable_quotes = []
usable_vols = []

for quote in quotes:
    if quote.bid <= 0 or quote.ask <= 0:
        continue
    if abs(log(quote.strike / forward_estimate.forward)) > PLOT_MONEYNESS_RANGE:
        continue
    try:
        vol = solver.solve(quote, forward_estimate)
    except ValueError:
        continue

    usable_quotes.append(quote)
    usable_vols.append(vol)
    if quote.option_type is OptionType.CALL:
        call_strikes.append(quote.strike)
        call_vols.append(vol)
    else:
        put_strikes.append(quote.strike)
        put_vols.append(vol)

slice_ = calibrator.calibrate(usable_quotes, usable_vols, forward_estimate, time_to_expiry)
client.close()

all_strikes = call_strikes + put_strikes
k_min = log(min(all_strikes) / forward_estimate.forward)
k_max = log(max(all_strikes) / forward_estimate.forward)

curve_strikes = []
curve_vols = []
num_curve_points = 200
for i in range(num_curve_points + 1):
    k = k_min + (k_max - k_min) * i / num_curve_points
    strike = forward_estimate.forward * exp(k)
    curve_strikes.append(strike)
    curve_vols.append(slice_.implied_vol(k, time_to_expiry))

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.scatter(call_strikes, call_vols, color=CALL_COLOR, label="Call (market)", zorder=3)
ax.scatter(put_strikes, put_vols, color=PUT_COLOR, label="Put (market)", zorder=3)
ax.plot(curve_strikes, curve_vols, color=FIT_COLOR, linewidth=2, label="SVI fit", zorder=2)
ax.axvline(forward_estimate.forward, color=FORWARD_COLOR, linestyle="--", linewidth=1, zorder=1)

ax.set_xlabel("Strike ($)")
ax.set_ylabel("Implied volatility")
ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
ax.xaxis.set_major_formatter(lambda value, _: f"${value / 1000:.0f}k")
ax.set_title(f"BTC volatility smile, expiry {expiry} (live Deribit data)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

Path("docs").mkdir(exist_ok=True)
fig.savefig("docs/smile.png")
print(f"Saved docs/smile.png (expiry {expiry}, forward {forward_estimate.forward:.2f})")
