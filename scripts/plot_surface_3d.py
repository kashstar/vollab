"""Generate the 3D volatility surface plot used in README.md.

Run with: python scripts/plot_surface_3d.py

Pulls every usable real, live BTC expiry, fits ForwardEstimator ->
ImpliedVolSolver -> SVICalibrator for each one, and builds one mesh
surface (strike, days-to-expiry, implied vol) from the fitted slices,
plus the real market points each slice was fit to. Saves to
docs/surface_3d.png via plotly + kaleido, with a fixed camera angle
since a static image can't be rotated the way the dashboard's version
can.
"""

from dataclasses import dataclass
from datetime import date
from math import exp, log
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from vollab.ingestion import DeribitClient
from vollab.ingestion.models import OptionQuote
from vollab.surface import ForwardEstimator, ImpliedVolSolver, SVICalibrator
from vollab.surface.models import ForwardEstimate
from vollab.surface.svi_slice import SVISlice

PUT_COLOR = "#eb6834"
MONEYNESS_RANGE = (-0.3, 0.3)
MONEYNESS_POINTS = 40


@dataclass
class FittedExpiry:
    expiry: date
    forward_estimate: ForwardEstimate
    slice: SVISlice
    quotes: list[OptionQuote]
    vols: list[float]
    time_to_expiry: float


client = DeribitClient()
estimator = ForwardEstimator()
solver = ImpliedVolSolver()
svi_calibrator = SVICalibrator()

snapshot_ts = None
fitted: list[FittedExpiry] = []
for expiry in client.get_expirations("BTC"):
    quotes = client.get_chain("BTC", expiry)
    if snapshot_ts is None:
        snapshot_ts = quotes[0].snapshot_ts

    try:
        forward_estimate = estimator.estimate(quotes, expiry, max_moneyness=0.15)
    except ValueError:
        continue

    time_to_expiry = (expiry - quotes[0].snapshot_ts.date()).days / 365.0
    usable_quotes = []
    vols = []
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

    fitted.append(
        FittedExpiry(expiry, forward_estimate, slice_, usable_quotes, vols, time_to_expiry)
    )
client.close()

assert snapshot_ts is not None
fitted.sort(key=lambda s: s.time_to_expiry)

k_grid = np.linspace(*MONEYNESS_RANGE, MONEYNESS_POINTS)
strike_mesh = np.zeros((len(fitted), MONEYNESS_POINTS))
days_mesh = np.zeros((len(fitted), MONEYNESS_POINTS))
vol_mesh = np.zeros((len(fitted), MONEYNESS_POINTS))

market_strikes: list[float] = []
market_days: list[int] = []
market_vols: list[float] = []

for row, s in enumerate(fitted):
    days = (s.expiry - snapshot_ts.date()).days
    forward = s.forward_estimate.forward
    for col, k in enumerate(k_grid):
        strike_mesh[row, col] = forward * exp(k)
        days_mesh[row, col] = days
        vol_mesh[row, col] = s.slice.implied_vol(k, s.time_to_expiry)

    # Only plot market points within the same moneyness band as the mesh
    # -- some far-dated expiries genuinely list strikes several multiples
    # of the forward away (real, sparse, illiquid Deribit listings),
    # which would otherwise blow out the strike axis and make the whole
    # chart unreadable, the same lesson learned building plot_smile.py.
    for quote, vol in zip(s.quotes, s.vols, strict=True):
        if abs(log(quote.strike / forward)) > MONEYNESS_RANGE[1]:
            continue
        market_strikes.append(quote.strike)
        market_days.append(days)
        market_vols.append(vol)

fig = go.Figure(
    data=[
        go.Surface(
            x=strike_mesh,
            y=days_mesh,
            z=vol_mesh,
            colorscale="Viridis",
            opacity=0.85,
            showscale=True,
            colorbar={"title": "Implied vol", "tickformat": ".0%"},
        ),
        go.Scatter3d(
            x=market_strikes,
            y=market_days,
            z=market_vols,
            mode="markers",
            marker={"size": 3, "color": PUT_COLOR},
            name="Market",
        ),
    ]
)
fig.update_layout(
    title="BTC volatility surface (live Deribit data)",
    scene={
        "xaxis_title": "Strike ($)",
        "yaxis_title": "Days to expiry",
        "zaxis_title": "Implied vol",
        "zaxis_tickformat": ".0%",
        "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 0.9}},
    },
    margin={"l": 0, "r": 0, "t": 40, "b": 0},
    width=1000,
    height=700,
)

Path("docs").mkdir(exist_ok=True)
fig.write_image(Path("docs/surface_3d.png"), scale=2)
print(f"Saved docs/surface_3d.png ({len(fitted)} expiries)")
