"""Check BootstrapSimulator against real historical BTC prices.

Run with: python scripts/check_bootstrap_simulator.py

Pulls a real year of BTC-PERPETUAL daily closes from Deribit, resamples
30-day paths, and checks that the resampled paths' return distribution
(mean, annualized std) roughly matches the real historical distribution
they were built from -- since a block bootstrap should preserve the
first and second moments of the source data, not distort them.
"""

import numpy as np

from vollab.hedging import BootstrapSimulator
from vollab.ingestion import DeribitClient

client = DeribitClient()
prices = client.get_historical_prices("BTC", days=365)
client.close()

print(f"Pulled {len(prices)} real daily closes, first={prices[0]:.2f} last={prices[-1]:.2f}")

historical_log_returns = np.diff(np.log(prices))
print(
    f"Historical daily log return: mean={historical_log_returns.mean():.6f} "
    f"std={historical_log_returns.std():.6f} "
    f"(annualized vol={historical_log_returns.std() * (365**0.5):.1%})"
)

simulator = BootstrapSimulator(prices, block_length=5, seed=7)
paths = simulator.simulate(num_paths=20_000, num_steps=30, time_to_expiry=30 / 365)

simulated_log_returns = np.diff(np.log(paths.spot), axis=1).flatten()
print(
    f"\nBootstrapped daily log return: mean={simulated_log_returns.mean():.6f} "
    f"std={simulated_log_returns.std():.6f} "
    f"(annualized vol={simulated_log_returns.std() * (365**0.5):.1%})"
)

spot0 = paths.spot[0, 0]
terminal = paths.spot[:, -1]
print(f"\nSpot today: {spot0:.2f}")
print(f"Mean simulated 30-day terminal spot: {terminal.mean():.2f}")
print(f"Median simulated 30-day terminal spot: {np.median(terminal):.2f}")
print(f"5th/95th percentile: {np.percentile(terminal, 5):.2f} / {np.percentile(terminal, 95):.2f}")

print(f"\nVariance at step 0 (mean across paths): {paths.variance[:, 0].mean():.4f}")
print(f"Variance at final step (mean across paths): {paths.variance[:, -1].mean():.4f}")
