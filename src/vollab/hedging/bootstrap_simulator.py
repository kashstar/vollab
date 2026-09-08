import math

import numpy as np

from vollab.hedging.market_simulator import MarketSimulator
from vollab.hedging.models import PathBatch

DEFAULT_BLOCK_LENGTH = 5
DEFAULT_REALIZED_VOL_WINDOW = 5
TRADING_DAYS_PER_YEAR = 365.0

MIN_HISTORICAL_RETURNS = 30


class BootstrapSimulator(MarketSimulator):
    """Generates paths by resampling real historical daily returns, via a
    stationary block bootstrap, rather than simulating from a model.

    This is deliberately real-world (P-measure), not risk-neutral
    (Q-measure) like HestonSimulator: it exists to check whether a
    hedging strategy that's theoretically sound under Heston's assumptions
    still performs reasonably against actual historical price behavior,
    which can have drift, fat tails, and momentum Heston doesn't capture.

    Assumes one simulated step corresponds to one historical trading day
    (dt = time_to_expiry / num_steps should be close to 1/365); this is
    not rescaled if a caller asks for a different step size.

    Variance isn't observed directly in raw price history, so it's proxied
    per path as a trailing realized-variance estimate computed from that
    path's own resampled returns (annualized, rolling window), not from
    any model. Known limitation: the first realized_vol_window steps use
    a shorter, noisier window (a 1-return window has zero variance by
    definition), so early-path variance readings are less reliable than
    later ones -- a strategy that reacts strongly to variance in the
    first few steps should be read with that in mind.
    """

    def __init__(
        self,
        historical_prices: list[float],
        spot0: float | None = None,
        block_length: int = DEFAULT_BLOCK_LENGTH,
        realized_vol_window: int = DEFAULT_REALIZED_VOL_WINDOW,
        seed: int | None = None,
    ) -> None:
        if len(historical_prices) < MIN_HISTORICAL_RETURNS + 1:
            raise ValueError(
                f"Need at least {MIN_HISTORICAL_RETURNS + 1} historical prices, "
                f"got {len(historical_prices)}."
            )
        prices = np.array(historical_prices)
        self._log_returns = np.diff(np.log(prices))
        self._spot0 = spot0 if spot0 is not None else historical_prices[-1]
        self._block_length = block_length
        self._realized_vol_window = realized_vol_window
        self._rng = np.random.default_rng(seed)

    def simulate(self, num_paths: int, num_steps: int, time_to_expiry: float) -> PathBatch:
        dt = time_to_expiry / num_steps
        returns = self._resample_returns(num_paths, num_steps)

        log_s = np.zeros((num_paths, num_steps + 1))
        log_s[:, 1:] = np.cumsum(returns, axis=1)
        log_s += math.log(self._spot0)

        variance = self._realized_variance(returns)

        return PathBatch(spot=np.exp(log_s), variance=variance, dt=dt)

    def _resample_returns(self, num_paths: int, num_steps: int) -> np.ndarray:
        """Stationary block bootstrap: for each path, concatenate random
        contiguous blocks of real historical returns (wrapping around the
        historical series if needed) until there are num_steps returns.
        """
        num_blocks = math.ceil(num_steps / self._block_length)
        n = len(self._log_returns)

        returns = np.empty((num_paths, num_blocks * self._block_length))
        starts = self._rng.integers(0, n, size=(num_paths, num_blocks))
        for b in range(num_blocks):
            block_indices = (starts[:, b : b + 1] + np.arange(self._block_length)) % n
            returns[:, b * self._block_length : (b + 1) * self._block_length] = (
                self._log_returns[block_indices]
            )

        return returns[:, :num_steps]

    def _realized_variance(self, returns: np.ndarray) -> np.ndarray:
        """Trailing realized variance at each step, from each path's own
        resampled returns, annualized. The first realized_vol_window steps
        use whatever history is available (a shorter window) rather than
        being undefined.
        """
        num_paths, num_steps = returns.shape
        variance = np.empty((num_paths, num_steps + 1))

        # No return history exists yet at t=0; use that path's own
        # whole-horizon variance as the starting proxy, not an undefined
        # or artificially-zero value.
        variance[:, 0] = np.var(returns, axis=1) * TRADING_DAYS_PER_YEAR

        for t in range(1, num_steps + 1):
            window_start = max(0, t - self._realized_vol_window)
            window = returns[:, window_start:t]
            variance[:, t] = np.var(window, axis=1) * TRADING_DAYS_PER_YEAR

        return variance
