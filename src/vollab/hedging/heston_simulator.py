import math

import numpy as np

from vollab.hedging.market_simulator import MarketSimulator
from vollab.hedging.models import PathBatch
from vollab.pricing.heston_paths import HestonPathStepper
from vollab.pricing.models import HestonParams


class HestonSimulator(MarketSimulator):
    """Generates (spot, variance) paths from calibrated Heston parameters,
    via the same HestonPathStepper (Andersen QE) used by MonteCarloPricer.

    Unlike MonteCarloPricer, this keeps every intermediate step rather
    than just the terminal value. HestonPathStepper's log-price step
    carries a small systematic drift bias (measured ~0.3-1%, see its
    docstring) that MonteCarloPricer corrects only at the terminal step,
    since only the terminal payoff distribution matters for pricing.
    That's not enough here: a hedging strategy reacts to every
    intermediate spot value, and a hedge's P&L accrues incrementally
    along the path, so a bias present at every step would silently
    contaminate every backtest result, not just the final price. This
    was caught by checking simulated paths against COSPricer (see
    scripts/check_heston_simulator.py) and measuring a real, systematic
    (not noise) ~3% gap in resulting option value before this fix.

    Fixed by applying the same martingale correction at every step, not
    just the last one: after each step, rescale that step's whole
    cross-section of spot values by a single constant factor so their
    mean exactly equals spot0. Since Heston's spot process is a
    risk-neutral martingale under the zero-rate convention used
    throughout this project's hedging tier (forward == today's spot,
    discount_factor == 1), spot0 is the correct target at every step, not
    just at expiry. The correction is a uniform multiplicative shift
    across all paths at that step -- it recenters the mean without
    distorting the spread (relative differences) between paths.
    """

    def __init__(self, params: HestonParams, spot0: float, seed: int | None = None) -> None:
        self._params = params
        self._spot0 = spot0
        self._rng = np.random.default_rng(seed)
        self._stepper = HestonPathStepper(self._rng)

    def simulate(self, num_paths: int, num_steps: int, time_to_expiry: float) -> PathBatch:
        dt = time_to_expiry / num_steps

        v = np.full((num_paths, num_steps + 1), np.nan)
        spot = np.full((num_paths, num_steps + 1), np.nan)
        v[:, 0] = self._params.v0
        spot[:, 0] = self._spot0
        log_s = np.full(num_paths, math.log(self._spot0))

        for t in range(num_steps):
            z1 = self._rng.standard_normal(num_paths)
            z2 = self._rng.standard_normal(num_paths)
            v[:, t + 1], log_s = self._stepper.step(v[:, t], log_s, z1, z2, self._params, dt)

            step_spot = np.exp(log_s)
            step_spot *= self._spot0 / step_spot.mean()
            spot[:, t + 1] = step_spot
            log_s = np.log(step_spot)

        return PathBatch(spot=spot, variance=v, dt=dt)
