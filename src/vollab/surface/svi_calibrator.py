from collections.abc import Sequence
from math import log, sqrt

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from vollab.ingestion.models import OptionQuote
from vollab.surface.models import ForwardEstimate
from vollab.surface.svi_slice import SVISlice

MIN_USABLE_POINTS = 5


class SVICalibrator:
    """Fits a 5-parameter SVI slice to a set of implied vols for one expiry.

    Known limitation: the initial guess (see _initial_guess) derives its
    wing-slope estimate from the single leftmost and single rightmost
    strike, which are also the noisiest, least liquid points in the chain.
    In live testing this occasionally lands the optimizer at a different,
    sometimes worse, local solution between snapshots (e.g. rho pinned near
    -1 in one run, near 0 in another), even though the fit quality in each
    individual run still looks reasonable near the money. The standard
    fixes are averaging the slope over several wing points instead of just
    the two extremes, or multi-start optimization (try several initial
    guesses, keep the best). Neither is implemented yet.
    """

    def calibrate(
        self,
        quotes: Sequence[OptionQuote],
        vols: Sequence[float],
        forward_estimate: ForwardEstimate,
        time_to_expiry: float,
    ) -> SVISlice:
        """Fit an SVISlice via weighted least squares.

        quotes and vols must be parallel: vols[i] is the implied vol for
        quotes[i] (as returned by ImpliedVolSolver). Points are weighted by
        Black-76 vega, so quotes whose price actually pins down volatility
        precisely count more than ones where it doesn't.

        Raises:
            ValueError: if fewer than MIN_USABLE_POINTS usable points are
                available (there's no fitting 5 parameters to fewer points
                than that).
        """
        log_moneyness, observed_w, weights = self._prepare_data(
            quotes,
            vols,
            forward_estimate.forward,
            forward_estimate.discount_factor,
            time_to_expiry,
        )

        if len(log_moneyness) < MIN_USABLE_POINTS:
            raise ValueError(
                f"Only {len(log_moneyness)} usable points, need at least "
                f"{MIN_USABLE_POINTS} to fit SVI's 5 parameters."
            )

        initial_guess = self._initial_guess(log_moneyness, observed_w)
        lower_bounds = [0.0, 0.0, -1.0, -np.inf, 1e-6]
        upper_bounds = [np.inf, np.inf, 1.0, np.inf, np.inf]

        result = least_squares(
            self._residuals,
            initial_guess,
            bounds=(lower_bounds, upper_bounds),
            args=(log_moneyness, observed_w, weights),
        )

        a, b, rho, m, sigma = result.x
        return SVISlice(a=a, b=b, rho=rho, m=m, sigma=sigma, expiry=forward_estimate.expiry)

    def _residuals(
        self,
        params: np.ndarray,
        log_moneyness: np.ndarray,
        observed_w: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """The SVI objective function: weighted residual at each point.

        For candidate parameters (a, b, rho, m, sigma), computes the SVI
        curve's predicted total variance at each observed strike, and
        returns how far off that is from what was actually observed,
        scaled by each point's weight. scipy.optimize.least_squares squares
        and sums these internally, so this returns the per-point weighted
        differences, not a single summed loss.
        """
        a, b, rho, m, sigma = params
        k = log_moneyness
        predicted_w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        residuals: np.ndarray = weights * (predicted_w - observed_w)
        return residuals

    def _initial_guess(
        self, log_moneyness: np.ndarray, observed_w: np.ndarray
    ) -> list[float]:
        """Rough starting point for the optimizer, derived from the data.

        a0/m0 come from whichever point has the lowest observed variance
        (the approximate bottom of the smile). b0/rho0 come from the
        empirical slope of the wings on each side of that point, using
        SVI's own asymptotic behavior: far to the right, w(k) grows at
        roughly b*(1+rho) per unit k; far to the left, at roughly
        b*(1-rho). Matching those two slopes to what the data actually
        shows gives a far better starting guess than a fixed placeholder.
        """
        min_index = int(np.argmin(observed_w))
        m0 = float(log_moneyness[min_index])
        a0 = float(observed_w[min_index])

        left_index = int(np.argmin(log_moneyness))
        right_index = int(np.argmax(log_moneyness))

        right_run = log_moneyness[right_index] - m0
        left_run = m0 - log_moneyness[left_index]

        right_slope = (observed_w[right_index] - a0) / right_run if right_run > 1e-6 else 0.1
        left_slope = (a0 - observed_w[left_index]) / left_run if left_run > 1e-6 else 0.1

        b0 = max((right_slope + left_slope) / 2, 0.01)
        rho0 = (right_slope - left_slope) / (right_slope + left_slope)
        rho0 = max(min(rho0, 0.9), -0.9)

        return [a0, b0, rho0, m0, 0.1]

    def _prepare_data(
        self,
        quotes: Sequence[OptionQuote],
        vols: Sequence[float],
        forward: float,
        discount_factor: float,
        time_to_expiry: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Turn (quote, vol) pairs into (log_moneyness, observed_w, weight) arrays.

        Weights are each point's Black-76 vega, computed using its own
        already-solved implied vol. Vega measures how much price changes
        per unit change in volatility: a high-vega point's implied vol is
        pinned down precisely by its price, while a low-vega point (deep
        in/out-of-the-money) leaves implied vol only loosely determined
        even from an exact price. Weighting by vega gives more say to the
        points that actually constrain the curve.
        """
        log_moneyness = []
        observed_w = []
        weights = []
        for quote, vol in zip(quotes, vols, strict=True):
            if quote.bid <= 0 or quote.ask <= 0:
                continue
            vega = self._vega(forward, quote.strike, time_to_expiry, discount_factor, vol)
            log_moneyness.append(log(quote.strike / forward))
            observed_w.append(vol**2 * time_to_expiry)
            weights.append(vega)

        return np.array(log_moneyness), np.array(observed_w), np.array(weights)

    def _vega(
        self,
        forward: float,
        strike: float,
        time_to_expiry: float,
        discount_factor: float,
        vol: float,
    ) -> float:
        """Black-76 vega: how much an option's price changes per unit of vol."""
        d1 = (log(forward / strike) + 0.5 * vol**2 * time_to_expiry) / (
            vol * sqrt(time_to_expiry)
        )
        return discount_factor * forward * norm.pdf(d1) * sqrt(time_to_expiry)
