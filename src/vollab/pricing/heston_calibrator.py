import math
import warnings
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares

from vollab.ingestion.models import OptionType
from vollab.pricing.cos_pricer import COSPricer
from vollab.pricing.models import HestonParams, OptionContract
from vollab.surface.black76 import black76_price
from vollab.surface.models import ForwardEstimate
from vollab.surface.svi_slice import SVISlice

MIN_MONEYNESS = -0.3
MAX_MONEYNESS = 0.3
DEFAULT_POINTS_PER_EXPIRY = 9


class HestonCalibrator:
    """Fits HestonParams to a fitted vol surface, via COS pricing inside
    weighted least squares.

    The calibration target is each SVI slice's own smooth curve (already
    checked for arbitrage), sampled at a grid of strikes and converted to
    Black-76 prices, not noisy raw quotes directly. Since SVI already
    smoothed away bid-ask noise, this target has none of the wing
    instability SVICalibrator had to fight; sample points are weighted
    uniformly rather than by vega.

    Known limitation, connected to SVICalibrator's own documented one:
    Heston fits a single set of 5 parameters across every expiry at once,
    so if any input slice is itself a product of SVICalibrator's
    documented instability (a parameter pinned at its bound, e.g.
    sigma near 0 or |rho| near 1 -- see svi_calibrator.py), Heston has
    to compromise trying to match a genuinely irregular target alongside
    well-behaved ones. In live testing this showed up as poor fits
    specifically on the near-dated expiries carrying that instability,
    while longer-dated (well-behaved) expiries fit within ~1 vol point.
    The real fix is upstream, in SVICalibrator's initial guess; this
    class doesn't attempt to filter or compensate for it.
    """

    def __init__(self, pricer: COSPricer) -> None:
        self._pricer = pricer

    def calibrate(
        self,
        slices: Sequence[SVISlice],
        forward_estimates: Sequence[ForwardEstimate],
        snapshot_ts: datetime,
        points_per_expiry: int = DEFAULT_POINTS_PER_EXPIRY,
    ) -> HestonParams:
        """Fit Heston's 5 parameters to the given SVI slices.

        slices and forward_estimates must be parallel: forward_estimates[i]
        describes the same expiry as slices[i].
        """
        contracts, targets = self._build_targets(
            slices, forward_estimates, snapshot_ts, points_per_expiry
        )

        atm_variance = self._average_atm_variance(slices, snapshot_ts)

        # kappa, theta, xi, rho, v0
        initial_guess = [2.0, atm_variance, 0.5, -0.3, atm_variance]
        lower_bounds = [1e-3, 1e-4, 1e-3, -0.999, 1e-4]
        upper_bounds = [20.0, 4.0, 5.0, 0.999, 4.0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = least_squares(
                self._residuals,
                initial_guess,
                bounds=(lower_bounds, upper_bounds),
                args=(contracts, targets),
            )

        kappa, theta, xi, rho, v0 = result.x
        return HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)

    def _residuals(
        self,
        param_vector: np.ndarray,
        contracts: list[OptionContract],
        targets: list[float],
    ) -> np.ndarray:
        """The Heston calibration objective: model price minus target price
        at every sampled point, for the candidate parameters in
        param_vector.
        """
        kappa, theta, xi, rho, v0 = param_vector
        params = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)

        residuals = [
            self._pricer.price(contract, params).price - target
            for contract, target in zip(contracts, targets, strict=True)
        ]
        return np.array(residuals)

    def _build_targets(
        self,
        slices: Sequence[SVISlice],
        forward_estimates: Sequence[ForwardEstimate],
        snapshot_ts: datetime,
        points_per_expiry: int,
    ) -> tuple[list[OptionContract], list[float]]:
        """Sample each slice at a grid of strikes and build (contract,
        target_price) pairs.

        Each sampled strike is priced with whichever of call/put is
        out-of-the-money there (calls above the forward, puts below) --
        the more liquid, more reliably priced side in real markets.
        """
        contracts = []
        targets = []

        for slice_, forward_estimate in zip(slices, forward_estimates, strict=True):
            time_to_expiry = (slice_.expiry - snapshot_ts.date()).days / 365.0
            if time_to_expiry <= 0:
                continue

            for i in range(points_per_expiry):
                k = MIN_MONEYNESS + (MAX_MONEYNESS - MIN_MONEYNESS) * i / (
                    points_per_expiry - 1
                )
                strike = forward_estimate.forward * math.exp(k)
                vol = slice_.implied_vol(k, time_to_expiry)
                option_type = OptionType.CALL if k >= 0 else OptionType.PUT

                target_price = black76_price(
                    forward=forward_estimate.forward,
                    strike=strike,
                    time_to_expiry=time_to_expiry,
                    discount_factor=forward_estimate.discount_factor,
                    vol=vol,
                    option_type=option_type,
                )

                contracts.append(
                    OptionContract(
                        strike=strike,
                        option_type=option_type,
                        forward=forward_estimate.forward,
                        discount_factor=forward_estimate.discount_factor,
                        time_to_expiry=time_to_expiry,
                    )
                )
                targets.append(target_price)

        return contracts, targets

    def _average_atm_variance(
        self, slices: Sequence[SVISlice], snapshot_ts: datetime
    ) -> float:
        """Average at-the-money total variance across slices, as a
        data-derived starting point for theta and v0 -- a generic
        placeholder guess is exactly what caused SVICalibrator's early
        instability, so this seeds from the real surface instead.
        """
        variances = []
        for slice_ in slices:
            time_to_expiry = (slice_.expiry - snapshot_ts.date()).days / 365.0
            if time_to_expiry <= 0:
                continue
            atm_vol = slice_.implied_vol(0.0, time_to_expiry)
            variances.append(atm_vol**2)

        return sum(variances) / len(variances)
