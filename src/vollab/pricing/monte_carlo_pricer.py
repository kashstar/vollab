import math

import numpy as np

from vollab.ingestion.models import OptionType
from vollab.pricing.models import HestonParams, OptionContract, PriceResult
from vollab.pricing.pricer import Pricer

DEFAULT_NUM_PATHS = 20_000
DEFAULT_NUM_STEPS = 100
PSI_CRITICAL = 1.5


class MonteCarloPricer(Pricer):
    """Prices European options under Heston by simulating price paths and
    averaging the discounted payoff.

    Variance is simulated with Andersen's QE (Quadratic-Exponential)
    scheme: it matches the true conditional mean and variance of Heston's
    variance process at each step while guaranteeing simulated variance
    never goes negative, unlike a naive step would. Price paths use
    antithetic variates -- every random draw is paired with its mirror
    image (negated) -- which reduces estimation noise for the same number
    of simulations, for free.

    Simplification worth knowing: the log-price step correlates price and
    variance shocks by reusing the same normal draw that drove variance's
    Gaussian branch, rather than Andersen's exact martingale-preserving
    construction. The QE variance scheme itself is implemented as
    published; only the price step is simplified. That simplification
    turned out to matter in practice: it introduced a measured ~1%
    systematic bias in E[S_T] versus the true forward, which is corrected
    for directly in price() (see the comment there) rather than by
    deriving Andersen's exact drift-correction coefficients.
    """

    def __init__(
        self,
        num_paths: int = DEFAULT_NUM_PATHS,
        num_steps: int = DEFAULT_NUM_STEPS,
        seed: int | None = None,
    ) -> None:
        self._num_paths = num_paths
        self._num_steps = num_steps
        self._rng = np.random.default_rng(seed)

    def price(self, contract: OptionContract, params: HestonParams) -> PriceResult:
        dt = contract.time_to_expiry / self._num_steps
        half_paths = self._num_paths // 2

        v = np.full(half_paths, params.v0)
        log_s = np.full(half_paths, math.log(contract.forward))
        v_anti = np.full(half_paths, params.v0)
        log_s_anti = np.full(half_paths, math.log(contract.forward))

        for _ in range(self._num_steps):
            z1 = self._rng.standard_normal(half_paths)
            z2 = self._rng.standard_normal(half_paths)

            v, log_s = self._step(v, log_s, z1, z2, params, dt)
            v_anti, log_s_anti = self._step(v_anti, log_s_anti, -z1, -z2, params, dt)

        terminal_price = np.concatenate([np.exp(log_s), np.exp(log_s_anti)])

        # Martingale correction: the discretized log-price step doesn't
        # exactly preserve E[S_T] = forward (a real, measured ~1% bias in
        # testing, not negligible), because v_avg in _step is correlated
        # with the same random shock driving the price step. Rescale the
        # whole sample so its mean matches the true forward exactly,
        # rather than trying to derive Andersen's closed-form correction
        # terms for the drift itself.
        terminal_price *= contract.forward / terminal_price.mean()

        if contract.option_type is OptionType.CALL:
            payoffs = np.maximum(terminal_price - contract.strike, 0.0)
        else:
            payoffs = np.maximum(contract.strike - terminal_price, 0.0)

        discounted = contract.discount_factor * payoffs
        price = float(np.mean(discounted))
        standard_error = float(np.std(discounted, ddof=1) / math.sqrt(len(discounted)))

        return PriceResult(price=price, standard_error=standard_error)

    def _step(
        self,
        v: np.ndarray,
        log_s: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray,
        params: HestonParams,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance variance and log-price by one time step of size dt."""
        v_next = self._step_variance(v, z1, params, dt)

        v_avg = 0.5 * (v + v_next)
        log_s_next = (
            log_s
            - 0.5 * v_avg * dt
            + params.rho * np.sqrt(v_avg * dt) * z1
            + math.sqrt(1 - params.rho**2) * np.sqrt(v_avg * dt) * z2
        )
        return v_next, log_s_next

    def _step_variance(
        self, v: np.ndarray, z: np.ndarray, params: HestonParams, dt: float
    ) -> np.ndarray:
        """Andersen's QE step for the variance process.

        Matches the CIR process's conditional mean m and variance s2 over
        one step exactly, then samples from whichever of two families best
        matches that mean/variance pair: a shifted-squared-Gaussian for
        the common case, or a Bernoulli/exponential mixture when variance
        is high relative to the mean (psi > PSI_CRITICAL) -- the regime
        where a Gaussian-based approximation would risk going negative.
        """
        kappa, theta, xi = params.kappa, params.theta, params.xi
        exp_kt = math.exp(-kappa * dt)

        m = theta + (v - theta) * exp_kt
        s2 = (
            v * xi**2 * exp_kt / kappa * (1 - exp_kt)
            + theta * xi**2 / (2 * kappa) * (1 - exp_kt) ** 2
        )
        psi = s2 / m**2

        v_next = np.empty_like(v)
        quadratic = psi <= PSI_CRITICAL

        psi_q = psi[quadratic]
        b2 = 2 / psi_q - 1 + np.sqrt(2 / psi_q) * np.sqrt(2 / psi_q - 1)
        a = m[quadratic] / (1 + b2)
        v_next[quadratic] = a * (np.sqrt(b2) + z[quadratic]) ** 2

        exponential = ~quadratic
        psi_e = psi[exponential]
        p = (psi_e - 1) / (psi_e + 1)
        beta = 2 / (m[exponential] * (psi_e + 1))
        u = self._rng.uniform(size=psi_e.shape)
        v_next[exponential] = np.where(u <= p, 0.0, np.log((1 - p) / (1 - u)) / beta)

        return v_next
