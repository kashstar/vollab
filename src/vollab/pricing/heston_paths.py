import math

import numpy as np

from vollab.pricing.models import HestonParams

PSI_CRITICAL = 1.5


class HestonPathStepper:
    """One Euler-ish time step of the Heston SDE, shared by MonteCarloPricer
    and (in the hedging tier) HestonSimulator so both use exactly the same
    validated numerical scheme rather than two copies that could quietly
    drift apart.

    Variance is stepped with Andersen's QE (Quadratic-Exponential) scheme:
    it matches the true conditional mean and variance of Heston's variance
    process at each step while guaranteeing simulated variance never goes
    negative, unlike a naive step would.

    Simplification worth knowing: the log-price step correlates price and
    variance shocks by reusing the same normal draw that drove variance's
    Gaussian branch, rather than Andersen's exact martingale-preserving
    construction. This measurably biases E[S_T] away from the true
    forward (~1% in testing) -- callers that need an unbiased terminal
    distribution (MonteCarloPricer) correct for it explicitly; callers
    that need full paths for hedging (HestonSimulator) do not, and should
    keep that bias in mind.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def step(
        self,
        v: np.ndarray,
        log_s: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray,
        params: HestonParams,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance variance and log-price by one time step of size dt."""
        v_next = self.step_variance(v, z1, params, dt)

        v_avg = 0.5 * (v + v_next)
        log_s_next = (
            log_s
            - 0.5 * v_avg * dt
            + params.rho * np.sqrt(v_avg * dt) * z1
            + math.sqrt(1 - params.rho**2) * np.sqrt(v_avg * dt) * z2
        )
        return v_next, log_s_next

    def step_variance(
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
