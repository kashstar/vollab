import math

import numpy as np

from vollab.ingestion.models import OptionType
from vollab.pricing.heston_paths import HestonPathStepper
from vollab.pricing.models import HestonParams, OptionContract, PriceResult
from vollab.pricing.pricer import Pricer

DEFAULT_NUM_PATHS = 20_000
DEFAULT_NUM_STEPS = 100


class MonteCarloPricer(Pricer):
    """Prices European options under Heston by simulating price paths and
    averaging the discounted payoff.

    Variance and log-price are stepped with HestonPathStepper (Andersen's
    QE scheme). Price paths use antithetic variates -- every random draw
    is paired with its mirror image (negated) -- which reduces estimation
    noise for the same number of simulations, for free.

    HestonPathStepper's log-price step doesn't exactly preserve
    E[S_T] = forward (a real, measured ~1% bias in testing, not
    negligible). That's corrected for directly below via a martingale
    correction (rescaling the terminal sample so its mean matches the
    true forward exactly), rather than deriving Andersen's exact
    closed-form drift-correction coefficients.
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
        self._stepper = HestonPathStepper(self._rng)

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

            v, log_s = self._stepper.step(v, log_s, z1, z2, params, dt)
            v_anti, log_s_anti = self._stepper.step(v_anti, log_s_anti, -z1, -z2, params, dt)

        terminal_price = np.concatenate([np.exp(log_s), np.exp(log_s_anti)])

        # Martingale correction: see class docstring.
        terminal_price *= contract.forward / terminal_price.mean()

        if contract.option_type is OptionType.CALL:
            payoffs = np.maximum(terminal_price - contract.strike, 0.0)
        else:
            payoffs = np.maximum(contract.strike - terminal_price, 0.0)

        discounted = contract.discount_factor * payoffs
        price = float(np.mean(discounted))
        standard_error = float(np.std(discounted, ddof=1) / math.sqrt(len(discounted)))

        return PriceResult(price=price, standard_error=standard_error)
