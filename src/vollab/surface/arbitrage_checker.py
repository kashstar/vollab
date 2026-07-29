from collections.abc import Sequence

from vollab.surface.models import ArbitrageViolation
from vollab.surface.svi_slice import SVISlice

DEFAULT_K_RANGE = (-1.5, 1.5)
DEFAULT_NUM_SAMPLES = 300


class ArbitrageChecker:
    """Checks a set of SVI slices for no-arbitrage violations.

    Butterfly: within one expiry, the implied risk-neutral density must be
    non-negative everywhere (Breeden-Litzenberger, via SVISlice.density).
    Calendar: total implied variance must not decrease as expiry lengthens,
    at the same log-moneyness, since variance only accumulates over time.
    """

    def check(
        self,
        slices: Sequence[SVISlice],
        k_range: tuple[float, float] = DEFAULT_K_RANGE,
        num_samples: int = DEFAULT_NUM_SAMPLES,
    ) -> list[ArbitrageViolation]:
        """Run both checks and return every violation found.

        Returns an empty list if the surface is arbitrage-free across the
        sampled range of log-moneyness. Findings are structured
        (ArbitrageViolation), never just a pass/fail boolean, so the
        specific strike range and magnitude of a problem is visible.
        """
        k_grid = self._sample_grid(k_range, num_samples)

        violations = []
        for slice_ in slices:
            violations.extend(self._check_butterfly(slice_, k_grid))

        by_expiry = sorted(slices, key=lambda s: s.expiry)
        for earlier, later in zip(by_expiry, by_expiry[1:], strict=False):
            violations.extend(self._check_calendar(earlier, later, k_grid))

        return violations

    def _check_butterfly(
        self, slice_: SVISlice, k_grid: list[float]
    ) -> list[ArbitrageViolation]:
        violations = []
        for k in k_grid:
            density = slice_.density(k)
            if density < 0:
                violations.append(
                    ArbitrageViolation(
                        kind="butterfly",
                        expiry=slice_.expiry,
                        log_moneyness=k,
                        detail=f"negative implied density {density:.6f} at log-moneyness {k:.4f}",
                    )
                )
        return violations

    def _check_calendar(
        self, earlier: SVISlice, later: SVISlice, k_grid: list[float]
    ) -> list[ArbitrageViolation]:
        violations = []
        for k in k_grid:
            w_earlier = earlier.w(k)
            w_later = later.w(k)
            if w_later < w_earlier:
                violations.append(
                    ArbitrageViolation(
                        kind="calendar",
                        expiry=later.expiry,
                        other_expiry=earlier.expiry,
                        log_moneyness=k,
                        detail=(
                            f"total variance at log-moneyness {k:.4f} dropped from "
                            f"{w_earlier:.6f} ({earlier.expiry}) to {w_later:.6f} "
                            f"({later.expiry})"
                        ),
                    )
                )
        return violations

    def _sample_grid(self, k_range: tuple[float, float], num_samples: int) -> list[float]:
        """Evenly spaced log-moneyness values to check, across k_range."""
        k_min, k_max = k_range
        step = (k_max - k_min) / (num_samples - 1)
        return [k_min + i * step for i in range(num_samples)]
