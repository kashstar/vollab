from dataclasses import dataclass
from datetime import date
from math import exp, pi, sqrt


@dataclass
class SVISlice:
    """The raw SVI curve for one expiry's volatility smile.

    Parameterizes total implied variance as a function of log-moneyness
    k = ln(strike / forward):

        w(k) = a + b * (rho * (k - m) + sqrt((k - m) ** 2 + sigma ** 2))

    a: overall variance level
    b: wing slope, must be >= 0
    rho: skew/rotation, must be in [-1, 1]
    m: horizontal shift
    sigma: curvature at the minimum, must be > 0. This is SVI's own
        curvature parameter, not the implied vol itself.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    expiry: date

    def w(self, k: float) -> float:
        """Total implied variance (vol^2 * time_to_expiry) at log-moneyness k."""
        return self.a + self.b * (
            self.rho * (k - self.m) + sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def implied_vol(self, k: float, time_to_expiry: float) -> float:
        """Annualized implied volatility at log-moneyness k."""
        return sqrt(self.w(k) / time_to_expiry)

    def density(self, k: float) -> float:
        """Risk-neutral probability density at log-moneyness k.

        This is the Breeden-Litzenberger result: the second derivative of
        the option price with respect to strike gives the market-implied
        probability density of where the price lands at expiry. Written
        directly in terms of SVI's own w(k) and its derivatives (Gatheral
        & Jacquier's closed form), rather than differentiating an actual
        option price numerically.

        Must be >= 0 everywhere for this slice to be free of butterfly
        arbitrage -- see ArbitrageChecker, which is what actually checks
        that across a range of k.
        """
        w = self.w(k)
        w_prime = self._w_prime(k)
        w_double_prime = self._w_double_prime(k)

        g = (
            (1 - k * w_prime / (2 * w)) ** 2
            - (w_prime**2 / 4) * (1 / w + 0.25)
            + w_double_prime / 2
        )
        d2 = -k / sqrt(w) - sqrt(w) / 2
        return g / sqrt(2 * pi * w) * exp(-(d2**2) / 2)

    def _w_prime(self, k: float) -> float:
        """First derivative of w(k) with respect to k."""
        u = k - self.m
        s = sqrt(u**2 + self.sigma**2)
        return self.b * (self.rho + u / s)

    def _w_double_prime(self, k: float) -> float:
        """Second derivative of w(k) with respect to k."""
        u = k - self.m
        s = sqrt(u**2 + self.sigma**2)
        return self.b * self.sigma**2 / s**3
