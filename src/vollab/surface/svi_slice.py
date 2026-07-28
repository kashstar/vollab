from dataclasses import dataclass
from datetime import date
from math import sqrt


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
