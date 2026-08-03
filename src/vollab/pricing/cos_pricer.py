import cmath
import math

from vollab.ingestion.models import OptionType
from vollab.pricing.models import HestonParams, OptionContract, PriceResult
from vollab.pricing.pricer import Pricer

DEFAULT_NUM_TERMS = 160
DEFAULT_TRUNCATION_MULTIPLIER = 10.0


class COSPricer(Pricer):
    """Prices European options under Heston via the COS method (Fang &
    Oosterlee, 2008): expand the risk-neutral density as a truncated
    Fourier cosine series, get the series coefficients from Heston's
    closed-form characteristic function, and sum against the payoff's own
    cosine coefficients.
    """

    def __init__(
        self,
        num_terms: int = DEFAULT_NUM_TERMS,
        truncation_multiplier: float = DEFAULT_TRUNCATION_MULTIPLIER,
    ) -> None:
        self._num_terms = num_terms
        self._truncation_multiplier = truncation_multiplier

    def price(self, contract: OptionContract, params: HestonParams) -> PriceResult:
        """Price contract under Heston via the COS method.

        Work in x = ln(S_T / strike), truncated to [a, b] (see
        _truncation_range). The characteristic function of x is
        self.characteristic_function(u, ...) shifted by strike:

            phi_x(u) = characteristic_function(u, ...) * exp(-i*u*ln(strike))

        For k = 0 .. num_terms - 1, with u_k = k*pi / (b - a):

            price_call = discount_factor * sum_k weight_k
                         * Re[phi_x(u_k) * exp(-i*u_k*a)] * V_k

        where weight_k is 0.5 for k == 0 and 1.0 otherwise (the usual
        half-weight on the first term of a cosine series), and V_k is the
        call payoff's cosine coefficient on [0, b]:

            V_k = (2 / (b - a)) * strike * (chi_k(0, b) - psi_k(0, b))

            chi_k(c, d) = 1 / (1 + (k*pi/(b-a))**2) * (
                cos(k*pi*(d-a)/(b-a)) * exp(d) - cos(k*pi*(c-a)/(b-a)) * exp(c)
                + (k*pi/(b-a)) * sin(k*pi*(d-a)/(b-a)) * exp(d)
                - (k*pi/(b-a)) * sin(k*pi*(c-a)/(b-a)) * exp(c)
            )

            psi_k(c, d) = d - c                                    if k == 0
            psi_k(c, d) = (sin(k*pi*(d-a)/(b-a)) - sin(k*pi*(c-a)/(b-a)))
                          * (b - a) / (k*pi)                        if k != 0

        For a put, use put-call parity on the resulting call price rather
        than deriving a separate set of coefficients:

            put = call - discount_factor * (forward - strike)
        """
        a, b = self._truncation_range(
            params, contract.forward, contract.strike, contract.time_to_expiry
        )
        width = b - a
        log_strike = math.log(contract.strike)

        call_price = 0.0
        for k in range(self._num_terms):
            u_k = k * math.pi / width
            phi = self.characteristic_function(
                u_k, params, contract.forward, contract.time_to_expiry
            )
            phi_x = phi * cmath.exp(-1j * u_k * log_strike)
            cosine_term = (phi_x * cmath.exp(-1j * u_k * a)).real

            v_k = self._call_coefficient(k, a, b, contract.strike)
            weight = 0.5 if k == 0 else 1.0
            call_price += weight * cosine_term * v_k

        call_price *= contract.discount_factor

        if contract.option_type is OptionType.CALL:
            price = call_price
        else:
            price = call_price - contract.discount_factor * (contract.forward - contract.strike)

        return PriceResult(price=max(price, 0.0))

    def _call_coefficient(self, k: int, a: float, b: float, strike: float) -> float:
        """V_k: the call payoff's cosine coefficient on [0, b]."""
        chi = self._chi(k, 0.0, b, a, b)
        psi = self._psi(k, 0.0, b, a, b)
        return (2 / (b - a)) * strike * (chi - psi)

    def _chi(self, k: int, c: float, d: float, a: float, b: float) -> float:
        """Cosine coefficient of e^x on [c, d]."""
        freq = k * math.pi / (b - a)
        term_d = (math.cos(freq * (d - a)) + freq * math.sin(freq * (d - a))) * math.exp(d)
        term_c = (math.cos(freq * (c - a)) + freq * math.sin(freq * (c - a))) * math.exp(c)
        return (term_d - term_c) / (1 + freq**2)

    def _psi(self, k: int, c: float, d: float, a: float, b: float) -> float:
        """Cosine coefficient of the constant 1 on [c, d]."""
        if k == 0:
            return d - c
        freq = k * math.pi / (b - a)
        return (math.sin(freq * (d - a)) - math.sin(freq * (c - a))) / freq

    def characteristic_function(
        self,
        u: complex,
        params: HestonParams,
        forward: float,
        time_to_expiry: float,
    ) -> complex:
        """Heston's characteristic function of ln(S_T), evaluated at u.

        Uses the 'Little Trap' form (Albrecher, Mayer, Schoutens & Tistaert,
        2007), which avoids branch-cut discontinuities the original 1993
        formulation is prone to when evaluated numerically.
        """
        kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0
        t = time_to_expiry

        d = cmath.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi**2 * (1j * u + u**2))
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        c = (kappa * theta / xi**2) * (
            (kappa - rho * xi * 1j * u - d) * t
            - 2 * cmath.log((1 - g * cmath.exp(-d * t)) / (1 - g))
        )
        big_d = ((kappa - rho * xi * 1j * u - d) / xi**2) * (
            (1 - cmath.exp(-d * t)) / (1 - g * cmath.exp(-d * t))
        )

        return cmath.exp(c + big_d * v0 + 1j * u * cmath.log(forward))

    def _truncation_range(
        self, params: HestonParams, forward: float, strike: float, time_to_expiry: float
    ) -> tuple[float, float]:
        """Truncation interval [a, b] for x = ln(S_T / strike).

        A simplified heuristic, not Fang & Oosterlee's exact cumulant-based
        range: centered on ln(forward/strike), widened by
        truncation_multiplier standard deviations of a rough volatility
        scale (the larger of v0 and theta). Generous enough in practice
        given num_terms is also generous, but flagged here as a
        deliberate simplification rather than the textbook-exact method.
        """
        center = math.log(forward / strike)
        vol_scale = math.sqrt(max(params.v0, params.theta))
        half_width = self._truncation_multiplier * vol_scale * math.sqrt(time_to_expiry)
        return center - half_width, center + half_width
