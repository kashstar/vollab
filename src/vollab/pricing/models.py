import warnings

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vollab.ingestion.models import OptionType


class HestonParams(BaseModel):
    """Heston stochastic volatility model parameters.

        dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t
        correlation(price shock, variance shock) = rho

    kappa: mean-reversion speed of variance
    theta: long-run variance level
    xi: volatility of volatility
    rho: correlation between price and variance shocks, in [-1, 1]
    v0: current (initial) variance
    """

    model_config = ConfigDict(frozen=True)

    kappa: float = Field(gt=0)
    theta: float = Field(gt=0)
    xi: float = Field(gt=0)
    rho: float = Field(ge=-1, le=1)
    v0: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_feller_condition(self) -> "HestonParams":
        """Warn, don't reject, if 2*kappa*theta < xi^2.

        This is the Feller condition. When it holds, variance stays
        strictly positive almost surely. When it's violated, variance can
        touch zero under simulation -- not invalid, just a numerically
        trickier regime worth knowing about.
        """
        feller_lhs = 2 * self.kappa * self.theta
        feller_rhs = self.xi**2
        if feller_lhs < feller_rhs:
            warnings.warn(
                f"Feller condition violated (2*kappa*theta={feller_lhs:.4f} < "
                f"xi^2={feller_rhs:.4f}); variance can touch zero under these params.",
                stacklevel=2,
            )
        return self


class OptionContract(BaseModel):
    """A single European option to price, fully specified except for the
    pricing model's own parameters (HestonParams).
    """

    model_config = ConfigDict(frozen=True)

    strike: float = Field(gt=0)
    option_type: OptionType
    forward: float = Field(gt=0)
    discount_factor: float = Field(gt=0)
    time_to_expiry: float = Field(gt=0)


class PriceResult(BaseModel):
    """The result of pricing one contract.

    standard_error is None for a deterministic method (COS); set for a
    simulation-based method (Monte Carlo).
    """

    model_config = ConfigDict(frozen=True)

    price: float = Field(ge=0)
    standard_error: float | None = Field(default=None, ge=0)


class CrossCheckResult(BaseModel):
    """Result of comparing COSPricer and MonteCarloPricer on one contract.

    passed is True when the two prices differ by no more than
    CrossChecker's max_std_errors times Monte Carlo's own reported
    standard error -- i.e. the difference is plausibly just sampling
    noise, not a sign one of the two implementations is wrong.
    """

    model_config = ConfigDict(frozen=True)

    strike: float
    option_type: OptionType
    time_to_expiry: float
    cos_price: float
    mc_price: float
    mc_standard_error: float
    diff_in_std_errors: float
    passed: bool
