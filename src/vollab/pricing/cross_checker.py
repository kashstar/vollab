from collections.abc import Sequence

from vollab.pricing.cos_pricer import COSPricer
from vollab.pricing.models import CrossCheckResult, HestonParams, OptionContract
from vollab.pricing.monte_carlo_pricer import MonteCarloPricer

DEFAULT_MAX_STD_ERRORS = 3.0


class CrossChecker:
    """Checks that COSPricer and MonteCarloPricer agree across a set of
    contracts. Two independently-implemented pricing methods landing on
    the same answer is strong evidence both are correct; this class is
    the pricing tier's actual test suite, run against real numbers rather
    than fixtures.
    """

    def __init__(
        self,
        cos_pricer: COSPricer,
        mc_pricer: MonteCarloPricer,
        max_std_errors: float = DEFAULT_MAX_STD_ERRORS,
    ) -> None:
        self._cos_pricer = cos_pricer
        self._mc_pricer = mc_pricer
        self._max_std_errors = max_std_errors

    def check(
        self, contracts: Sequence[OptionContract], params: HestonParams
    ) -> list[CrossCheckResult]:
        """Price every contract both ways and report how well they agree.

        A contract passes if the two prices differ by no more than
        max_std_errors times Monte Carlo's own reported standard error --
        the same logic as asking "is this difference plausibly just
        sampling noise, or is something actually wrong."
        """
        results = []
        for contract in contracts:
            cos_result = self._cos_pricer.price(contract, params)
            mc_result = self._mc_pricer.price(contract, params)
            assert mc_result.standard_error is not None

            diff_in_std_errors = (
                mc_result.price - cos_result.price
            ) / mc_result.standard_error
            passed = abs(diff_in_std_errors) <= self._max_std_errors

            results.append(
                CrossCheckResult(
                    strike=contract.strike,
                    option_type=contract.option_type,
                    time_to_expiry=contract.time_to_expiry,
                    cos_price=cos_result.price,
                    mc_price=mc_result.price,
                    mc_standard_error=mc_result.standard_error,
                    diff_in_std_errors=diff_in_std_errors,
                    passed=passed,
                )
            )
        return results
