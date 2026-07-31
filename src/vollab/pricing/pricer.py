from abc import ABC, abstractmethod

from vollab.pricing.models import HestonParams, OptionContract, PriceResult


class Pricer(ABC):
    """Interface for pricing a European option under Heston.

    This is one of the few places an ABC is introduced ahead of a second
    implementation actually existing -- but COSPricer and MonteCarloPricer
    are both being built specifically to cross-check each other, so the
    second implementation isn't speculative, it's the whole point.
    """

    @abstractmethod
    def price(self, contract: OptionContract, params: HestonParams) -> PriceResult:
        """Price contract under the Heston model with params."""
        raise NotImplementedError
