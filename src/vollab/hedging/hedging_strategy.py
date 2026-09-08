from abc import ABC, abstractmethod

from vollab.hedging.models import HedgePosition, HedgeState


class HedgingStrategy(ABC):
    """Interface for a hedging strategy: given the current state of one
    path, what position should be held right now.

    SPEC.md originally sketched this as position(state) -> float. It's
    HedgePosition here instead (underlying units, plus an optional second
    "hedge option" leg) because DeltaGammaHedger genuinely needs a second
    instrument to neutralize gamma, not just the underlying -- a
    deliberate, documented deviation from the original design note.
    """

    @abstractmethod
    def position(self, state: HedgeState) -> HedgePosition:
        """Return the target position for state."""
        raise NotImplementedError
