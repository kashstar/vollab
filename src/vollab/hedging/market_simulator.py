from abc import ABC, abstractmethod

from vollab.hedging.models import PathBatch


class MarketSimulator(ABC):
    """Interface for generating simulated (spot, variance) paths for a
    hedging backtest.

    A second implementation (BootstrapSimulator) exists specifically to
    check HedgeBacktester results against real historical dynamics, not
    just Heston's own idealized world, so this ABC isn't speculative.
    """

    @abstractmethod
    def simulate(self, num_paths: int, num_steps: int, time_to_expiry: float) -> PathBatch:
        """Generate num_paths paths, each num_steps steps, spanning
        time_to_expiry years.
        """
        raise NotImplementedError
