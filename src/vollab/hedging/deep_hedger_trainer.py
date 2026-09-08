import torch

from vollab.hedging.deep_hedger import DeepHedger
from vollab.hedging.greeks import price_at
from vollab.hedging.heston_simulator import HestonSimulator
from vollab.hedging.models import CostModel, PathBatch
from vollab.ingestion.models import OptionType
from vollab.pricing.cos_pricer import COSPricer
from vollab.pricing.models import HestonParams

DEFAULT_LEARNING_RATE = 1e-3


class DeepHedgerTrainer:
    """Trains a DeepHedger by rolling it forward across simulated Heston
    paths entirely in torch (so gradients flow back through every hedging
    decision to the network's parameters), minimizing a risk measure of
    terminal P&L.

    Per SPEC.md's division of labor, the loss/objective itself (see
    loss() below) is the one piece left unimplemented here -- everything
    else (data generation, the differentiable per-step rollout, the
    optimizer loop) is built.
    """

    def __init__(
        self,
        network: DeepHedger,
        simulator: HestonSimulator,
        pricer: COSPricer,
        params: HestonParams,
        strike: float,
        option_type: OptionType,
        cost_model: CostModel,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        self._network = network
        self._simulator = simulator
        self._pricer = pricer
        self._params = params
        self._strike = strike
        self._option_type = option_type
        self._cost_model = cost_model
        self._optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train(
        self, num_epochs: int, num_paths: int, num_steps: int, time_to_expiry: float
    ) -> list[float]:
        """Train for num_epochs, each on a freshly simulated batch of
        num_paths paths. Returns the loss value from each epoch.
        """
        loss_history = []
        for _ in range(num_epochs):
            paths = self._simulator.simulate(num_paths, num_steps, time_to_expiry)
            terminal_pnl = self._simulate_hedged_pnl(paths)
            loss = self.loss(terminal_pnl)

            self._optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # torch's own stub gap, not ours
            self._optimizer.step()

            loss_history.append(float(loss.item()))
        return loss_history

    def _simulate_hedged_pnl(self, paths: PathBatch) -> torch.Tensor:
        """Rolls the network forward across every step of paths, entirely
        in torch, and returns one terminal P&L value per path (a 1D
        tensor, differentiable with respect to the network's parameters).

        Underlying-only, matching DeepHedger itself: only the option's
        own terminal payoff is needed, not a per-step repriced
        hedge-option leg, so no COSPricer calls happen inside this loop
        (only once, for the fixed initial premium).
        """
        num_paths, num_steps = paths.num_paths, paths.num_steps
        spot = torch.tensor(paths.spot, dtype=torch.float32)
        variance = torch.tensor(paths.variance, dtype=torch.float32)

        premium = price_at(
            self._pricer,
            float(spot[0, 0]),
            self._strike,
            self._option_type,
            num_steps * paths.dt,
            self._params,
        )

        prev_position = torch.zeros(num_paths)
        pnl = torch.full((num_paths,), premium, dtype=torch.float32)

        for t in range(num_steps):
            remaining = (num_steps - t) * paths.dt
            log_moneyness = torch.log(spot[:, t] / self._strike)
            vol = torch.sqrt(variance[:, t].clamp(min=0.0))
            remaining_t = torch.full((num_paths,), remaining, dtype=torch.float32)
            features = torch.stack([log_moneyness, remaining_t, vol, prev_position], dim=1)

            position = self._network(features).squeeze(-1)

            trade = position - prev_position
            cost = self._cost_model.spread_bps / 10_000 * spot[:, t] * trade.abs()
            pnl = pnl - cost
            pnl = pnl + position * (spot[:, t + 1] - spot[:, t])

            prev_position = position

        if self._option_type is OptionType.CALL:
            payoff = torch.clamp(spot[:, -1] - self._strike, min=0.0)
        else:
            payoff = torch.clamp(self._strike - spot[:, -1], min=0.0)

        pnl = pnl - payoff
        return pnl

    def loss(self, terminal_pnl: torch.Tensor) -> torch.Tensor:
        """Your piece: the DeepHedger training objective.

        terminal_pnl is a 1D tensor, one value per simulated path in this
        batch (positive is good, from the seller's point of view -- same
        sign convention as HedgeBacktester's P&L). Everything above this
        method (data generation, the differentiable per-step rollout, the
        optimizer step in train()) is built; all that's missing is
        turning terminal_pnl into a single scalar loss to minimize, one
        that rewards a tight, safe P&L distribution rather than just a
        high average one -- naively minimizing -mean(terminal_pnl) alone
        would happily learn a policy with a great average and a
        catastrophic tail.

        A standard choice (Buehler et al. 2019) is CVaR_alpha of the loss
        (-terminal_pnl): the expected loss in the worst (1 - alpha)
        fraction of outcomes. Rockafellar & Uryasev's 2000 formulation
        makes this differentiable by introducing an auxiliary scalar w
        (approximating VaR) optimized jointly with the network:

            C(w) = w + (1 / (1 - alpha)) * E[relu(-terminal_pnl - w)]

        minimized over both w and the network's own parameters at once
        (w can be a plain nn.Parameter added to DeepHedger's __init__, or
        estimated more simply each step via torch.quantile -- noisier as
        a gradient signal, but no extra parameter to manage).

        Raises:
            NotImplementedError: always, until you implement this.
        """
        raise NotImplementedError(
            "Implement the CVaR loss/objective described in the docstring above."
        )
