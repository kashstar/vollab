import numpy as np

from vollab.hedging.greeks import price_at
from vollab.hedging.hedging_strategy import HedgingStrategy
from vollab.hedging.models import CostModel, HedgePnLReport, HedgePosition, HedgeState, PathBatch
from vollab.ingestion.models import OptionType
from vollab.pricing.cos_pricer import COSPricer
from vollab.pricing.models import HestonParams

MIN_REMAINING_TIME = 1e-12


def _payoff(strike: float, option_type: OptionType, spot: float) -> float:
    if option_type is OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


class HedgeBacktester:
    """Runs a HedgingStrategy over simulated paths and reports the
    resulting P&L distribution, from the point of view of someone who
    sold one option and is hedging their exposure to it.

    Scope note: SPEC.md originally sketched HedgePnLReport as including a
    delta/gamma/theta/costs/residual P&L attribution. What's implemented
    here is the total P&L distribution plus total transaction costs, not
    that full Greek-by-Greek breakdown -- a real but separate piece of
    work (it needs its own second-order Taylor-based accounting) left for
    later rather than rushed.

    If hedge_strike/hedge_option_type are given, the strategy's
    hedge_option leg is marked to market at every step using COSPricer,
    the same way the main option is. If they're None, any nonzero
    hedge_option a strategy returns is an error -- there'd be nothing to
    price it against.
    """

    def __init__(
        self,
        strategy: HedgingStrategy,
        pricer: COSPricer,
        params: HestonParams,
        cost_model: CostModel,
        strike: float,
        option_type: OptionType,
        hedge_strike: float | None = None,
        hedge_option_type: OptionType | None = None,
    ) -> None:
        self._strategy = strategy
        self._pricer = pricer
        self._params = params
        self._cost_model = cost_model
        self._strike = strike
        self._option_type = option_type
        self._hedge_strike = hedge_strike
        self._hedge_option_type = hedge_option_type

    def run(self, paths: PathBatch) -> HedgePnLReport:
        """Run the backtest and summarize the resulting P&L distribution."""
        pnl, total_costs = self.simulate_pnl(paths)
        return self._summarize(pnl, total_costs)

    def simulate_pnl(self, paths: PathBatch) -> tuple[np.ndarray, float]:
        """Run the backtest and return the raw per-path P&L array plus
        total transaction costs, without summarizing. Exposed separately
        from run() so callers that want the full distribution (e.g. for
        plotting) don't need to duplicate this loop.
        """
        num_paths = paths.num_paths
        num_steps = paths.num_steps
        horizon = num_steps * paths.dt
        use_hedge_option = self._hedge_strike is not None

        spot0 = float(paths.spot[0, 0])
        premium = price_at(
            self._pricer, spot0, self._strike, self._option_type, horizon, self._params
        )

        pnl = np.empty(num_paths)
        total_costs = 0.0

        for p in range(num_paths):
            path_pnl = premium
            prev_position = HedgePosition(underlying=0.0, hedge_option=0.0)
            hedge_price = (
                price_at(
                    self._pricer,
                    spot0,
                    self._hedge_strike,  # type: ignore[arg-type]
                    self._hedge_option_type,  # type: ignore[arg-type]
                    horizon,
                    self._params,
                )
                if use_hedge_option
                else 0.0
            )

            for t in range(num_steps):
                spot_t = float(paths.spot[p, t])
                remaining = (num_steps - t) * paths.dt

                state = HedgeState(
                    spot=spot_t,
                    variance=float(paths.variance[p, t]),
                    strike=self._strike,
                    option_type=self._option_type,
                    time_to_expiry=remaining,
                    prev_position=prev_position.underlying,
                )
                position = self._strategy.position(state)

                trade_underlying = position.underlying - prev_position.underlying
                cost = self._cost_model.cost(spot_t, trade_underlying)
                path_pnl -= cost
                total_costs += cost

                if use_hedge_option:
                    trade_hedge = position.hedge_option - prev_position.hedge_option
                    hedge_cost = self._cost_model.cost(hedge_price, trade_hedge)
                    path_pnl -= hedge_cost
                    total_costs += hedge_cost
                elif position.hedge_option != 0.0:
                    raise ValueError(
                        "Strategy returned a nonzero hedge_option position, but "
                        "this backtester has no hedge_strike/hedge_option_type to "
                        "price it against."
                    )

                spot_next = float(paths.spot[p, t + 1])
                path_pnl += position.underlying * (spot_next - spot_t)

                if use_hedge_option:
                    remaining_next = remaining - paths.dt
                    if remaining_next > MIN_REMAINING_TIME:
                        hedge_price_next = price_at(
                            self._pricer,
                            spot_next,
                            self._hedge_strike,  # type: ignore[arg-type]
                            self._hedge_option_type,  # type: ignore[arg-type]
                            remaining_next,
                            self._params,
                        )
                    else:
                        hedge_price_next = _payoff(
                            self._hedge_strike,  # type: ignore[arg-type]
                            self._hedge_option_type,  # type: ignore[arg-type]
                            spot_next,
                        )
                    path_pnl += position.hedge_option * (hedge_price_next - hedge_price)
                    hedge_price = hedge_price_next

                prev_position = position

            spot_final = float(paths.spot[p, num_steps])
            path_pnl -= _payoff(self._strike, self._option_type, spot_final)
            pnl[p] = path_pnl

        return pnl, total_costs

    def _summarize(self, pnl: np.ndarray, total_costs: float) -> HedgePnLReport:
        sorted_pnl = np.sort(pnl)
        num_paths = len(pnl)

        def tail_mean(fraction: float) -> float:
            count = max(1, int(num_paths * fraction))
            return float(sorted_pnl[:count].mean())

        return HedgePnLReport(
            num_paths=num_paths,
            mean_pnl=float(pnl.mean()),
            std_pnl=float(pnl.std(ddof=1)),
            cvar_50=tail_mean(0.50),
            cvar_95=tail_mean(0.05),
            worst_decile_mean=tail_mean(0.10),
            total_transaction_costs=total_costs,
        )
