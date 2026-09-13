"""VolLab dashboard: live vol surface, Heston pricing, and hedging backtest.

Run with: streamlit run scripts/dashboard.py

Every number on this page comes from the real, live Deribit API and the
project's own classes, the same ones scripts/check_*.py exercise. There's
no synthetic or cached-from-file data here except Streamlit's own
short-TTL caching, used to avoid re-fetching/re-calibrating on every
widget interaction.
"""

from datetime import date, datetime
from math import exp, log

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.axes import Axes

from vollab.hedging import (
    CostModel,
    DeltaGammaHedger,
    DeltaHedger,
    HedgeBacktester,
    HestonSimulator,
)
from vollab.ingestion import DeribitClient
from vollab.ingestion.models import OptionQuote, OptionType
from vollab.pricing import (
    COSPricer,
    HestonCalibrator,
    HestonParams,
    MonteCarloPricer,
    OptionContract,
)
from vollab.surface import ArbitrageChecker, ForwardEstimator, ImpliedVolSolver, SVICalibrator
from vollab.surface.models import ForwardEstimate
from vollab.surface.svi_slice import SVISlice

CALL_COLOR = "#2a78d6"
PUT_COLOR = "#eb6834"
FIT_COLOR = "#1baf7a"
DELTA_GAMMA_COLOR = "#1baf7a"

SURFACE_CACHE_TTL = 30
HESTON_NUM_TERMS = 64
SURFACE_3D_MONEYNESS_RANGE = (-0.3, 0.3)
SURFACE_3D_MONEYNESS_POINTS = 40

st.set_page_config(page_title="VolLab", layout="wide")


class SurfaceSlice:
    """One expiry's fitted forward, SVI slice, and the quotes/vols it was
    built from -- bundled together since every tab needs some subset of
    these, and they're all produced by the same fitting pass.
    """

    def __init__(
        self,
        expiry: date,
        forward_estimate: ForwardEstimate,
        slice_: SVISlice,
        quotes: list[OptionQuote],
        vols: list[float],
        time_to_expiry: float,
    ) -> None:
        self.expiry = expiry
        self.forward_estimate = forward_estimate
        self.slice = slice_
        self.quotes = quotes
        self.vols = vols
        self.time_to_expiry = time_to_expiry


@st.cache_data(ttl=SURFACE_CACHE_TTL, show_spinner=False)
def load_surface(currency: str) -> tuple[datetime, list[SurfaceSlice], list[tuple[str, str]]]:
    """Fetch live chains for every expiry and fit ForwardEstimator ->
    ImpliedVolSolver -> SVICalibrator for each one that's usable.

    Returns (snapshot_ts, fitted slices, [(expiry_str, skip_reason), ...]).
    """
    client = DeribitClient()
    estimator = ForwardEstimator()
    solver = ImpliedVolSolver()
    svi_calibrator = SVICalibrator()

    snapshot_ts = None
    fitted = []
    skipped = []
    try:
        for expiry in client.get_expirations(currency):
            quotes = client.get_chain(currency, expiry)
            if snapshot_ts is None:
                snapshot_ts = quotes[0].snapshot_ts

            try:
                forward_estimate = estimator.estimate(quotes, expiry, max_moneyness=0.15)
            except ValueError as exc:
                skipped.append((str(expiry), str(exc)))
                continue

            time_to_expiry = (expiry - quotes[0].snapshot_ts.date()).days / 365.0
            usable_quotes = []
            vols = []
            for quote in quotes:
                if quote.bid <= 0 or quote.ask <= 0:
                    continue
                try:
                    vol = solver.solve(quote, forward_estimate)
                except ValueError:
                    continue
                usable_quotes.append(quote)
                vols.append(vol)

            try:
                slice_ = svi_calibrator.calibrate(
                    usable_quotes, vols, forward_estimate, time_to_expiry
                )
            except ValueError as exc:
                skipped.append((str(expiry), str(exc)))
                continue

            fitted.append(
                SurfaceSlice(
                    expiry, forward_estimate, slice_, usable_quotes, vols, time_to_expiry
                )
            )
    finally:
        client.close()

    assert snapshot_ts is not None
    return snapshot_ts, fitted, skipped


@st.cache_data(ttl=SURFACE_CACHE_TTL, show_spinner=False)
def calibrate_heston(currency: str) -> HestonParams:
    """Calibrate Heston to the whole live surface. Cached alongside
    load_surface (same TTL) since both change together, and both tabs
    that need Heston params (Pricing, Hedging) share this one fit.
    """
    snapshot_ts, fitted, _ = load_surface(currency)
    slices = [s.slice for s in fitted]
    forward_estimates = [s.forward_estimate for s in fitted]
    return HestonCalibrator(COSPricer(num_terms=HESTON_NUM_TERMS)).calibrate(
        slices, forward_estimates, snapshot_ts
    )


def style_axes(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_surface_figure(
    fitted: list[SurfaceSlice], snapshot_ts: datetime, currency: str
) -> go.Figure:
    """Build a 3D (strike, days-to-expiry, implied vol) figure: a mesh
    surface from each expiry's fitted SVI slice, plus the real market
    points that went into fitting it.

    Each expiry contributes one row of the mesh, sampled at the same
    grid of log-moneyness values (so every row lines up), but at that
    expiry's own forward -- so the strike axis is in real dollars, not
    moneyness, matching how a trading desk actually looks at a surface.
    """
    fitted_sorted = sorted(fitted, key=lambda s: s.time_to_expiry)
    k_grid = np.linspace(*SURFACE_3D_MONEYNESS_RANGE, SURFACE_3D_MONEYNESS_POINTS)

    strike_mesh = np.zeros((len(fitted_sorted), SURFACE_3D_MONEYNESS_POINTS))
    days_mesh = np.zeros((len(fitted_sorted), SURFACE_3D_MONEYNESS_POINTS))
    vol_mesh = np.zeros((len(fitted_sorted), SURFACE_3D_MONEYNESS_POINTS))

    market_strikes: list[float] = []
    market_days: list[int] = []
    market_vols: list[float] = []

    for row, s in enumerate(fitted_sorted):
        days = (s.expiry - snapshot_ts.date()).days
        forward = s.forward_estimate.forward
        for col, k in enumerate(k_grid):
            strike_mesh[row, col] = forward * exp(k)
            days_mesh[row, col] = days
            vol_mesh[row, col] = s.slice.implied_vol(k, s.time_to_expiry)

        # Only plot market points within the same moneyness band as the
        # mesh -- some far-dated expiries genuinely list strikes several
        # multiples of the forward away (real, sparse, illiquid Deribit
        # listings), which would otherwise blow out the strike axis and
        # make the whole chart unreadable, the same lesson learned
        # building plot_smile.py.
        for quote, vol in zip(s.quotes, s.vols, strict=True):
            if abs(log(quote.strike / forward)) > SURFACE_3D_MONEYNESS_RANGE[1]:
                continue
            market_strikes.append(quote.strike)
            market_days.append(days)
            market_vols.append(vol)

    fig = go.Figure(
        data=[
            go.Surface(
                x=strike_mesh,
                y=days_mesh,
                z=vol_mesh,
                colorscale="Viridis",
                opacity=0.85,
                name="SVI fit",
                showscale=True,
                colorbar={"title": "Implied vol", "tickformat": ".0%"},
            ),
            go.Scatter3d(
                x=market_strikes,
                y=market_days,
                z=market_vols,
                mode="markers",
                marker={"size": 3, "color": PUT_COLOR},
                name="Market",
            ),
        ]
    )
    fig.update_layout(
        title=f"{currency} volatility surface",
        scene={
            "xaxis_title": "Strike ($)",
            "yaxis_title": "Days to expiry",
            "zaxis_title": "Implied vol",
            "zaxis_tickformat": ".0%",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=650,
    )
    return fig


st.title("VolLab")
st.caption(
    "Live BTC/ETH options research: implied vol surface, Heston pricing, "
    "and a hedging backtest, built on Deribit's public market data."
)

currency = st.sidebar.selectbox("Underlying", ["BTC", "ETH"])
if st.sidebar.button("Refresh live data"):
    load_surface.clear()
    calibrate_heston.clear()

with st.spinner(f"Fetching live {currency} chain and fitting the surface..."):
    snapshot_ts, fitted, skipped = load_surface(currency)

if not fitted:
    st.error(
        f"No usable {currency} expiries right now (all skipped: "
        f"{[reason for _, reason in skipped]}). Try refreshing."
    )
    st.stop()

st.sidebar.caption(f"Snapshot: {snapshot_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
st.sidebar.caption(f"{len(fitted)} expiries fitted, {len(skipped)} skipped")

tab_smile, tab_surface3d, tab_term, tab_pricing, tab_hedging = st.tabs(
    ["Volatility Smile", "3D Surface", "Term Structure", "Heston Pricing", "Hedging Backtest"]
)

with tab_smile:
    st.subheader(f"{currency} volatility smile")

    expiry_labels = [str(s.expiry) for s in fitted]
    chosen_label = st.selectbox("Expiry", expiry_labels)
    chosen = fitted[expiry_labels.index(chosen_label)]

    fe = chosen.forward_estimate
    slice_ = chosen.slice

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Forward", f"${fe.forward:,.0f}")
    col2.metric("Discount factor", f"{fe.discount_factor:.4f}")
    col3.metric("Strike pairs used", fe.num_pairs)
    col4.metric("Forward fit R2", f"{fe.r_squared:.4f}")

    call_strikes: list[float] = []
    call_vols: list[float] = []
    put_strikes: list[float] = []
    put_vols: list[float] = []
    for quote, vol in zip(chosen.quotes, chosen.vols, strict=True):
        if quote.option_type is OptionType.CALL:
            call_strikes.append(quote.strike)
            call_vols.append(vol)
        else:
            put_strikes.append(quote.strike)
            put_vols.append(vol)

    all_strikes = call_strikes + put_strikes
    if len(all_strikes) >= 2:
        k_min = log(min(all_strikes) / fe.forward)
        k_max = log(max(all_strikes) / fe.forward)
        curve_strikes = []
        curve_vols = []
        for i in range(200):
            k = k_min + (k_max - k_min) * i / 199
            curve_strikes.append(fe.forward * exp(k))
            curve_vols.append(slice_.implied_vol(k, chosen.time_to_expiry))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.scatter(call_strikes, call_vols, color=CALL_COLOR, label="Call (market)", zorder=3)
        ax.scatter(put_strikes, put_vols, color=PUT_COLOR, label="Put (market)", zorder=3)
        ax.plot(curve_strikes, curve_vols, color=FIT_COLOR, linewidth=2, label="SVI fit", zorder=2)
        ax.axvline(fe.forward, color="#898781", linestyle="--", linewidth=1)
        ax.set_xlabel("Strike ($)")
        ax.set_ylabel("Implied volatility")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.xaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
        ax.legend(frameon=False)
        style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Not enough usable strikes to plot a smile for this expiry.")

    st.caption(
        f"SVI params: a={slice_.a:.5f}  b={slice_.b:.5f}  "
        f"rho={slice_.rho:+.4f}  m={slice_.m:+.4f}  sigma={slice_.sigma:.5f}"
    )

    checker = ArbitrageChecker()
    violations = checker.check([slice_])
    if violations:
        st.warning(f"{len(violations)} butterfly-arbitrage violation(s) found in this slice.")
    else:
        st.success("No butterfly-arbitrage violations found in this slice.")

    if skipped:
        with st.expander(f"{len(skipped)} expiries skipped"):
            for expiry_str, reason in skipped:
                st.text(f"{expiry_str}: {reason}")

with tab_surface3d:
    st.subheader(f"{currency} volatility surface")
    st.caption(
        "One mesh row per fitted expiry (its own SVI slice, sampled across "
        "log-moneyness), plus the real market points each slice was fit "
        "to. Drag to rotate, scroll to zoom."
    )

    if len(fitted) >= 2:
        surface_fig = build_surface_figure(fitted, snapshot_ts, currency)
        st.plotly_chart(surface_fig, use_container_width=True)
    else:
        st.warning("Need at least 2 fitted expiries to build a surface.")

with tab_term:
    st.subheader(f"{currency} term structure")

    days = [(s.expiry - snapshot_ts.date()).days for s in fitted]
    atm_vols = [s.slice.implied_vol(0.0, s.time_to_expiry) for s in fitted]
    forwards = [s.forward_estimate.forward for s in fitted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(days, atm_vols, marker="o", color=FIT_COLOR)
    ax1.set_xlabel("Days to expiry")
    ax1.set_ylabel("ATM implied volatility")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_axes(ax1)

    ax2.plot(days, forwards, marker="o", color=CALL_COLOR)
    ax2.set_xlabel("Days to expiry")
    ax2.set_ylabel("Forward ($)")
    ax2.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
    style_axes(ax2)

    fig.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        {
            "Expiry": [str(s.expiry) for s in fitted],
            "Days": days,
            "Forward": [f"${f:,.0f}" for f in forwards],
            "ATM vol": [f"{v:.1%}" for v in atm_vols],
            "a": [f"{s.slice.a:.5f}" for s in fitted],
            "b": [f"{s.slice.b:.5f}" for s in fitted],
            "rho": [f"{s.slice.rho:+.4f}" for s in fitted],
        },
        hide_index=True,
    )

with tab_pricing:
    st.subheader(f"Heston pricing ({currency})")

    with st.spinner("Calibrating Heston to the live surface..."):
        heston_params = calibrate_heston(currency)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("kappa", f"{heston_params.kappa:.4f}")
    col2.metric("theta", f"{heston_params.theta:.4f}")
    col3.metric("xi", f"{heston_params.xi:.4f}")
    col4.metric("rho", f"{heston_params.rho:+.4f}")
    col5.metric("v0", f"{heston_params.v0:.4f}")

    feller_lhs = 2 * heston_params.kappa * heston_params.theta
    feller_rhs = heston_params.xi**2
    if feller_lhs < feller_rhs:
        st.warning(
            f"Feller condition violated ({feller_lhs:.4f} < {feller_rhs:.4f}): "
            "variance can touch zero under these calibrated params."
        )

    price_expiry_labels = [str(s.expiry) for s in fitted]
    price_col1, price_col2, price_col3 = st.columns(3)
    price_expiry_label = price_col1.selectbox("Expiry", price_expiry_labels, key="price_expiry")
    price_chosen = fitted[price_expiry_labels.index(price_expiry_label)]
    default_strike = float(round(price_chosen.forward_estimate.forward, -2))
    strike = price_col2.number_input("Strike ($)", value=default_strike, step=500.0)
    option_type = OptionType(price_col3.radio("Type", ["call", "put"]))

    contract = OptionContract(
        strike=strike,
        option_type=option_type,
        forward=price_chosen.forward_estimate.forward,
        discount_factor=price_chosen.forward_estimate.discount_factor,
        time_to_expiry=price_chosen.time_to_expiry,
    )
    cos_result = COSPricer(num_terms=HESTON_NUM_TERMS).price(contract, heston_params)
    mc_result = MonteCarloPricer(num_paths=20_000, num_steps=100, seed=7).price(
        contract, heston_params
    )
    assert mc_result.standard_error is not None
    diff_in_stderr = (mc_result.price - cos_result.price) / mc_result.standard_error

    p1, p2, p3 = st.columns(3)
    p1.metric("COS price", f"${cos_result.price:,.2f}")
    p2.metric(
        "Monte Carlo price",
        f"${mc_result.price:,.2f}",
        f"stderr {mc_result.standard_error:.2f}",
    )
    cross_check_label = "agree" if abs(diff_in_stderr) < 3 else "DISAGREE"
    p3.metric("Cross-check", f"{diff_in_stderr:+.2f} stderr", cross_check_label)

with tab_hedging:
    st.subheader(f"Hedging backtest ({currency})")
    st.caption(
        "DeltaHedger vs DeltaGammaHedger on simulated paths from the live-"
        "calibrated Heston surface. Scaled down from the full headline "
        "experiment (scripts/run_headline_experiment.py) for interactive use."
    )

    with st.spinner("Calibrating Heston to the live surface..."):
        heston_params = calibrate_heston(currency)

    h1, h2, h3, h4 = st.columns(4)
    spread_bps = h1.slider("Spread (bps)", 0.0, 20.0, 5.0, step=1.0)
    num_paths = h2.slider("Paths", 100, 1000, 300, step=100)
    num_steps = h3.slider("Rehedge steps", 5, 30, 10, step=5)
    horizon_days = h4.slider("Horizon (days)", 10, 60, 30, step=5)

    if st.button("Run backtest", type="primary"):
        spot0 = fitted[0].forward_estimate.forward
        strike0 = spot0
        hedge_strike = spot0 * 1.05
        pricer = COSPricer(num_terms=HESTON_NUM_TERMS)
        cost_model = CostModel(spread_bps=spread_bps)

        with st.spinner(f"Simulating {num_paths} paths and running both backtests..."):
            simulator = HestonSimulator(heston_params, spot0, seed=42)
            paths = simulator.simulate(num_paths, num_steps, horizon_days / 365)

            delta_bt = HedgeBacktester(
                DeltaHedger(pricer, heston_params),
                pricer,
                heston_params,
                cost_model,
                strike=strike0,
                option_type=OptionType.CALL,
            )
            delta_pnl, delta_costs = delta_bt.simulate_pnl(paths)

            dg_bt = HedgeBacktester(
                DeltaGammaHedger(pricer, heston_params, hedge_strike, OptionType.CALL),
                pricer,
                heston_params,
                cost_model,
                strike=strike0,
                option_type=OptionType.CALL,
                hedge_strike=hedge_strike,
                hedge_option_type=OptionType.CALL,
            )
            dg_pnl, dg_costs = dg_bt.simulate_pnl(paths)

        st.session_state["hedge_results"] = (delta_pnl, dg_pnl, delta_costs, dg_costs)

    if "hedge_results" in st.session_state:
        delta_pnl, dg_pnl, delta_costs, dg_costs = st.session_state["hedge_results"]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(
            delta_pnl, bins=30, alpha=0.55, color=CALL_COLOR, label="Delta hedge", density=True
        )
        ax.hist(
            dg_pnl,
            bins=30,
            alpha=0.55,
            color=DELTA_GAMMA_COLOR,
            label="Delta-gamma hedge",
            density=True,
        )
        ax.axvline(float(np.mean(delta_pnl)), color=CALL_COLOR, linestyle="--", linewidth=1.5)
        ax.axvline(float(np.mean(dg_pnl)), color=DELTA_GAMMA_COLOR, linestyle="--", linewidth=1.5)
        ax.set_xlabel("Terminal P&L ($)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Delta hedge**")
            st.metric("Mean P&L", f"${np.mean(delta_pnl):,.2f}")
            st.metric("Std P&L", f"${np.std(delta_pnl, ddof=1):,.2f}")
            st.metric("Total costs", f"${delta_costs:,.2f}")
        with d2:
            st.markdown("**Delta-gamma hedge**")
            st.metric("Mean P&L", f"${np.mean(dg_pnl):,.2f}")
            st.metric("Std P&L", f"${np.std(dg_pnl, ddof=1):,.2f}")
            st.metric("Total costs", f"${dg_costs:,.2f}")
    else:
        st.info("Set your parameters and click Run backtest.")
