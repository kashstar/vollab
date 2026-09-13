"""VolLab dashboard: live vol surface, Heston pricing, and hedging backtest.

Run with: streamlit run scripts/dashboard.py

Every number on this page comes from the real, live Deribit API and the
project's own classes, the same ones scripts/check_*.py exercise. There's
no synthetic or cached-from-file data here except Streamlit's own
short-TTL caching, used to avoid re-fetching/re-calibrating on every
widget interaction.

Every chart is Plotly, not matplotlib, specifically so it's interactive
in the browser (hover for exact values, drag to zoom/pan, drag to rotate
in 3D) rather than a static image. Every "what does this mean" expander
and metric help= tooltip is plain-language, aimed at someone learning
the material alongside the code, not a glossary for people who already
know it.
"""

from datetime import date, datetime
from math import exp, log

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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

LEGEND_TOP = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0}

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
                hovertemplate=(
                    "Strike: $%{x:,.0f}<br>Days to expiry: %{y}<br>"
                    "Fitted vol: %{z:.1%}<extra>SVI fit</extra>"
                ),
            ),
            go.Scatter3d(
                x=market_strikes,
                y=market_days,
                z=market_vols,
                mode="markers",
                marker={"size": 3, "color": PUT_COLOR},
                name="Market",
                hovertemplate=(
                    "Strike: $%{x:,.0f}<br>Days to expiry: %{y}<br>"
                    "Market vol: %{z:.1%}<extra>Market</extra>"
                ),
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
        legend=LEGEND_TOP,
    )
    return fig


st.title("VolLab")
st.caption(
    "Live BTC/ETH options research: implied vol surface, Heston pricing, "
    "and a hedging backtest, built on Deribit's public market data."
)

currency = st.sidebar.selectbox(
    "Underlying",
    ["BTC", "ETH"],
    help="Which crypto's option chain to pull from Deribit. Everything on "
    "the page refits from scratch when you change this.",
)
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

    with st.expander("What is a volatility smile?"):
        st.markdown(
            "Black-Scholes assumes one constant volatility applies to every "
            "strike. Real markets disagree: solve backwards from each "
            "strike's actual traded price for the volatility that would "
            "reproduce it, and you get a different number at every strike. "
            "Plotted against strike, that curve typically dips near the "
            "money and rises on both wings, forming a smile (or an "
            "asymmetric skew, common in crypto) -- the market pricing in "
            "more tail risk than a constant-vol model would predict. The "
            "green line is **SVI**, a 5-parameter curve fit through the "
            "noisy real points, giving one smooth, usable description of "
            "the whole smile."
        )

    expiry_labels = [str(s.expiry) for s in fitted]
    chosen_label = st.selectbox("Expiry", expiry_labels)
    chosen = fitted[expiry_labels.index(chosen_label)]

    fe = chosen.forward_estimate
    slice_ = chosen.slice

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Forward",
        f"${fe.forward:,.0f}",
        help="The market's expected price for this expiry, recovered from "
        "put-call parity -- not just today's spot price.",
    )
    col2.metric(
        "Discount factor",
        f"{fe.discount_factor:.4f}",
        help="The time-value-of-money factor implied by the same fit. "
        "Close to 1 for short-dated crypto options, since there's no "
        "strong risk-free carry the way there is for equities.",
    )
    col3.metric(
        "Strike pairs used",
        fe.num_pairs,
        help="How many call/put strikes had a real two-sided market and "
        "went into fitting the forward above.",
    )
    col4.metric(
        "Forward fit R2",
        f"{fe.r_squared:.4f}",
        help="How well the strikes actually fall on the theoretical "
        "put-call-parity line. 1.0 is a perfect fit; lower means noisier "
        "quotes went into this expiry's forward.",
    )

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

        smile_fig = go.Figure()
        smile_fig.add_trace(
            go.Scatter(
                x=call_strikes,
                y=call_vols,
                mode="markers",
                name="Call (market)",
                marker={"color": CALL_COLOR, "size": 8},
                hovertemplate="Strike: $%{x:,.0f}<br>Implied vol: %{y:.1%}<extra>Call</extra>",
            )
        )
        smile_fig.add_trace(
            go.Scatter(
                x=put_strikes,
                y=put_vols,
                mode="markers",
                name="Put (market)",
                marker={"color": PUT_COLOR, "size": 8},
                hovertemplate="Strike: $%{x:,.0f}<br>Implied vol: %{y:.1%}<extra>Put</extra>",
            )
        )
        smile_fig.add_trace(
            go.Scatter(
                x=curve_strikes,
                y=curve_vols,
                mode="lines",
                name="SVI fit",
                line={"color": FIT_COLOR, "width": 2},
                hovertemplate="Strike: $%{x:,.0f}<br>Fitted vol: %{y:.1%}<extra>SVI fit</extra>",
            )
        )
        smile_fig.add_vline(x=fe.forward, line_dash="dash", line_color="#898781")
        smile_fig.update_layout(
            xaxis_title="Strike ($)",
            yaxis_title="Implied volatility",
            yaxis_tickformat=".0%",
            xaxis_tickformat="$,.0f",
            height=450,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            legend=LEGEND_TOP,
        )
        st.plotly_chart(smile_fig, use_container_width=True)
    else:
        st.warning("Not enough usable strikes to plot a smile for this expiry.")

    st.markdown("**SVI fit parameters**")
    svi_col1, svi_col2, svi_col3, svi_col4, svi_col5 = st.columns(5)
    svi_col1.metric(
        "a", f"{slice_.a:.5f}", help="Overall variance level -- shifts the whole curve up/down."
    )
    svi_col2.metric(
        "b", f"{slice_.b:.5f}", help="Wing steepness. Larger b means the smile rises faster "
        "away from the money."
    )
    svi_col3.metric(
        "rho",
        f"{slice_.rho:+.4f}",
        help="Skew/rotation, from -1 to 1. Negative tilts the smile so downside "
        "strikes carry higher vol than upside ones (crash-risk pricing).",
    )
    svi_col4.metric(
        "m", f"{slice_.m:+.4f}", help="Horizontal shift of the smile's minimum."
    )
    svi_col5.metric(
        "sigma",
        f"{slice_.sigma:.5f}",
        help="Curvature right at the smile's minimum -- how sharply it bends "
        "near the money.",
    )

    with st.expander("What is the arbitrage check?"):
        st.markdown(
            "A fitted curve isn't automatically economically sane just "
            "because it fits the points well. **Butterfly arbitrage** "
            "means the curve implies a negative probability somewhere -- "
            "mathematically impossible, and a sign the fit (not the real "
            "market) has gone wrong. `ArbitrageChecker` derives the "
            "implied probability density from the curve's own shape "
            "(Breeden-Litzenberger) and checks it never goes negative."
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

    with st.expander("What is this?"):
        st.markdown(
            "The smile in the first tab is a snapshot of one expiry. But "
            "every expiry has its own smile, and they're not independent -- "
            "near-dated smiles are usually steeper (less time for extreme "
            "moves to look proportionally large), while far-dated ones "
            "flatten out. Stack every expiry's smile side by side and you "
            "get a surface: strike on one axis, time to expiry on another, "
            "implied vol on the third. This is the actual object a vol "
            "trader thinks in terms of, not any single smile in isolation."
        )

    st.caption(
        "One mesh row per fitted expiry (its own SVI slice, sampled across "
        "log-moneyness), plus the real market points each slice was fit "
        "to. Drag to rotate, scroll to zoom, hover for exact values."
    )

    if len(fitted) >= 2:
        surface_fig = build_surface_figure(fitted, snapshot_ts, currency)
        st.plotly_chart(surface_fig, use_container_width=True)
    else:
        st.warning("Need at least 2 fitted expiries to build a surface.")

with tab_term:
    st.subheader(f"{currency} term structure")

    with st.expander("What is term structure?"):
        st.markdown(
            "This is the at-the-money slice of the surface above: how "
            "implied volatility changes with time to expiry, holding "
            "strike fixed at the forward. A near-term event (a big macro "
            "print, an exchange-specific risk) shows up as a bump in the "
            "very front of the curve; the far end usually reflects a "
            "steadier, longer-run view of volatility. The forward curve "
            "alongside it shows the market's expected price at each "
            "expiry, which usually drifts only slightly from spot for "
            "crypto's short-to-medium horizons."
        )

    days = [(s.expiry - snapshot_ts.date()).days for s in fitted]
    atm_vols = [s.slice.implied_vol(0.0, s.time_to_expiry) for s in fitted]
    forwards = [s.forward_estimate.forward for s in fitted]

    term_fig = make_subplots(
        rows=1, cols=2, subplot_titles=["ATM implied volatility", "Forward price"]
    )
    term_fig.add_trace(
        go.Scatter(
            x=days,
            y=atm_vols,
            mode="lines+markers",
            marker={"color": FIT_COLOR},
            showlegend=False,
            hovertemplate="Days: %{x}<br>ATM vol: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    term_fig.add_trace(
        go.Scatter(
            x=days,
            y=forwards,
            mode="lines+markers",
            marker={"color": CALL_COLOR},
            showlegend=False,
            hovertemplate="Days: %{x}<br>Forward: $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    term_fig.update_yaxes(tickformat=".0%", row=1, col=1)
    term_fig.update_yaxes(tickformat="$,.0f", row=1, col=2)
    term_fig.update_xaxes(title_text="Days to expiry", row=1, col=1)
    term_fig.update_xaxes(title_text="Days to expiry", row=1, col=2)
    term_fig.update_layout(height=420, margin={"l": 0, "r": 0, "t": 40, "b": 0})
    st.plotly_chart(term_fig, use_container_width=True)

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

    with st.expander("What is the Heston model, and why cross-check two ways?"):
        st.markdown(
            "SVI is a curve fit: useful for reading off vols at strikes "
            "that trade, useless for pricing something that's never "
            "traded, like an exotic payoff. Heston fixes that by modeling "
            "*how* the price and its volatility evolve together over "
            "time, as an actual stochastic process, not just a snapshot. "
            "Its price has no simple closed form, so it's computed two "
            "completely independent ways here: `COSPricer` (a fast "
            "Fourier-series expansion) and `MonteCarloPricer` (literally "
            "simulating thousands of possible futures and averaging the "
            "payoff). If a Heston implementation has a subtle bug, it "
            "tends to still produce a plausible-looking number -- so "
            "instead of trusting either method alone, the two are checked "
            "against each other below. Agreement is real evidence both "
            "are right, not proof either one is."
        )

    with st.spinner("Calibrating Heston to the live surface..."):
        heston_params = calibrate_heston(currency)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "kappa",
        f"{heston_params.kappa:.4f}",
        help="Mean-reversion speed: how quickly variance gets pulled back "
        "toward its long-run average (theta) after a shock.",
    )
    col2.metric(
        "theta",
        f"{heston_params.theta:.4f}",
        help="Long-run average variance that the process reverts to. "
        "Roughly, the market's steady-state view of vol-squared.",
    )
    col3.metric(
        "xi",
        f"{heston_params.xi:.4f}",
        help="Vol-of-vol: how much variance itself randomly fluctuates, "
        "not the price. Higher xi means a wider, more uncertain vol path.",
    )
    col4.metric(
        "rho",
        f"{heston_params.rho:+.4f}",
        help="Correlation between price shocks and variance shocks. "
        "Negative is the usual crypto/equity pattern: vol tends to rise "
        "when price falls.",
    )
    col5.metric(
        "v0",
        f"{heston_params.v0:.4f}",
        help="Today's starting variance -- where the process begins before "
        "any mean reversion has had time to act.",
    )

    feller_lhs = 2 * heston_params.kappa * heston_params.theta
    feller_rhs = heston_params.xi**2
    if feller_lhs < feller_rhs:
        st.warning(
            f"Feller condition violated ({feller_lhs:.4f} < {feller_rhs:.4f}): "
            "variance can touch zero under these calibrated params. This is "
            "a real mathematical property of the fitted parameters "
            "(`2 * kappa * theta` should be >= `xi^2` to guarantee variance "
            "never hits exactly zero), not a bug -- it just means this "
            "particular calibration lands in the numerically trickier regime."
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
    p1.metric(
        "COS price",
        f"${cos_result.price:,.2f}",
        help="Price from the Fourier-cosine expansion method -- fast, "
        "deterministic, no randomness.",
    )
    p2.metric(
        "Monte Carlo price",
        f"${mc_result.price:,.2f}",
        f"stderr {mc_result.standard_error:.2f}",
        help="Price from simulating 20,000 random price paths and "
        "averaging the discounted payoff. stderr is the standard error "
        "of that average -- how much this estimate would wobble if you "
        "reran it with a different random seed.",
    )
    cross_check_label = "agree" if abs(diff_in_stderr) < 3 else "DISAGREE"
    p3.metric(
        "Cross-check",
        f"{diff_in_stderr:+.2f} stderr",
        cross_check_label,
        help="The two prices' difference, measured in Monte Carlo standard "
        "errors. Within about 3 is consistent with pure sampling noise; "
        "much larger would mean one of the two pricers is actually wrong.",
    )

with tab_hedging:
    st.subheader(f"Hedging backtest ({currency})")

    with st.expander("What is delta vs delta-gamma hedging?"):
        st.markdown(
            "Selling an option leaves you exposed to the underlying's "
            "price moving. **Delta hedging** offsets that by holding "
            "shares of the underlying equal to the option's delta (its "
            "price sensitivity to a $1 move), rebalanced at every step. "
            "**Delta-gamma hedging** goes further: delta itself changes "
            "as price moves (that curvature is gamma), so it also trades "
            "a second, fixed option to neutralize that second-order "
            "exposure, not just the first-order one. The tradeoff is "
            "real, not theoretical: gamma hedging usually reduces P&L "
            "variance, but trading a second instrument every step adds "
            "its own cost and drag, so it doesn't automatically win on "
            "average -- that's exactly what this backtest measures."
        )

    st.caption(
        "DeltaHedger vs DeltaGammaHedger on simulated paths from the live-"
        "calibrated Heston surface. Scaled down from the full headline "
        "experiment (scripts/run_headline_experiment.py) for interactive use."
    )

    with st.spinner("Calibrating Heston to the live surface..."):
        heston_params = calibrate_heston(currency)

    h1, h2, h3, h4 = st.columns(4)
    spread_bps = h1.slider(
        "Spread (bps)",
        0.0,
        20.0,
        5.0,
        step=1.0,
        help="Transaction cost per trade, as basis points of trade value "
        "(100bps = 1%). 0 means frictionless -- costs isolate purely from "
        "rebalancing.",
    )
    num_paths = h2.slider(
        "Paths",
        100,
        1000,
        300,
        step=100,
        help="How many independent simulated price scenarios to average "
        "the P&L distribution over. More paths means a less noisy "
        "estimate, at the cost of a slower run.",
    )
    num_steps = h3.slider(
        "Rehedge steps",
        5,
        30,
        10,
        step=5,
        help="How many times each strategy rebalances its position over "
        "the option's life. More steps tracks the option's true delta "
        "more closely, but also means more transaction costs.",
    )
    horizon_days = h4.slider(
        "Horizon (days)",
        10,
        60,
        30,
        step=5,
        help="Time to expiry of the option being hedged.",
    )

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

        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=delta_pnl,
                name="Delta hedge",
                marker_color=CALL_COLOR,
                opacity=0.55,
                histnorm="probability density",
                hovertemplate="P&L: $%{x:,.0f}<extra>Delta hedge</extra>",
            )
        )
        hist_fig.add_trace(
            go.Histogram(
                x=dg_pnl,
                name="Delta-gamma hedge",
                marker_color=DELTA_GAMMA_COLOR,
                opacity=0.55,
                histnorm="probability density",
                hovertemplate="P&L: $%{x:,.0f}<extra>Delta-gamma hedge</extra>",
            )
        )
        hist_fig.add_vline(x=float(np.mean(delta_pnl)), line_dash="dash", line_color=CALL_COLOR)
        hist_fig.add_vline(
            x=float(np.mean(dg_pnl)), line_dash="dash", line_color=DELTA_GAMMA_COLOR
        )
        hist_fig.update_layout(
            barmode="overlay",
            xaxis_title="Terminal P&L ($)",
            yaxis_title="Density",
            height=420,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            legend=LEGEND_TOP,
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        pnl_help = (
            "Average terminal P&L across all simulated paths, from the "
            "seller's point of view (premium received, minus the option "
            "payoff owed, minus hedging costs, plus/minus hedge P&L)."
        )
        std_help = (
            "Standard deviation of that P&L across paths -- how spread out "
            "the outcomes are. Lower means more predictable, tighter risk."
        )
        costs_help = "Total transaction costs paid across all paths and rehedges."

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Delta hedge**")
            st.metric("Mean P&L", f"${np.mean(delta_pnl):,.2f}", help=pnl_help)
            st.metric("Std P&L", f"${np.std(delta_pnl, ddof=1):,.2f}", help=std_help)
            st.metric("Total costs", f"${delta_costs:,.2f}", help=costs_help)
        with d2:
            st.markdown("**Delta-gamma hedge**")
            st.metric("Mean P&L", f"${np.mean(dg_pnl):,.2f}", help=pnl_help)
            st.metric("Std P&L", f"${np.std(dg_pnl, ddof=1):,.2f}", help=std_help)
            st.metric("Total costs", f"${dg_costs:,.2f}", help=costs_help)
    else:
        st.info("Set your parameters and click Run backtest.")
