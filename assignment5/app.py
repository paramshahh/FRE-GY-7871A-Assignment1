"""Streamlit dashboard: voter sentiment, prediction markets and sector ETFs.

    streamlit run app.py

Reads the last pipeline run from outputs/results.pkl (written by
build_report.py) and can re-run the pipeline or pull live venue quotes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src import config
from src import granger_econometrics as ge
from src import prediction_markets as pmk
from src.pipeline import RESULTS_PATH, load_results, run_pipeline

st.set_page_config(page_title="Midterm Sentiment & Markets", layout="wide")


@st.cache_resource(show_spinner="Loading results")
def get_results(_stamp: float):
    return load_results()


@st.cache_data(ttl=300, show_spinner="Fetching live quotes")
def live_quotes() -> dict:
    """Latest P(Dem) per chamber from Polymarket and Kalshi (5-minute cache)."""
    s = requests.Session()
    out = {}
    end = pd.Timestamp.now(tz="UTC")
    start = (end - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    for ch in config.CHAMBERS:
        for venue, fn in (("Polymarket", lambda c: pmk.fetch_polymarket(c, s)),
                          ("Kalshi", lambda c: pmk.fetch_kalshi(c, s, start,
                                                                end.strftime("%Y-%m-%d")))):
            try:
                ser = fn(ch)
                out[(ch, venue)] = (float(ser.iloc[-1]), ser.index[-1])
            except Exception as exc:
                out[(ch, venue)] = (np.nan, repr(exc))
    return out


def gauge(value: float, title: str, ref: float | None = None) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if ref is not None else "gauge+number",
        value=100 * value, number={"suffix": "%", "valueformat": ".1f"},
        delta={"reference": 100 * ref, "valueformat": ".1f"} if ref is not None else None,
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f4e9c"},
               "steps": [{"range": [0, 50], "color": "#f3d6d6"},
                         {"range": [50, 100], "color": "#d6e2f3"}],
               "threshold": {"line": {"color": "black", "width": 2}, "value": 50}}))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def pval_heatmap(df: pd.DataFrame, value: str, title: str) -> go.Figure:
    m = df.pivot(index="cause", columns="effect", values=value)
    z = -np.log10(m.clip(lower=1e-6))
    fig = px.imshow(z, color_continuous_scale="Blues", aspect="auto", zmin=0, zmax=3,
                    labels=dict(color=f"-log10 {value}"), title=title)
    fig.update_traces(text=m.map(lambda p: f"{p:.3f}").values, texttemplate="%{text}")
    fig.update_layout(height=120 + 40 * len(m), margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
st.title("Voter Sentiment, Topic Salience & Election Prediction Markets")
st.caption("US 2026 midterms · FRE-GY 7871 Assignment 5")

with st.sidebar:
    st.header("Pipeline")
    method = st.selectbox("Topic model", ["nmf", "lda", "bertopic"])
    lag_rule = st.selectbox("Lag selection", ["bic", "aic", 1, 2, 3, 5])
    cov = st.selectbox("Covariance", ["HC1", "HAC", "nonrobust"])
    hq = st.slider("High-news quantile (Rigobon-Sack H regime)", 0.5, 0.95,
                   config.HIGH_NEWS_QUANTILE, 0.05)
    refresh = st.checkbox("Download fresh data", value=False)
    if st.button("Run pipeline"):
        try:
            with st.spinner("Running pipeline"):
                run_pipeline(refresh=refresh, topic_method=method, lags=lag_rule, cov=cov,
                             high_quantile=hq)
            st.cache_resource.clear()
            st.success("Done")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

if not RESULTS_PATH.exists():
    st.warning("No results yet. Run `python build_report.py` (or the sidebar button) once the "
               "data sources are reachable. See README.md for the hosts the pipeline needs.")
    st.stop()

R = get_results(RESULTS_PATH.stat().st_mtime)
st.caption(f"Results built {R.params['built']} · window {R.params['start']} to "
           f"{R.params['end']} · market data: {R.market.source} · text: {R.text_sources}")

tab_m, tab_t, tab_g, tab_s, tab_tab = st.tabs(
    ["Prediction markets", "Topics & tone", "Granger tests", "Sector reactions", "Tables"])

with tab_m:
    st.subheader("P(Democratic control)")
    use_live = st.toggle("Pull live quotes now", value=False)
    quotes = live_quotes() if use_live else {}
    cols = st.columns(len(config.CHAMBERS))
    for c, ch in zip(cols, config.CHAMBERS):
        series = R.pm.consensus[ch].dropna()
        last = float(series.iloc[-1])
        prev = float(series.iloc[-6]) if len(series) > 6 else None
        live = [v for (cc, _), (v, _) in quotes.items() if cc == ch and np.isfinite(v)]
        val = float(np.mean(live)) if live else last
        label = f"{ch} ({'live' if live else 'as of ' + series.index[-1].strftime('%d %b')})"
        c.plotly_chart(gauge(val, label, prev), width="stretch")
    if quotes:
        st.dataframe(pd.DataFrame([{"chamber": k[0], "venue": k[1], "P(Dem)": v[0],
                                    "as of / error": str(v[1])} for k, v in quotes.items()]))
    long = R.pm.by_venue.dropna()
    for ch in config.CHAMBERS:
        fig = px.line(long[long["chamber"] == ch], x="date", y="prob_dem", color="venue",
                      labels={"prob_dem": "P(Dem)"}, height=320, title=ch)
        fig.add_scatter(x=R.pm.consensus.index, y=R.pm.consensus[ch], name="consensus",
                        line=dict(color="black", width=2))
        st.plotly_chart(fig, width="stretch")
    st.write("Venue coverage (trading days with a quote):")
    st.dataframe(R.pm.coverage)
    if R.pm.errors:
        with st.expander("Venue errors on the last run"):
            st.json(R.pm.errors)

with tab_t:
    st.subheader("Daily topic salience")
    sal = R.daily.salience.copy()
    sal.index = sal.index.strftime("%Y-%m-%d")
    st.plotly_chart(px.imshow(sal.T, aspect="auto", color_continuous_scale="YlOrRd",
                              labels=dict(color="share")), width="stretch")
    st.subheader("Topic-conditioned tone, 5-day mean")
    picks = st.multiselect("Topics", list(R.tone_filled.columns), list(R.tone_filled.columns)[:3])
    st.plotly_chart(px.line(R.daily.tone[picks].rolling(5, min_periods=2).mean()),
                    width="stretch")
    emo = st.selectbox("Emotion", list(R.daily.emotions))
    st.plotly_chart(px.line(R.daily.emotions[emo].rolling(5, min_periods=2).mean(),
                            labels={"value": f"share {emo}"}), width="stretch")
    c1, c2 = st.columns(2)
    c1.write("Topic components and labels")
    c1.dataframe(pd.DataFrame({"label": R.topic_model.component_labels,
                               "top terms": [", ".join(t[:8]) for t in R.topic_model.top_terms]}))
    c2.write("Quality-filter waterfall")
    c2.dataframe(R.filter_report)
    c2.write("Topic-model comparison")
    c2.dataframe(R.model_comparison)

with tab_g:
    stat = st.radio("Show", ["p_value", "q_value"], horizontal=True)
    st.plotly_chart(pval_heatmap(R.stage1, stat, "Stage 1: Tone_k → ΔProb"), width="stretch")
    st.plotly_chart(pval_heatmap(R.stage1_reverse, stat, "Reverse: ΔProb → Tone_k"),
                    width="stretch")
    st.plotly_chart(pval_heatmap(R.stage2_ret, stat, "Stage 2: ΔProb → excess return"),
                    width="stretch")
    st.plotly_chart(pval_heatmap(R.stage2_vol, stat, "Stage 2: ΔProb → range volatility"),
                    width="stretch")
    st.write("Lag-order criteria (VAR of ΔProb and tone)")
    st.dataframe(R.lag_table)

with tab_s:
    ch = st.selectbox("Chamber", config.CHAMBERS)
    n_ev = st.slider("Largest |ΔProb| days", 3, 25, 10)
    es = ge.event_study(R.market.excess, R.pm.dprob[ch], n_events=n_ev, window=5)
    if len(es):
        fig = px.line(es * 100, labels={"value": "cum. excess return (%)", "day": "day"},
                      title=f"Excess returns around big {ch} probability moves (signed)")
        fig.add_vline(x=0, line_dash="dot")
        st.plotly_chart(fig, width="stretch")
    etf = st.selectbox("Impulse response: ETF", list(R.market.excess.columns))
    lag = int(st.number_input("VAR lags", 1, 5, 2))
    irf = ge.var_irf(pd.concat([R.pm.dprob[ch].rename("dProb"),
                                R.market.excess[etf].rename(etf)], axis=1), lag)
    fig = go.Figure([go.Scatter(x=irf.h, y=irf.hi, line=dict(width=0), showlegend=False),
                     go.Scatter(x=irf.h, y=irf.lo, fill="tonexty", line=dict(width=0),
                                name="95% band"),
                     go.Scatter(x=irf.h, y=irf.irf, name="IRF", line=dict(color="black"))])
    fig.update_layout(title=f"Response of {etf} excess return to a 1-sd ΔProb_{ch} shock",
                      xaxis_title="days")
    st.plotly_chart(fig, width="stretch")
    st.write("Rigobon-Sack heteroskedasticity estimates")
    st.dataframe(R.rigobon_sack)

with tab_tab:
    for name, df in R.tables.items():
        st.subheader(name.replace("_", " ").title())
        st.dataframe(df)
