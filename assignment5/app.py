"""Streamlit dashboard: LSA topics, sentiment, election markets and the midterm basket.

    streamlit run app.py

Reads the last pipeline run (outputs/results.pkl, written by build_report.py) and can
pull live Polymarket/Kalshi quotes for the gauges.
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
from src.pipeline import RESULTS_PATH, load_results

st.set_page_config(page_title="Midterm Topics & Markets", layout="wide")


@st.cache_resource(show_spinner="Loading results")
def get_results(_stamp: float):
    return load_results()


@st.cache_data(ttl=300, show_spinner="Fetching live quotes")
def live_quotes() -> dict:
    """Latest P(Dem) per contract from Polymarket and Kalshi (cached for 5 minutes)."""
    s = requests.Session()
    end = pd.Timestamp.now(tz=config.TZ)
    start = (end - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    out = {}
    for ch in config.CONTRACTS:
        for venue, fn in (("Polymarket", pmk.fetch_polymarket), ("Kalshi", pmk.fetch_kalshi)):
            try:
                ser = fn(ch, s, start, end.strftime("%Y-%m-%d"))
                out[(ch, venue)] = (float(ser.iloc[-1]), str(ser.index[-1]))
            except Exception as exc:
                out[(ch, venue)] = (np.nan, repr(exc)[:120])
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
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def q_heatmap(df: pd.DataFrame, rows: str, cols: str, stat: str, title: str) -> go.Figure:
    m = df.pivot(index=rows, columns=cols, values=stat)
    m.index = [str(i).replace("S_", "").replace("dProb_", "") for i in m.index]
    m.columns = [str(c).replace("S_", "").replace("dProb_", "").replace("vol_", "")
                 for c in m.columns]
    fig = px.imshow(-np.log10(m.clip(lower=1e-4)), color_continuous_scale="Blues", zmin=0,
                    zmax=3, aspect="auto", labels=dict(color=f"-log10 {stat}"), title=title)
    fig.update_traces(text=m.map(lambda p: f"{p:.2f}").values, texttemplate="%{text}")
    fig.update_layout(height=140 + 32 * len(m), margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
st.title("Topic Modeling, Sentiment & Election Markets — 2026 US Midterms")
if not RESULTS_PATH.exists():
    st.warning("No results yet: run `python build_report.py` (see README.md).")
    st.stop()
R = get_results(RESULTS_PATH.stat().st_mtime)
st.caption(f"Built {R.params['built'][:16]} · {R.params['start']} to {R.params['end']} · "
           f"{R.params['n_tweets']:,} filtered tweets · LSA k = {R.params['k']}")

tabs = st.tabs(["Election markets", "LSA topics", "Sentiment & salience", "Granger tests",
                "Basket reactivity", "Tables"])

with tabs[0]:
    live = st.toggle("Pull live Polymarket / Kalshi quotes", value=False)
    quotes = live_quotes() if live else {}
    cols = st.columns(len(config.CONTRACTS))
    for c, ch in zip(cols, config.CONTRACTS):
        s = R.pm.consensus[ch].dropna()
        lv = [v for (cc, _), (v, _) in quotes.items() if cc == ch and np.isfinite(v)]
        val = float(np.mean(lv)) if lv else float(s.iloc[-1])
        label = f"P(Dem) {ch} — {'live' if lv else 'close ' + s.index[-1].strftime('%d %b')}"
        c.plotly_chart(gauge(val, label, float(s.iloc[-6]) if len(s) > 6 else None),
                       width="stretch")
    if quotes:
        st.dataframe(pd.DataFrame([{"contract": k[0], "venue": k[1], "P(Dem)": v[0],
                                    "as of / error": v[1]} for k, v in quotes.items()]))
    long = R.pm.by_venue.dropna(subset=["prob_dem"])
    for ch in config.CONTRACTS:
        fig = px.line(long[long["chamber"] == ch], x="date", y="prob_dem", color="venue",
                      labels={"prob_dem": "P(Dem)"}, height=280, title=ch)
        fig.add_scatter(x=R.pm.consensus.index, y=R.pm.consensus[ch], name="consensus",
                        line=dict(color="black", width=2))
        st.plotly_chart(fig, width="stretch")
    st.write("Sessions with a quote at or before 16:00 ET")
    st.dataframe(R.pm.coverage)

with tabs[1]:
    m = R.lsa_model
    c1, c2 = st.columns(2)
    fig = px.line(x=np.arange(1, len(R.spectrum) + 1), y=R.spectrum, markers=True,
                  labels={"x": "component", "y": "singular value"},
                  title="Singular value spectrum (uncentred TF-IDF)")
    fig.add_vline(x=R.params["k"], line_dash="dot")
    c1.plotly_chart(fig, width="stretch")
    st_ = R.stability.assign(pole=lambda d: d["label"] + " " + d["component"].astype(str)
                             + np.where(d["sign"] > 0, "+", "-"))
    c2.plotly_chart(px.bar(st_, x="stability", y="pole", orientation="h", range_x=[0, 1],
                           title="Pole stability across 5 subsample refits"), width="stretch")
    lab = m.poles[~m.poles["label"].isin(["Other", "General"])]
    n_terms = st.slider("Terms per pole", 3, 10, 5)
    terms = []
    for _, p in lab.iterrows():
        u = p["sign"] * m.U[:, p["component"]]
        terms += [m.terms[i] for i in np.argsort(u)[::-1][:n_terms]]
    terms = list(dict.fromkeys(terms))
    pos = {t: i for i, t in enumerate(m.terms)}
    H = pd.DataFrame([[max(p["sign"] * m.U[pos[t], p["component"]], 0) for t in terms]
                      for _, p in lab.iterrows()], columns=terms,
                     index=[f"{p.label} {p.component}{'+' if p.sign > 0 else '-'}"
                            for p in lab.itertuples()])
    st.plotly_chart(px.imshow(H, aspect="auto", color_continuous_scale="Blues",
                              title="Topic loading heatmap (U_k, positive part)"),
                    width="stretch")
    st.dataframe(R.tables["table1a_svd_topics"])

with tabs[2]:
    sal = R.daily.salience[R.signals.columns].rolling(5, min_periods=1).mean()
    sal.index = sal.index.strftime("%Y-%m-%d")
    st.plotly_chart(px.imshow(sal.T, aspect="auto", color_continuous_scale="YlOrRd",
                              title="Issue salience (general stratum, 5-session mean)"),
                    width="stretch")
    picks = st.multiselect("Issues", list(R.signals.columns), list(R.signals.columns)[:4])
    st.plotly_chart(px.line(R.signals[picks], title="Topic-conditioned sentiment S_k,t "
                            "(5-session pooled)"), width="stretch")
    emo = st.selectbox("Emotion", list(R.daily.emotions))
    st.plotly_chart(px.line(R.daily.emotions[emo][picks].rolling(5, min_periods=2).mean(),
                            title=f"Loading-weighted share of {emo}"), width="stretch")
    st.dataframe(R.tables["table1b_issue_sentiment"])
    st.subheader("News salience check (GDELT)")
    if len(getattr(R, "salience_validation", pd.DataFrame())):
        st.write("Does X discourse track the news agenda? (5-session means)")
        st.dataframe(R.salience_validation.round(3))
        st.write("GDELT news salience / tone → ΔProb (Table 2B)")
        st.dataframe(R.tables.get("table2b_news_salience"))
        st.caption("Series availability")
        st.dataframe(R.news_status)
    else:
        st.info("No GDELT issue series in this run.")

with tabs[3]:
    stat = st.radio("Statistic", ["q_value", "p_value"], horizontal=True)
    st.plotly_chart(q_heatmap(R.stage1, "cause", "effect", stat,
                              "Stage 1: sentiment S_k → ΔProb"), width="stretch")
    st.plotly_chart(q_heatmap(R.stage1_reverse, "effect", "cause", stat,
                              "Reverse: ΔProb → S_k"), width="stretch")
    st.plotly_chart(q_heatmap(R.stage2_ret, "cause", "effect", stat,
                              "Stage 2: ΔProb → excess return"), width="stretch")
    st.plotly_chart(q_heatmap(R.stage2_vol, "cause", "effect", stat,
                              "Stage 2: ΔProb → range volatility"), width="stretch")
    st.subheader("Covariance SVD factors")
    cs = R.cov_svd
    st.write(f"Singular values {np.round(cs.S, 3)}, permutation p {np.round(cs.p_values, 3)}, "
             f"trained to {cs.train_end.date()}")
    st.dataframe(cs.U.round(3))
    st.dataframe(R.factor_oos.round(3))

with tabs[4]:
    st.subheader("Robustness battery for FDR survivors")
    st.dataframe(R.robustness.round(4))
    ch = st.selectbox("Contract", config.CONTRACTS, index=2)
    assets = st.multiselect("Assets", list(R.market.excess.columns),
                            ["XLF", "UNH", "XLK", "ICLN", "XLE", "LMT"])
    n_ev = st.slider("Largest |ΔProb| sessions", 3, 25, 10)
    es = ge.event_study(R.market.excess[assets], R.pm.dprob[ch], n_events=n_ev, window=5)
    if len(es):
        fig = px.line(es * 100, labels={"value": "cumulative excess return (%)",
                                        "day": "sessions from event"},
                      title=f"Basket around large {ch} moves (signed by the move)")
        fig.add_vline(x=0, line_dash="dot")
        st.plotly_chart(fig, width="stretch")
    asset = st.selectbox("Impulse response: asset", list(R.market.excess.columns), index=3)
    irf = ge.var_irf(pd.concat([R.pm.dprob[ch].rename("dProb"),
                                R.market.excess[asset].rename(asset)], axis=1), 1)
    fig = go.Figure([go.Scatter(x=irf.h, y=irf.hi, line=dict(width=0), showlegend=False),
                     go.Scatter(x=irf.h, y=irf.lo, fill="tonexty", line=dict(width=0),
                                name="95% band"),
                     go.Scatter(x=irf.h, y=irf.irf, name="IRF", line=dict(color="black"))])
    fig.update_layout(title=f"{asset} excess return after a 1-sd ΔP({ch}) shock",
                      xaxis_title="sessions")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Rigobon–Sack (H = top 20% GDELT news sessions)")
    st.dataframe(R.rigobon_sack.round(3))

with tabs[5]:
    for name, df in R.tables.items():
        st.subheader(name.replace("_", " ").title())
        st.dataframe(df)
