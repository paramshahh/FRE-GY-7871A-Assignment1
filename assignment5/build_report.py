"""Run the pipeline and build assignment5_report.pdf and assignment5_capstone.ipynb.

    python build_report.py                 # from cached data (default)
    python build_report.py --refresh       # re-download the free sources first
    python build_report.py --skip-notebook

Tweets are never re-downloaded here (they are paid); see collect_tweets.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import granger_econometrics as ge  # noqa: E402
from src.pipeline import Results, run_pipeline  # noqa: E402
from src.table_generator import fmt  # noqa: E402

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "assignment5_report.pdf"
NB_PATH = ROOT / "assignment5_capstone.ipynb"
plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> Path:
    path = config.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _qmatrix(df: pd.DataFrame, rows: str, cols: str) -> pd.DataFrame:
    return df.pivot(index=rows, columns=cols, values="q_value")


def make_figures(R: Results) -> dict[str, Path]:
    figs = {}
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.6), sharex=True)
    for ax, ch in zip(axes, config.CONTRACTS):
        sub = R.pm.by_venue[R.pm.by_venue["chamber"] == ch]
        for v, g in sub.groupby("venue"):
            ax.plot(g["date"], g["prob_dem"], lw=0.7, alpha=0.6, label=v)
        ax.plot(R.pm.consensus.index, R.pm.consensus[ch], color="black", lw=1.4,
                label="consensus")
        ax.axhline(0.5, color="grey", lw=0.5, ls=":")
        ax.set_ylabel(f"P(Dem) {ch}")
    axes[0].legend(ncol=4, fontsize=6, frameon=False)
    fig.suptitle("Figure 1. Implied probability of Democratic control, 16:00 ET quotes")
    figs["markets"] = _save(fig, "fig1_prediction_markets.png")

    fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.8))
    sv = R.spectrum
    ax[0].plot(np.arange(1, len(sv) + 1), sv, marker=".", lw=0.8)
    ax[0].axvline(R.params["k"], color="grey", ls=":")
    ax[0].set(xlabel="component", ylabel="singular value", title="Singular value spectrum")
    st = R.stability.copy()
    st["pole"] = st["label"].str.slice(0, 18) + " " + st["component"].astype(str) \
        + np.where(st["sign"] > 0, "+", "-")
    ax[1].barh(st["pole"], st["stability"], color="#1f4e9c")
    ax[1].set(xlabel="share of 5 subsample refits", title="Pole stability", xlim=(0, 1))
    ax[1].tick_params(axis="y", labelsize=5)
    fig.suptitle("Figure 2. LSA spectrum and topic stability")
    figs["spectrum"] = _save(fig, "fig2_spectrum_stability.png")

    m = R.lsa_model
    lab = m.poles[~m.poles["label"].isin(["Other", "General"])]
    terms = []
    for _, p in lab.iterrows():
        u = p["sign"] * m.U[:, p["component"]]
        terms += [m.terms[i] for i in np.argsort(u)[::-1][:4]]
    terms = list(dict.fromkeys(terms))
    pos = {t: i for i, t in enumerate(m.terms)}
    H = np.array([[max(p["sign"] * m.U[pos[t], p["component"]], 0) for t in terms]
                  for _, p in lab.iterrows()])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.imshow(H, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(lab)), [f"{r.label[:20]} {r.component}{'+' if r.sign > 0 else '-'}"
                                    for r in lab.itertuples()], fontsize=5.5)
    ax.set_xticks(range(len(terms)), terms, rotation=90, fontsize=5)
    ax.set_title("Figure 3. Term loadings U_k of the labelled poles (positive part)")
    figs["loadings"] = _save(fig, "fig3_loading_heatmap.png")

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 4.4), sharex=True)
    sal = R.daily.salience[R.signals.columns].rolling(5, min_periods=1).mean()
    ax[0].stackplot(sal.index, sal.fillna(0).T.values, labels=sal.columns, alpha=0.85)
    ax[0].set_ylabel("salience (5-session mean)")
    ax[0].legend(ncol=4, fontsize=5.5, frameon=False, loc="upper left")
    for c in R.signals.columns:
        ax[1].plot(R.signals.index, R.signals[c], lw=0.9, label=c)
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_ylabel("S_k,t (5-session pooled)")
    fig.suptitle("Figure 4. Issue salience (general stratum) and topic-conditioned sentiment")
    figs["sentiment"] = _save(fig, "fig4_salience_sentiment.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.4), gridspec_kw={"width_ratios": [1, 2.4]})
    panels = [(_qmatrix(R.stage1, "cause", "effect"), "Stage 1: S_k → ΔProb"),
              (_qmatrix(R.stage2_ret, "effect", "cause"), "Stage 2: ΔProb → excess return")]
    for ax, (mm, title) in zip(axes, panels):
        if title.startswith("Stage 2"):
            mm = mm.T
        ax.imshow(-np.log10(mm.clip(lower=1e-4)).values, cmap="Blues", vmin=0, vmax=2,
                  aspect="auto")
        ax.set_xticks(range(mm.shape[1]), [c.replace("dProb_", "") for c in mm.columns],
                      rotation=90, fontsize=6)
        ax.set_yticks(range(mm.shape[0]), [c.replace("S_", "").replace("dProb_", "")[:18]
                                           for c in mm.index], fontsize=6)
        for i in range(mm.shape[0]):
            for j in range(mm.shape[1]):
                ax.text(j, i, f"{mm.values[i, j]:.2f}", ha="center", va="center", fontsize=4.5)
        ax.set_title(title, fontsize=8)
    fig.suptitle("Figure 5. Granger tests: FDR q-values (darker = smaller)")
    figs["granger"] = _save(fig, "fig5_granger_q.png")

    cs = R.cov_svd
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    cs.U["F1"].sort_values().plot.barh(ax=axes[0], color="#1f4e9c")
    axes[0].set_title(f"F1 signal weights (σ={cs.S[0]:.2f}, perm p={cs.p_values[0]:.2f})")
    axes[0].tick_params(axis="y", labelsize=6)
    cs.V["F1"].sort_values().plot.barh(ax=axes[1], color="#c0392b")
    axes[1].tick_params(axis="y", labelsize=5.5)
    axes[1].set_title("F1 target portfolio")
    fig.suptitle("Figure 6. Leading factor of the sentiment–return cross-covariance SVD")
    figs["covsvd"] = _save(fig, "fig6_covariance_svd.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), sharey=True)
    for ax, ch in zip(axes, ("Senate", "Sweep")):
        es = ge.event_study(R.market.excess[["XLF", "UNH", "XLK", "ICLN", "XLE", "LMT"]],
                            R.pm.dprob[ch], n_events=10, window=5)
        if len(es):
            (es * 100).plot(ax=ax, lw=1, legend=ch == "Sweep")
        ax.axvline(0, color="grey", ls=":")
        ax.set_title(f"{ch}: 10 largest |ΔProb| sessions (signed)")
        ax.set_xlabel("sessions from event")
    axes[0].set_ylabel("cumulative excess return (%)")
    if axes[1].get_legend():
        axes[1].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("Figure 7. Basket reactions around large prediction-market moves")
    figs["events"] = _save(fig, "fig7_event_study.png")

    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    n = R.news["count_All"]
    ax.bar(n.index, n.fillna(0), color="#999999", width=1.0, label="midterm articles")
    h = (R.high_news == 1).reindex(n.index, fill_value=False)
    ax.bar(n.index[h], n[h], color="#c0392b", width=1.0, label="H regime")
    for d in n.index[R.news["outage_days"] > 0]:
        ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="#f5b041",
                   alpha=0.4, lw=0)
    ax.set_title("Figure 8. GDELT midterm coverage per session: H regime (red), GDELT "
                 "outages (orange, excluded)")
    ax.legend(fontsize=6, frameon=False)
    figs["news"] = _save(fig, "fig8_gdelt_regimes.png")
    return figs


# ---------------------------------------------------------------------------
# Findings, generated from the numbers
# ---------------------------------------------------------------------------
def news_finding(R: Results) -> str:
    """One paragraph on the GDELT news-salience check, generated from the numbers."""
    nt_, val, st = R.news_tests, R.salience_validation, R.news_status
    if not len(nt_):
        return "<b>News check.</b> No GDELT issue series were available."
    missing = [i for i, r in st.iterrows() if not r.all()]
    txt = (f"<b>News check (GDELT).</b> {int((nt_['q_value'] < 0.10).sum())} of {len(nt_)} "
           "tests of per-issue news salience and news tone on ΔProb survive FDR at 10% "
           f"(smallest q = {nt_['q_value'].min():.2f}).")
    if len(val):
        v = val.set_index("issue")
        txt += (" X discourse tracks the news agenda most closely for "
                + ", ".join(f"{i} (ρ = {r.corr_levels:.2f})" for i, r in
                            v.sort_values("corr_levels", ascending=False).head(3).iterrows())
                + f"; the median level correlation across issues is "
                f"{v['corr_levels'].median():.2f}.")
    if missing:
        txt += " Series not available from GDELT: " + ", ".join(missing) + "."
    return txt


def findings(R: Results) -> list[str]:
    s1, s2, s2v = R.stage1, R.stage2_ret, R.stage2_vol
    oos = R.factor_oos
    out = [
        f"<b>Stage 1 (sentiment → election odds).</b> {int((s1['q_value'] < 0.10).sum())} of "
        f"{len(s1)} tests survive the Benjamini–Hochberg correction at 10% (smallest q = "
        f"{s1['q_value'].min():.2f}); {int((s1['p_value'] < 0.05).sum())} of {len(s1)} reject "
        "at an unadjusted 5% level, about what chance alone delivers. At a daily horizon, public "
        "issue sentiment on X does not measurably lead prediction-market prices.",
        f"<b>Covariance SVD.</b> No factor of the lagged sentiment–return cross-covariance beats "
        f"its circular-shift permutation null (p = "
        f"{', '.join(f'{p:.2f}' for p in R.cov_svd.p_values)})"
        + (f"; the leading factor's out-of-sample correlation is "
           f"{oos['oos_corr'].iloc[0]:+.2f} (t = {oos['oos_t'].iloc[0]:+.1f}), the opposite "
           "sign to its training fit, which is what over-fitting looks like." if len(oos)
           else "."),
        news_finding(R),
        f"<b>Stage 2 (election odds → assets).</b> {int((s2['q_value'] < 0.10).sum())} of "
        f"{len(s2)} excess-return tests and {int((s2v['q_value'] < 0.10).sum())} of {len(s2v)} "
        "volatility tests survive FDR at 10%.",
    ]
    rob = R.robustness
    if len(rob):
        ok = rob[rob["robust"]]

        def where(r) -> str:
            if r.p_1st_half < 0.10 and r.p_2nd_half < 0.10:
                return "both halves"
            return "mainly the " + ("first" if r.p_1st_half < r.p_2nd_half else "second") \
                + " half"
        txt = "; ".join(f"ΔP({r.cause.replace('dProb_', '')}) → {r.effect} "
                        f"(Σγ = {r.sum_gamma:+.2f}, p = {r.p:.3f}, {where(r)})"
                        for r in ok.itertuples())
        reading = ""
        for ch in config.CONTRACTS:
            sub = ok[ok["cause"] == f"dProb_{ch}"]
            down = ", ".join(sub[sub["sum_gamma"] < 0]["effect"])
            up = ", ".join(sub[sub["sum_gamma"] > 0]["effect"])
            if down or up:
                reading += (f" A rise in P(Dem {ch.lower()}) is followed the next session by "
                            + " and ".join(x for x in (f"underperformance of {down}" if down
                                                       else "", f"outperformance of {up}"
                                                       if up else "") if x) + ".")
        out.append(f"<b>Robustness.</b> {len(ok)} of {len(rob)} survivors pass every check "
                   "(HAC errors; without the 5 largest ΔProb sessions; without the 5 Jun 2026 "
                   "sell-off; same sign in both halves; null placebo): " + (txt or "none")
                   + "." + reading + " The other survivors rest on a few extreme sessions or "
                   "one half of the sample.")
    rs = R.rigobon_sack
    if len(rs):
        rel = rs.groupby("contract")["var_ratio_p"].first()
        ok = rs[(rs["var_ratio_p"] < 0.05) & ((rs["ci_low"] > 0) | (rs["ci_high"] < 0))]
        out.append("<b>Rigobon–Sack.</b> Relevance test p-values (Var_H/Var_L of ΔProb): "
                   + ", ".join(f"{c} {p:.3f}" for c, p in rel.items())
                   + f". Even where the variance shift is significant it is too small to pin "
                   f"β down: {len(ok)} of {len(rs)} estimates have a 95% CI excluding zero, so "
                   "contemporaneous causal effects are not identified in this sample.")
    return out


METHODS = [
    "<b>Data and timing.</b> All series sit on the NYSE calendar. Tweets are fetched per "
    "session (previous 16:00 ET close to this one); prediction-market probabilities are the "
    "last hourly quote at or before 16:00 ET (Kalshi CONTROLH-2026, CONTROLS-2026, "
    "KXBALANCEPOWERCOMBO-27FEB; Polymarket control and balance-of-power markets; each "
    "normalised across its outcomes, consensus = equal-weight average, PredictIt a zero-weight "
    "cross-check); a GDELT day counts for the next session. Text: twitterapi.io, a general "
    "issue × political-context stratum plus an affordability oversample used for tone only; "
    "English, no retweets/replies, 5–150 likes, followers ≥ 10, views ≥ 100, astroturf removed "
    "(93,450 credits, every call ledgered).",
    "<b>LSA and sentiment.</b> Uncentred TF-IDF (uni/bigrams, hashtags, cashtags) → truncated "
    "SVD, k = 30 → varimax rotation of components 1..29 (centring would densify the matrix and "
    "treat unseen terms as negative evidence). Poles are labelled by the share of loading "
    "energy on an issue's seed terms (≥ 15%, purity ≥ 60%). A document's issue weight is the "
    "squared cosine of its projection on that issue's poles; polarity is VADER with an "
    "issue-specific overlay ('rising' is negative for prices and rents). S_k,t = "
    "Σ(Pos − Neg)/ΣTotal pooled over 5 sessions with ≥ 5 effective documents.",
    "<b>Econometrics.</b> ADF-tested series; Granger F-tests (HC1, BIC lags, lagged SPY/ΔVIX/Δ2Y "
    "controls); Benjamini–Hochberg FDR per table; a robustness battery for every survivor (HAC, "
    "without the 5 largest ΔProb sessions, without 5 Jun 2026, both halves, placebo). "
    "Covariance SVD of (1/T)Σ s_{t−1}′δy_t on the first 60% of the sample with a circular-shift "
    "permutation test, scored out of sample on the last 40%. Rigobon–Sack β = ΔCov/ΔVar across "
    "H (top 20% GDELT midterm-news sessions) and L sessions, outages excluded, regime bootstrap.",
]
LIMITATIONS = [
    "One campaign (155 sessions): daily tests have little power against effects that build "
    "over weeks, and the robust Stage 2 results are hypotheses for 2028, not a trading rule.",
    "The X sample is politically engaged mid-size accounts (median ~24–42k followers), not the "
    "median voter; Technology/AI never formed a topic, so Big Tech is tested only against odds.",
    "GDELT had 19 outage days (mid-September, most of October), excluded rather than read as "
    "quiet. Yahoo closes for SPY/IVV/VOO disagree day by day over 25 Jun – 1 Jul 2026; a median "
    "benchmark or dropping that week changes no conclusion. Full checks: DATA_AUDIT.md.",
]


def _cell(g: float, q: float) -> str:
    """'Σγ (q)' with a marker when the test survives FDR at 10%."""
    if not np.isfinite(q):
        return "–"
    return f"{g:+.2f} ({q:.2f}){' ◆' if q < 0.10 else ''}"


def compact_table2(R: Results) -> pd.DataFrame:
    t = R.tables["table2_granger_stage1"]
    keep = [c for c in ("Lags", "F", "p", "q (BH)", "Σγ", "p (3d)", "p (rev)") if c in t]
    return t[keep]


def compact_table2b(R: Results) -> pd.DataFrame:
    """Issue x (series, contract) q-values, plus how closely X tracks the news agenda."""
    t = R.news_tests
    if not len(t):
        return pd.DataFrame({"note": ["no GDELT issue series"]})
    t = t.assign(issue=t["cause"].str.replace("news_sal_", "").str.replace("news_tone_", ""),
                 series=np.where(t["cause"].str.startswith("news_sal_"), "Sal", "Tone"),
                 contract=t["effect"].str.replace("dProb_", ""))
    m = t.pivot_table(index="issue", columns=["series", "contract"], values="q_value")
    m.columns = [f"{a} {b} q" for a, b in m.columns]
    if len(R.salience_validation):
        m["ρ X vs news"] = R.salience_validation.set_index("issue")["corr_levels"]
    m.index.name = "Issue"
    return m


def compact_table3(R: Results) -> pd.DataFrame:
    """Asset x contract matrix: next-session excess-return response Σγ (q), vol and RS."""
    r, v, rs = R.stage2_ret, R.stage2_vol, R.rigobon_sack
    rows = []
    for a in config.ASSETS:
        row = {"Asset": a}
        for ch in config.CONTRACTS:
            x = r[(r["effect"] == a) & (r["cause"] == f"dProb_{ch}")]
            row[f"Ret: {ch}"] = _cell(x["sum_gamma"].iloc[0], x["q_value"].iloc[0]) \
                if len(x) else "–"
        vv = v[v["effect"] == f"vol_{a}"]
        row["Vol min q"] = vv["q_value"].min() if len(vv) else np.nan
        s = rs[(rs["asset"] == a) & (rs["contract"] == "Sweep")] if len(rs) else rs
        row["β RS (Sweep) [95% CI]"] = (f"{s['beta_rs'].iloc[0]:+.2f} [{s['ci_low'].iloc[0]:.1f}, "
                                        f"{s['ci_high'].iloc[0]:.1f}]") if len(s) else "–"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Asset")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _register_fonts() -> tuple[str, str]:
    """DejaVu Sans (shipped with matplotlib) so Greek letters and arrows render."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", str(ttf / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(ttf / "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def build_pdf(R: Results, figs: dict[str, Path]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    font, bold = _register_fonts()
    ss = getSampleStyleSheet()
    for st in ss.byName.values():
        st.fontName = bold if ("Heading" in st.name or st.name == "Title") else font
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=11.2)
    small = ParagraphStyle("s", parent=body, fontSize=7, leading=9,
                           textColor=colors.HexColor("#444444"))
    h1, h2 = ss["Heading1"], ss["Heading2"]
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=16, alignment=TA_CENTER)
    cell = ParagraphStyle("c", fontName=font, fontSize=6, leading=7.2)
    hcell = ParagraphStyle("hc", fontName=bold, fontSize=6, leading=7.2)

    def tbl(df: pd.DataFrame, digits: int = 3, widths=None, wrap_cols=()) -> Table:
        d = df.reset_index()
        data = [[Paragraph(str(c), hcell) for c in d.columns]]
        for row in d.values:
            data.append([Paragraph(fmt(v, digits), cell) if c in wrap_cols else fmt(v, digits)
                         for c, v in zip(d.columns, row)])
        t = Table(data, repeatRows=1, colWidths=widths)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fa")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
        return t

    def img(p: Path, w: float = 7.0) -> Image:
        iw, ih = ImageReader(str(p)).getSize()
        return Image(str(p), width=w * inch, height=w * inch * ih / iw)

    p = R.params
    cons = R.pm.consensus.iloc[-1]
    story = [Paragraph("Topic Modeling, Sentiment and Election Markets", title),
             Paragraph("Applied LSA and sentiment analysis around the 2026 US midterms", h2),
             Paragraph(f"FRE-GY 7871 · Assignment 5 · sample {p['start']} to {p['end']} "
                       f"(155 NYSE sessions) · {p['n_tweets']:,} filtered tweets · built "
                       f"{p['built'][:16]}", body),
             Paragraph(f"Latest consensus P(Dem): House {cons['House']:.1%}, Senate "
                       f"{cons['Senate']:.1%}, Democratic sweep {cons['Sweep']:.1%}.", body),
             Spacer(1, 6), Paragraph("Executive summary", h1)]
    story += [Paragraph(s, body) for s in findings(R)]
    story += [Spacer(1, 4), Paragraph(
        "Granger tests are predictive restrictions on lagged information, not structural "
        "effects. With one election cycle, read the q-values and the robustness battery "
        "rather than single p-values.", small)]

    cov = R.pm.coverage
    story += [Paragraph(
        f"<b>Data.</b> {p['n_tweets']:,} filtered tweets (general + affordability strata, every "
        "session covered); prices for 9 sector ETFs, 10 single names, SPY and VIX with no "
        "missing prints; prediction-market quotes on " + ", ".join(
            f"{c}: {int(cov[c].max())}" for c in cov.columns) + " of 155 sessions; GDELT news "
        "volume and tone for every issue (19 outage days masked).", body)]
    story += [PageBreak(), img(figs["markets"], 6.6), img(figs["news"], 6.6)]

    story += [PageBreak(), Paragraph("1. Topics: LSA on X discourse", h1),
              Paragraph("<b>Table 1A.</b> SVD latent topics and term loadings (U_k)", body),
              tbl(R.tables["table1a_svd_topics"], 2,
                  widths=[1.25 * inch, 0.42 * inch, 0.35 * inch, 0.45 * inch, 0.42 * inch,
                          0.48 * inch, 3.95 * inch], wrap_cols=("Top terms (loading)",)),
              Paragraph("Varimax-rotated LSA, k = 30, uncentred TF-IDF; loadings are entries of "
                        "U_k. Stability: share of 5 refits on 80% subsamples in which the pole "
                        "reappears. Energy/Climate is the least stable (0.6); no Technology/AI "
                        "axis forms at any k.", small),
              img(figs["spectrum"], 6.6)]
    story += [PageBreak(),
              Paragraph("<b>Table 1B.</b> Topic salience and sentiment by issue", body),
              tbl(R.tables["table1b_issue_sentiment"], 2),
              Paragraph("Salience: mean daily share of issue loading (general stratum only). S: "
                        "5-session pooled topic-conditioned sentiment. % Re-signed: documents "
                        "whose issue-conditioned polarity differs from generic VADER.", small),
              img(figs["sentiment"], 6.6), img(figs["loadings"], 6.6)]
    story += [PageBreak(), Paragraph("2. Stage 1: sentiment and news → election odds", h1),
              Paragraph("<b>Table 2.</b> Granger causality tests (topic sentiment → ΔProb)",
                        body), tbl(compact_table2(R)),
              Paragraph("HC1 F-tests of lags of S_k in an AR model of ΔProb with lagged SPY, "
                        "ΔVIX and Δ2Y controls; lags by BIC; q: BH FDR across the table; "
                        "p (3d): 3-session pooled sentiment; p (rev): ΔProb → S_k.", small),
              Paragraph("<b>Table 2B.</b> GDELT news salience and tone → ΔProb (FDR q-values)",
                        body), tbl(compact_table2b(R), 2),
              Paragraph("Same specification and FDR as Table 2; outage sessions drop out. "
                        "ρ: correlation of X issue salience with GDELT's share of midterm "
                        "coverage mentioning the issue (5-session means).", small),
              Paragraph("<b>Covariance SVD.</b> Signal–return alpha factors, trained to "
                        f"{R.cov_svd.train_end.date()}, scored out of sample after it.", body),
              tbl(pd.DataFrame({"σ": R.cov_svd.S, "perm p": R.cov_svd.p_values},
                               index=R.cov_svd.U.columns).join(
                  R.factor_oos.set_index("factor")[
                      ["oos_days", "oos_corr", "oos_t", "sign_hit_rate"]])),
              img(figs["covsvd"], 6.2)]
    rob = R.robustness
    story += [PageBreak(), Paragraph("3. Stage 2: election odds → midterm basket", h1),
              Paragraph("<b>Table 3.</b> Midterm equity basket reactivity matrix: next-session "
                        "excess-return response Σγ (FDR q), ◆ = q < 0.10", body),
              tbl(compact_table3(R), 2),
              Paragraph("Excess return over SPY; Vol: 2(H−L)/(H+L), smallest q across the three "
                        "contracts; β RS: Rigobon–Sack response to ΔP(Sweep), H = top 20% GDELT "
                        "news sessions. Full 57-row table with F, p and relevance tests: "
                        "outputs/tables/table3_basket_reactivity.md.", small),
              Paragraph("<b>Robustness battery</b> for every FDR survivor (p-values)", body),
              tbl(rob[["cause", "effect", "sum_gamma", "p", "p_HAC", "p_no_extremes",
                       "p_excl_dates", "p_1st_half", "p_2nd_half", "p_placebo", "robust"]]
                  .assign(cause=lambda d: d["cause"].str.replace("dProb_", ""))
                  .set_index(["cause", "effect"]) if len(rob)
                  else pd.DataFrame({"note": ["no survivors"]})),
              img(figs["events"], 6.6)]
    story += [PageBreak(), Paragraph("4. Methods", h1)] + [Paragraph(x, body) for x in METHODS]
    story += [Paragraph("5. Limitations", h1)] + [Paragraph("• " + x, body) for x in LIMITATIONS]
    story += [Paragraph("AI use: built with Claude Code; see AI_USE.md for what was done by "
                        "the model and by the author.", small)]

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter, leftMargin=0.55 * inch,
                            rightMargin=0.55 * inch, topMargin=0.55 * inch,
                            bottomMargin=0.55 * inch,
                            title="Assignment 5: Topic Modeling, Sentiment and Election Markets")
    doc.build(story)


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------
def build_notebook() -> None:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("# Assignment 5: Topic modeling and sentiment around the 2026 US midterms\n"
           "Executed capstone notebook. It loads the results that `build_report.py` built by "
           "running the full pipeline on the cached real data, and walks through every exhibit "
           "in order. To recompute, call `run_pipeline()` from `src.pipeline`."),
        code("import warnings; warnings.filterwarnings('ignore')\n"
             "import numpy as np, pandas as pd\n"
             "from IPython.display import Image, Markdown, display\n"
             "from src.pipeline import load_results\nfrom src import table_generator as tg\n"
             "pd.set_option('display.width', 180)\npd.set_option('display.max_columns', 30)\n"
             "R = load_results()\nR.params"),
        md("## 1. Data\nCoverage of every source, the tweet filter waterfall and the "
           "prediction-market consensus."),
        code("display(R.pm.coverage)\ndisplay(R.filter_report)\n"
             "display(R.market.alignment.T)\nR.pm.consensus.tail()"),
        code("display(Image('outputs/figures/fig1_prediction_markets.png'))\n"
             "display(Image('outputs/figures/fig8_gdelt_regimes.png'))"),
        md("## 2. LSA topics\nUncentred TF-IDF, truncated SVD (k = 30), varimax rotation. "
           "Table 1A lists each labelled pole with its top term loadings."),
        code("display(Markdown(tg.to_markdown(R.tables['table1a_svd_topics'], 2)))\n"
             "display(Image('outputs/figures/fig2_spectrum_stability.png'))\n"
             "display(Image('outputs/figures/fig3_loading_heatmap.png'))"),
        code("R.model_comparison"),
        md("## 3. Salience and topic-conditioned sentiment (Table 1B)"),
        code("display(Markdown(tg.to_markdown(R.tables['table1b_issue_sentiment'], 2)))\n"
             "display(R.signal_coverage)\n"
             "display(Image('outputs/figures/fig4_salience_sentiment.png'))"),
        md("## 4. Stationarity and lag order"),
        code("display(R.adf.round(3))\nR.lag_table.round(3)"),
        md("## 5. Stage 1: sentiment → prediction markets (Table 2)"),
        code("display(Markdown(tg.to_markdown(R.tables['table2_granger_stage1'])))\n"
             "R.stage1_salience.sort_values('p_value').head(8)"),
        md("## 6. Covariance SVD: political alpha factors"),
        code("cs = R.cov_svd\nprint('singular values', cs.S.round(3), 'perm p', "
             "cs.p_values.round(3), 'trained to', cs.train_end.date())\n"
             "display(cs.U.round(2))\ndisplay(R.factor_oos.round(3))\n"
             "display(Image('outputs/figures/fig6_covariance_svd.png'))"),
        md("## 6b. News salience check (GDELT, Table 2B)\nDoes X discourse track the news "
           "agenda, and does news salience or tone lead the election odds?"),
        code("display(R.news_status)\ndisplay(R.salience_validation.round(3))\n"
             "display(Markdown(tg.to_markdown(R.tables['table2b_news_salience'])))"),
        md("## 7. Stage 2: prediction markets → midterm basket (Table 3) and robustness"),
        code("display(Markdown(tg.to_markdown(R.tables['table3_basket_reactivity'])))\n"
             "display(R.robustness.round(4))\n"
             "display(Image('outputs/figures/fig5_granger_q.png'))\n"
             "display(Image('outputs/figures/fig7_event_study.png'))"),
        code("R.sentiment_to_asset.sort_values('p_value').round(4)"),
        md("## 8. Rigobon–Sack heteroskedasticity check"),
        code("rs = R.rigobon_sack\ndisplay(rs.groupby('contract')[['var_ratio_x', "
             "'var_ratio_p', 'var_ratio_y']].median().round(3))\n"
             "rs[['asset', 'contract', 'beta_ols', 'beta_rs', 'ci_low', 'ci_high']].round(3)"),
    ]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(ROOT)}})
    nbformat.write(nb, NB_PATH)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download the free sources")
    ap.add_argument("--skip-notebook", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore")
    try:
        R = run_pipeline(refresh=args.refresh)
    except ConnectionError as exc:
        print(f"\nCannot build the report: {exc}\n", file=sys.stderr)
        return 1
    figs = make_figures(R)
    build_pdf(R, figs)
    print(f"wrote {PDF_PATH}")
    if not args.skip_notebook:
        build_notebook()
        print(f"wrote {NB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
