"""Run the pipeline and build assignment5_report.pdf and assignment5_capstone.ipynb.

    python build_report.py                 # download fresh data, then build
    python build_report.py --no-refresh    # rebuild from the cache in data/raw/
    python build_report.py --skip-notebook

Fails with the list of unreachable sources if real data cannot be obtained;
nothing is ever simulated.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import granger_econometrics as ge  # noqa: E402
from src.pipeline import Results, run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "assignment5_report.pdf"
NB_PATH = ROOT / "assignment5_capstone.ipynb"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(R: Results) -> dict[str, Path]:
    figs = {}

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
    for ax, ch in zip(axes, config.CHAMBERS):
        sub = R.pm.by_venue[R.pm.by_venue["chamber"] == ch]
        for v, g in sub.groupby("venue"):
            ax.plot(g["date"], g["prob_dem"], lw=0.8, alpha=0.7, label=v)
        ax.plot(R.pm.consensus.index, R.pm.consensus[ch], color="black", lw=1.6,
                label="consensus")
        ax.axhline(0.5, color="grey", lw=0.5, ls=":")
        ax.set_ylabel(f"P(Dem) {ch}")
    axes[0].legend(ncol=4, fontsize=7, frameon=False)
    fig.suptitle("Figure 1. Implied probability of Democratic control, by venue")
    figs["fig1"] = _save(fig, "fig1_prediction_markets.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    sal = R.daily.salience.T
    im = ax.imshow(sal.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(sal.index)), sal.index)
    ticks = np.linspace(0, sal.shape[1] - 1, 8).astype(int)
    ax.set_xticks(ticks, [sal.columns[i].strftime("%d %b") for i in ticks])
    fig.colorbar(im, ax=ax, label="share of documents")
    ax.set_title("Figure 2. Daily topic salience")
    figs["fig2"] = _save(fig, "fig2_salience_heatmap.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for k in R.daily.tone.columns:
        ax.plot(R.daily.tone.index, R.daily.tone[k].rolling(5, min_periods=2).mean(),
                lw=1, label=k)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Tone (5-day mean)")
    ax.legend(ncol=3, fontsize=7, frameon=False)
    ax.set_title("Figure 3. Topic-conditioned tone, (Pos - Neg) / Total")
    figs["fig3"] = _save(fig, "fig3_tone.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4),
                             gridspec_kw={"width_ratios": [1, 2.2]})
    for ax, (df, title) in zip(axes, [(R.stage1, "Tone → ΔProb"),
                                      (R.stage2_ret, "ΔProb → excess return")]):
        m = df.pivot(index="cause", columns="effect", values="p_value")
        ax.imshow(-np.log10(m.clip(lower=1e-4)).values, cmap="Blues", vmin=0, vmax=3,
                  aspect="auto")
        ax.set_xticks(range(m.shape[1]), [c.replace("dProb_", "") for c in m.columns],
                      rotation=45, ha="right")
        ax.set_yticks(range(m.shape[0]), [c.replace("tone_", "").replace("dProb_", "")
                                          for c in m.index])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
    fig.suptitle("Figure 4. Granger p-values (darker = smaller)")
    figs["fig4"] = _save(fig, "fig4_granger_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True)
    for ax, ch in zip(axes, config.CHAMBERS):
        es = ge.event_study(R.market.excess, R.pm.dprob[ch], n_events=10, window=5)
        if len(es):
            (es * 100).plot(ax=ax, lw=1, legend=ch == config.CHAMBERS[-1])
        ax.axvline(0, color="grey", ls=":")
        ax.set_title(f"{ch}: 10 largest |ΔProb| days")
        ax.set_xlabel("trading days from event")
    axes[0].set_ylabel("cum. signed excess return (%)")
    if axes[-1].get_legend():
        axes[-1].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("Figure 5. Sector reactions to large prediction-market moves")
    figs["fig5"] = _save(fig, "fig5_event_study.png")
    return figs


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


def _save(fig, name: str) -> Path:
    path = config.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Summary statements, generated from the numbers
# ---------------------------------------------------------------------------
def findings(R: Results) -> list[str]:
    out = []
    s1 = R.stage1
    sig = s1[s1["q_value"] < 0.10]
    out.append(f"Stage 1: {len(sig)} of {len(s1)} tone → ΔProb tests reject at an FDR of 10% "
               f"(BH); {int((s1['p_value'] < 0.05).sum())} reject at an unadjusted 5% level.")
    for _, r in sig.sort_values("p_value").head(4).iterrows():
        out.append(f"  {r['cause'].replace('tone_', '')} → {r['effect'].replace('dProb_', '')}: "
                   f"F = {r['F']:.2f}, p = {r['p_value']:.3f}, q = {r['q_value']:.3f}, "
                   f"Σγ = {r['sum_gamma']:+.4f}.")
    rv = R.stage1_reverse
    out.append(f"Reverse direction (ΔProb → tone): {int((rv['q_value'] < 0.10).sum())} of "
               f"{len(rv)} reject at FDR 10%.")
    for name, df in (("excess returns", R.stage2_ret), ("range volatility", R.stage2_vol)):
        k = df[df["q_value"] < 0.10]
        pairs = ", ".join(f"{a.replace('dProb_', '')}→{b}"
                          for a, b in zip(k["cause"], k["effect"]))
        out.append(f"Stage 2 ({name}): {len(k)} of {len(df)} ΔProb → sector tests reject at "
                   f"FDR 10%" + (f": {pairs}." if pairs else "."))
    rs = R.rigobon_sack
    if len(rs):
        relevant = rs.groupby("chamber")["var_ratio_p"].max() < 0.05
        ok = rs[rs["chamber"].isin(relevant[relevant].index)]
        excl = ok[(ok["ci_low"] > 0) | (ok["ci_high"] < 0)]
        out.append(f"Rigobon-Sack: ΔProb variance is significantly higher on high-news days "
                   f"for {int(relevant.sum())} of {len(relevant)} chambers (F-test p < 0.05); "
                   f"for those, {len(excl)} of {len(ok)} sector betas have a 95% CI excluding "
                   f"zero.")
    return out


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def build_pdf(R: Results, figs: dict[str, Path]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    font, bold = _register_fonts()
    ss = getSampleStyleSheet()
    for st in ss.byName.values():
        st.fontName = bold if "Heading" in st.name or st.name == "Title" else font
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5,
                           textColor=colors.HexColor("#444444"))
    h1, h2 = ss["Heading1"], ss["Heading2"]
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=17, alignment=TA_CENTER)

    regular = font

    def tbl(df: pd.DataFrame, font: float = 6.8, digits: int = 3) -> Table:
        from src.table_generator import fmt

        d = df.reset_index()
        hs = ParagraphStyle("h", fontName=bold, fontSize=font, leading=font + 1.5)
        data = [[Paragraph(str(c), hs) for c in d.columns]]
        data += [[fmt(v, digits) for v in row] for row in d.values]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), regular),
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fa")]),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
        return t

    def img(p: Path, w: float = 7.0) -> Image:
        from reportlab.lib.utils import ImageReader

        iw, ih = ImageReader(str(p)).getSize()
        return Image(str(p), width=w * inch, height=w * inch * ih / iw)

    p = R.params
    story = [Paragraph("Voter Sentiment, Topic Salience and Election Prediction Markets", title),
             Paragraph("Quantifying their financial impact during the 2026 US midterms", h2),
             Paragraph(f"FRE-GY 7871 · Assignment 5 · Sample {p['start']} to {p['end']} · "
                       f"built {p['built'][:16]}", body), Spacer(1, 10),
             Paragraph("Executive summary", h1)]
    story += [Paragraph(s, body) for s in findings(R)]
    story += [Spacer(1, 6), Paragraph(
        "All tests are predictive (Granger) restrictions on lagged information and do not by "
        "themselves identify causal effects. Sample length is one campaign season, so power "
        "is limited and individual rejections should be read alongside the FDR-adjusted "
        "q-values.", small)]

    story += [Paragraph("Data and provenance", h1), Paragraph(
        f"<b>Markets:</b> {R.market.source} (sector ETFs {', '.join(config.SECTOR_ETFS)}; "
        f"SPY, ^VIX; FRED DGS2, DGS3MO). <b>Prediction markets:</b> Polymarket CLOB price "
        "history, Kalshi daily candles (CONTROLH-2026, CONTROLS-2026), PredictIt exports and "
        f"snapshots. <b>Text:</b> {R.text_sources}.", body), Spacer(1, 4),
        Paragraph("Venue coverage (trading days with a quote)", h2), tbl(R.pm.coverage),
        Spacer(1, 4), Paragraph("Text quality-filter waterfall", h2),
        tbl(R.filter_report.set_index("step")),
        Spacer(1, 4), Paragraph("Alignment of market series to the NYSE calendar", h2),
        tbl(R.market.alignment)]
    if R.pm.errors:
        story.append(Paragraph("Venue errors on this run: " + "; ".join(
            f"{k}: {v[:120]}" for k, v in R.pm.errors.items()), small))

    story += [PageBreak(), Paragraph("Results", h1), img(figs["fig1"])]
    t1 = R.tables["table1_topic_salience"]
    story += [Paragraph("<b>Table 1.</b> Topic salience and sentiment distribution across "
                        "midterm priority issues", body), tbl(t1, digits=2),
              Paragraph("Salience: mean daily share of filtered documents. Tone = (Pos - Neg)/"
                        "Total per topic-day. % Re-signed: documents whose topic-conditioned "
                        "polarity differs from generic VADER.", small),
              img(figs["fig2"]), img(figs["fig3"])]
    story += [PageBreak(), Paragraph("<b>Table 2.</b> Granger causality: public tone &rarr; "
                                     "prediction-market probabilities", body),
              tbl(R.tables["table2_granger_stage1"]),
              Paragraph(f"OLS F-tests ({p['cov']} covariance) of lags of Tone_k in an AR model "
                        "of dProb with lagged SPY return, dVIX and dDGS2 controls; lag order by "
                        f"{str(p['lags']).upper()}. q: Benjamini-Hochberg FDR-adjusted. "
                        "p (reverse): dProb &rarr; Tone_k. *** p&lt;0.01 ** p&lt;0.05 * p&lt;0.10.",
                        small), img(figs["fig4"])]
    story += [PageBreak(), Paragraph("<b>Table 3.</b> Sector ETF reactivity matrix: "
                                     "prediction-market shifts &rarr; excess returns and "
                                     "volatility", body),
              tbl(R.tables["table3_sector_reactivity"], font=6.2),
              Paragraph("Ret: excess return over SPY; Vol: 2(H-L)/(H+L), differenced if the ADF "
                        "test does not reject a unit root. &beta; RS: Rigobon-Sack (2003) "
                        "estimate, dCov/dVar across high-news (top "
                        f"{int(100 * (1 - p['high_quantile']))}% discourse volume) and quiet "
                        "days, with a regime-stratified bootstrap 95% CI. VarH/VarL is the "
                        "relevance condition: it must be well above 1.", small),
              img(figs["fig5"])]
    story += [PageBreak(), Paragraph("Methods", h1)]
    for para in METHODS:
        story.append(Paragraph(para, body))
    story += [Paragraph("Topic-model comparison", h2), tbl(R.model_comparison.set_index("method")),
              Paragraph("Topic components", h2), tbl(pd.DataFrame({
                  "label": R.topic_model.component_labels,
                  "top terms": [", ".join(t[:7]) for t in R.topic_model.top_terms]}), font=6.5),
              Paragraph("Lag-order selection (VAR of dProb and tone)", h2), tbl(R.lag_table)]
    story += [Paragraph("Limitations", h1)] + [Paragraph(x, body) for x in LIMITATIONS]

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter, leftMargin=0.6 * inch,
                            rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title="Assignment 5: Voter Sentiment and Election Markets")
    doc.build(story)


METHODS = [
    "<b>Calendar.</b> Every series lives on the NYSE trading calendar. Text and quotes "
    "stamped at or after 16:00 ET, or on a non-trading day, belong to the next session, so "
    "weekend discourse is information for Monday's close, never Friday's. Returns are never "
    "computed across a missing print; FRED yields are forward-filled at most three days.",
    "<b>Prediction markets.</b> Each venue's D and R contracts are normalised to sum to one "
    "(overround removed). The consensus is a fixed-weight average (Polymarket 0.45, Kalshi "
    "0.45, PredictIt 0.10) renormalised over the venues quoting that day; ΔProb is its daily "
    "change in probability points.",
    "<b>Text.</b> Tweets from twitterapi.io advanced search (English, -is:retweet, four 6-hour "
    "windows per day) and GDELT headlines. Filters: followers ≥ 10, views ≥ 100, identical "
    "normalised text from ≥ 5 accounts (astroturf), fewer than 4 tokens.",
    "<b>Topics.</b> TF-IDF (uni+bigrams) NMF with 12 components, each component named by its "
    "overlap with seed vocabularies for the priority issues; components without overlap are "
    "Other. LDA (and BERTopic when installed) are fitted for comparison on UMass coherence and "
    "topic diversity.",
    "<b>Tone.</b> VADER with an issue-specific lexicon overlay: words whose valence depends on "
    "the issue (\"rising\", \"cuts\", \"surge\") are re-scored per topic. This is the lesson "
    "from off-the-shelf FinBERT, which reads macro growth language as dovish: a generic "
    "polarity model is a measurement error that correlates with the topic. Emotions (anxiety, "
    "anger, optimism) are lexicon flags. Tone_k,t = (Pos - Neg)/Total ∈ [-1, 1].",
    "<b>Econometrics.</b> Series are ADF-tested and differenced where needed. Granger tests are "
    "OLS F/Wald tests with robust covariance, lags by BIC (max 5), with lagged macro controls, "
    "and a joint VAR test as a cross-check. Each table is FDR-corrected (Benjamini-Hochberg). "
    "The Rigobon-Sack estimator β = (Cov_H - Cov_L)/(Var_H(ΔProb) - Var_L(ΔProb)) uses the "
    "rise in the variance of political-news shocks on high-discourse days to identify the "
    "contemporaneous effect of ΔProb on returns despite simultaneity.",
]
LIMITATIONS = [
    "One election cycle gives roughly 150 trading days; tests have low power and a handful of "
    "rejections among dozens of tests is what chance alone would deliver, hence the q-values.",
    "Prediction-market prices were far from 50% for the House for most of the window, so "
    "probability moves were small and the instrument for the House is weak; the Senate is "
    "the more informative chamber.",
    "The H regime is defined from discourse volume, which is itself partly a response to "
    "market moves; event dates (debates, CPI prints) are a cleaner alternative.",
    "twitterapi.io search returns a sample of matching posts, not the full firehose, and "
    "GDELT's DOC API only covers about three months, so salience is measured with error.",
]


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------
def build_notebook() -> None:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("# Assignment 5: Voter sentiment, prediction markets and sector ETFs\n"
           "Executed capstone notebook. It re-runs the pipeline from the cached real data "
           "in `data/raw/` (run `python build_report.py` first) and displays every exhibit."),
        code("import pandas as pd\nfrom IPython.display import Image, Markdown, display\n"
             "from src.pipeline import run_pipeline\nfrom src import table_generator as tg\n"
             "pd.set_option('display.width', 160)\nR = run_pipeline(refresh=False, save=False)\n"
             "R.params, R.text_sources, R.market.source"),
        md("## 1. Data coverage and quality filters"),
        code("display(R.pm.coverage)\ndisplay(R.filter_report)\ndisplay(R.market.alignment)"),
        md("## 2. Prediction markets"),
        code("display(Image('outputs/figures/fig1_prediction_markets.png'))\n"
             "R.pm.consensus.describe()"),
        md("## 3. Topics, salience and tone"),
        code("display(R.model_comparison)\n"
             "pd.DataFrame({'label': R.topic_model.component_labels,\n"
             "              'top terms': [', '.join(t[:8]) for t in R.topic_model.top_terms]})"),
        code("display(Markdown(tg.to_markdown(R.tables['table1_topic_salience'], 2)))\n"
             "display(Image('outputs/figures/fig2_salience_heatmap.png'))\n"
             "display(Image('outputs/figures/fig3_tone.png'))"),
        md("## 4. Stationarity and lag order"),
        code("display(R.adf)\nR.lag_table"),
        md("## 5. Stage 1: tone → prediction-market probabilities (Table 2)"),
        code("display(Markdown(tg.to_markdown(R.tables['table2_granger_stage1'])))\n"
             "display(Image('outputs/figures/fig4_granger_matrix.png'))\nR.stage1_salience"),
        md("## 6. Stage 2: prediction markets → sector ETFs (Table 3)"),
        code("display(Markdown(tg.to_markdown(R.tables['table3_sector_reactivity'])))\n"
             "display(R.tone_to_sector)\n"
             "display(Image('outputs/figures/fig5_event_study.png'))"),
        md("## 7. Rigobon-Sack heteroskedasticity check"),
        code("R.rigobon_sack.round(4)"),
    ]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(ROOT)}})
    nbformat.write(nb, NB_PATH)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-refresh", action="store_true", help="use cached data only")
    ap.add_argument("--topic-method", default="nmf", choices=["nmf", "lda", "bertopic"])
    ap.add_argument("--skip-notebook", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        R = run_pipeline(refresh=not args.no_refresh, topic_method=args.topic_method)
    except ConnectionError as exc:
        print(f"\nCannot build the report without real data.\n{exc}\n", file=sys.stderr)
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
