"""Tables 2-6 and Figure 1.

Each function returns a frame ready to print, so the notebook stays a narrative
rather than a pile of pandas. Regression specifications are written out in the
docstrings because the report has to state them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .analysis import newey_west_trend

MEASURES = {
    "prop_negative": "Negative, proportional",
    "tfidf_negative": "Negative, tf.idf",
    "prop_uncertainty": "Uncertainty, proportional",
    "tfidf_uncertainty": "Uncertainty, tf.idf",
}
CONTROLS = ["log_size", "log_dollar_vol", "pre_exret", "is_10k"]


# ---------------------------------------------------------------------------
# Table 2 -- summary statistics
# ---------------------------------------------------------------------------
def table2(panel: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for both measures on both lexicons, by form type."""
    rows = []
    for form in ("10-K", "10-Q", "All"):
        sub = panel if form == "All" else panel[panel["form"] == form]
        for col, label in MEASURES.items():
            s = sub[col].dropna()
            rows.append({
                "form": form, "measure": label, "n": len(s),
                "mean": s.mean(), "sd": s.std(), "min": s.min(),
                "p25": s.quantile(.25), "median": s.median(),
                "p75": s.quantile(.75), "max": s.max(),
                "cv": s.std() / s.mean() if s.mean() else np.nan,
            })
    return pd.DataFrame(rows)


def measure_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Correlation between the sentiment and uncertainty measures (Q2)."""
    rows = []
    for scheme, a, b in [
        ("Proportional", "prop_negative", "prop_uncertainty"),
        ("tf.idf", "tfidf_negative", "tfidf_uncertainty"),
    ]:
        for form in ("All", "10-K", "10-Q"):
            sub = panel if form == "All" else panel[panel["form"] == form]
            s = sub[[a, b]].dropna()
            rows.append({
                "weighting": scheme, "form": form, "n": len(s),
                "pearson": s[a].corr(s[b]),
                "spearman": s[a].corr(s[b], method="spearman"),
            })
    # and the correlation between the two weightings of the same lexicon
    for lex, a, b in [
        ("Negative", "prop_negative", "tfidf_negative"),
        ("Uncertainty", "prop_uncertainty", "tfidf_uncertainty"),
    ]:
        s = panel[[a, b]].dropna()
        rows.append({"weighting": f"{lex}: prop vs tf.idf", "form": "All",
                     "n": len(s), "pearson": s[a].corr(s[b]),
                     "spearman": s[a].corr(s[b], method="spearman")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 3 -- the thirty most frequent words on each list
# ---------------------------------------------------------------------------
def table3(wordfreq: pd.DataFrame, top: int = 30) -> pd.DataFrame:
    """Most frequent lexicon words in the corpus, with each word's share."""
    out = []
    for lex, sub in wordfreq.groupby("lexicon"):
        sub = sub.sort_values("count", ascending=False).head(top).copy()
        sub["rank"] = range(1, len(sub) + 1)
        sub["cumulative_share"] = sub["share_of_lexicon_count"].cumsum()
        out.append(sub)
    return pd.concat(out, ignore_index=True)[
        ["lexicon", "rank", "word", "count", "share_of_lexicon_count", "cumulative_share"]
    ]


def concentration(wordfreq: pd.DataFrame) -> pd.DataFrame:
    """Top-10 and top-30 share of each lexicon's total count (Q1)."""
    rows = []
    for lex, sub in wordfreq.groupby("lexicon"):
        sub = sub.sort_values("count", ascending=False)
        total = sub["count"].sum()
        rows.append({
            "lexicon": lex,
            "distinct_words_used": int((sub["count"] > 0).sum()),
            "total_occurrences": int(total),
            "top10_share": sub["count"].head(10).sum() / total,
            "top30_share": sub["count"].head(30).sum() / total,
            "top50_share": sub["count"].head(50).sum() / total,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1 -- measures by quarter, with VIX
# ---------------------------------------------------------------------------
def quarterly_series(panel: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Quarterly means by form, the firm-demeaned version, and quarterly VIX.

    Two composition corrections, both needed before the picture means anything:
    10-Ks are longer and heavier in risk language and cluster in Q1, so pooling
    the forms manufactures a calendar sawtooth; and firms differ in baseline
    tone, so a changing mix of filers moves the pooled mean on its own. The
    demeaned series subtracts each firm's own average.
    """
    df = panel.copy()
    df["q"] = pd.PeriodIndex(df["cal_quarter"], freq="Q")

    by_form = {}
    for form in ("10-K", "10-Q"):
        sub = df[df["form"] == form]
        by_form[form] = sub.groupby("q")[list(MEASURES)].mean()

    pooled = df.groupby("q")[list(MEASURES)].mean()

    demeaned = df.copy()
    for col in MEASURES:
        demeaned[col] = demeaned[col] - demeaned.groupby(["firm", "form"])[col].transform("mean")
    dm = {f: demeaned[demeaned["form"] == f].groupby("q")[list(MEASURES)].mean()
          for f in ("10-K", "10-Q")}

    vix = prices["^VIX"].dropna()
    vix_q = vix.groupby(pd.PeriodIndex(vix.index, freq="Q")).mean()
    vix_q = vix_q[(vix_q.index >= pooled.index.min()) & (vix_q.index <= pooled.index.max())]

    df["year"] = df["q"].dt.year
    annual = {f: df[df["form"] == f].groupby("year")[list(MEASURES)].mean()
              for f in ("10-K", "10-Q")}

    return {"by_form": by_form, "pooled": pooled, "demeaned": dm, "vix": vix_q,
            "annual": annual,
            "n_by_form_q": df.groupby(["q", "form"]).size().unstack(fill_value=0),
            "n_by_form_year": df.groupby(["year", "form"]).size().unstack(fill_value=0)}


def plot_figure1(series: dict, path: str | None = None):
    """One panel per measure. 10-Q quarterly, 10-K annual, VIX shaded behind.

    The forms are drawn on different frequencies on purpose. 10-Ks cluster in
    the first calendar quarter -- 2021Q1 carries 57 of them against 8 10-Qs --
    so outside Q1 a "quarterly 10-K mean" is the average of one or two filings
    and swings by more than any real change in language. Plotting that series
    quarterly produces a violent sawtooth that is an artefact of the filing
    calendar, not a fact about tone. The annual 10-K mean rests on ~77 filings
    a point and is comparable across years; the 10-Q mean rests on 60-80 a
    quarter and is fine as it is.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True)
    vix = series["vix"]
    vx = [p.to_timestamp() for p in vix.index]

    for ax, (col, label) in zip(axes.ravel(), MEASURES.items()):
        axv = ax.twinx()
        axv.fill_between(vx, vix.to_numpy(), color="0.87", zorder=0, lw=0)
        axv.set_ylim(0, max(vix) * 2.3)
        axv.set_yticks([10, 20, 30])
        axv.tick_params(axis="y", labelsize=7, colors="0.5")
        axv.set_ylabel("VIX", fontsize=7, color="0.5")

        q = series["by_form"]["10-Q"][col]
        ax.plot([p.to_timestamp() for p in q.index], q.to_numpy(),
                marker="s", ms=3.2, lw=1.4, color="#2B5FB4",
                label="10-Q (quarterly)", zorder=3)

        a = series["annual"]["10-K"][col]
        ax.plot([pd.Timestamp(f"{y}-07-01") for y in a.index], a.to_numpy(),
                marker="o", ms=6.0, lw=2.0, color="#B44B2B",
                label="10-K (annual)", zorder=4)

        ax.set_zorder(axv.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(alpha=.25, lw=.6)
        ax.tick_params(labelsize=8)
        if col.startswith("prop_"):
            ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.2f}%")

    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Figure 1  Tone measures over 2021-2025, by form type, with VIX shaded behind\n"
                 "10-Q quarterly (60-80 filings a point); 10-K annual (~77 a point) because "
                 "10-Ks cluster in Q1",
                 fontsize=10.5, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .945))
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight")
    return fig


def plot_composition(series: dict, path: str | None = None):
    """Figure 1b: filings per quarter by form -- the composition being corrected for.

    This is the evidence for the correction rather than an assertion of it. The
    Q1 spike in 10-Ks is what would drive a sawtooth through any pooled series,
    because 10-Ks are both longer and heavier in risk language than 10-Qs.
    """
    import matplotlib.pyplot as plt

    n = series["n_by_form_q"]
    x = [p.to_timestamp() for p in n.index]
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.bar(x, n["10-Q"], width=70, color="#2B5FB4", label="10-Q")
    ax.bar(x, n["10-K"], width=70, bottom=n["10-Q"], color="#B44B2B", label="10-K")
    ax.set_title("Figure 1b  Filings per quarter by form: 10-Ks cluster in Q1",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=.25, lw=.6)
    ax.tick_params(labelsize=8)
    ax.set_ylabel("filings", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Table 4 -- trend tests
# ---------------------------------------------------------------------------
def table4(panel: pd.DataFrame, series: dict, form: str | None = None) -> pd.DataFrame:
    """Aggregate (OLS and Newey-West) and within-firm trend tests.

    Aggregate: the quarterly mean of the measure regressed on a linear time
    trend, t = 0..19. Twenty observations of a persistent series will produce a
    confident-looking OLS t-statistic whether or not a trend exists, because
    the residuals are autocorrelated; the Newey-West t-statistic is the one to
    read.

    Within-firm: the filing-level measure on a time trend in years with firm
    fixed effects and a 10-K dummy, standard errors clustered by firm. This
    asks whether the *same company* changed its language, which is the question
    the trend is supposed to be about, and is the specification to lead with.
    """
    df = panel if form is None else panel[panel["form"] == form]
    rows = []
    for col, label in MEASURES.items():
        agg_src = series["by_form"][form][col] if form else series["pooled"][col]
        agg = newey_west_trend(agg_src)

        d = df.dropna(subset=[col]).copy()
        d["t_years"] = (d["day0"] - d["day0"].min()).dt.days / 365.25
        rhs = "t_years + C(firm)" + ("" if form else " + is_10k")
        fit = smf.ols(f"{col} ~ {rhs}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["firm"]})

        rows.append({
            "measure": label,
            "form": form or "All",
            "agg_n_quarters": agg["n_obs"],
            "agg_slope_per_year": agg["slope_per_year"],
            "agg_t_ols": agg["t_ols"],
            "agg_t_nw": agg["t_nw"],
            "agg_nw_lags": agg["nw_lags"],
            "within_n": int(fit.nobs),
            "within_slope_per_year": float(fit.params["t_years"]),
            "within_t": float(fit.tvalues["t_years"]),
            "within_p": float(fit.pvalues["t_years"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 5 -- does uncertainty language predict volatility?
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def table5(panel: pd.DataFrame, form: str | None = None,
           time_fe: str = "quarter") -> pd.DataFrame:
    """Post-filing realised volatility on the uncertainty measure, two specs.

        (1) post_vol ~ unc + size + liquidity + pre_exret + 10-K + firm + quarter FE
        (2) the same, plus pre_vol

    The uncertainty measure is standardised so the coefficient reads as the
    change in annualised post-filing volatility per one-standard-deviation of
    uncertainty language. The gap between (1) and (2) is the result: without
    the control, a large coefficient mostly rediscovers that volatile firms
    write hedged filings.

    ``time_fe`` is "quarter" for the pooled sample and "year" for the form
    splits. Firms file one 10-K a year, so on the 10-K subsample firm and
    quarter dummies are very nearly collinear, the design matrix comes back
    rank-deficient, and the reported standard errors stop meaning anything.
    """
    df = panel if form is None else panel[panel["form"] == form]
    ctrl = [c for c in CONTROLS if not (form and c == "is_10k")]
    rows = []
    for col in ("prop_uncertainty", "tfidf_uncertainty"):
        need = ["post_vol", col, "pre_vol", "firm", time_fe] + ctrl
        d = df.dropna(subset=need).copy()
        d["x"] = _z(d[col])
        d["pre_vol_z"] = _z(d["pre_vol"])
        pooled = " + ".join(ctrl)
        base = pooled + f" + C(firm) + C({time_fe})"
        # The first pair leaves the between-firm variation in, which is where
        # the "volatile firms write hedged filings" channel lives and where the
        # pre-filing-volatility control actually bites. The second pair adds
        # firm and quarter fixed effects, which absorb that channel themselves,
        # so the control has much less left to do.
        for name, rhs in [
            ("(1) no FE, without pre-filing vol", f"x + {pooled}"),
            ("(2) no FE, with pre-filing vol", f"x + pre_vol_z + {pooled}"),
            (f"(3) firm+{time_fe} FE, without pre-filing vol", f"x + {base}"),
            (f"(4) firm+{time_fe} FE, with pre-filing vol", f"x + pre_vol_z + {base}"),
        ]:
            fit = smf.ols(f"post_vol ~ {rhs}", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["firm"]})
            rows.append({
                "measure": MEASURES[col], "form": form or "All", "spec": name,
                "n": int(fit.nobs),
                "coef_per_sd": float(fit.params["x"]),
                "se": float(fit.bse["x"]),
                "t": float(fit.tvalues["x"]),
                "p": float(fit.pvalues["x"]),
                "pre_vol_coef": float(fit.params["pre_vol_z"]) if "pre_vol_z" in fit.params else np.nan,
                "pre_vol_t": float(fit.tvalues["pre_vol_z"]) if "pre_vol_z" in fit.params else np.nan,
                "r2": float(fit.rsquared),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 6 -- does sentiment predict the filing-period return?
# ---------------------------------------------------------------------------
def table6(panel: pd.DataFrame) -> pd.DataFrame:
    """[0,+3] excess return on the negative-tone measure, with controls.

    Specifications mirror Table 5. The measure is standardised, so the
    coefficient is the excess return per one standard deviation of negative
    tone, in the same units as the dependent variable.
    """
    rows = []
    for col in ("prop_negative", "tfidf_negative", "prop_uncertainty", "tfidf_uncertainty"):
        need = ["exret_0_3", col, "pre_vol", "firm", "quarter"] + CONTROLS
        d = panel.dropna(subset=need).copy()
        d["x"] = _z(d[col])
        d["pre_vol_z"] = _z(d["pre_vol"])
        for name, rhs in [
            ("(1) measure only", "x"),
            ("(2) + controls", "x + " + " + ".join(CONTROLS) + " + pre_vol_z"),
            ("(3) + controls + FE", "x + " + " + ".join(CONTROLS) +
             " + pre_vol_z + C(firm) + C(quarter)"),
        ]:
            fit = smf.ols(f"exret_0_3 ~ {rhs}", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["firm"]})
            rows.append({
                "measure": MEASURES[col], "spec": name, "n": int(fit.nobs),
                "coef_per_sd": float(fit.params["x"]), "se": float(fit.bse["x"]),
                "t": float(fit.tvalues["x"]), "p": float(fit.pvalues["x"]),
                "r2": float(fit.rsquared),
            })
    return pd.DataFrame(rows)


def power_arithmetic(panel: pd.DataFrame, table6_df: pd.DataFrame) -> pd.DataFrame:
    """Minimum detectable effect for the return test, stated before interpreting.

    At 80% power and a 5% two-sided test the detectable coefficient is
    (1.96 + 0.84) = 2.80 standard errors. Comparing that to the standard error
    the regression actually delivers says what this test could ever have found.
    """
    y = panel["exret_0_3"].dropna()
    rows = []
    for _, r in table6_df[table6_df["spec"] == "(3) + controls + FE"].iterrows():
        rows.append({
            "measure": r["measure"],
            "n": r["n"],
            "sd_of_exret_0_3": y.std(),
            "se_of_coef": r["se"],
            "mde_80pct_power": 2.80 * r["se"],
            "mde_in_bp": 2.80 * r["se"] * 1e4,
            "observed_coef_bp": r["coef_per_sd"] * 1e4,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q2 -- filings that score high on one measure and low on the other
# ---------------------------------------------------------------------------
def divergent_filings(panel: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Filings where negative tone and uncertainty disagree most.

    Both measures are standardised within form type, so a 10-K is compared to
    other 10-Ks. The gap is z(negative) - z(uncertainty); the extremes at each
    end are the filings worth reading to see what the two lists are picking up
    when they part company.
    """
    d = panel.dropna(subset=["prop_negative", "prop_uncertainty"]).copy()
    for col in ("prop_negative", "prop_uncertainty"):
        d[f"z_{col}"] = d.groupby("form")[col].transform(lambda s: (s - s.mean()) / s.std())
    d["gap"] = d["z_prop_negative"] - d["z_prop_uncertainty"]
    cols = ["ticker", "company", "form", "filing_date", "n_words_scored",
            "prop_negative", "prop_uncertainty", "z_prop_negative",
            "z_prop_uncertainty", "gap"]
    top = d.nlargest(n, "gap")[cols].assign(direction="high negative, low uncertainty")
    bot = d.nsmallest(n, "gap")[cols].assign(direction="low negative, high uncertainty")
    return pd.concat([top, bot], ignore_index=True)
