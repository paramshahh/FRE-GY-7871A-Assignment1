"""Tables 1-3 as tidy frames, plus Markdown and LaTeX (booktabs) renderers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(x, digits: int = 3) -> str:
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}f}"
    return str(x)


def to_markdown(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(v, digits) for v in r.values) + " |")
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def to_latex(df: pd.DataFrame, caption: str, label: str, digits: int = 3,
             notes: str = "", index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [_tex_escape(str(c)) for c in d.columns]
    spec = "l" + "r" * (len(cols) - 1)
    body = [" & ".join(_tex_escape(fmt(v, digits)) for v in r.values) + r" \\"
            for _, r in d.iterrows()]
    out = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{_tex_escape(caption)}}}",
           rf"\label{{{label}}}", r"\small", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(cols) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    if notes:
        out.append(rf"\par\smallskip\footnotesize\emph{{Notes:}} {_tex_escape(notes)}")
    out.append(r"\end{table}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The three exhibits
# ---------------------------------------------------------------------------
def table1_loadings(model, stability: pd.DataFrame, n_terms: int = 6) -> pd.DataFrame:
    """Table 1A: each labelled LSA pole with its top terms and their loadings in U_k."""
    rows = []
    st = stability.set_index(["component", "sign"])["stability"] \
        if len(stability) else pd.Series(dtype=float)
    for _, p in model.poles.iterrows():
        if p["label"] in ("Other",):
            continue
        u = p["sign"] * model.U[:, p["component"]]
        idx = np.argsort(u)[::-1][:n_terms]
        axis = f"{p['component']}{'+' if p['sign'] > 0 else '-'}"
        rows.append({"Issue": p["label"], "Axis": axis,
                     "σ": model.singular_values[p["component"]],
                     "Seed share": p["seed_share"], "Purity": p["purity"],
                     "Stability": st.get((p["component"], p["sign"]), np.nan),
                     "Top terms (loading)": "; ".join(f"{model.terms[i]} ({u[i]:.2f})"
                                                      for i in idx if u[i] > 0)})
    out = pd.DataFrame(rows)
    out = out[~((out["Issue"] == "General") & (out["Axis"].str.endswith("-")))]
    return out.sort_values(["Issue", "Axis"]).set_index(["Issue", "Axis"])


def table1_issues(daily, signals: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Table 1B: salience and sentiment distribution per issue."""
    d = daily.docs
    rows = []
    for k in coverage.index:
        dk = d[d["topic"] == k]
        n = len(dk)
        rows.append({
            "Issue": k, "Docs": n,
            "Salience": daily.salience[k].mean() if k in daily.salience else np.nan,
            "S mean": signals[k].mean() if k in signals else np.nan,
            "S sd": signals[k].std() if k in signals else np.nan,
            "% Pos": 100 * (dk["polarity"] > 0).mean() if n else np.nan,
            "% Neg": 100 * (dk["polarity"] < 0).mean() if n else np.nan,
            "% Anxiety": 100 * dk["anxiety"].mean() if n else np.nan,
            "% Anger": 100 * dk["anger"].mean() if n else np.nan,
            "% Optimism": 100 * dk["optimism"].mean() if n else np.nan,
            "% Re-signed": 100 * (dk["polarity"] != dk["polarity_generic"]).mean() if n else np.nan,
            "Coverage": coverage.loc[k, "coverage"],
            "Used": "yes" if coverage.loc[k, "used"] else "no (too sparse)",
        })
    return pd.DataFrame(rows).set_index("Issue")


def table2_stage1(stage1: pd.DataFrame, reverse: pd.DataFrame | None = None,
                  robust: pd.DataFrame | None = None) -> pd.DataFrame:
    """Granger tests, topic sentiment -> dProb, with robustness and reverse direction."""
    t = stage1.copy()
    t["Contract"] = t["effect"].str.replace("dProb_", "", regex=False)
    t["Issue"] = t["cause"].str.replace("S_", "", regex=False)
    out = pd.DataFrame({"Issue": t["Issue"], "Contract": t["Contract"], "Lags": t["lags"],
                        "N": t["nobs"], "F": t["F"], "p": t["p_value"], "q (BH)": t["q_value"],
                        "Sig.": t["p_value"].map(stars), "Σγ": t["sum_gamma"],
                        "ΔR²": t["r2_gain"]})
    for name, df, col_c, col_e in (("p (3d)", robust, "cause", "effect"),
                                   ("p (rev)", reverse, "effect", "cause")):
        if df is not None and len(df):
            m = pd.DataFrame({"Issue": df[col_c].str.replace("S_", "", regex=False),
                              "Contract": df[col_e].str.replace("dProb_", "", regex=False),
                              name: df["p_value"]})
            out = out.merge(m, on=["Issue", "Contract"], how="left")
    return out.set_index(["Issue", "Contract"]).sort_index()


def table2b_news(tests: pd.DataFrame) -> pd.DataFrame:
    """GDELT news salience / tone -> dProb, in the layout of Table 2."""
    if tests is None or not len(tests):
        return pd.DataFrame({"note": ["no GDELT issue series available"]})
    kind = np.where(tests["cause"].str.startswith("news_sal_"), "salience", "tone")
    issue = tests["cause"].str.replace("news_sal_", "", regex=False) \
        .str.replace("news_tone_", "", regex=False)
    out = pd.DataFrame({"Issue": issue, "Series": kind,
                        "Contract": tests["effect"].str.replace("dProb_", "", regex=False),
                        "Lags": tests["lags"], "N": tests["nobs"], "F": tests["F"],
                        "p": tests["p_value"], "q (BH)": tests["q_value"],
                        "Sig.": tests["p_value"].map(stars), "Σγ": tests["sum_gamma"]})
    return out.set_index(["Issue", "Series", "Contract"]).sort_index()


def table3_basket(stage2_ret: pd.DataFrame, stage2_vol: pd.DataFrame,
                  rs: pd.DataFrame) -> pd.DataFrame:
    """Basket reactivity: dProb -> excess returns and volatility, plus Rigobon-Sack betas."""
    rows = []
    for asset in config.ASSETS:
        for ch in config.CONTRACTS:
            cause = f"dProb_{ch}"
            r = stage2_ret[(stage2_ret["effect"] == asset) & (stage2_ret["cause"] == cause)]
            v = stage2_vol[(stage2_vol["effect"] == f"vol_{asset}")
                           & (stage2_vol["cause"] == cause)]
            s = rs[(rs["asset"] == asset) & (rs["contract"] == ch)] if len(rs) else rs
            if r.empty and v.empty:
                continue
            rows.append({
                "Asset": asset, "Contract": ch,
                "Ret F": _first(r, "F"), "Ret p": _first(r, "p_value"),
                "Ret q": _first(r, "q_value"), "Ret Σγ": _first(r, "sum_gamma"),
                "Vol F": _first(v, "F"), "Vol p": _first(v, "p_value"),
                "Vol q": _first(v, "q_value"),
                "β OLS": _first(s, "beta_ols"), "β RS": _first(s, "beta_rs"),
                "RS 95% CI": (f"[{_first(s, 'ci_low'):.2f}, {_first(s, 'ci_high'):.2f}]"
                              if len(s) else "–"),
                "VarH/VarL": _first(s, "var_ratio_x"), "Relevance p": _first(s, "var_ratio_p"),
            })
    return pd.DataFrame(rows).set_index(["Asset", "Contract"])


def _first(d: pd.DataFrame, col: str) -> float:
    return d[col].iloc[0] if len(d) else np.nan


def write_tables(tables: dict[str, tuple[pd.DataFrame, str, str]]) -> None:
    """Write each table as .md and .tex under outputs/tables/."""
    for name, (df, caption, notes) in tables.items():
        (config.TABLE_DIR / f"{name}.md").write_text(
            f"**{caption}**\n\n{to_markdown(df)}\n\n_{notes}_\n", encoding="utf-8")
        (config.TABLE_DIR / f"{name}.tex").write_text(
            to_latex(df, caption, f"tab:{name}", notes=notes), encoding="utf-8")
