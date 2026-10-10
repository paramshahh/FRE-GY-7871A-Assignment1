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
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:.{digits}f}"


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
def table1_topics(scored: pd.DataFrame, salience: pd.DataFrame, tone: pd.DataFrame) -> pd.DataFrame:
    """Topic salience and sentiment distribution across priority issues."""
    rows = []
    for k in salience.columns:
        d = scored[scored["topic"] == k]
        n = len(d)
        rows.append({
            "Topic": k, "Docs": n,
            "Salience (mean)": salience[k].mean(),
            "Tone (mean)": tone[k].mean(), "Tone (sd)": tone[k].std(),
            "% Pos": 100 * (d["polarity"] > 0).mean() if n else np.nan,
            "% Neg": 100 * (d["polarity"] < 0).mean() if n else np.nan,
            "% Anxiety": 100 * d["anxiety"].mean() if n else np.nan,
            "% Anger": 100 * d["anger"].mean() if n else np.nan,
            "% Optimism": 100 * d["optimism"].mean() if n else np.nan,
            "% Re-signed": 100 * (d["polarity"] != d["polarity_generic"]).mean() if n else np.nan,
        })
    return pd.DataFrame(rows).set_index("Topic")


def table2_stage1(stage1: pd.DataFrame, reverse: pd.DataFrame | None = None) -> pd.DataFrame:
    """Granger tests, topic tone -> P(Dem) changes, with the reverse direction."""
    t = stage1.copy()
    t["Chamber"] = t["effect"].str.replace("dProb_", "", regex=False)
    t["Topic"] = t["cause"].str.replace("tone_", "", regex=False)
    out = t.assign(**{"F": t["F"], "p": t["p_value"], "q (BH)": t["q_value"],
                      "Sig.": t["p_value"].map(stars), "Σγ": t["sum_gamma"],
                      "ΔR²": t["r2_gain"], "Lags": t["lags"], "N": t["nobs"]})
    cols = ["Topic", "Chamber", "Lags", "N", "F", "p", "q (BH)", "Sig.", "Σγ", "ΔR²"]
    out = out[cols]
    if reverse is not None and len(reverse):
        r = reverse.assign(Chamber=reverse["cause"].str.replace("dProb_", "", regex=False),
                           Topic=reverse["effect"].str.replace("tone_", "", regex=False))
        out = out.merge(r[["Topic", "Chamber", "p_value"]].rename(
            columns={"p_value": "p (reverse)"}), on=["Topic", "Chamber"], how="left")
    return out.set_index(["Topic", "Chamber"]).sort_index()


def table3_sectors(stage2_ret: pd.DataFrame, stage2_vol: pd.DataFrame,
                   rs: pd.DataFrame) -> pd.DataFrame:
    """Sector ETF reactivity: dProb -> excess returns and range volatility, plus RS betas."""
    rows = []
    for etf in config.SECTOR_ETFS:
        for ch in config.CHAMBERS:
            cause = f"dProb_{ch}"
            r = stage2_ret[(stage2_ret["effect"] == etf) & (stage2_ret["cause"] == cause)]
            v = stage2_vol[(stage2_vol["effect"] == f"vol_{etf}") & (stage2_vol["cause"] == cause)]
            s = rs[(rs["etf"] == etf) & (rs["chamber"] == ch)] if len(rs) else rs
            if r.empty and v.empty:
                continue
            rows.append({
                "ETF": etf, "Chamber": ch,
                "Ret F": _first(r, "F"), "Ret p": _first(r, "p_value"),
                "Ret q": _first(r, "q_value"),
                "Ret Σγ": _first(r, "sum_gamma"),
                "Vol F": _first(v, "F"), "Vol p": _first(v, "p_value"),
                "Vol q": _first(v, "q_value"),
                "β OLS": _first(s, "beta_ols"), "β RS": _first(s, "beta_rs"),
                "RS 95% CI": (f"[{_first(s, 'ci_low'):.2f}, {_first(s, 'ci_high'):.2f}]"
                              if len(s) else "–"),
                "VarH/VarL": _first(s, "var_ratio_x"),
            })
    return pd.DataFrame(rows).set_index(["ETF", "Chamber"])


def _first(d: pd.DataFrame, col: str) -> float:
    return d[col].iloc[0] if len(d) else np.nan


def write_tables(tables: dict[str, tuple[pd.DataFrame, str, str]]) -> None:
    """Write each table as .md and .tex under outputs/tables/."""
    for name, (df, caption, notes) in tables.items():
        (config.TABLE_DIR / f"{name}.md").write_text(
            f"**{caption}**\n\n{to_markdown(df)}\n\n_{notes}_\n", encoding="utf-8")
        (config.TABLE_DIR / f"{name}.tex").write_text(
            to_latex(df, caption, f"tab:{name}", notes=notes), encoding="utf-8")
