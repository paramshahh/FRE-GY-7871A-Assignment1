# Assignment 1 report

### Uncertainty and Sentiment Analysis of Quarterly and Annual Financial Reports

FRE-GY 7871 A · NLP and the Investment Process

**Name:** Param Shah
**NetID:** pks9550
**GitHub repo:** https://github.com/paramshahh/FRE-GY-7871A-Assignment1

---

## 1. What I did

I scored every 10-K and 10-Q filed between 2021 and 2025 by the SEC-registered operating companies
held in the six ARK ETFs, using the two Loughran-McDonald word lists the assignment specifies:
Fin-Neg, 2,355 words, and Fin-Unc, 297. Each filing gets four scores — each list weighted two ways,
once as a simple proportion of the document's words and once by the tf.idf scheme of equation (1).
I then asked three things of those scores: whether either measure trended over the five years,
whether uncertainty language in a filing predicts the volatility that follows it, and whether
negative language predicts the return over the four trading days beginning with the first day the
filing was actually tradable. The estimation sample is 1,542 filings from 91 firms across 20
quarters. The short answer is that the trend depends entirely on which form type you look at, the
volatility result is purely cross-sectional and disappears within firms, and the return test is too
underpowered to have resolved an effect of the size it found.

## 2. Data construction

The universe is the frozen snapshot of the six ARK funds' published holdings that ships with the
repository, so the sample is the class sample rather than a live pull. Table 1 carries the count
from that snapshot through to the estimation sample without a gap.

**Table 1 — sample waterfall**

| Step | Removed | Remaining |
|---|---:|---:|
| ARK holdings positions (6 funds, frozen snapshot) | — | 226 |
| Unique US-listed tickers after cleaning | 102 | 124 |
| Drop: no CIK on file (foreign or private filer) | 7 | 117 |
| Drop: CIK found, no 10-K or 10-Q in 2021–2025 | 24 | 93 |
| Candidate 10-K / 10-Q filings, 2021–2025 | — | 1,748 |
| 1. Drop amendments (10-K/A, 10-Q/A) and parse failures | 46 | 1,702 |
| 1b. Drop duplicate accessions (same CIK, two share classes) | 20 | 1,682 |
| 2. Drop filings under the word-count floor | 0 | 1,682 |
| 3. Keep one filing per company per calendar quarter | 14 | 1,668 |
| 4. Require a usable day 0 and a day −1 price of at least $3 | 103 | 1,565 |
| 5. Require 60 trading days of returns before and after day 0 | 23 | **1,542** |

The 93 tickers resolve to 92 distinct CIKs, and the final sample contains 91 firms — 384 10-Ks and
1,158 10-Qs, spanning 2021Q1 to 2025Q4.

Two of those rows are not in the brief's numbered list, and both are there because a filter that
gets applied without being reported is worse than one that was never applied. The first is
amendments: script `02` calls `list_filings(include_amendments=False)`, so 10-K/A and 10-Q/A filings
were excluded before anything was downloaded and would otherwise never have appeared in a waterfall
at all. Re-listing the same CIKs with amendments included counts 46 of them. The second is duplicate
accessions. Script `02` iterates over tickers, and ARK holds Alphabet as both GOOG and GOOGL, which
share one CIK, so Alphabet's twenty filings were downloaded twice under two labels. These are the
same document, not two filings; I kept the Class A line and dropped the other twenty rows. Left
alone, filter 3 would have absorbed them silently and the waterfall would have been quietly wrong
about what filter 3 does.

Two of the brief's own filters are worth a sentence. Filter 2 removes nothing: the shortest 10-Q in
the corpus runs to 4,181 words and the shortest 10-K to 17,774, both far above the 1,000 and 2,000
floors. The filter is real, it simply does not bind on a universe of this size. Filter 4 is the
expensive one at 103 filings — 28 lacked a tradable day 0 or a day −1 close, and the other 75 were
priced below $3, which is what one should expect from a fund family whose 2021 vintage was repriced
hard. Filter 5 costs 23 filings, every one of them short on the *pre*-filing side because the firm
had recently listed; none are lost for want of post-filing data, so the price file's 2026-03-30
endpoint does not bind. One firm, Kodiak AI, drops out entirely: all eleven of its filings fail the
60-day pre-period test against a 128-day price history.

**Parsing.** I used `src/parse.py` unchanged, which strips inline-XBRL scaffolding and drops any
table whose non-space characters are more than 15% digits. No filing failed to parse and none scored
zero words, so the parse-failure count in Table 1 is zero. Tokenisation cost something worth naming:
`tokenize` emits uppercase alphabetic tokens and preserves hyphens and apostrophes, but neither
Loughran-McDonald list contains a single hyphenated or apostrophe form. So `THIRD-PARTY`,
`LONG-TERM` and `COMPANY'S` — 1.23% of all tokens in the corpus — can never match a lexicon word no
matter what they mean, and enter the measures only by inflating the denominator of every
proportional score. The effect is small and uniform across filings, but it is a downward bias in the
level of every proportional measure, and it is a property of the tokeniser rather than of the
filings.

**Point-in-time.** Day 0 is the first trading day on or after the later of the EDGAR filing date and
the acceptance date, where `acceptance_datetime` is converted from UTC to America/New_York and
pushed forward one calendar day when it lands at or after 16:00 Eastern. This moves day 0 for **862
of 1,542 filings, 55.9% of the sample**. The reason it moves so many is visible in the acceptance
timestamps: **1,155 filings, 74.9%, are accepted at or after 16:00 Eastern**, and 721 of them land in
the 16:00 hour alone. Companies file into the close. Treating the EDGAR filing date as tradable
would misdate the event for more than half the sample, and in a consistent direction. Of the moved
filings, 700 shift by one calendar day and 141 by three, the latter being Friday-after-the-close
filings that first become tradable on Monday. The trading calendar itself is built from the days
actually present in the downloaded SPY series rather than from an external calendar package.

**Share counts** come from `data/prices/shares.csv`, joined to each filing on its accession number,
so the count used is the one printed on the document being scored. 1,497 of 1,542 filings (97.1%)
carry one; the 45 that do not are left missing rather than filled forward from a later filing, and
they drop out of any specification using log size. Of those matched, 1,014 come from the cover-page
tag `dei:EntityCommonStockSharesOutstanding`, 459 from
`us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding`, and 24 from two further fallbacks. The
weighted-average-diluted figure is not strictly a point-in-time share count — it is a period average
— so for those 459 filings market cap is measured with some error. It is still a number reported *on
the filing being scored*, which is the property that matters for avoiding look-ahead.

Three further choices the brief leaves open. The $3 screen is applied to split- and
dividend-adjusted closes, because `prices.csv` is downloaded with `auto_adjust=True` and raw closes
are not in the dataset; for a firm that split during the window, this is not the price that actually
traded. "Calendar quarter" in filter 3 means the quarter of the filing date, not the report date,
which is consistent with breaking ties on the earliest filing date and with Figure 1 being about
when filings land. And a window [a, b] is measured from the close on day a − 1 throughout — the
brief specifies this for the [0, +3] return and I apply the same convention to the [−60, −6] control
so the two are comparable.

## 3. Word lists

Fin-Neg contains 2,355 words and Fin-Unc 297. **Forty words sit on both lists** — 13.5% of Fin-Unc
and 1.7% of Fin-Neg — and they are words like ANOMALY, DEVIATION, CONFUSION and DESTABILIZING that
carry both a negative valence and a hedging function. In the corpus, 1,565 of the Negative words and
243 of the Uncertainty words appear at least once.

A 13.5% overlap in the *lists* puts only a weak floor under the correlation of the *measures*, and
it turns out to be nowhere near the binding constraint. What actually drives the two measures
together is not the shared vocabulary but the fact that both lists are dominated by the language of
a risk-factor section, which §5 takes up.

## 4. Method

The proportional measure is the count of list words in a filing over its total word count. The
tf.idf measure is equation (1), with natural logs throughout:

> tfidf(w, j) = [ (1 + ln tf(w, j)) / (1 + ln a_j) ] × ln( N / df(w) )

summed over every list word present in the document. Words with tf = 0 are skipped, so ln 0 never
arises. The two terms the brief flags as ambiguous I read as follows. **a_j is the average word count
within document j** — its total words over its distinct words — a per-document quantity computed
from that filing alone, not a corpus-wide average. **N and df(w) are computed on the corpus being
scored**, which I take to be the final 1,542-filing estimation sample: the same documents the
regressions run on. Deriving document frequencies from the pre-filter corpus and then regressing on
the post-filter one would be exactly the mismatch the brief penalises, so the code computes both
from the filtered sample and they cannot drift apart. Verified against the brief's three-document
example, my implementation returns 0.8480, 0.2885 and 0.5026.

Aggregation to quarters is done separately by form type throughout, for reasons §6 sets out. For the
volatility and return tests the tone measure is standardised, so a coefficient reads per one
standard deviation of the measure and the proportional and tf.idf specifications sit on comparable
scales.

Standard errors are clustered on the firm. The panel is wide in firms and short in quarters — 91
firms, 20 quarters — and the residuals of repeated filings by the same company are obviously
correlated, both because firms have persistent writing styles and because their volatility is
persistent. Clustering on the firm addresses the dimension where the dependence actually is.

The aggregate trend test needs Newey-West for a different reason. Regressing twenty quarterly
observations of a highly persistent series on a linear time trend will produce a confident-looking
OLS t-statistic whether or not a trend exists, because the OLS standard error assumes independent
residuals and the residuals of a persistent series are nothing of the kind. Newey-West with a lag
truncation of 2 (the 4(T/100)^(2/9) rule at T = 20) corrects for that autocorrelation. Reporting
only the OLS t-statistic on a 20-observation quarterly series would be close to meaningless.

## 5. What the measures are made of

**Table 2 — summary statistics**

| Form | Measure | n | Mean | SD | Median | CV |
|---|---|---:|---:|---:|---:|---:|
| 10-K | Negative, proportional | 384 | 2.547% | 0.451% | 2.576% | 0.177 |
| 10-K | Negative, tf.idf | 384 | 167.19 | 61.29 | 152.98 | 0.367 |
| 10-K | Uncertainty, proportional | 384 | 2.113% | 0.263% | 2.136% | 0.124 |
| 10-K | Uncertainty, tf.idf | 384 | 27.55 | 9.71 | 26.03 | 0.353 |
| 10-Q | Negative, proportional | 1,158 | 2.172% | 1.078% | 1.773% | 0.496 |
| 10-Q | Negative, tf.idf | 1,158 | 75.86 | 72.90 | 34.42 | 0.961 |
| 10-Q | Uncertainty, proportional | 1,158 | 1.933% | 0.661% | 1.762% | 0.342 |
| 10-Q | Uncertainty, tf.idf | 1,158 | 12.70 | 8.67 | 9.90 | 0.683 |

**Table 3 — ten most frequent words on each list** (full thirty in `outputs/table3_top_words.csv`)

| Rank | Negative | Count | Share | Uncertainty | Count | Share |
|---:|---|---:|---:|---|---:|---:|
| 1 | LOSS | 57,793 | 4.75% | MAY | 373,348 | **37.37%** |
| 2 | ADVERSELY | 57,066 | 4.69% | COULD | 206,338 | **20.66%** |
| 3 | ADVERSE | 35,207 | 2.89% | RISK | 50,347 | 5.04% |
| 4 | CLAIMS | 32,908 | 2.71% | RISKS | 49,849 | 4.99% |
| 5 | HARM | 26,192 | 2.15% | BELIEVE | 31,369 | 3.14% |
| 6 | AGAINST | 26,019 | 2.14% | APPROXIMATELY | 18,161 | 1.82% |
| 7 | UNABLE | 25,990 | 2.14% | ASSUMPTIONS | 13,140 | 1.32% |
| 8 | FAILURE | 24,699 | 2.03% | FLUCTUATIONS | 12,656 | 1.27% |
| 9 | LOSSES | 23,712 | 1.95% | MIGHT | 12,473 | 1.25% |
| 10 | LITIGATION | 22,980 | 1.89% | UNCERTAINTIES | 12,313 | 1.23% |

**Q1. What is each measure actually made of?**

The two lists are not comparably concentrated, and it is not close. The Negative list spreads its
1,216,382 occurrences across 1,565 distinct words, with the top ten accounting for **27.3%** and the
top thirty for **46.1%**. The Uncertainty list puts 998,968 occurrences through only 243 distinct
words, and the top ten account for **78.1%**, the top thirty for **92.4%**. The uncertainty measure
is not a measure over a 297-word vocabulary in any meaningful sense. It is very nearly a count of two
words: **MAY and COULD together are 58.0% of every uncertainty word in the corpus**, and adding RISK
and RISKS takes it to 68.1%.

That settles the second half of the question. MAY, COULD, RISK and RISKS are not management
expressing doubt about a particular outcome; they are the grammatical skeleton of a risk-factor
section — "we may be unable to", "could adversely affect our business" — drafted by counsel and
carried forward from year to year with minimal edits. The Negative list is only marginally better
off: its own top entries are ADVERSELY, ADVERSE, CLAIMS, HARM and LITIGATION, which is the same
template seen from the other side. What the two measures mostly share is that both are counting how
much risk-factor boilerplate a document contains.

There is a mechanical consequence that matters for everything downstream and that I did not expect
before computing it. The idf term is ln(N/df), so a word appearing in *every* filing has df = N and
an idf of **exactly zero**. In this corpus MAY, COULD, RISK, RISKS — and, on the Negative list, LOSS
— each appear in all 1,542 filings. Their tf.idf contribution is not small; it is nil. The
proportional and tf.idf versions of the Uncertainty measure are therefore built from nearly disjoint
vocabulary: the proportional measure is 58% two words that the tf.idf measure ignores entirely. That
is not a subtlety of weighting, it is two different variables wearing the same name, and it explains
why the two correlate at only 0.63 and why §6 finds their trends running in opposite directions.

**Q2. Are sentiment and uncertainty measuring different things?**

Largely not. Across the pooled sample the two **proportional** measures correlate at **0.887**
(Spearman 0.894), and the two **tf.idf** measures at **0.934** (Spearman 0.937). Within form type
the proportional correlation is 0.768 for 10-Ks and 0.892 for 10-Qs. By contrast the two weightings
of the *same* list correlate at 0.794 for Negative and only **0.629** for Uncertainty — so the
choice of lexicon matters less to what you end up measuring than the choice of weighting does, which
is an odd thing to be able to say about two lists meant to capture different constructs.

The filings that pull apart show what the lists catch when they disagree. DraftKings' 10-Q of 8
November 2024 scores 5.04% negative (z = +2.66 among 10-Qs) against 1.64% uncertainty (z = −0.44).
Its negative words are DAMAGES (57), COMPLAINT (42), PLAINTIFF (37), DISMISS (36) and CLAIMS (29) —
a legal-proceedings note describing litigation that has already happened. Concrete, realised bad
news, with nothing hedged about it. Illumina's 10-K of 18 February 2022 is the mirror image: 2.04%
negative (z = −1.13) against 2.38% uncertainty (z = +1.03), driven by MAY (193), COULD (125), RISK
(41), RISKS (36) — and by CONTINGENT (34) and INTANGIBLE (28), which are accounting nouns for
contingent consideration and intangible assets rather than expressions of doubt about anything. So
the uncertainty list also misfires on ordinary balance-sheet vocabulary.

Honestly, then: with the two measures correlating at 0.89 and 0.93, most of what follows is one
result reported twice, not two independent findings. Where Tables 4 to 6 show the negative and
uncertainty measures agreeing, that agreement carries almost no additional information. The places
worth attending to are the ones where they *disagree*, and those turn out to be driven by the
weighting scheme rather than by the lexicon.

## 6. Trends, 2021–2025

**Composition corrections, stated before describing anything.** Three were necessary and the first
two change the answer. Form mix: 10-Ks are twice as long as 10-Qs and carry markedly more risk
language (2.55% vs 2.17% negative), and they cluster in the first calendar quarter — 2021Q1 holds 57
10-Ks against 8 10-Qs. Pooling the forms therefore manufactures a calendar sawtooth that is an
artefact of the filing calendar and nothing else, so Figure 1 draws the forms separately.
Observation count per point: even after separating, a *quarterly* 10-K mean outside Q1 is the
average of one or two filings and swings violently for no substantive reason, so the 10-K series is
plotted annually (~77 filings a point) while the 10-Q series stays quarterly (60–80 a point). My
first version of Figure 1 plotted both quarterly and showed a dramatic zigzag in the 10-K line that
was pure sample-size noise; Figure 1b plots filings per quarter by form so the reader can see the
composition being corrected for rather than take my word for it. Firm mix: firms differ in baseline
tone and the set of filers changes, which the within-firm specification in Table 4 handles.

**Table 4 — trend tests** (pp/year for proportional measures)

| Sample | Measure | Aggregate slope/yr | t (OLS) | t (NW) | Within-firm slope/yr | t (clustered) |
|---|---|---:|---:|---:|---:|---:|
| All | Negative, proportional | −0.011 pp | −0.37 | −0.42 | −0.010 pp | −0.65 |
| All | Uncertainty, proportional | −0.009 pp | −0.66 | −0.88 | −0.011 pp | −1.30 |
| All | Uncertainty, tf.idf | −0.91 | −0.98 | −1.56 | −0.53 | **−4.46** |
| 10-K | Negative, proportional | +0.058 pp | 2.14 | **3.84** | +0.081 pp | **8.27** |
| 10-K | Uncertainty, proportional | +0.029 pp | 1.96 | **2.72** | +0.037 pp | **5.81** |
| 10-K | Uncertainty, tf.idf | −0.87 | −1.21 | −1.87 | −0.35 | −2.38 |
| 10-Q | Negative, proportional | −0.032 pp | −2.27 | −2.16 | −0.036 pp | −1.88 |
| 10-Q | Uncertainty, proportional | −0.026 pp | −3.02 | −3.24 | −0.026 pp | −2.40 |
| 10-Q | Uncertainty, tf.idf | −0.32 | −1.10 | −1.50 | −0.57 | **−4.07** |

**Q3. Did either measure trend over 2021–2025?**

On the pooled sample, no — every aggregate t-statistic is below 1.6 in absolute value and the
within-firm tests are null too. But that pooled null is an artefact of averaging, and it is the
clearest illustration in this report of why the composition correction is not a formality. **The
pooled series is flat because a rising 10-K series and a falling 10-Q series cancel.**

Split by form, 10-K negative tone rises by **+0.081 percentage points a year** within firm, with a
clustered t of **8.27**; the annual 10-K mean climbs from 2.30% in 2021 to 2.69% in 2025, a 17%
relative increase over five years. 10-K uncertainty rises by +0.037 pp/year, t = 5.81. The aggregate
versions survive Newey-West at 3.84 and 2.72. 10-Qs move the other way, more weakly: negative tone
falls 0.036 pp/year (within-firm t = −1.88, marginal) and uncertainty falls 0.026 pp/year (t =
−2.40).

The specification I trust is the **within-firm** one, for both of the reasons that make it the right
question. It asks whether the same company changed its own language, which is what "the filings got
more negative" ought to mean, and it is immune to the changing composition of ARK's holdings — a
fund that rotates into distressed names would push up a pooled mean without any firm writing
differently. It is also estimated on 384 and 1,158 observations rather than 20, so it has real
power. The aggregate test is reported because the brief asks for it and because agreement between
the two is reassuring, but with 20 quarterly observations of a persistent series I would not rest
anything on the aggregate alone: note that for 10-K negative tone the OLS t of 2.14 would be
reported as significant while Newey-West, correcting for exactly the autocorrelation one should
expect, gives 3.84 — in this instance stronger, but the point is that the OLS figure was not
informative either way.

Which reading does the evidence support? The deflationary one. A steady, monotone, five-year rise in
10-K risk language that shows up in the proportional measure but **not** in tf.idf — 10-K
uncertainty tf.idf actually *falls*, t = −2.38 — is the signature of accumulating boilerplate rather
than of accumulating risk. Recall that tf.idf assigns zero weight to words in every filing and
positive weight to rarer ones. A firm that each year appends a new risk factor to an existing
section, in the same MAY/COULD/ADVERSELY register, raises its proportional score while leaving its
tf.idf score flat or lower. That is precisely the pattern observed. Risk-factor sections ratchet:
language is added after each new exposure and almost never removed. I do not think these filings are
telling us that 2025 was more dangerous than 2021; I think they are telling us that the 10-K
risk-factor section is 17% longer in the words that already dominated it.

Against VIX there is no visible relationship in Figure 1 in either direction. VIX falls from the low
20s in 2021–22 to the mid-teens by 2024 while 10-K tone rises monotonically through the same period.
Whatever the filings are tracking, it is not the market's contemporaneous assessment of risk.

## 7. Uncertainty, volatility and returns

**Table 5 — post-filing realised volatility [+4, +63] on the uncertainty measure**, standardised,
coefficients in annualised volatility points per SD, firm-clustered standard errors, n = 1,497

| Measure | Specification | Coef | SE | t |
|---|---|---:|---:|---:|
| Uncertainty, proportional | (1) no FE, without pre-filing vol | **+0.0493** | 0.0123 | **3.99** |
| Uncertainty, proportional | (2) no FE, with pre-filing vol | **+0.0388** | 0.0106 | **3.66** |
| Uncertainty, proportional | (3) firm+quarter FE, without pre-filing vol | −0.0088 | 0.0070 | −1.26 |
| Uncertainty, proportional | (4) firm+quarter FE, with pre-filing vol | −0.0091 | 0.0068 | −1.34 |
| Uncertainty, tf.idf | (1) no FE, without pre-filing vol | **+0.0684** | 0.0160 | **4.28** |
| Uncertainty, tf.idf | (2) no FE, with pre-filing vol | **+0.0532** | 0.0140 | **3.79** |
| Uncertainty, tf.idf | (3) firm+quarter FE, without pre-filing vol | +0.0002 | 0.0097 | 0.02 |
| Uncertainty, tf.idf | (4) firm+quarter FE, with pre-filing vol | −0.0004 | 0.0092 | −0.04 |

**Q4. Does uncertainty language predict volatility?**

Taken at face value, specification (1) says yes and says it loudly: a one-standard-deviation
increase in uncertainty language is followed by 4.9 annualised volatility points more realised
volatility over the following quarter, t = 3.99. Adding the pre-filing volatility control knocks
that down to 3.9 points, a **21% reduction**, and the coefficient stays significant at t = 3.66. On
the brief's framing, the gap between those two is the answer, and a fifth of the raw effect is
simply the fact that firms which were volatile before the filing are volatile after it, and also
write more hedged filings.

But the more informative comparison is the one the fixed effects make, and it is more damaging. Once
firm and quarter fixed effects are included, the coefficient collapses from +0.049 to **−0.009 and
loses all significance** (t = −1.26); for the tf.idf measure it goes to +0.0002, t = 0.02. The
entire relationship is cross-sectional. It says that firms which habitually write hedged filings are
firms which are habitually volatile — a fact about which companies these are, not about what any
particular filing disclosed. Within a given firm, a filing with more uncertainty language than that
firm's own norm is **not** followed by more volatility than that firm's own norm.

This is why the pre-filing-volatility control looks weak here when the brief anticipates it doing
more work: in specifications (3) and (4) the firm fixed effect has already absorbed the same
between-firm channel that pre_vol was there to absorb, so adding it moves the coefficient by 0.0003.
The control and the fixed effect are substitutes, and the fixed effect is the more complete
instrument. Reading specification (2) alone — significant, and apparently robust to the control the
brief names — would have led me to the wrong conclusion. The answer to Q4 is that uncertainty
language does not predict volatility; it identifies firms.

**Table 6 — [0, +3] excess return on tone**, standardised, firm-clustered, n = 1,497

| Measure | (1) measure only | (2) + controls | (3) + controls + FE |
|---|---:|---:|---:|
| Negative, proportional | −0.47% (t −1.58) | −0.51% (t −1.80) | −1.16% (t −1.97) |
| Negative, tf.idf | −0.55% (t −1.83) | −0.71% (t −2.01) | −1.17% (t −1.45) |
| Uncertainty, proportional | −0.60% (t −2.02) | −0.65% (t −2.35) | −1.12% (t −1.68) |
| Uncertainty, tf.idf | −0.57% (t −2.06) | −0.84% (t −2.25) | −1.50% (t −2.05) |

**The power arithmetic, before interpreting.** The dependent variable has a standard deviation of
11.46% across 1,497 filings. In the fully specified regression the standard error on a standardised
tone coefficient runs from 0.59% to 0.80%. At 80% power and a 5% two-sided test, the minimum
detectable effect is 2.80 standard errors, which is **165 to 224 basis points** depending on the
measure. The observed coefficients are **112 to 149 basis points**. In other words the effects this
design estimates are *smaller than the smallest effect it was capable of reliably detecting* — every
one of them sits below its own MDE. A four-day excess return is mostly noise, and 1,497 observations
of an 11.5% standard deviation is not enough to pull a sub-1.5% signal out of it with confidence.

So the marginal significance scattered through Table 6 should not be read as a finding. The signs
are uniformly negative, which is the direction theory predicts — more negative language, lower
filing-period return — and that consistency across four measures and three specifications is mildly
suggestive. But individual t-statistics oscillating around −2 as controls are added and removed, in
a design whose MDE exceeds its own point estimates, is what an underpowered test looks like whether
or not anything is there. A null result here is the correct answer, and it is what the brief said to
expect.

## 8. 10-K versus 10-Q

**Q5. Do 10-Qs behave like 10-Ks?**

No, and the differences run in both directions.

On trends they are opposed, as §6 sets out: 10-K tone rises (+0.081 pp/year within firm, t = 8.27)
while 10-Q tone falls (−0.036 pp/year, t = −1.88), and pooling them produces a spurious null. On the
volatility test they behave alike, which is itself informative — both show the same pattern of a
strong cross-sectional coefficient (10-K: +0.069, t = 4.85; 10-Q: +0.052, t = 3.86) collapsing to
nothing under fixed effects (10-K: −0.011, t = −0.43; 10-Q: +0.013, t = 0.93). The finding that the
relationship is purely cross-sectional is not an artefact of one form type. Note that on the form
splits I use year rather than quarter fixed effects: firms file one 10-K a year, so on the 10-K
subsample firm and quarter dummies are nearly collinear and the design matrix comes back
rank-deficient, which makes the standard errors meaningless. That is a real constraint on what can
be identified from 384 annual filings, not a modelling preference.

**Variance of a proportional measure.** The prediction is that shorter, more templated documents
should produce a *more* stable proportional score, since the denominator is dominated by fixed
boilerplate. The data say the opposite, emphatically. The coefficient of variation of proportional
negative tone is **0.177 for 10-Ks and 0.496 for 10-Qs** — nearly three times as dispersed — and for
tf.idf it is 0.367 against 0.961. The reason is that 10-Qs are heterogeneous in a way 10-Ks are not.
A 10-K always contains a full Item 1A risk-factor section; a 10-Q may incorporate risk factors by
reference, restate them in full, or update only those that changed. So 10-Q negative tone ranges from
0.37% to 5.04% across the sample, and much of that range reflects a drafting convention about
whether the risk factors were reprinted rather than anything about the quarter. The 10-Q measure is
noisier, and the noise is structural rather than statistical.

**Is the 10-Q filing-date reaction even separately identified?** Essentially not. A 10-Q is filed
days after the earnings release that already disclosed the quarter's numbers, and the [0, +3] window
around the 10-Q filing will often sit inside or adjacent to the post-earnings drift. The text of the
10-Q is largely a formalisation of information the market priced at the earnings call. Without
conditioning on the earnings date — which this dataset does not contain — any 10-Q filing-date
return effect is confounded with the earnings reaction, and I would not attribute a 10-Q coefficient
in Table 6 to the filing's language. The 10-K is better identified, though not cleanly: it too
follows a Q4 earnings release, but by a longer and more variable gap, and it carries substantial
genuinely new content in the risk factors and MD&A.

**Which form should carry more textual signal?** The 10-K, on every count. It is the document with
new narrative content rather than an update, it is better separated from the earnings release, its
proportional measure is three times less noisy, and it is where the only strong trend result in this
report appears. The 10-Qs contribute three-quarters of the sample size and, I think, rather less than
three-quarters of the information.

## 9. Limitations

**Survivorship, and the sharper version.** The universe is ARK's holdings as of the frozen snapshot
dates in 2026, not as of each filing date. Every company ARK bought and sold between 2021 and 2025
is missing, and companies get sold out of a fund like ARKK disproportionately after they fall.
Something of the scale is visible in the universe itself: of 124 current US-listed holdings only 93
filed a 10-K or 10-Q across the whole window, and 24 of the rest had no 10-X in the period at all
because they listed partway through — so even the surviving universe is tilted toward recent
entrants. The sharper version matters specifically for §6: the firms whose 2021–2025 language I can
observe are the firms still worth holding in 2026. If deteriorating firms write increasingly
negative filings and are then sold, the trend I measure is computed on exactly the sample that
excludes them, and the true cross-sectional trend in negative tone is more positive than +0.081
pp/year, not less. The direction of the bias is knowable even though its size is not.

**Twenty quarters is a short series.** The aggregate trend tests rest on 20 observations, which is
why I lead with the within-firm specification throughout. Five years is also a single macro episode
— a post-COVID recovery, an inflation shock, a rate-hiking cycle and an AI boom — so a "trend" here
is not separable from a regime.

**Benchmark.** Excess returns are measured against SPY. For a set of high-beta, long-duration growth
names, SPY is a poor risk model: much of what I am calling excess return is beta exposure that a
market-model or factor-adjusted return would strip out. ARKK is available in the dataset as a
thematic-peer benchmark and would be the natural robustness check; I did not run it.

**Everything I ran, not only what is reported.** Four measures × (aggregate + within-firm) trend
tests on three samples is 24 trend tests; Table 5 is 4 specifications × 2 measures × 3 samples, or
24 regressions; Table 6 is 3 specifications × 4 measures, or 12. That is roughly 60 coefficients,
and at a 5% threshold three would be expected to reach significance by chance. I have not applied a
multiple-testing correction. This is the main reason I treat the scattered t ≈ −2 values in Table 6
as noise rather than findings, and the reason I weight the 10-K trend result (t = 8.27, and monotone
in the raw annual means) far more heavily than anything sitting near the threshold. Two further
choices were made and not varied: the 15% numeric-table threshold in `parse.py`, and the decision to
score the filtered rather than the full corpus.

**Q6. Which of my results do I believe?**

Going test by test, since they do not have the same power and do not fail in the same direction.

*The 10-K trend (Table 4) — I believe it as a description, not as an interpretation.* A within-firm
t of 8.27 on 384 observations, monotone in the raw annual means from 2.30% to 2.69%, is not a
fragile result, and it survives the composition corrections that killed the pooled version. What I
do not believe is that it measures rising risk. The proportional and tf.idf measures point in
opposite directions, which is exactly what accumulating boilerplate looks like and not what rising
risk looks like. So: high confidence in the fact, low confidence in the story.

*The volatility result (Table 5) — I believe the null, more than I believed the positive.* The
collapse from +0.049 (t = 3.99) to −0.009 (t = −1.26) under fixed effects is a large, clean movement
in the direction that a spurious cross-sectional correlation should move. The failure mode I worry
about is the opposite one: that firm fixed effects on 20 quarters absorb real within-firm signal
along with the noise. But the coefficient does not merely shrink, it crosses zero, which is hard to
explain as over-controlling.

*The return test (Table 6) — I believe nothing either way, and that is the honest answer.* Every
point estimate sits below its own minimum detectable effect. The uniformly negative signs are weak
corroboration of the expected direction, but I cannot distinguish them from zero and would not trade
on them. Critically, this is a failure of power, not evidence of absence: the test is equally
consistent with a real 1% effect and with nothing at all.

*The concentration results (Tables 2 and 3) — I believe these most of all*, and they are the least
statistical things in the report. That MAY and COULD are 58% of the uncertainty count, and that
they carry an idf of exactly zero, are arithmetic facts about the corpus with no standard errors
attached. They are also the results that most change how I read everything else.

## 10. What I would do next

The single change with the largest return is a point-in-time universe. ARK publishes its holdings
daily and the historical files are available, so reconstructing what each fund held on each filing
date — rather than what it holds now — would remove the survivorship bias that §9 argues is biasing
the central trend result in a known direction. It is perhaps a day of work: download and parse five
years of daily holdings files, then match each filing to the funds holding that name on that date.
Second, and cheaper, I would add the earnings announcement date from EDGAR 8-K filings, which would
let the 10-Q filing-date reaction be separated from the post-earnings drift and would make Table 6's
10-Q half interpretable rather than merely underpowered.
