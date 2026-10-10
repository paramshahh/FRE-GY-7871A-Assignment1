# Assignment 5: Brainstorming and design evaluation

Applied topic modelling and sentiment analysis around the 2026 US midterms (election day
3 Nov 2026). Sample: 2 Mar – 9 Oct 2026, 155 NYSE trading days. Everything below was
checked against the live APIs while building the data layer, not assumed. Where an
assumption failed, the failure and the fix are recorded.

---

## 1. Topic engine: LSA (truncated SVD) vs LDA vs BERTopic

| | LSA / truncated SVD | LDA | BERTopic |
|---|---|---|---|
| Model | Low-rank approximation X ≈ U_k S_k V_kᵀ of the TF-IDF matrix | Bayesian mixture: documents are distributions over topics, topics over words | Sentence embeddings → UMAP → HDBSCAN clusters → c-TF-IDF labels |
| Output | Signed, orthogonal axes; each axis has two poles | Non-negative topic shares that sum to 1 | Hard cluster per document (+ outliers) |
| Deterministic | Yes (up to sign; randomised solver is seeded) | No (sampling / variational init) | No (UMAP, HDBSCAN) |
| Short texts (tweets, headlines) | Works: bigrams and IDF carry the signal | Weak: too few words per document to estimate a mixture | Strong semantically, but needs a transformer model (~400MB) and a GPU to be quick |
| Linear algebra we can reason about | Yes: projection, energy shares, the same SVD used later on the covariance matrix | No | No |
| Cost | Seconds on ~6k docs | Minutes | Minutes plus model download |

**Choice: LSA as the primary engine**, LDA and NMF as comparison baselines
(coherence and topic diversity are reported). LSA is linear, so a day's topic exposure is
simply the projection of that day's text onto the term loadings. The same tool (an SVD)
is then reused on the sentiment-return cross-covariance, so the whole pipeline is one
piece of mathematics. BERTopic stays an optional extra (`pip install bertopic`).

### 1.1 Why LSA does not demean the term-document matrix

PCA centres the data matrix; LSA does not. There are two reasons:

1. **Sparsity.** The TF-IDF matrix here is more than 99% zeros. Subtracting column means
   makes every entry non-zero, so a 6,000 × 20,000 matrix with ~100k non-zeros becomes
   120M dense floats. That breaks `scipy.sparse.linalg.svds` / randomised SVD on a laptop.
2. **The right prior for unobserved terms is zero.** A zero in TF-IDF means "the term did
   not occur", which carries no evidence. After centring, a term a document never uses
   gets a negative value: "this document is *less* about mortgages than average".
   That makes absence look like evidence, and it is noisier the shorter the document.
   Tweets are short, so this matters.

**Consequence that has to be handled:** without centring, the first singular vector is
roughly the mean-document direction. For a non-negative matrix it is one-signed
(Perron–Frobenius), and it measures *how much* political text there is, not *which*
issue it is about. The code labels component 0 `General` **only after checking**:
≥90% of its loading energy must sit on one sign, and no single issue may dominate it.
A unit test showed why the check is needed. When `max_df` strips the shared campaign
vocabulary, the matrix becomes almost block-diagonal and the first axis can be a real
issue (in the test it was healthcare affordability). Forcing it to `General` would have
silently deleted that issue.

### 1.2 Poles, labels and weights

SVD signs are arbitrary and every axis has two ends. Each axis is oriented so that its
largest-|loading| term is positive. Each **pole** is then labelled by its *energy share*:
the fraction of Σ max(±u, 0)² falling on terms that contain an issue's seed words. A
label needs ≥15% share and ≥60% purity (the best issue's share of all seed energy on the
pole); otherwise the pole is `Other`. A first version counted seed words among the top-25
terms. It labelled mixed housing+healthcare poles as housing, so it was replaced.

A document's weight on issue k is w_dk = Σ over k's poles of max(±cos_dj, 0)², where
cos_dj is the document's projection on axis j divided by its TF-IDF norm. The square is
deliberate. With linear weights, shared campaign words gave every document a small
loading on every pole, so the tone of immigration posts leaked into housing sentiment
(the unit test moved from −0.32 to −0.95 when this was fixed).

### 1.3 Daily projection, salience and topic-conditioned sentiment

* β_t = v_t U_k, where v_t is the sum of the TF-IDF rows of the documents assigned to
  trading session t. This is the spec's "project daily documents onto the singular vectors".
* Salience_k,t = Σ_d∈t w_dk / Σ_j Σ_d∈t w_dj, i.e. the share of the day's issue mass.
* **S_k,t = (Pos_k,t − Neg_k,t) / Total_k,t ∈ [−1, 1]**, with Pos_k,t = Σ_d w_dk·1[pol_d > 0].
  The polarity pol_d comes from VADER with an **issue-specific lexicon overlay** for the
  document's dominant issue.

**The FinBERT lesson.** Off-the-shelf FinBERT reads macro growth language ("rates rising",
"strong growth") as dovish or hawkish according to finance-news conventions. A generic
model mislabels in the same way. "Rents are rising" and "coverage is rising" are
opposites, and "Medicaid cuts" is negative for voters even though "cuts" is neutral or
positive in a rates context. The overlay re-scores ~15 words per issue (`TOPIC_LEXICON`).
`% re-signed` in Table 1 reports how often the issue-conditioned polarity disagrees with
generic VADER. Emotional tone is tracked as three lexicon flags (anxiety, anger,
optimism) on top of polarity.

### 1.4 SVD of the signal-return cross-covariance

With s_t (T × K standardised topic-sentiment signals) and δy_t (T × N standardised
targets: ΔProb and excess returns),

  C = (1/T) Σ_t s_{t−1}ᵀ δy_t = U_k S_k V_kᵀ.

U_k gives K → k signal combinations ("political alpha factors") and V_k gives the target
portfolios each factor predicts. f_t = s_t U_k replaces K noisy regressors with k
factors before any regression. This is the reduced-rank / partial-least-squares idea,
and it fights the multiple-testing problem of running K × N Granger tests.

Three things the textbook formula does not say, and that the implementation enforces:

1. **Lag.** The formula pairs s_t with δy_t. Contemporaneous pairing finds co-movement,
   not prediction, so the code pairs s_{t−1} with δy_t.
2. **No look-ahead.** U, S, V and the standardisation moments are estimated on the first
   60% of the sample only. Factors are then formed for every date with those fixed weights.
3. **How many factors.** Singular values of a pure-noise K × N cross-covariance are not
   zero. Each S_i is compared with its distribution under **circular shifts** of the
   signal dates. That keeps each block's autocorrelation and breaks only their alignment.
   Factors with permutation p > 0.05 are reported but not used.

### 1.5 Choosing k, and rotating (decided on the real tweets)

On the 5,816 filtered tweets (TF-IDF 5,816 × 20,000, 0.49% dense) the spectrum has no
elbow. σ₁ = 10.7 is the General axis, σ₂ = 5.8, and from σ₃ ≈ 4.4 the values decay slowly.
That is typical of short texts, so k is chosen on interpretability and stability rather
than an energy threshold (k = 30 keeps only 8.3% of the TF-IDF energy, and that is fine:
the rest is idiosyncratic wording).

Raw SVD axes mix issues (e.g. one axis = Iran war + inflation + tariffs). A **varimax
rotation** of components 1..k−1 (component 0 untouched) is an orthogonal rotation inside
the same subspace. Fitted values and explained variance are unchanged, and each axis
concentrates on fewer terms. Results:

| k | labelled poles, raw | labelled poles, varimax | issues found |
|---|---|---|---|
| 20 | 8 / 38 | 13 / 38 | 6 |
| **30** | 12 / 58 | **17 / 58** | **7** (adds Energy/Climate) |
| 35 | – | – | 7 (no new issue) |

**Stability.** Each labelled pole is refitted on 5 random 80% subsamples. It counts as
reappearing if a pole with the same label has loading cosine ≥ 0.6. At k = 30, 16 of
17 poles reappear 5/5 and Energy/Climate 3/5.

**Technology/AI never forms an axis** at any k. Only 486 tweets contain a tech seed word,
and "AI" is not a voter-salient midterm issue in this sample. The Big Tech basket is
therefore tested against prediction-market moves and the General/Economy signals, not
against a tech-sentiment series that the data cannot support.

Two seed fixes came from inspecting the poles: "drug prices" was split between
Healthcare Affordability ("drug") and Economy ("prices"), so `drugs`, `medications` and
`seniors` were added to Healthcare Affordability. The "Affordable Care Act / health care"
pole was unlabelled because `care` was not a seed.

---

## 2. Data sources: what was evaluated and what was found

### 2.1 Social media: twitterapi.io

* Endpoint `GET https://api.twitterapi.io/twitter/tweet/advanced_search`, header
  `X-API-Key`, params `query`, `queryType` (Latest/Top) and `cursor`. It returns
  ≈20 tweets per page plus `has_next_page` and `next_cursor`. An MCP wrapper exists, but the
  REST endpoint is what the pipeline needs (scriptable, cacheable).
* **Pricing:** 15 credits per returned tweet, with a 15-credit minimum per call.
  The budget was 100k of the 115k credits on the account, i.e. ≈6,600 tweets, so the
  sample design is the main methodological decision. The account balance updates late,
  so a local **ledger** of every paid call is the authority and has a hard stop.
* **Query limit:** 512 characters including dates and operators. A 551-character query
  returned a 400 (not billed); a guard now raises before any call that would exceed it.
* **Quality filters.** Server-side (free): `lang:en -is:retweet -is:reply min_faves:5
  -min_faves:150`. Client-side: followers ≥ 10, views ≥ 100, identical normalised text
  from ≥ 5 accounts (astroturf), and fewer than 4 tokens. The server-side engagement floor
  means almost nothing is paid for and then thrown away (5 of 3,100 removed).
* **Who is in the sample.** Pilots showed "Top" ranking with a 5–1,000 like band returns
  mostly members of Congress and pundits (median 65–120k followers). Capping at 150 likes
  lowers that but does not remove it (median ≈ 40k). "Latest" was worse: the last 35
  minutes before the cutoff, more noise, median still 26k. On X, posts that earn ≥5 likes
  mostly come from accounts with an audience. **The sample is politically engaged
  mid-size accounts, not the median voter**, and the report says so.
* **Design (≈94k credits):**
  * *General stratum:* one "Top" page per trading session, for the window from the previous
    session's 16:00 ET close to this session's close (weekends land on Monday). The
    query is issue terms × political context. 3,100 tweets.
  * *Affordability stratum:* one page per session of a housing and healthcare
    affordability query, because the general stratum had < 1 tweet per day on those
    issues. It is used for tone only. **Salience comes from the general stratum only**,
    so the oversample cannot inflate it. A tweet returned by both strata is kept once,
    attributed to the general stratum.
* **Failure found and fixed:** one session returned 0 tweets with `has_next_page=false`
  (a normal Friday, so an upstream hiccup). It had been cached as complete. Empty pages are
  now never cached and are retried (≤3 attempts); the retry returned a full page.

### 2.2 News

* **GDELT DOC 2.0** (free, no key). `timelinevolraw` gives exact daily article counts
  per query plus `norm` (all articles GDELT monitored that day). `timelinetone` gives
  GDELT's average tone. One query per issue (midterm coverage that also mentions the
  issue), so news salience is an exact count, not a sample.
* GDELT throttles hard (HTTP 429 in plain text). Through the cloud proxy's shared IP it
  took minutes per request. The collector now caches each (query, mode) response as it
  lands and resumes on re-run. The first version kept everything in memory until the
  end and lost an hour of progress when restarted.
* A UTC day ends at 20:00 ET, after the close, so a GDELT day's counts are information
  for the **next** session. Days where `norm` falls below 30% of its 15-day rolling median
  are feed outages and are masked, not read as quiet news days.
* **RSS feeds** were considered and rejected for the backtest. A feed is a snapshot of the
  latest items with no history, so it cannot rebuild March–October. It suits the live
  dashboard going forward, not the econometrics.

### 2.3 Prediction markets

| Contract | Polymarket | Kalshi | PredictIt |
|---|---|---|---|
| House control | "Will the Democratic / Republican Party control the House after the 2026 Midterm elections?" (found via Gamma `/public-search`) | `CONTROLH-2026-D` / `-R` | market 8157 |
| Senate control | same, Senate | `CONTROLS-2026-D` / `-R` | market 8155 |
| **Democratic sweep** | "2026 Balance of Power: D Senate, D House" + DR, RD, RR, Other | `KXBALANCEPOWERCOMBO-27FEB-DD` + DR, RD, RR | – |

* **Timing is the critical choice.** Daily bars close at midnight ET (Kalshi) or around
  20:00 ET (Polymarket). Either way they include the hours after the 16:00 equity close,
  so a daily ΔProb would overlap two return days. Both venues provide **hourly** history
  (Kalshi 60-minute candles requested ≤100 days at a time; Polymarket CLOB
  `prices-history` with `fidelity=60` in ≤14-day windows, the API maximum). The
  probability used is **the last quote at or before 16:00 ET**, and its timestamp is kept.
  Median staleness: Kalshi 0 hours, Polymarket 1 hour, worst case 7 hours.
* Kalshi price = bid/ask midpoint when the spread is ≤ 5¢ (the price *at* the close),
  otherwise last trade.
* Each venue's outcome prices are normalised to sum to one (overround removed). The sweep
  is normalised over all four (Polymarket five) balance-of-power outcomes.
* **PredictIt** only offers daily bars (90-day CSV) that close at midnight ET, and in our
  sample sits 6.6pp above the other two venues for the House (fees and the $850 cap).
  It is kept as a cross-check series with **zero weight** in the consensus.
* Consensus = equal-weight Polymarket/Kalshi average. The two venues agree on levels
  (correlation 0.94–0.96) much more than on daily changes (0.38–0.44, partly because Kalshi
  ticks in 1¢ steps). That disagreement is a reason to average them.

### 2.4 Markets and the midterm basket

| Exposure | ETFs | Single names |
|---|---|---|
| Clean vs traditional energy | ICLN vs XLE | ENPH vs XOM |
| Healthcare / pharma | XLV | UNH |
| Financials / bank regulation | XLF | JPM |
| Defense & aerospace | XAR, ITA | LMT |
| Big Tech / AI policy | XLK | NVDA, MSFT, GOOGL, META, AAPL |
| Housing affordability | ITB, XHB | – |

Controls: SPY, ^VIX (Yahoo), DGS2 and DGS3MO (FRED). Excess return R_i,t − R_SPY,t;
range volatility 2(H − L)/(H + L).

**Verification without a second vendor.** Stooq blocked us after a burst of downloads.
Each fund is instead triangulated against same-exposure funds from other issuers
(SPY ↔ IVV, VOO; XLE ↔ VDE; XLV ↔ VHT; XLF ↔ VFH; XLK ↔ VGT; XAR ↔ ITA; ITB ↔ XHB).
This caught something real. On 2026-06-26 the S&P 500 *index* (Yahoo ^GSPC and FRED
SP500, one upstream source) shows −4.7bp while SPY, IVV and VOO all fall 73–86bp. Three
issuers agreeing means **the index close is stale and the ETF data is right**.

---

## 3. Econometric transmission hypotheses

**Stage 1 (sentiment → markets that price the election).**
H1: issue-conditioned sentiment S_k,t Granger-causes ΔProb_t (House, Senate, Sweep):

  ΔProb_t = α + Σ_{j=1}^{p} β_j ΔProb_{t−j} + Σ_{j=1}^{p} γ_j S_{k,t−j} + controls + ε_t,
  H0: γ_1 = … = γ_p = 0.

Expected signs: worse pocketbook sentiment (economy, housing, healthcare affordability)
hurts the incumbent party (Republican in 2026), so γ < 0 for ΔP(Dem). Immigration and
defense sentiment have ambiguous signs.

**Stage 2 (election odds → asset prices).**
H2: ΔProb_t and S_k,t Granger-cause excess returns and range volatility of the basket.
Priors on ΔP(Dem sweep): ICLN/ENPH +, XLE/XOM −, UNH/XLV − (drug-pricing and ACA
risk), JPM/XLF − (bank regulation), LMT/XAR − (defense budget), Big Tech
ambiguous (antitrust vs AI regulation). Housing (ITB/XHB) ambiguous.

**Design.** Lag order by BIC (max 5). HC1 robust F-tests. Lagged SPY, ΔVIX and ΔDGS2 as
controls. Reverse-direction tests are reported alongside (ΔProb → sentiment: does
discourse follow the odds?). Benjamini–Hochberg FDR across each table, because with
≈8 issues × 3 contracts in Stage 1 and ≈20 assets × 3 contracts × 2 outcomes in Stage 2,
several 5% rejections are expected by chance. The covariance-SVD factors (§1.4) give a
low-dimensional alternative test with far fewer comparisons.

**Heteroskedasticity check (Rigobon & Sack 2003).** Election odds and sector returns are
plausibly simultaneous: a defense rally can move "defense-hawk" odds as much as the
reverse. OLS of R on ΔProb is then biased. If the variance of political news shocks rises
on high-news days (H) while the variance of other return shocks does not,

  β_RS = (Cov_H(ΔProb, R) − Cov_L(ΔProb, R)) / (Var_H(ΔProb) − Var_L(ΔProb))

identifies the effect. Equivalently, it is an IV estimate with instrument
w_t = ΔProb_t·(1_H/T_H − 1_L/T_L). Relevance condition: Var_H(ΔProb)/Var_L(ΔProb) must be
well above 1 (F-test reported). Exclusion: return-shock variance must be similar across
regimes (ratio reported). H days are the top 20% of **GDELT midterm news volume**, which
is exogenous to the tweet sample. Event dates (debates, primaries, CPI prints) are an
alternative definition. The bootstrap CI resamples within regime.

**Is it warranted here?** Partly. The House contract sat at 75–92% all window, so its
moves are small and the instrument is weak. The Senate and Sweep contracts (≈ 40–65%)
move enough to be informative. The decision is made from the variance-ratio F-test, not
assumed.

---

## 4. Risks and limitations known before estimation

* ≈150 trading days and many tests: low power. Read q-values, not single p-values.
* The tweet sample is engaged mid-size accounts, ≈ 20–40 posts per session, so tone is
  noisy at daily frequency; 5-day smoothing is shown as a robustness check.
* GDELT DOC timelines are throttled from the cloud environment; outages are masked.
* Prediction-market liquidity differs across contracts. House odds near 90% compress
  ΔProb.
* Granger causality is predictive, not structural. Only the Rigobon–Sack estimates speak
  to contemporaneous causal effects, and only where the relevance test passes.
