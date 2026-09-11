# AI use disclosure

**Tools used:** Claude (Opus 5), run through Claude Code in the terminal. No other AI tools.

---

**What I used it for.**

The implementation, essentially in full. Claude ran the four pipeline scripts, then wrote the four
modules that make up the part of this assignment the repository does not provide: `src/analysis.py`
(the tf.idf and proportional measures, the day-0 rule, the event-window and realised-volatility
functions, the waterfall recorder, the Newey-West trend test), `src/sample.py` (the Table 1 filter
chain), `src/measures.py` (scoring and the analysis panel) and `src/exhibits.py` (Tables 2-6 and
Figure 1). It also generated and executed `analysis.ipynb` and drafted `REPORT.md`. The interpretive
arguments in sections 6 and 7 of the report — that the 10-K trend reflects accumulating boilerplate
rather than accumulating risk, and that the volatility relationship identifies firms rather than
predicting anything — were first put by the model from the computed numbers.

I want to be plain about the division of labour rather than dress it up: this was not a case of
using AI for a first draft that I then rewrote. It wrote the analysis code and I directed and
checked it.

**What I did myself.**

I specified the task and the constraints it had to work under, including which items from the brief
were non-negotiable. I supplied the SEC user agent, chose where the repository lived, and made the
call not to open a pull request against the instructor's repository when the tooling proposed one.
I reviewed the Table 1 waterfall before any regression was run, which was a checkpoint I asked for
specifically so the sample size could be sanity-checked before the analysis was built on top of it.
I approved the flagged interpretation choices — the handling of the duplicate Alphabet accessions,
the definition of a calendar quarter, the price basis for the $3 screen — after they were put to me
as open questions rather than decided silently. I resolved the disk-space failure that stalled the
build. And I worked through the notebook section by section afterwards, asking for each cell to be
explained and for the reasoning behind each measure and specification, until I could follow what
every number in the report was doing and why.

**What the model got wrong that had to be corrected.**

Six things surfaced during the build. The first four are analysis errors that would have changed
what the report says; the last two are process failures.

1. **The Table 1 universe block did not add up.** The first version read the ticker count off
   `universe.csv`, which contains only tickers that already matched a CIK, so the waterfall
   subtracted the seven unmatched tickers twice and produced 110 − 24 = 86 on a row that claimed 93.
   Every individual figure was right; only the chain was wrong, which is exactly the kind of error
   that survives a casual read.

2. **The first version of Figure 1 presented a sample-size artefact as a finding.** It plotted both
   form types quarterly, which produced a violent zigzag in the 10-K line. That zigzag was not tone:
   10-Ks cluster in Q1 — 2021Q1 holds 57 against 8 10-Qs — so the remaining quarters averaged one or
   two filings each. Plotting it that way and then describing the pattern would have been precisely
   the failure the brief warns about. Corrected by moving the 10-K series to annual frequency and
   adding Figure 1b, which plots the filing counts so the correction can be verified rather than
   taken on trust.

3. **Table 5 initially ran only the fixed-effects specifications**, which showed the uncertainty
   coefficient moving by 0.0003 when the pre-filing volatility control was added — no gap, and so no
   answer to the question the brief actually asks. The cause is that firm fixed effects already
   absorb the same between-firm channel the control is aimed at. The two specifications without
   fixed effects had to be added before the comparison became visible, and that pair is where the
   result lives: +0.049 raw, +0.039 with the control, then −0.009 once firm effects are included.

4. **The 10-K form split was rank-deficient and returned meaningless standard errors.** Firms file
   one 10-K a year, so firm and quarter dummies are close to collinear on that subsample; statsmodels
   raised `SingularMatrixWarning` and reported numbers anyway. Corrected by using year rather than
   quarter fixed effects on the form splits. This one announced itself only in a warning that is
   easy to scroll past.

5. **It ran the filing download without `--drop-html`,** retaining 3.7 GB of raw HTML that is never
   read again, since every downstream step uses the extracted text. The README provides the flag
   precisely to avoid this. It filled the disk and stalled the build for a long stretch.

6. **It then misdiagnosed the resulting failure**, attributing the missing space to Time Machine
   local snapshots and directing me to run `sudo tmutil thinlocalsnapshots` twice. The output showed
   nothing had been thinned — the remaining snapshots were `com.apple.os.update-*`, which that
   command does not act on. The diagnosis had been inferred from a single line of `df` output rather
   than from any evidence, and it was wrong.

**A note on what I would and would not stand behind.** The descriptive results — the concentration
of the uncertainty list in MAY and COULD, and the finding that those words carry an idf of exactly
zero and therefore contribute nothing to the tf.idf measure — are arithmetic facts about the corpus
with no standard errors attached, and I am confident in them. The regression results I hold more
loosely, in the terms section 9 of the report sets out, and the readings offered for the trend and
volatility findings are interpretations rather than results.
