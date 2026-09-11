# AI use disclosure

**Tools used:** Claude (Opus 5), via Claude Code in the terminal. No other AI tools.

---

**What I used it for.**

Effectively the whole implementation. Claude ran the four pipeline scripts, then wrote the four
modules that constitute the assignment's own work — `src/analysis.py` (tf.idf and the proportional
measure, the day-0 rule, the event-window and volatility functions, the waterfall recorder),
`src/sample.py` (the Table 1 filter chain), `src/measures.py` (scoring and the analysis panel) and
`src/exhibits.py` (Tables 2–6, Figure 1, the power arithmetic). It also generated and executed
`analysis.ipynb` and drafted `REPORT.md`.

It was not used as an autocomplete. The work it did that mattered most was catching things in its
own output, listed below.

**What I did myself.**

> **[PARAM — EDIT THIS SECTION BEFORE SUBMITTING.]** I am writing this file as part of the same
> session that produced the code, so I cannot honestly characterise your contribution for you, and
> inventing one in an integrity disclosure would be exactly the wrong thing to do. What is true from
> the transcript: you specified the task and its constraints, supplied the SEC user agent, chose
> where the repository lived, made the call not to open a pull request against the instructor's
> repository, directed the build to continue at the checkpoints, and resolved the disk-space failure
> on your machine. Anything beyond that — reading the output, checking the numbers, deciding you
> agreed with the interpretations — only you can attest to. Please rewrite this paragraph in your
> own words, and delete this block.

**What the model got wrong that had to be corrected.**

Six things, all caught during the build. The first four are analysis errors that would have changed
the results; the last two are process failures.

1. **The Table 1 universe block was internally inconsistent.** The first version read the ticker
   count off `universe.csv`, which only contains tickers that already matched a CIK, so the
   waterfall subtracted the 7 unmatched tickers twice and arrived at 110 − 24 = 86 where the row
   said 93. Caught by checking that each row's arithmetic actually chained, which is the only reason
   it surfaced — every individual number was right, only the sequence was wrong.

2. **Figure 1's first version was a sample-size artefact presented as a finding.** It plotted both
   form types quarterly, which produced a violent zigzag in the 10-K series. That zigzag was not
   tone: 10-Ks cluster in Q1, so the non-Q1 quarterly "means" were averages of one or two filings.
   Plotting it that way and then describing the pattern would have been precisely the failure the
   brief warns about. Fixed by moving the 10-K series to an annual frequency and adding Figure 1b,
   which shows the filing counts so the reader can check the claim rather than trust it.

3. **Table 5 originally ran only the fixed-effects specifications**, which showed the coefficient on
   uncertainty moving by 0.0003 when the pre-filing volatility control was added — i.e. no gap at
   all, and no answer to Q4. The reason is that firm fixed effects already absorb the same
   between-firm channel the control targets. The two specifications without fixed effects had to be
   added before the question the brief actually asks could be answered, and that pair is where the
   result lives (+0.049 → +0.039 with the control, then → −0.009 once firm effects are included).

4. **The 10-K form split was rank-deficient and reported meaningless standard errors.** Firms file
   one 10-K a year, so firm and quarter dummies are nearly collinear on that subsample; statsmodels
   raised `SingularMatrixWarning` and returned numbers anyway. Switched to year fixed effects on the
   form splits. Worth noting that this one announced itself in a warning that is easy to scroll past.

5. **It ran the filing download without `--drop-html`,** keeping 3.7 GB of raw HTML that is never
   read again — every downstream step uses the extracted text. The README offers the flag precisely
   to avoid this. It filled the disk and stalled the build for a long stretch.

6. **It then misdiagnosed the resulting disk failure**, attributing the missing space to Time
   Machine local snapshots and sending me to run `sudo tmutil thinlocalsnapshots` twice. The output
   showed nothing was thinned — the remaining snapshots were `com.apple.os.update-*`, which that
   command does not touch. The diagnosis was inferred from a single `df` line rather than from any
   evidence, and it was wrong.

**On the substance of the findings.** The interpretations in §6 and §7 of the report — that the
10-K trend is accumulating boilerplate rather than accumulating risk, and that the volatility
relationship identifies firms rather than predicting anything — were argued by the model from the
computed numbers. They are consistent with the evidence as far as I can check, but they are readings
rather than results, and the underlying numbers (the idf-of-zero finding in particular) are the part
I would stand behind unconditionally.
