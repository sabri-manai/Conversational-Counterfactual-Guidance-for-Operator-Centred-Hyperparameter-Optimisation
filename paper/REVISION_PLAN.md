# Revision plan — FGCS-D-26-04298

Deadline 28 Oct 2026 (8 days left).

## Must fix — these block resubmission

1. **MVTec confirmation doesn't match how it was trained.** The HPO logged
   mean F1 across 4 categories, each with its own model and threshold. The
   main confirmation instead pools all 4 into one model with one threshold.
   Same config, two very different scores (0.57 vs 0.80 F1). Both reviewers
   are circling this from different angles (R2.1, R2.3, R2.5, R2.6). Need to
   re-run the 24 MVTec candidates the right way and rewrite §4.4–4.7, the
   abstract, and the conclusion based on what actually comes out.
2. **Recorded score vs. rerun score.** Reviewer 2 wants to see whether logged
   trial scores hold up on retraining. Rerun ~40 logged trials plus a plain
   no-HPO reference config on the 4 categories. Cheap (under 2 hours)
   and it's evidence we should have.
3. **The study numbers describe the wrong parser.** The logs show the study
   ran on Qwen-2.5-7B through OpenRouter with zero fallbacks and candidates
   mostly pulled from retrieval, not the 3B model with 87.5% fallback that
   the paper reports as the "study-time front-end." Pull the real numbers
   from the logs we already have.
4. **The paper's numbers don't match what's saved in the repo** (Table 4
   Proditec, surrogate fidelity, the cost table). Pick one pipeline run,
   freeze it, and regenerate every number in the paper from a single script
   so this can't happen again.


## Worth doing — cheap and actually strengthens the paper

- Run the same query on both datasets, confirmed at 2–3 anchors — one
  targeted experiment, not a full sweep (R2.5, R2.6).
- Check variability across 5 seeds on a subset of candidates before claiming
  one method beats another by a small margin (R2.4).
- Add a plain "filter by query, sort by recorded F1" baseline with no
  surrogate — it's cheap, directly requested, and fills a real gap (R2.8).
- Swap in the GPU latency numbers we already measured for the cost table
  instead of the CPU ones currently used (R1.7).
- Track down the Table 10 outlier (P-D166E1, which has a duplicate
  P-D2166E1) and fix or footnote it.
- Redraw Figure 3 without the truncated axis so the spread actually shows,
  and add the MVTec surrogate-vs-confirmed scatter.

## Answer in the text / response letter — no new experiments

- GP surrogates and calibrated uncertainty: add a paragraph acknowledging
  this as a known alternative, cite the one genuinely relevant DOI (the
  relative-RMSE paper), and politely pass on the others — they're out of
  domain, and building a GP surrogate from scratch this week isn't worth it.
- SHAP / feature importance without budget: CatBoost already produces this.
  Just report what we have, no new modelling.
- Log-density sensitivity: write a short limitation paragraph explaining why
  dense logs were used rather than running a 50/25/10/5% sweep. Low payoff
  for the time it'd cost.
- Explanation faithfulness: show 1–2 worked examples and add an explicit
  "unvalidated" note in the interface text instead of building a checker.
- Dense writing in §4.7/4.9: just restructure with summary sentences — an
  editing pass, not new content.
- Review-budget vs. top-k inspection: one paragraph positioning this against
  existing industrial QC practice.
- R² within budget bands: report it if a quick groupby gives it to us,
  otherwise call it future work.
- Over-reliance: acknowledge it as an unmeasured limitation. A proper study
  of this is its own paper.
- Everything else small — tie-breaking in Eq. 3, renaming "corruption
  level," Proditec dataset details, the ethics-basis statement, p-value
  caveats, disclosing the LLM judge's model/version/prompt — just fix
  directly in the text.

## Not doing this round

- A full multi-anchor sweep across 5 anchors on both datasets.
- Thinning the log to several density levels with full confirmation at each.
- Building a new GP or ensemble surrogate from scratch.
- A dedicated study of operator over-reliance.

Flag these as future work instead.
