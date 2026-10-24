# `review/` — stand-alone material for the FGCS-D-26-04298 major revision

Manuscript, decision letter with the reviews and the revision plan are in [`../paper/`](../paper/).

Everything here is new and independent of the existing notebooks: nothing outside `review/` is modified, and every
notebook reads the committed artefacts of the repository. Notebooks follow the repository style (markdown intro, `# ====`
banner cells, `save_table` / `save_fig`, `SEED = 42`). Launch Jupyter from the repo root or from `review/` (kernel
`pdt_hpo_xai_patchcore`, i.e. the pins in `requirements.txt`).

## Why these notebooks exist (audit findings that drive the reruns)

| # | Finding (from `R00` / `R06`) | Consequence for the paper |
|---|---|---|
| 1 | The MVTec HPO objective trains **one PatchCore and one review-budget threshold per category** and averages four F1s; the MVTec confirmation trains **one pooled memory bank and one pooled threshold**. | The −0.20 MVTec "surrogate error" mixes surrogate error with a protocol change → `R01`. |
| 2 | The paper's PatchCore F1 values come from `decomposition_per_candidate_*.csv`, not from the `*_patchcore_evaluation.csv` "rerun". The max rerun difference is 0.07 (MVTec) / 0.15 (Proditec), not 0.019; DiCE/ACFX review budgets differ between artefacts for 14/24 and 13/24 candidates. | One frozen table, one pipeline, several seeds → `R02`. |
| 3 | Table 4 LogTopK distances (5.65, 3.82) are the maxima of the anchor sweep; the primary-anchor values are 1.41 and 0.22. The primary anchor is the best logged trial. | Fix the table; multi-anchor confirmation → `R03`. |
| 4 | `mvtec/revision_outputs/tables/baseline_confirmation_mvtec.csv` looks like it was run on Proditec data (implied prevalence 0.339 vs 0.702 for MVTec test, AUROC ≈ 0.975, 588 s). | §4.7 / §4.9 / Table 6 claims are unsupported until `R03` block E reruns it. |
| 5 | MVTec surrogate R² collapses to 0.10 without the review budget (budget SHAP share 75 %). Proditec test R² is 0.936, not 0.96. | Report honestly → `R04`. |
| 6 | The study interface ran `qwen/qwen-2.5-7b-instruct` with the `llm_only_with_repair` parser, 0 fallbacks. Table 8's "study-time" row (3B, 87.5 % fallback) describes a different parser. 19/34 Task-3 parses are numerically unfaithful without being flagged as fallback; 15 of 20 relative-phrased attempts returned 0 options. | Correct Table 8 / Table 10; rerun parser evaluation on the 7B model → `R06`, `R07`. |
| 7 | Table 10 "mean attempts" for condition B is the cumulative session count, not per task (per-task: 1.75 / 1.42 / 3.09). No questionnaire item survives Holm correction; conditions were run in non-overlapping time blocks (paper: "randomly assigned"). | Rewrite the user-study text → `R06`. |

## Notebooks, run order and cost

| Notebook | Needs | Time | Reviewer item | Main outputs (`outputs/tables`, `outputs/figures`) |
|---|---|---|---|---|
| `R00_audit_repo_vs_paper` | CPU | ~1 min | internal consistency, R2.2, R2.9 | `R00_table4_*`, `R00_table5_*`, `R00_confirmation_artefact_consistency_*`, `R00_sectionR_implied_prevalence`, `R00_mvtec_protocol_gap` |
| `R04_surrogate_diagnostics` | CPU | 5–10 min | R2.2 (SHAP, R² w/o budget, within budget bands), R1.1 (GP vs CatBoost), R1.2 (relative RMSE), §4.7 winner's curse | `R04_*` |
| `R05_filter_sort_baseline_density_cost` | CPU | 10–15 min | R2.8 (filter-and-sort baseline), R1.6 (log density), R1.7 (cost break-even) | `R05_*` |
| `R06_user_study_audit` | CPU | ~1 min | R1 study limitations, R2.7 (parser vs repair workflow), over-reliance | `R06_*` |
| `R07_parser_eval_7b` | CPU + `OPENROUTER_API_KEY` for the LLM variants (~40 short requests plus retries, cached) | ~2 min | R2.7, R1 parser evaluation | `R07_*` |
| `R01_mvtec_per_category_confirmation` | **GPU + MVTec AD images** | 1–2 h | R1/R2 single-category confirmation, R2.3 (HPO vs no-HPO at equal budget), R2.5 | `R01_*` |
| `R02_frozen_table_rerun` | **GPU + both datasets** | 2–4 h / dataset, 3 seeds | confirmation reproducibility, noise floor | `R02_*` |
| `R03_anchors_and_baseline_confirmation` | **GPU + both datasets** | 2–5 h / dataset | "one anchor drives the negative finding", baseline-retrieval confirmation | `R03_*` |

Recommended order: `R00 → R06 → R04 → R05 → R07 → R01 → R02 → R03`. `R01–R03` are resumable (every PatchCore run is
cached in `outputs/cache/`), so an interrupted run can simply be restarted.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `MVTEC_ROOT` | R01–R03 | folder containing `bottle/ cable/ capsule/ hazelnut/` (MVTec AD). The absolute paths in the committed split CSVs are re-rooted onto it. |
| `PDT_ROOT` | R02, R03 | folder containing `train/` and `test/` of the Proditec public dataset. |
| `REVIEW_DEVICE` | R01–R03 | `""` = auto (CUDA if available), or `cuda:1` / `cpu`. |
| `REVIEW_WORKERS` | R01–R03 | DataLoader workers (default 2; use 0 on macOS). |
| `REVIEW_SEEDS` | R02 (default 3), R03 (default 1) | number of PatchCore seeds. |
| `REVIEW_N_LOGGED`, `REVIEW_N_TOP` | R02 | stratified logged trials re-trained per dataset (40) and extra best-N trials (8). |
| `REVIEW_N_PER_CELL` | R03 | candidates per anchor × method (3). |
| `REVIEW_SMOKE=1` | R01–R03, R05 | tiny plumbing run; writes to `outputs/smoke/` and **never** to the real outputs. Results are not valid. |
| `OPENROUTER_API_KEY` | R07 | read from the environment only (never stored). Without it R07 runs only the offline blocks and says so. |
| `R07_MODEL` | R07 | OpenRouter model id (default `qwen/qwen-2.5-7b-instruct`, the model used in the study). |
| `R07_MOCK=1` | R07 | deterministic mock model for a plumbing test; writes to `outputs/smoke/`. |

### Running on the original study server

The split CSVs in `mvtec/mvtec_splits/` and `proditec/patchcore_splits/` still contain the server's absolute image paths
(`/home/jovyan/shared/...`), so **on the original server no dataset variable has to be set**: paths that exist are used
unchanged. Set `MVTEC_ROOT` / `PDT_ROOT` only on a machine where the images live elsewhere. Pick the GPU the original
runs used with `CUDA_VISIBLE_DEVICES` (the original notebooks used physical GPU 1, seen as `cuda:0`).

```bash
pip install -r requirements.txt            # same pins as the original study (Python 3.11, torch 2.10 / CUDA 12.8)
export CUDA_VISIBLE_DEVICES=1              # as in the original runs; adjust if needed
export OPENROUTER_API_KEY=...              # only for R07's 7B variants; never commit it
cd review
./run_all.sh                               # R00 R06 R04 R05 R07 R01 R02 R03, resumable
./run_all.sh R01 R03                       # or only some notebooks
REVIEW_SMOKE=1 ./run_all.sh R01            # 2-minute plumbing check first (invalid results, writes to outputs/smoke/)
```

`run_all.sh` executes each notebook with `jupyter nbconvert` and stores the executed copy (with outputs) in
`outputs/executed_notebooks/`; the committed notebooks stay output-free. Tables, figures and `frozen_numbers.json`
are written to `outputs/`. The first PatchCore run downloads the torchvision backbone weights (network needed once).

## Files

* `patchcore_review_lib.py` — shared PatchCore code (feature extraction, memory bank, review-budget threshold, per-category
  and pooled runners, prevalence-matched F1, resumable `run_many`). The PatchCore settings are the ones of the
  committed pipelines.
* `outputs/tables/*.csv|tex`, `outputs/figures/*.pdf|png` — results. `outputs/frozen_numbers.json` — one section per
  notebook with every number meant for the manuscript; take paper numbers from here, not from the console.
* `outputs/cache/` — PatchCore run cache (large; git-ignored). `outputs/smoke/` — plumbing-test outputs (git-ignored, never cite). `outputs/executed_notebooks/` — executed copies written by `run_all.sh` (git-ignored).
* `run_all.sh` — runs the notebooks in the recommended order.

## Status of the outputs shipped in this folder

* `R00`, `R04`, `R06` and the offline part of `R07` were executed on the real repository data.
* `R05` was executed in full (fractions 0.5 / 0.25 / 0.1 / 0.05, 3 repetitions).
* `R07` LLM variants have **not** been run (no API key available when the notebook was written); run it with the key to
  produce the 7B numbers. The relative-constraint overlay in R07 was designed after seeing the benchmark, so its score on the
  40 utterances is not held-out evidence; use a fresh held-out set before claiming it.
* `R01`, `R02`, `R03` were only plumbing-tested on synthetic images (they need the datasets and a GPU). No result of these
  notebooks exists yet. **Do not cite anything from `outputs/smoke/`.**

## Turning results into paper numbers

1. Run the notebooks; check `outputs/frozen_numbers.json` for every section.
2. Replace the manuscript numbers by the values in the `.tex`/`.csv` tables (Table 4/5 → `R00` + `R02`/`R03`; Table 8 →
   `R06` + `R07`; Table 10 → `R06_table10_recomputed`).
3. Re-state claims that the audit did not support (§4.7/§4.9/Table 6 on the baseline confirmation, the 0.019 rerun
   tolerance, "randomly assigned", the over-reliance wording) according to what the new tables show.
4. Note for R2.5: MVTec test prevalence is 0.70 and Proditec's 0.345, so raw cross-dataset F1 is confounded; use the
   prevalence-matched F1 and the no-HPO comparison from `R01`.
