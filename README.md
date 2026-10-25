# CF-XHPO — Conversational Counterfactual Guidance for Operator-Centred Hyperparameter Optimisation in Industrial Anomaly Detection

Code, data, and analysis notebooks accompanying the paper *"Conversational Counterfactual
Guidance for Operator-Centred Hyperparameter Optimisation in Industrial Anomaly Detection"*
(Sabri Manai, Szymon Bobek, Grzegorz J. Nalepa — Jagiellonian University; Horizon Europe PEER
project), *Future Generation Computer Systems*. A live demonstration of the conversational
interface is hosted separately at https://huggingface.co/spaces/smana/cf-xai-hpo.

## Where the paper's numbers come from

Every number in the paper is taken from one frozen result artefact,
[`review/outputs/frozen_numbers.json`](review/outputs/frozen_numbers.json), and the tables in
[`review/outputs/tables/`](review/outputs/tables/). Both are produced by the reproduction notebooks
`review/R00`–`R07` (see [`review/README.md`](review/README.md)), which read the committed logs,
surrogates, and candidate files of `mvtec/` and `proditec/` and re-run every confirmation that the
paper reports. The pipeline notebooks in `mvtec/`, `proditec/`, and `shared/` produce those inputs.

One protocol note matters when reading the pipeline notebooks: the MVTec HPO objective trains one
PatchCore model and one review-budget threshold **per category** and averages the four F1 scores
(paper, Eq. 4). `mvtec/05_confirmation_evaluation.ipynb` confirms candidates with one pooled memory
bank and one pooled threshold, which measures a different quantity. The paper's MVTec confirmation
numbers therefore come from `review/R01` and `review/R02`, which use the per-category protocol; the
pooled numbers appear in the paper only "for reference" (Table 4).

## Layout

```
mvtec/        MVTec AD benchmark pipeline (4 categories: bottle, cable, capsule, hazelnut)
proditec/     Proditec production use case (derived logs, features, and models; images are proprietary)
shared/       Analyses that operate on both datasets together, and the conversational interface
user_study/   Section 5 human-participant study: pseudonymised logs + analysis notebook
review/       Reproduction notebooks R00-R07, frozen result tables, figure script (source of all paper numbers)
demo_app/     Streamlit source for the live HF Spaces demo (deployed separately)
paper/        LaTeX source of the manuscript (main.tex, references.bib, figures/)
```

Each of `mvtec/` and `proditec/` is self-contained: the notebooks and their data artifacts
(`outputs_patchcore/`, `surrogate_ready_dataset/`, `surrogate_pkl_cfs_metadata/`, `tables_*/`,
`figures_*/`, `revision_outputs/`) sit side by side, exactly as the code's own relative-path and
`rglob`-based discovery expects. Run notebooks with their containing folder as the working
directory (e.g. launch Jupyter from `mvtec/` to run `mvtec/01...06`).

## Paper → source mapping

| Paper item | Source |
|---|---|
| Table 1 (configuration space) | `objective()` in `*/01_data_collection.ipynb`; schema in `*/02_preprocessing.ipynb` |
| Table 2 (surrogate ablation), §4.3 (SHAP, GP check, within-band R², rRMSE, winner's curse) | `review/R04_surrogate_diagnostics.ipynb` |
| §4.3 development-split benchmark of 13 regressors | `*/03_surrogate_modeling.ipynb` |
| §4.3 GroupKFold cross-validation, duplicate rate | `shared/07_cross_dataset_revision_analysis.ipynb` |
| Candidate sets (24 per dataset, DiCE / LogTopK / ACFX) | `*/04_candidate_generation.ipynb` → `*/surrogate_pkl_cfs_metadata/*_counterfactuals/` |
| §4.2 three-seed confirmation, noise floor, 96 logged trials re-run; Table 3; Figure 2 | `review/R02_frozen_table_rerun.ipynb`; distances checked in `review/R00_audit_repo_vs_paper.ipynb` |
| Table 4 (protocol diagnosis), Table 5 (per-category MVTec, no-HPO reference, F1 bounds, prevalence-matched F1); Figure 3 | `review/R01_mvtec_per_category_confirmation.ipynb` |
| §4.8 anchor sweep (60 retrainings); Table 6 (retrieval vs generation, confirmed) | `review/R03_anchors_and_baseline_confirmation.ipynb` |
| §4.9 twelve-query suite | `shared/07_cross_dataset_revision_analysis.ipynb` (section S7) |
| §4.9 RecordedTopK baseline, log-density sweep; Table 7 (cost, break-even K) | `review/R05_filter_sort_baseline_density_cost.ipynb` |
| Table 8 (parser), study-time column | `review/R07_parser_eval_7b.ipynb` |
| Table 8 (parser), released column; parser latency | `shared/09_conversational_interface_and_parser_eval.ipynb` |
| §4.11 live parse audit, §5.3 over-reliance proxies, Table 9 attempt counts | `review/R06_user_study_audit.ipynb` |
| Table 9 (task validity), questionnaire, completion time, §5.4 qualitative bundle | `user_study/data_analysis.ipynb` |
| §6.3 numeric audit of the 51 study-time explanations | `review/audit_explanations.py` |
| Figures 2–4 (as typeset) | `review/make_paper_figures.py` |
| Figure 1 (framework overview) | `paper/figures/framework_overview.png` |
| §3.6 conversational interface | `demo_app/app.py` (deployed), `shared/09` (offline front-end) |

**Data Availability checklist** (as stated in the paper): the optimisation-space definition
(`Table 1` — see the `objective()` functions in `01_data_collection.ipynb` and the schema in
`02_preprocessing.ipynb`), preprocessing and surrogate code, the candidate-production and
retrieval implementations, the twelve-query suite (`shared/07`, section S7), the parser prompts
and the 40-utterance labelled parser-evaluation set (both inline in `shared/09`), per-candidate
confirmation results (`*/surrogate_pkl_cfs_metadata/*_counterfactuals_evaluation/`), the analysis and reproduction notebooks with their frozen result tables (`review/`), seeds,
and this dependency list are all included. Trial-level configuration
and metric data are released for both use cases; **only the proprietary Proditec images are
withheld** (the Proditec pipeline notebooks reference them but the release ships only the
derived logs, features, and models — see `proditec/01_data_collection.ipynb`'s header). MVTec AD
itself is public ([Bergmann et al. 2019](https://www.mvtec.com/company/research/datasets/mvtec-ad)).

## Run order

1. `mvtec/01_data_collection.ipynb` / `proditec/01_data_collection.ipynb` are **safe to run**:
   they check the trial count already stored in the committed `.db` (`*/outputs_patchcore/*.db`)
   against the target (10,000 for MVTec, 3,000 for Proditec) before touching any images, and
   since that target is already met they print a confirmation and skip straight to reporting the
   existing results — no retraining happens and no raw images are needed. They only attempt
   actual PatchCore retraining (tens of GPU-hours) if you lower the target or delete the `.db`
   so more trials are genuinely needed. Proditec's images are proprietary and not included in
   this release, so its notebook will always take the "already collected" path here.
2. `02_preprocessing` → `03_surrogate_modeling` → `04_candidate_generation` →
   `05_confirmation_evaluation` (per dataset; `05` retrains PatchCore 24 times — expect roughly
   30-70 s per candidate, matching Table 7). Run `05`'s "Section R" cells (bottom of the
   notebook) after `shared/07`'s section S8 has produced `baseline_confirm_candidates_<ds>.csv`.
3. `06_revision_diagnostics` (per dataset) — needs `05`'s confirmation CSV.
4. `shared/07_cross_dataset_revision_analysis.ipynb` — run from `shared/` (or anywhere under
   this release root); it auto-discovers both `mvtec/` and `proditec/` via `rglob`. Re-run its
   S8-S9 cells after `05`'s Section R has produced `baseline_confirmation_<ds>.csv`.
5. `shared/09_conversational_interface_and_parser_eval.ipynb` — needs a GPU for
   Qwen2.5-3B-Instruct (or edit `MODEL_ID`/`LLM_MODEL_OPTIONS` for a CPU-friendly alternative).
   Writes `parser_eval_latency.json` to `shared/revision_outputs/`; copy that file into
   `mvtec/revision_outputs/` and `proditec/revision_outputs/` before re-running `shared/07`
   section S11, so the end-to-end latency figures (Table 7) pick up the measured LLM latency
   instead of falling back to `NaN`.
6. `shared/08_final_results_export.ipynb` — run last; consolidates everything above into
   `final_results/`.
7. `user_study/data_analysis.ipynb` — independent of steps 1-6; only needs `user_study/logs/`.
8. `review/` — the reproduction notebooks that produce the paper's numbers; see
   [`review/README.md`](review/README.md) for their order, cost, and environment variables.

## What was changed from the working repository

This release was built by copying each canonical source notebook's cells verbatim and applying
a small number of precise, verified edits — nothing was retyped from memory. In summary:

- **Made the data-collection notebooks (`*/01_data_collection.ipynb`) idempotent and runnable.**
  Originally these needed sibling `fast`/`shared` directories that don't exist in this layout and
  immediately asserted on the (not-included) raw image directory before doing anything else,
  including checking whether more trials were even needed. Both notebooks were restructured so
  the trial-count check against the committed `.db` happens first (cheap, no images touched);
  every subsequent cell that needs raw images is now gated behind that check and is skipped
  automatically once the target is already met, which it is for the logs shipped here. The
  Proditec notebook was also missing `import numpy`/`pandas`/`optuna`/`time`/`zlib` entirely —
  it referenced all five without importing any of them anywhere in the file, so it could not
  have run in a fresh kernel; the imports are now present. Both GPU-selection cells fall back to
  GPU 0 or CPU if the hardcoded second GPU device does not exist.
- **Removed** two early-stage prototype folders (ACFX/DiCE/CFNOW experiments on generic tabular
  benchmarks — Adult income, breast cancer, phishing, bank marketing, fit with XGBoost) that
  predate and are unconnected to the PatchCore/MVTec/Proditec pipeline the paper reports. No
  notebook in this release references them, and the paper does not report results on those
  datasets.
- **Removed** superseded duplicate/conference-era notebooks once a later, more complete version
  was confirmed to be the one the rest of the pipeline actually consumes (e.g. an early DiCE-only
  candidate-generation notebook superseded by the 3-method version; an early single-method
  evaluation notebook superseded by the version with the retrieval-baseline confirmation
  appended; a stray leftover copy of each dataset's revision-diagnostics notebook that had been
  accidentally duplicated into the other dataset's folder during earlier reorganisation; an
  early monolithic prototype of the conversational interface superseded by the deployed
  `demo_app/app.py`).
- **Fixed real bugs**, each documented in the corresponding notebook's header cell: a DiCE
  candidate-generation call that declared a seed but never used it; a hardcoded `"mvtec_"`
  filename prefix in the Proditec confirmation-evaluation notebook that produced a
  misleadingly-named output file (and the already-generated file has been renamed accordingly);
  a fragile exact-filename evaluation-CSV lookup in the MVTec revision notebook that could
  silently pick up a wrong-dataset file (ported the more robust pattern already used correctly
  on the Proditec side); a missing metadata-JSON export in the Proditec surrogate notebook that
  left a downstream schema lookup silently falling back to a hardcoded schema; a handful of
  leftover copy-paste labels/comments (e.g. a Proditec notebook cell whose diagnostic printout
  said `"dataset": "mvtec_4class"`); and a report-formatting bug that always printed
  "speedup 0-0x" once all measured speed-ups fell below 1.
- **Removed energy/CO2 "savings" estimates** from the revision-diagnostics and final-export
  notebooks. The paper's Section 4.10 explicitly states that *no power telemetry was collected
  and no energy or CO2 saving is claimed* — the removed cells computed such figures from
  assumed, unmeasured power constants, and a consistency check on their own numbers showed the
  measured LLM parse/explain latency alone would make every method a net energy **cost**, not a
  saving, under those assumptions. The legitimate, measured wall-clock **timing** (which Table 7
  does report as the "component speed-up" and "complete interaction range") is kept.
- **Dropped dead/duplicate cells** (byte-identical duplicate diagnostic-plot cells, trailing
  empty cells) and **stripped all cell outputs** so the shipped notebooks are lightweight and
  unambiguous about what will be produced on re-run.
- **Renamed a full set of misnamed Proditec output artifacts.** The "paper-ready tables and
  plots" section of the Proditec confirmation-evaluation notebook (paper table, method-average
  metrics, both soft-parameter-profile tables/figures, and the confirmation CSV itself) was
  copy-pasted from the MVTec notebook and had kept hardcoded `"mvtec_"`-prefixed output
  filenames throughout — six tables and five figures were being written under a misleading
  dataset name. Fixed in code (now `"proditec_"`-prefixed, or derived from the actual input
  filename) and the already-generated files in this release renamed to match.
- **Pruned stray data artifacts**: an empty, wrong-dataset SQLite file that had been copied into
  the Proditec output folder; orphaned outputs from an earlier, narrower DiCE-only run that the
  current candidate-generation notebook can no longer reproduce or explain; ad hoc scratch files
  left over from interactive demo-app sessions; a duplicate, unprefixed copy of the Proditec
  surrogate model; the stale energy/CO2 tables described above; and the "avoided compute for
  interactive workloads" tables, which came from a part of the Part A cell that was removed
  along with the energy computation (the accompanying per-query timing figure is unaffected and
  kept).

## Known limitations (documented, not silently patched)

- **TPE startup-trial counts differ between the datasets.** `mvtec/01_data_collection.ipynb` uses
  `n_startup_trials=4000` and the Proditec notebook 1,500; the paper (Section 4.2) reports both
  values. They are properties of the already-collected logs and are left unchanged.
- **`shared/07` section S7's `INCLUDE_ACFX_IN_SUITE` flag** is not wired to the query-suite job
  list (there is no effect either way); ACFX is excluded from the 12-query sweep for cost
  reasons regardless, consistent with Section 4.9 of the paper.
- **Section 5.4 (qualitative/LLM-as-judge analysis)**: `user_study/data_analysis.ipynb` prepares
  the anonymised justification export and a ready-made LLM prompt but does not call an LLM
  itself. Feed `tables/qualitative_comments/llm_prompt_for_qualitative_summary.txt` /
  `..._for_llm.md` to an LLM client of your choice to reproduce the qualitative summary.
- **`user_study/data_analysis.ipynb` "Reviewer 2, item 7" cells** (near the end of the notebook)
  still write `revision_outputs/` relative to the launch directory rather than the fixed
  `DATA_ROOT` used earlier in the notebook; launch the notebook from `user_study/` for these
  specific cells to land in `user_study/revision_outputs/`.
- **Provenance paths in already-generated JSON/manifest files** (e.g. `final_results/*.json`,
  `*_metadata.json`) may still contain the original run machine's absolute paths
  (`/home/jovyan/...`); these are data, not code, and are overwritten with local paths the next
  time the producing notebook runs.
- **If you deliberately lower `TARGET_TRIALS` to extend data collection**, the data-collection
  notebooks' `DATASET_DIR` still defaults to the original machine's path convention
  (`shared/pdt/dataset_public` for Proditec; an absolute path for MVTec) and needs to be pointed
  at your own local copy of the images first; see each notebook's header cell.

## Environment

See `requirements.txt` (pinned to the versions in the paper's Section 4.2: Python 3.11.9,
PyTorch 2.10.0/CUDA 12.8, torchvision 0.25.0, scikit-learn 1.6.1, Optuna 4.2.0, CatBoost 1.2.8,
dice-ml 0.12, acfx 0.3.6, interpret 0.7.3, transformers 5.1.0). `demo_app/requirements.txt`
lists the smaller dependency set for the standalone Streamlit demo. All experiments were run on
a single workstation (1x NVIDIA RTX A5500, power-capped to 70%; AMD Threadripper PRO 5995WX;
256 GB RAM).

## License

Apache License 2.0 — see `LICENSE`.
