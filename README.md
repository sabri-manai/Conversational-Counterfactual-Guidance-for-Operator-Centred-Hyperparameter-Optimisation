# CF-XHPO — Conversational Counterfactual Guidance for Operator-Centred Hyperparameter Optimisation in Industrial Anomaly Detection

Code, data, and analysis notebooks accompanying the paper *"Conversational Counterfactual
Guidance for Operator-Centred Hyperparameter Optimisation in Industrial Anomaly Detection"*
(Sabri Manai, Szymon Bobek, Grzegorz J. Nalepa — Jagiellonian University; Horizon Europe PEER
project). A live demonstration of the conversational interface is hosted separately at
https://huggingface.co/spaces/smana/cf-xai-hpo.

This is a cleaned, deduplicated release built from the project's working repository. Every
notebook is organised around the two-dataset pipeline described in the paper (Section 3-4):
optimisation-log collection → preprocessing → surrogate modelling → counterfactual candidate
generation → real-model confirmation → cross-cutting diagnostics, plus the conversational
interface/parser evaluation (Section 3.6/4.11) and the human-participant study (Section 5).

## Layout

```
mvtec/        MVTec AD benchmark pipeline (4 categories: bottle, cable, capsule, hazelnut)
proditec/     Proditec production use case
shared/       Analyses that operate on both datasets together, and the conversational interface
user_study/   Section 5 human-participant study: raw anonymised logs + analysis notebook
demo_app/     Streamlit source for the live HF Spaces demo (deployed separately)
paper/        Submitted manuscript, decision letter with the reviews, revision plan, LaTeX source
review/       Revision (FGCS-D-26-04298): audit + re-run notebooks R00-R07, outputs and run script (see review/README.md)
```

Each of `mvtec/` and `proditec/` is self-contained: the notebooks and their data artifacts
(`outputs_patchcore/`, `surrogate_ready_dataset/`, `surrogate_pkl_cfs_metadata/`, `tables_*/`,
`figures_*/`, `revision_outputs/`) sit side by side, exactly as the code's own relative-path and
`rglob`-based discovery expects. Run notebooks with their containing folder as the working
directory (e.g. launch Jupyter from `mvtec/` to run `mvtec/01...06`).

## Notebook → paper mapping

| Notebook | Produces | Paper linkage |
|---|---|---|
| `mvtec/01_data_collection.ipynb` | MVTec HPO log (9,976 COMPLETE trials) | Section 3.2/3.3, Section 4.2 |
| `mvtec/02_preprocessing.ipynb` | Surrogate-ready MVTec table | Section 3.3 |
| `mvtec/03_surrogate_modeling.ipynb` | CatBoost surrogate + dev-split benchmark | **Table 2** (dev-split rows), Section 3.4 |
| `mvtec/04_candidate_generation.ipynb` | 24 MVTec candidates (DiCE/LogTopK/ACFX, K=8 each) | Section 3.5, Section 4.2 |
| `mvtec/05_confirmation_evaluation.ipynb` | Real-PatchCore confirmation of the 24 candidates | **Table 3** (MVTec rows), Section 4.4-4.6 |
| `mvtec/06_revision_diagnostics.ipynb` | Gap decomposition, per-category AUROC, anchor sensitivity | **Table 4**, **Figure 3**, Section 4.7-4.8 |
| `proditec/01_data_collection.ipynb` | Proditec HPO log (2,999 COMPLETE trials) | Section 3.2/3.3, Section 4.2 |
| `proditec/02_preprocessing.ipynb` | Surrogate-ready Proditec table | Section 3.3 |
| `proditec/03_surrogate_modeling.ipynb` | CatBoost surrogate + dev-split benchmark | **Table 2** (dev-split rows) |
| `proditec/04_candidate_generation.ipynb` | 24 Proditec candidates | Section 3.5, Section 4.2 |
| `proditec/05_confirmation_evaluation.ipynb` | Real-PatchCore confirmation + retrieval-baseline confirmation (Section R) | **Table 3** (Proditec rows), **Table 5** (6 of 12 rows) |
| `proditec/06_revision_diagnostics.ipynb` | Gap decomposition, anchor sensitivity | **Table 4** (Proditec column), Section 4.7-4.8 |
| `shared/07_cross_dataset_revision_analysis.ipynb` | GroupKFold CV, duplication rate, 12-query suite, generation-vs-retrieval, end-to-end latency | Section 4.3 (primary R2), Section 4.9, **Table 5/6** |
| `shared/08_final_results_export.ipynb` | Consolidated `final_results/` bundle | — |
| `shared/09_conversational_interface_and_parser_eval.ipynb` | Conversational interface + **40-utterance labelled parser evaluation** | Section 3.6, **Table 7**, Section 4.11 |
| `user_study/data_analysis.ipynb` | Task validity, completion time, questionnaire, qualitative-analysis evidence bundle | **Table 8/9**, **Figure 4**, Section 5 |
| `demo_app/app.py` | Streamlit conversational-interface demo (source for the hosted HF Space) | Section 3.6 |

**Data Availability checklist** (as stated in the paper): the optimisation-space definition
(`Table 1` — see the `objective()` functions in `01_data_collection.ipynb` and the schema in
`02_preprocessing.ipynb`), preprocessing and surrogate code, the candidate-production and
retrieval implementations, the twelve-query suite (`shared/07`, section S7), the parser prompts
and the 40-utterance labelled parser-evaluation set (both inline in `shared/09`), per-candidate
confirmation results (`*/surrogate_pkl_cfs_metadata/*_counterfactuals_evaluation/`), the
analysis notebooks, seeds, and this dependency list are all included. Trial-level configuration
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
   30-70 s per candidate, matching Table 6). Run `05`'s "Section R" cells (bottom of the
   notebook) after `shared/07`'s section S8 has produced `baseline_confirm_candidates_<ds>.csv`.
3. `06_revision_diagnostics` (per dataset) — needs `05`'s confirmation CSV.
4. `shared/07_cross_dataset_revision_analysis.ipynb` — run from `shared/` (or anywhere under
   this release root); it auto-discovers both `mvtec/` and `proditec/` via `rglob`. Re-run its
   S8-S9 cells after `05`'s Section R has produced `baseline_confirmation_<ds>.csv`.
5. `shared/09_conversational_interface_and_parser_eval.ipynb` — needs a GPU for
   Qwen2.5-3B-Instruct (or edit `MODEL_ID`/`LLM_MODEL_OPTIONS` for a CPU-friendly alternative).
   Writes `parser_eval_latency.json` to `shared/revision_outputs/`; copy that file into
   `mvtec/revision_outputs/` and `proditec/revision_outputs/` before re-running `shared/07`
   section S11, so the end-to-end latency figures (Table 6) pick up the measured LLM latency
   instead of falling back to `NaN`.
6. `shared/08_final_results_export.ipynb` — run last; consolidates everything above into
   `final_results/`.
7. `user_study/data_analysis.ipynb` — independent of steps 1-6; only needs `user_study/logs/`.

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
  saving, under those assumptions. The legitimate, measured wall-clock **timing** (which Table 6
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

- **MVTec TPE startup-trial count.** `mvtec/01_data_collection.ipynb` uses
  `n_startup_trials=4000`; the Proditec collection notebook and the paper's stated protocol both
  use 1,500. This is a property of the *already-collected* MVTec log (not re-run as part of this
  release) and is left as an accurate record rather than edited retroactively.
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
