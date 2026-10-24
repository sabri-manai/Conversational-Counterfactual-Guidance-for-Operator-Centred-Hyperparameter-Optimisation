# Running the revision notebooks on the study server

Short run guide. Background, findings and the full variable list are in [`README.md`](README.md).
Run everything from the repo root after `git pull`.

## 1. Setup (once)

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name pdt_hpo_xai_patchcore --display-name "Python 3.11 (PDT HPO XAI PatchCore)"
export CUDA_VISIBLE_DEVICES=1        # the GPU the original runs used; check it is the right one
export OPENROUTER_API_KEY=...        # only needed for R07; never commit it
cd review
```

The kernel install is required. The notebooks and `run_all.sh` expect the kernel name `pdt_hpo_xai_patchcore`, and
`run_all.sh` fails if it does not exist.

`MVTEC_ROOT` and `PDT_ROOT` are not needed on the study server, because the split CSVs already hold its
`/home/jovyan/shared/...` image paths. Set them only on a machine where the images live elsewhere.

## 2. Plumbing check (about 2 minutes)

```bash
REVIEW_SMOKE=1 ./run_all.sh R01
```

This writes only to `outputs/smoke/`, and its numbers are invalid. It confirms that the images are found, the GPU works
and the backbone weights download.

## 3. Real runs, in this order

```bash
./run_all.sh R01      # MVTec per-category confirmation: 1-2 h (the most important fix)
./run_all.sh R02      # frozen table rerun, 3 seeds: 2-4 h per dataset
./run_all.sh R03      # multi-anchor and baseline confirmation: 2-5 h per dataset
./run_all.sh R07      # 7B parser evaluation: ~2 min, needs OPENROUTER_API_KEY
```

To run the GPU notebooks back to back in the background:

```bash
nohup ./run_all.sh R01 R02 R03 > run.log 2>&1 &
```

They are resumable: if a run is interrupted, start it again and it continues from the cache.

R00, R04, R05 and R06 already ran on real repo data and their results are committed. Rerun them only if their inputs
change, for example after R02 or R03 produce new confirmation CSVs:

```bash
./run_all.sh R00 R04 R05 R06
```

R07 reads `outputs/tables/R06_parse_audit_per_attempt.csv`, so on a fresh clone run R06 before R07 (the default order does).

Not verified: GPU memory use. If the server GPU has less than about 24 GB, check the PatchCore batch size before the full runs.

## Where the results go

Everything is under `review/outputs/`:

| Path | Content | In git |
|---|---|---|
| `outputs/tables/R0x_*.csv`, `*.tex` | result tables | yes |
| `outputs/figures/R0x_*.pdf`, `*.png` | figures | yes |
| `outputs/frozen_numbers.json` | one section per notebook; the source for manuscript numbers | yes |
| `outputs/executed_notebooks/` | executed copies with outputs | no (git-ignored) |
| `outputs/cache/` | PatchCore run cache (large) | no (git-ignored) |
| `outputs/smoke/` | plumbing-test output, never cite | no (git-ignored) |

The committed notebooks are never overwritten by `run_all.sh`.

## After the runs

Commit `review/outputs/tables`, `review/outputs/figures` and `review/outputs/frozen_numbers.json`, then pull them back
to the working copy. Compare the new numbers with the paper and update `paper/main.tex`. Deadline: 28 Oct 2026.
