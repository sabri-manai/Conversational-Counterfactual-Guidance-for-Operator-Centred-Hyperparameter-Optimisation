#!/usr/bin/env bash
# Execute the revision notebooks in the recommended order (resumable: R01-R03 cache every PatchCore run).
#   ./run_all.sh              all notebooks      ./run_all.sh R01 R03     only the listed ones
# Env: REVIEW_KERNEL (default pdt_hpo_xai_patchcore), REVIEW_TIMEOUT seconds per cell (default -1 = none),
#      plus the variables documented in README.md (MVTEC_ROOT, PDT_ROOT, REVIEW_SMOKE, OPENROUTER_API_KEY, ...).
set -euo pipefail
cd "$(dirname "$0")"

KERNEL="${REVIEW_KERNEL:-pdt_hpo_xai_patchcore}"
TIMEOUT="${REVIEW_TIMEOUT:--1}"
ORDER=(R00 R06 R04 R05 R07 R01 R02 R03)
SELECT=("${@:-${ORDER[@]}}")

if [[ "${REVIEW_SMOKE:-0}" == "1" || "${R07_MOCK:-0}" == "1" ]]; then
  DEST="outputs/smoke/executed_notebooks"   # plumbing runs never mix with real results
else
  DEST="outputs/executed_notebooks"
fi
mkdir -p "$DEST"

for id in "${SELECT[@]}"; do
  nb=$(ls "${id}"_*.ipynb 2>/dev/null | head -n1 || true)
  if [[ -z "$nb" ]]; then echo "no notebook for '$id'" >&2; exit 1; fi
  echo "=== $nb ($(date '+%H:%M:%S')) ==="
  jupyter nbconvert --to notebook --execute "$nb" --output-dir "$DEST" \
    --ExecutePreprocessor.kernel_name="$KERNEL" --ExecutePreprocessor.timeout="$TIMEOUT"
done
echo "done -> tables/figures in outputs/, executed copies in $DEST"
