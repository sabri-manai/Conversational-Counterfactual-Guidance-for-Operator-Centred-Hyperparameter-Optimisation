"""Numeric audit of the study-time explanations (paper, Section 6.3 "Explanation faithfulness").

The study-time interface rendered each explanation from a deterministic template (``raw_explanation``)
and then passed it through the language model for fluency (``explanation``, the text participants saw).
This script compares the decimal numbers in the two texts for every generated attempt that had an
explanation, and writes one row per attempt plus a summary line.

Usage (from the repository root or from review/):
    python review/audit_explanations.py

Output: review/outputs/tables/R06_explanation_numeric_audit.csv
Covers numbers only, not comparative wording.
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "user_study" / "logs" / "interface"
OUT = ROOT / "review" / "outputs" / "tables" / "R06_explanation_numeric_audit.csv"
NUM = re.compile(r"\d+\.\d+")


def main() -> None:
    rows = []
    for f in sorted(glob.glob(str(LOGS / "*_interface_session.json"))):
        session = json.load(open(f))
        for attempt in session.get("generated_attempts", []):
            raw = attempt.get("raw_explanation") or ""
            shown = attempt.get("explanation") or ""
            # attempts without options carry a fixed "Only 0 valid ..." message, not an explanation
            if not raw or raw.startswith("Only 0 valid"):
                continue
            r, s = sorted(NUM.findall(raw)), sorted(NUM.findall(shown))
            added = [x for x in s if x not in r]
            dropped = [x for x in r if x not in s]
            rows.append(dict(participant_id=session["participant_id"], task=attempt.get("task_name"),
                             numbers_identical=not added and not dropped,
                             numbers_added=" ".join(added), numbers_dropped=" ".join(dropped)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["numbers_identical"] for r in rows)
    print(f"{len(rows)} explanations audited, {n_ok} reproduce every template number exactly -> {OUT}")
    for r in rows:
        if not r["numbers_identical"]:
            print(f"  {r['participant_id']} {r['task']}: added [{r['numbers_added']}] dropped [{r['numbers_dropped']}]")


if __name__ == "__main__":
    main()
