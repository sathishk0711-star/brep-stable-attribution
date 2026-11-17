#!/usr/bin/env python3
"""Audit research evidence without generating or modifying measured outcomes.

Usage: python audit_artifacts.py --summary full_summary.json --expected-seeds 11 23 37 53 71
Exit 1 when the available artifact cannot support the specified five-seed claim.
"""
import argparse
import json
import math
import sys
from pathlib import Path

def audit(path, expected):
    problems = []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    found = data.get("seeds", [])
    if len(found) != len(set(found)):
        problems.append("Duplicate seeds in summary")
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing:
        problems.append("Missing required seeds: " + ", ".join(map(str, missing)))
    if extra:
        problems.append("Unexpected seeds: " + ", ".join(map(str, extra)))
    for method, vals in data.get("methods", {}).items():
        for field in ("deletion", "insertion", "completeness", "mass"):
            v = vals.get(field)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append(f"{method}: invalid or missing {field}")
        score = vals.get("score")
        if not isinstance(score, list) or len(score) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in score):
            problems.append(f"{method}: invalid score and confidence limits")
        elif not (score[1] <= score[0] <= score[2]):
            problems.append(f"{method}: confidence interval does not contain mean")
        if isinstance(vals.get("deletion"), (int, float)) and isinstance(vals.get("insertion"), (int, float)) and isinstance(score, list) and len(score) == 3:
            if abs((vals["insertion"] - vals["deletion"]) - score[0]) > 0.015:
                problems.append(f"{method}: reported score inconsistent with insertion-minus-deletion AUC")
    return found, problems

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="full_summary.json")
    p.add_argument("--expected-seeds", nargs="+", type=int, default=[11, 23, 37, 53, 71])
    args = p.parse_args()
    found, problems = audit(args.summary, args.expected_seeds)
    print("Evidence summary:", args.summary)
    print("Recorded seeds:", found)
    if problems:
        for issue in problems:
            print("NOT VERIFIED:", issue)
        print("Do not treat this summary as independent support for all manuscript experiments.")
        return 1
    print("PASS: structural evidence audit only; raw part-level records and provenance remain necessary.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
