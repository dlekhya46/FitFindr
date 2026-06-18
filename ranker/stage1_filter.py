"""
Stage 1: Hard filter.

The filter logic itself is computed during feature extraction in ingest.py
(_hard_filter_score). This module provides the predicate and a diagnostic
summary for logging.
"""


def passes_hard_filter(feats: dict) -> bool:
    """Return True if candidate passes Stage 1 (hard_filter_score > 0)."""
    return feats["hard_filter_score"] > 0.0


def filter_summary(all_feats: list) -> dict:
    """Return counts of why candidates were eliminated — useful for debugging."""
    passed   = sum(1 for f in all_feats if passes_hard_filter(f))
    total    = len(all_feats)

    reasons = {
        "too_junior":         0,
        "inactive_unreachable": 0,
        "domain_mismatch":    0,
        "location_miss":      0,
        "passed":             passed,
        "total":              total,
    }

    for f in all_feats:
        if f["years_of_experience"] < 2.0:
            reasons["too_junior"] += 1
        elif f["days_inactive"] > 365 and not f["open_to_work"]:
            reasons["inactive_unreachable"] += 1
        elif not passes_hard_filter(f):
            # Domain mismatch is the dominant remaining case
            reasons["domain_mismatch"] += 1

    return reasons
