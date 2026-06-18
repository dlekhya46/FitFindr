"""
Output layer: score normalization, tie-breaking, and CSV serialization.

Rules enforced here:
  - Scores are monotonically non-increasing with rank
  - Each rank 1–100 appears exactly once
  - Tie-breaking by candidate_id ascending (as required by submission spec)
  - Score range [0.20, 0.99]
"""

import csv


# ─────────────────────────────────────────────────────────────────────────────
# Score normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scores(top_100: list) -> list:
    """
    Assign ranks 1–100 and normalized scores [0.20, 0.99].
    Guarantees non-increasing scores and tie-breaking by candidate_id.
    """
    # Primary sort: final_score descending; secondary: candidate_id ascending (tie-break)
    top_100 = sorted(top_100, key=lambda x: (-x["final_score"], x["candidate_id"]))

    raw = [c["final_score"] for c in top_100]
    mn, mx = min(raw), max(raw)
    spread = mx - mn + 1e-9

    prev_score = 1.0
    for i, cand in enumerate(top_100):
        norm = 0.20 + 0.79 * (cand["final_score"] - mn) / spread
        norm = min(norm, prev_score)      # enforce non-increasing
        cand["score"] = round(norm, 4)
        cand["rank"]  = i + 1
        prev_score    = norm

    return top_100


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_submission(top_100: list, out_path: str) -> None:
    """Write the final submission CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cand in top_100:
            writer.writerow([
                cand["candidate_id"],
                cand["rank"],
                cand["score"],
                cand.get("reasoning", ""),
            ])
    print(f"  Written: {out_path} ({len(top_100)} rows)")
