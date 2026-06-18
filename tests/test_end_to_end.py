"""
End-to-end integration test using sample_candidates.json.

Verifies the full pipeline produces a valid submission CSV on the
provided 50-candidate sample — without needing sentence-transformers.

Run: python -m pytest tests/test_end_to_end.py -v
  or: python tests/test_end_to_end.py
"""

import sys
import csv
import json
import os
import re
import tempfile
from pathlib import Path

FITFINDR = Path(__file__).parent.parent
sys.path.insert(0, str(FITFINDR))

CONFIG_PATH = FITFINDR / "artifacts" / "jd_config.json"

# Locate sample_candidates.json via env var or default sibling-folder convention.
# Override by setting: export FITFINDR_TEST_DATA=/path/to/folder/containing/sample_candidates.json
_test_data_dir = Path(os.environ.get("FITFINDR_TEST_DATA", FITFINDR.parent / "india_runs"))
SAMPLE_PATH = _test_data_dir / "sample_candidates.json"


def load_sample_as_jsonl(tmpdir: str) -> str:
    """Convert sample_candidates.json to a tmp JSONL file for testing."""
    out_path = os.path.join(tmpdir, "sample.jsonl")
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        candidates = json.load(f)
    with open(out_path, "w", encoding="utf-8") as out:
        for c in candidates:
            out.write(json.dumps(c) + "\n")
    return out_path


def run_pipeline(jsonl_path: str, out_path: str, config: dict) -> list:
    """Run the full pipeline (Stages 1-3 + honeypot + reasoning + output)."""
    from ranker.ingest         import stream_candidates, extract_features
    from ranker.stage2_score   import compute_composite_score
    from ranker.stage3_semantic import semantic_rerank
    from ranker.honeypot       import detect_and_demote_honeypots
    from ranker.reasoning      import generate_reasoning
    from ranker.output         import normalize_scores, write_submission

    # Stage 1
    passed = []
    for cand in stream_candidates(jsonl_path):
        feats = extract_features(cand, config)
        if feats["hard_filter_score"] > 0.0:
            passed.append(feats)

    if not passed:
        return []

    # Stage 2
    for f in passed:
        f["composite_score"] = compute_composite_score(f, config)
    passed.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
    top_k = passed[:min(50, len(passed))]

    # Stage 3 (no embedding — falls back to composite ordering)
    top_n = semantic_rerank(top_k, config, artifacts_dir=str(FITFINDR / "artifacts"))

    # Post-processing
    top_n = detect_and_demote_honeypots(top_n)
    for c in top_n:
        c["reasoning"] = generate_reasoning(c, config)
    top_n = normalize_scores(top_n)
    write_submission(top_n, out_path)

    return top_n


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pipeline_produces_valid_output():
    if not SAMPLE_PATH.exists():
        print(f"SKIP: sample_candidates.json not found at {SAMPLE_PATH}")
        return

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = load_sample_as_jsonl(tmpdir)
        out_path   = os.path.join(tmpdir, "test_submission.csv")
        results    = run_pipeline(jsonl_path, out_path, config)

        assert len(results) > 0, "Pipeline should produce at least 1 result"
        assert os.path.exists(out_path), "Output CSV should be created"

        # Verify CSV structure
        with open(out_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows   = list(reader)

        assert header == ["candidate_id", "rank", "score", "reasoning"], \
            f"Header mismatch: {header}"

        # All candidate IDs valid format
        id_pat = re.compile(r"^CAND_\d{7}$")
        for row in rows:
            assert id_pat.match(row[0]), f"Invalid candidate_id: {row[0]}"

        # Ranks are unique integers
        ranks = [int(r[1]) for r in rows]
        assert len(set(ranks)) == len(ranks), "Duplicate ranks found"
        assert min(ranks) == 1,              "Ranks must start at 1"

        # Scores are non-increasing
        scores = [float(r[2]) for r in rows]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Scores not non-increasing at rank {i+1}: {scores[i]:.4f} < {scores[i+1]:.4f}"

        # Scores in valid range
        for score in scores:
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

        print(f"\n  Passed: {len(results)} candidates ranked")
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  Top 3:")
        for row in rows[:3]:
            print(f"    #{row[1]}  {row[0]}  score={row[2]}  {row[3][:80]}")


def test_stage1_eliminates_domain_mismatch():
    """Verify that obvious non-fits (HR Managers, etc.) are eliminated."""
    if not SAMPLE_PATH.exists():
        return

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    with open(SAMPLE_PATH, encoding="utf-8") as f:
        candidates = json.load(f)

    from ranker.ingest import extract_features

    eliminated = []
    for cand in candidates:
        feats = extract_features(cand, config)
        if feats["hard_filter_score"] == 0.0:
            eliminated.append((cand["candidate_id"], cand["profile"]["current_title"]))

    print(f"\n  Eliminated by Stage 1: {len(eliminated)}/{len(candidates)}")
    for cid, title in eliminated[:5]:
        print(f"    {cid}  {title}")

    # At least some should be eliminated from the 50-candidate sample
    # (CAND_0000002 is an Operations Manager at Wipro — should be eliminated)
    assert len(eliminated) > 0, "At least some candidates should be eliminated"


def test_scores_reflect_ai_relevance():
    """Top-ranked candidates should have higher AI career fraction than bottom-ranked."""
    if not SAMPLE_PATH.exists():
        return

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = load_sample_as_jsonl(tmpdir)
        out_path   = os.path.join(tmpdir, "test_submission.csv")
        results    = run_pipeline(jsonl_path, out_path, config)

    if len(results) < 4:
        return

    top_ai_frac = sum(c["ai_career_fraction"] for c in results[:3]) / 3
    bot_ai_frac = sum(c["ai_career_fraction"] for c in results[-3:]) / 3
    assert top_ai_frac >= bot_ai_frac, \
        f"Top candidates should have higher AI fraction ({top_ai_frac:.2f}) " \
        f"than bottom ({bot_ai_frac:.2f})"


if __name__ == "__main__":
    tests = [
        ("Pipeline produces valid output",           test_pipeline_produces_valid_output),
        ("Stage 1 eliminates domain mismatches",     test_stage1_eliminates_domain_mismatch),
        ("Scores reflect AI relevance",              test_scores_reflect_ai_relevance),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\nRunning: {name}")
        try:
            fn()
            print(f"  PASS")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
