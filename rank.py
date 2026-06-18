#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking System
========================================================

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints (enforced by design):
  CPU only  ·  ≤ 5 min wall-clock  ·  ≤ 16 GB RAM  ·  no network during ranking

Pipeline:
  Stage 1  Stream 100K candidates → hard-filter rules  → ~5K–10K candidates
  Stage 2  Multi-signal scoring   → top-K composite    → top 500
  Stage 3  Semantic re-ranking    → cosine similarity  → top 100
  Post     Honeypot detection · Reasoning · Normalize · CSV output
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure package root is on the path when run from any working directory
sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Rank 100K candidates for the Redrob Senior AI Engineer role.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidates", required=True,
                   help="Path to candidates.jsonl or candidates.jsonl.gz")
    p.add_argument("--out", required=True,
                   help="Output CSV path  (e.g. team_xxx.csv)")
    p.add_argument("--config", default="artifacts/jd_config.json",
                   help="JD config JSON path")
    p.add_argument("--artifacts-dir", default="artifacts",
                   help="Directory with jd_embedding.npy and model_cache/")
    p.add_argument("--top-k", type=int, default=500,
                   help="Candidates carried from Stage 2 into Stage 3")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.time()

    _banner("Redrob Candidate Ranker")
    _print(f"Candidates : {args.candidates}")
    _print(f"Output     : {args.out}")
    _print(f"Config     : {args.config}")

    # ── Load JD config ────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        _error(f"JD config not found: {config_path}\nRun: python generate_jd_artifacts.py")

    with open(config_path, encoding="utf-8") as f:
        jd_config = json.load(f)

    n_must = len(jd_config.get("must_have_skills", []))
    _print(f"JD config  : loaded  ({n_must} must-have skills)")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Stream + Hard Filter
    # ═════════════════════════════════════════════════════════════════════════
    _section("Stage 1 — Streaming & Hard Filter")

    from ranker.ingest import stream_candidates, extract_features

    passed  = []
    total   = 0
    t_s1    = time.time()

    for candidate in stream_candidates(args.candidates):
        total += 1
        feats  = extract_features(candidate, jd_config)
        if feats["hard_filter_score"] > 0.0:
            passed.append(feats)
        if total % 10_000 == 0:
            _print(f"  {total:>7,} processed  |  {len(passed):>6,} passed  |  "
                   f"{time.time() - t_s1:.1f}s")

    t1 = time.time()
    pass_rate = 100 * len(passed) / max(total, 1)
    _print(f"  Done: {len(passed):,}/{total:,} passed ({pass_rate:.1f}%)  "
           f"[{t1 - t0:.1f}s]")

    if len(passed) < 100:
        _error("Fewer than 100 candidates passed Stage 1. "
               "Check jd_config.json disqualifier rules.")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Multi-Signal Composite Scoring
    # ═════════════════════════════════════════════════════════════════════════
    _section("Stage 2 — Multi-Signal Scoring")

    from ranker.stage2_score import compute_composite_score

    for feats in passed:
        feats["composite_score"] = compute_composite_score(feats, jd_config)

    passed.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
    top_k = passed[:args.top_k]

    t2 = time.time()
    _print(f"  Top {len(top_k):,} selected  |  score range "
           f"[{top_k[-1]['composite_score']:.3f}, {top_k[0]['composite_score']:.3f}]  "
           f"[{t2 - t1:.1f}s]")

    # Log score breakdown for the #1 candidate
    if top_k:
        best = top_k[0]
        ss = best.get("sub_scores", {})
        _print(f"  #1 sub-scores — role_fit:{ss.get('role_fit',0):.3f}  "
               f"tech:{ss.get('tech_depth',0):.3f}  "
               f"behav:{ss.get('behavioral',0):.3f}  "
               f"ctx:{ss.get('context',0):.3f}")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Semantic Re-ranking
    # ═════════════════════════════════════════════════════════════════════════
    _section("Stage 3 — Semantic Re-ranking")

    from ranker.stage3_semantic import semantic_rerank

    top_100 = semantic_rerank(top_k, jd_config, args.artifacts_dir)

    t3 = time.time()
    _print(f"  Top 100 selected  |  final score range "
           f"[{top_100[-1]['final_score']:.3f}, {top_100[0]['final_score']:.3f}]  "
           f"[{t3 - t2:.1f}s]")

    # ═════════════════════════════════════════════════════════════════════════
    # Honeypot detection
    # ═════════════════════════════════════════════════════════════════════════
    _section("Honeypot Detection")

    from ranker.honeypot import detect_and_demote_honeypots

    top_100   = detect_and_demote_honeypots(top_100)
    n_suspect = sum(1 for c in top_100 if c.get("honeypot_score", 0) >= 3)
    honeypot_rate = 100 * n_suspect / 100
    _print(f"  {n_suspect} suspect(s) demoted  |  estimated honeypot rate ≤ {honeypot_rate:.0f}%")
    if n_suspect > 9:
        _print("  WARNING: >9 suspects detected — review scoring rules.")

    # ═════════════════════════════════════════════════════════════════════════
    # Reasoning generation
    # ═════════════════════════════════════════════════════════════════════════
    _section("Reasoning Generation")

    from ranker.reasoning import generate_reasoning

    for cand in top_100:
        cand["reasoning"] = generate_reasoning(cand, jd_config)
    _print(f"  Generated {len(top_100)} reasoning strings")

    # ═════════════════════════════════════════════════════════════════════════
    # Output
    # ═════════════════════════════════════════════════════════════════════════
    _section("Output")

    from ranker.output import normalize_scores, write_submission

    top_100 = normalize_scores(top_100)
    write_submission(top_100, args.out)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    _banner("Complete")
    _print(f"  Total time  : {total_time:.1f}s")
    _print(f"  Stage 1     : {t1 - t0:.1f}s  ({total:,} candidates streamed)")
    _print(f"  Stage 2     : {t2 - t1:.1f}s  ({len(passed):,} scored)")
    _print(f"  Stage 3     : {t3 - t2:.1f}s  (top {len(top_k):,} re-ranked)")
    _print(f"  Output file : {args.out}")
    _print()
    _print(f"  Top 5 candidates:")
    for c in top_100[:5]:
        _print(f"    #{c['rank']:>3}  {c['candidate_id']}  "
               f"score={c['score']:.4f}  {c['current_title'][:40]}")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def _section(msg: str):
    print(f"\n[{msg}]")


def _print(msg: str = ""):
    print(msg)


def _error(msg: str):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
