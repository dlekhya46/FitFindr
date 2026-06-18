"""
Unit tests for the honeypot detection module.

Run: python -m pytest tests/test_honeypot.py -v
"""

import sys
from pathlib import Path
FITFINDR = Path(__file__).parent.parent
sys.path.insert(0, str(FITFINDR))

import json
import pytest
from ranker.honeypot import _honeypot_score, detect_and_demote_honeypots


def _base_feats(overrides=None):
    """Return a plausible, non-honeypot feature dict."""
    raw = {
        "redrob_signals": {
            "profile_completeness_score": 78.0,
            "github_activity_score": 45.0,
            "recruiter_response_rate": 0.65,
            "interview_completion_rate": 0.80,
            "offer_acceptance_rate": 0.75,
            "endorsements_received": 40,
            "connection_count": 300,
            "last_active_date": "2026-05-10",
            "saved_by_recruiters_30d": 12,
        },
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 40, "duration_months": 48},
            {"name": "embeddings", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
        ],
        "career_history": [
            {
                "company": "TechCo",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "AI",
                "company_size": "201-500",
                "description": "Deployed retrieval systems.",
            }
        ],
        "education": [
            {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2014, "end_year": 2018, "tier": "tier_1"}
        ],
    }
    feats = {
        "candidate_id": "CAND_0000001",
        "_raw": raw,
        "ai_career_fraction": 0.80,
        "honeypot_score": 0,
    }
    if overrides:
        feats.update(overrides)
    return feats


# ── Normal candidate should have low honeypot score ──────────────────────────

def test_normal_candidate_low_score():
    feats = _base_feats()
    score = _honeypot_score(feats)
    assert score < 3, f"Normal candidate should not be flagged (score={score})"


# ── Metric perfection ─────────────────────────────────────────────────────────

def test_metric_perfection_flagged():
    feats = _base_feats()
    feats["_raw"]["redrob_signals"].update({
        "profile_completeness_score": 100.0,
        "github_activity_score": 100.0,
        "recruiter_response_rate": 1.0,
        "interview_completion_rate": 1.0,
        "offer_acceptance_rate": 1.0,
    })
    score = _honeypot_score(feats)
    assert score >= 3, f"Metric perfection should be flagged (score={score})"


# ── Endorsement anomaly ───────────────────────────────────────────────────────

def test_endorsement_anomaly_flagged():
    feats = _base_feats()
    feats["_raw"]["redrob_signals"]["endorsements_received"] = 5000
    feats["_raw"]["redrob_signals"]["connection_count"] = 10
    score = _honeypot_score(feats)
    assert score >= 1, f"Endorsement anomaly should contribute to score (score={score})"


# ── Skill-career contradiction ────────────────────────────────────────────────

def test_skill_career_contradiction_flagged():
    feats = _base_feats()
    feats["ai_career_fraction"] = 0.0   # no AI/ML career history
    feats["_raw"]["skills"] = [
        {"name": "machine learning", "proficiency": "expert", "endorsements": 80, "duration_months": 60},
        {"name": "deep learning",    "proficiency": "expert", "endorsements": 70, "duration_months": 60},
        {"name": "nlp",              "proficiency": "expert", "endorsements": 65, "duration_months": 60},
    ]
    score = _honeypot_score(feats)
    assert score >= 2, f"Skill-career contradiction should be flagged (score={score})"


# ── Date impossibility ────────────────────────────────────────────────────────

def test_impossible_career_dates_flagged():
    feats = _base_feats()
    feats["_raw"]["career_history"] = [
        {
            "company": "Corp",
            "title": "Engineer",
            "start_date": "2022-06-01",
            "end_date": "2021-01-01",     # ends before it starts
            "duration_months": 18,
            "is_current": False,
            "industry": "Tech",
            "company_size": "201-500",
            "description": "...",
        }
    ]
    score = _honeypot_score(feats)
    assert score >= 2, f"Impossible career dates should be flagged (score={score})"


# ── Future last_active ────────────────────────────────────────────────────────

def test_future_last_active_flagged():
    feats = _base_feats()
    feats["_raw"]["redrob_signals"]["last_active_date"] = "2099-12-31"
    score = _honeypot_score(feats)
    assert score >= 2, f"Future last_active_date should be flagged (score={score})"


# ── Demotion logic ────────────────────────────────────────────────────────────

def test_suspects_demoted_to_bottom():
    clean_candidates = [_base_feats({"candidate_id": f"CAND_{i:07d}", "final_score": 0.9 - i * 0.01})
                        for i in range(91)]
    suspect = _base_feats({"candidate_id": "CAND_SUSPECT", "final_score": 0.95})
    suspect["_raw"]["redrob_signals"].update({
        "profile_completeness_score": 100.0,
        "github_activity_score": 100.0,
        "recruiter_response_rate": 1.0,
        "interview_completion_rate": 1.0,
        "offer_acceptance_rate": 1.0,
    })

    top_100 = clean_candidates + [suspect]
    result  = detect_and_demote_honeypots(top_100)

    # Suspect should be in the last 10 ranks
    suspect_rank = next(i for i, c in enumerate(result) if c["candidate_id"] == "CAND_SUSPECT")
    assert suspect_rank >= 90, f"Suspect should be in bottom 10, got index {suspect_rank}"


def test_honeypot_rate_below_10_percent():
    """Even if 9 suspects are detected, they're < 10% of top-100."""
    clean = [_base_feats({"candidate_id": f"CAND_{i:07d}", "final_score": 0.9}) for i in range(91)]
    suspects = []
    for j in range(9):
        s = _base_feats({"candidate_id": f"CAND_S{j:06d}", "final_score": 0.95})
        s["_raw"]["redrob_signals"].update({
            "profile_completeness_score": 100.0, "github_activity_score": 100.0,
            "recruiter_response_rate": 1.0, "interview_completion_rate": 1.0,
            "offer_acceptance_rate": 1.0,
        })
        suspects.append(s)

    top_100 = clean + suspects
    result  = detect_and_demote_honeypots(top_100)
    n_flagged = sum(1 for c in result if c.get("honeypot_score", 0) >= 3)
    assert n_flagged / 100 <= 0.10, f"Honeypot rate {n_flagged/100:.1%} exceeds 10%"


if __name__ == "__main__":
    tests = [
        ("Normal candidate low score",         test_normal_candidate_low_score),
        ("Metric perfection flagged",          test_metric_perfection_flagged),
        ("Endorsement anomaly flagged",        test_endorsement_anomaly_flagged),
        ("Skill-career contradiction flagged", test_skill_career_contradiction_flagged),
        ("Impossible dates flagged",           test_impossible_career_dates_flagged),
        ("Future last_active flagged",         test_future_last_active_flagged),
        ("Suspects demoted to bottom",         test_suspects_demoted_to_bottom),
        ("Honeypot rate stays below 10%",      test_honeypot_rate_below_10_percent),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
