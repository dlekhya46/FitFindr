"""
Unit tests for Stage 2 multi-dimensional scoring.

Verifies that strong candidates score higher than weak ones on each dimension,
and that the anti-stuffing logic catches keyword stuffers.

Run: python -m pytest tests/test_stage2.py -v
"""

import sys
from pathlib import Path
FITFINDR = Path(__file__).parent.parent
sys.path.insert(0, str(FITFINDR))

import json
import pytest
from ranker.ingest import extract_features
from ranker.stage2_score import compute_composite_score

CONFIG_PATH = FITFINDR / "artifacts" / "jd_config.json"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _score(candidate, config):
    feats = extract_features(candidate, config)
    feats["composite_score"] = compute_composite_score(feats, config)
    return feats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_signals(**overrides):
    base = {
        "profile_completeness_score": 80.0,
        "signup_date": "2025-01-01",
        "last_active_date": "2026-05-15",
        "open_to_work_flag": True,
        "profile_views_received_30d": 30,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.70,
        "avg_response_time_hours": 10.0,
        "skill_assessment_scores": {"Python": 80.0, "embeddings": 75.0},
        "connection_count": 400,
        "endorsements_received": 60,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 65.0,
        "search_appearance_30d": 200,
        "saved_by_recruiters_30d": 10,
        "interview_completion_rate": 0.85,
        "offer_acceptance_rate": 0.80,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    base.update(overrides)
    return base


def _make(title, yoe, company, career_desc, skills_list, country="India",
          location="Bangalore", signals_override=None):
    signals = _base_signals(**(signals_override or {}))
    return {
        "candidate_id": "CAND_TEST",
        "profile": {
            "anonymized_name": "Test",
            "headline": title,
            "summary": career_desc[:200],
            "location": location,
            "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": company,
            "current_company_size": "201-500",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": company,
                "title": title,
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": int(yoe * 12),
                "is_current": True,
                "industry": "AI/ML",
                "company_size": "201-500",
                "description": career_desc,
            }
        ],
        "education": [
            {"institution": "IIT Delhi", "degree": "B.Tech",
             "field_of_study": "CS", "start_year": 2014,
             "end_year": 2018, "tier": "tier_1"}
        ],
        "skills": [
            {"name": s[0], "proficiency": s[1], "endorsements": s[2], "duration_months": s[3]}
            for s in skills_list
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": signals,
    }


# ── Core ranking tests ────────────────────────────────────────────────────────

def test_strong_ai_engineer_outscores_weak(cfg):
    """A candidate with production AI experience should score higher than one without."""
    strong = _make(
        "ML Engineer", 7.0, "ProductAI",
        "Deployed FAISS-based embeddings retrieval to production for 5M users. "
        "Built hybrid search with Elasticsearch. A/B tested ranking changes using NDCG. "
        "Fine-tuned BERT for semantic search. Vector database Milvus. Shipped to production.",
        [("Python", "expert", 80, 72), ("embeddings", "expert", 60, 48),
         ("FAISS", "advanced", 40, 36), ("Elasticsearch", "advanced", 30, 30),
         ("NDCG", "advanced", 20, 24)],
    )
    weak = _make(
        "ML Engineer", 7.0, "Corp",
        "Worked on machine learning research projects. Explored transformer models in notebooks.",
        [("Python", "intermediate", 10, 24), ("NLP", "beginner", 5, 12)],
    )
    strong_feats = _score(strong, cfg)
    weak_feats   = _score(weak, cfg)
    assert strong_feats["composite_score"] > weak_feats["composite_score"], \
        "Strong production AI engineer should outscore weak candidate"


def test_keyword_stuffer_scores_lower_than_genuine(cfg):
    """Anti-stuffing: skills listed but not in career history should be penalized."""
    # Stuffer: every AI keyword in skills, but career is HR work
    stuffer = _make(
        "HR Manager", 6.0, "HRCorp",
        "Managed recruitment processes, employee onboarding, HR operations and payroll.",
        [("embeddings", "expert", 50, 60), ("FAISS", "expert", 40, 48),
         ("Python", "expert", 60, 60), ("Pinecone", "expert", 30, 36),
         ("NDCG", "advanced", 20, 24), ("LLM", "expert", 45, 48)],
    )
    # Genuine: fewer skills, but they appear in actual work descriptions
    genuine = _make(
        "ML Engineer", 5.0, "TechStart",
        "Deployed semantic search using sentence transformers and FAISS index. "
        "Production recommendation system with 1M daily active users. "
        "Measured quality with NDCG and ran A/B experiments.",
        [("Python", "expert", 60, 60), ("embeddings", "advanced", 30, 36),
         ("FAISS", "advanced", 25, 30), ("recommendation", "advanced", 20, 24)],
    )
    stuffer_feats = _score(stuffer, cfg)
    genuine_feats = _score(genuine, cfg)
    assert genuine_feats["composite_score"] > stuffer_feats["composite_score"], \
        "Genuine AI engineer should outscore keyword stuffer"


def test_experience_sweet_spot_scores_highest(cfg):
    """YoE 6-8 should score higher on context_fit than 1yr or 20yr."""
    def make_yoe(yoe):
        return _make(
            "ML Engineer", yoe, "TechCo",
            "Built production embedding retrieval system. Deployed to 1M users. NDCG A/B tests.",
            [("Python", "expert", 50, int(yoe*10)), ("embeddings", "advanced", 30, 30)],
        )

    feats_sweet  = _score(make_yoe(7.0),  cfg)
    feats_junior = _score(make_yoe(1.5),  cfg)
    feats_senior = _score(make_yoe(20.0), cfg)

    assert feats_sweet["sub_scores"]["context"] >= feats_junior["sub_scores"]["context"]
    assert feats_sweet["sub_scores"]["context"] >= feats_senior["sub_scores"]["context"]


def test_high_response_rate_scores_higher_behavioral(cfg):
    """Higher recruiter response rate should yield higher behavioral score."""
    cand_high = _make(
        "ML Engineer", 6.0, "Co",
        "Production ML systems with FAISS and embeddings.",
        [("Python", "expert", 50, 60)],
        signals_override={"recruiter_response_rate": 0.95, "interview_completion_rate": 0.95,
                          "notice_period_days": 15},
    )
    cand_low = _make(
        "ML Engineer", 6.0, "Co",
        "Production ML systems with FAISS and embeddings.",
        [("Python", "expert", 50, 60)],
        signals_override={"recruiter_response_rate": 0.05, "interview_completion_rate": 0.20,
                          "notice_period_days": 120},
    )
    feats_high = _score(cand_high, cfg)
    feats_low  = _score(cand_low,  cfg)
    assert feats_high["sub_scores"]["behavioral"] > feats_low["sub_scores"]["behavioral"], \
        "High engagement signals should yield higher behavioral score"


def test_preferred_city_scores_higher_context(cfg):
    """Candidate in Pune/Noida should score higher on context than candidate in London."""
    pune   = _make("ML Engineer", 7.0, "Co", "Production ML.", [("Python","expert",40,48)],
                   country="India", location="Pune")
    london = _make("ML Engineer", 7.0, "Co", "Production ML.", [("Python","expert",40,48)],
                   country="UK", location="London", signals_override={"willing_to_relocate": False})

    feats_pune   = _score(pune,   cfg)
    feats_london = _score(london, cfg)
    assert feats_pune["sub_scores"]["context"] > feats_london["sub_scores"]["context"]


def test_composite_score_in_valid_range(cfg):
    """Composite score must always be in [0, 1]."""
    for yoe in [0.5, 3.0, 7.0, 15.0]:
        cand  = _make("ML Engineer", yoe, "Co",
                      "Machine learning and embeddings work.", [("Python","intermediate",10,12)])
        feats = _score(cand, cfg)
        assert 0.0 <= feats["composite_score"] <= 1.0, \
            f"Score {feats['composite_score']} out of range for yoe={yoe}"


if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        c = json.load(f)

    tests = [
        ("Strong AI engineer outscores weak",          lambda: test_strong_ai_engineer_outscores_weak(c)),
        ("Keyword stuffer scored lower than genuine",  lambda: test_keyword_stuffer_scores_lower_than_genuine(c)),
        ("Experience sweet spot 6-8yr best context",  lambda: test_experience_sweet_spot_scores_highest(c)),
        ("High response rate → better behavioral",    lambda: test_high_response_rate_scores_higher_behavioral(c)),
        ("Preferred city → better context",           lambda: test_preferred_city_scores_higher_context(c)),
        ("Composite always in [0,1]",                 lambda: test_composite_score_in_valid_range(c)),
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
