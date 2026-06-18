"""
Unit tests for Stage 1 hard filter rules.
Tests cover every disqualification path in ingest._hard_filter_score().

Run: python -m pytest tests/test_stage1.py -v
"""

import sys
from pathlib import Path
FITFINDR = Path(__file__).parent.parent
sys.path.insert(0, str(FITFINDR))

import json
import pytest
from ranker.ingest import extract_features

# ── Load the real JD config ───────────────────────────────────────────────────
CONFIG_PATH = FITFINDR / "artifacts" / "jd_config.json"

@pytest.fixture(scope="module")
def jd_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Candidate builder ─────────────────────────────────────────────────────────

def _make_candidate(
    cid="CAND_0000001",
    title="ML Engineer",
    yoe=6.0,
    company="StartupAI",
    country="India",
    location="Bangalore",
    last_active="2026-05-01",
    open_to_work=True,
    willing_to_relocate=True,
    notice_period=30,
    career=None,
    skills=None,
):
    career = career or [
        {
            "company": company,
            "title": title,
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": int(yoe * 12),
            "is_current": True,
            "industry": "AI/ML",
            "company_size": "51-200",
            "description": (
                "Built and deployed embeddings-based retrieval system to production. "
                "Vector search with FAISS, semantic similarity, NDCG evaluation."
            ),
        }
    ]
    skills = skills or [
        {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
        {"name": "embeddings", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
    ]
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": f"{title} | AI",
            "summary": "ML engineer building retrieval systems in production.",
            "location": location,
            "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": company,
            "current_company_size": "51-200",
            "current_industry": "Technology",
        },
        "career_history": career,
        "education": [
            {"institution": "IIT Delhi", "degree": "B.Tech",
             "field_of_study": "Computer Science", "start_year": 2014,
             "end_year": 2018, "grade": "8.5 CGPA", "tier": "tier_1"}
        ],
        "skills": skills,
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 85.0,
            "signup_date": "2025-01-01",
            "last_active_date": last_active,
            "open_to_work_flag": open_to_work,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.75,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {"Python": 82.0, "embeddings": 78.0},
            "connection_count": 300,
            "endorsements_received": 45,
            "notice_period_days": notice_period,
            "expected_salary_range_inr_lpa": {"min": 25.0, "max": 45.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": willing_to_relocate,
            "github_activity_score": 72.0,
            "search_appearance_30d": 150,
            "saved_by_recruiters_30d": 8,
            "interview_completion_rate": 0.90,
            "offer_acceptance_rate": 0.80,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


# ── Tests: should PASS ────────────────────────────────────────────────────────

def test_strong_ai_engineer_passes(jd_config):
    cand  = _make_candidate(title="ML Engineer", yoe=7.0, company="TechCorp")
    feats = extract_features(cand, jd_config)
    assert feats["hard_filter_score"] > 0.0, "Strong ML Engineer should pass"


def test_data_scientist_passes(jd_config):
    cand  = _make_candidate(title="Data Scientist", yoe=5.5)
    feats = extract_features(cand, jd_config)
    assert feats["hard_filter_score"] > 0.0


def test_senior_engineer_passes(jd_config):
    cand  = _make_candidate(title="Senior Software Engineer", yoe=8.0)
    feats = extract_features(cand, jd_config)
    assert feats["hard_filter_score"] > 0.0


# ── Tests: should FAIL (return 0.0) ──────────────────────────────────────────

def test_too_junior_eliminated(jd_config):
    cand  = _make_candidate(title="ML Engineer", yoe=1.0)
    feats = extract_features(cand, jd_config)
    assert feats["hard_filter_score"] == 0.0, "Under 2 yrs experience should be eliminated"


def test_hr_manager_no_ai_eliminated(jd_config):
    career = [
        {
            "company": "Corp",
            "title": "HR Manager",
            "start_date": "2016-01-01",
            "end_date": None,
            "duration_months": 96,
            "is_current": True,
            "industry": "HR",
            "company_size": "1001-5000",
            "description": "Managed recruitment, onboarding, and HR operations.",
        }
    ]
    skills = [
        {"name": "Recruiting", "proficiency": "expert", "endorsements": 40, "duration_months": 96},
        {"name": "Excel", "proficiency": "advanced", "endorsements": 20, "duration_months": 96},
    ]
    cand  = _make_candidate(title="HR Manager", yoe=8.0, career=career, skills=skills)
    feats = extract_features(cand, jd_config)
    assert feats["hard_filter_score"] == 0.0, "HR Manager with no AI career should be eliminated"


def test_marketing_manager_no_ai_eliminated(jd_config):
    career = [
        {
            "company": "BrandCo",
            "title": "Marketing Manager",
            "start_date": "2018-01-01",
            "end_date": None,
            "duration_months": 72,
            "is_current": True,
            "industry": "Marketing",
            "company_size": "201-500",
            "description": "Led digital marketing campaigns, SEO, content strategy.",
        }
    ]
    cand  = _make_candidate(title="Marketing Manager", yoe=6.0, career=career, skills=[])
    feats = extract_features(cand, jd_config)
    assert feats["hard_filter_score"] == 0.0, "Marketing Manager with no AI should be eliminated"


def test_completely_inactive_eliminated(jd_config):
    cand  = _make_candidate(
        last_active="2023-01-01",   # >2 years inactive
        open_to_work=False,
    )
    feats = extract_features(cand, jd_config)
    assert feats["hard_filter_score"] == 0.0, "Inactive >1yr + not open to work should be eliminated"


# ── Tests: penalties (passes but with reduced score) ─────────────────────────

def test_consulting_only_gets_penalty(jd_config):
    career = [
        {
            "company": "TCS",
            "title": "ML Engineer",
            "start_date": "2018-01-01",
            "end_date": None,
            "duration_months": 72,
            "is_current": True,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": "Built recommendation models. Deployed embeddings retrieval with FAISS.",
        }
    ]
    cand  = _make_candidate(title="ML Engineer", yoe=6.0, company="TCS", career=career)
    feats = extract_features(cand, jd_config)
    assert 0.0 < feats["hard_filter_score"] < 1.0, "TCS-only career should pass with penalty"
    assert feats["consulting_penalty"] < 1.0


def test_outside_india_no_relocate_penalty(jd_config):
    cand  = _make_candidate(country="USA", location="New York", willing_to_relocate=False)
    feats = extract_features(cand, jd_config)
    assert 0.0 < feats["hard_filter_score"] < 0.5, "Outside India + no relocation should be penalized"


def test_outside_india_willing_to_relocate_less_penalty(jd_config):
    cand_no   = _make_candidate(country="USA", location="New York", willing_to_relocate=False)
    cand_yes  = _make_candidate(country="USA", location="New York", willing_to_relocate=True)
    feats_no  = extract_features(cand_no,  jd_config)
    feats_yes = extract_features(cand_yes, jd_config)
    assert feats_yes["hard_filter_score"] > feats_no["hard_filter_score"], \
        "Willing to relocate should have higher score than not willing"


if __name__ == "__main__":
    # Quick smoke test without pytest
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    tests = [
        ("Strong ML Engineer passes",         lambda: test_strong_ai_engineer_passes(cfg)),
        ("Too junior eliminated",              lambda: test_too_junior_eliminated(cfg)),
        ("HR Manager eliminated",             lambda: test_hr_manager_no_ai_eliminated(cfg)),
        ("Marketing Manager eliminated",      lambda: test_marketing_manager_no_ai_eliminated(cfg)),
        ("Completely inactive eliminated",    lambda: test_completely_inactive_eliminated(cfg)),
        ("Consulting penalty applied",        lambda: test_consulting_only_gets_penalty(cfg)),
    ]

    passed_n = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed_n += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")

    print(f"\n{passed_n}/{len(tests)} tests passed")
