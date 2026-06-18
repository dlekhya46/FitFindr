"""
Candidate ingestion: streaming JSONL reader + feature extraction.

Streams candidates one line at a time to stay within the 16 GB RAM constraint.
Extracts five feature domains per candidate into a flat dict:
  identity, career, skills, behavioral, context
"""

import math
import json
import gzip
from datetime import date
from pathlib import Path
from typing import Iterator

# ─────────────────────────────────────────────────────────────────────────────
# Streaming reader
# ─────────────────────────────────────────────────────────────────────────────

def stream_candidates(path: str) -> Iterator[dict]:
    """Yield one candidate dict per line from .jsonl or .jsonl.gz."""
    p = Path(path)
    if p.suffix == ".gz":
        opener, mode = gzip.open, "rt"
    else:
        opener, mode = open, "rb"

    with opener(p, mode) as f:
        for line in f:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ─────────────────────────────────────────────────────────────────────────────
# Top-level feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(candidate: dict, jd_config: dict) -> dict:
    """
    Transform a raw candidate record into a flat feature dict.
    Stores _raw for downstream reasoning generation.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])

    feats: dict = {
        "candidate_id": candidate["candidate_id"],
        "_raw": candidate,          # retained for reasoning; not a model input
    }

    # ── Identity ──────────────────────────────────────────────────────────────
    feats["current_title"]       = profile.get("current_title", "")
    feats["years_of_experience"] = float(profile.get("years_of_experience", 0))
    feats["current_company"]     = profile.get("current_company", "")
    feats["current_industry"]    = profile.get("current_industry", "")
    feats["location"]            = profile.get("location", "")
    feats["country"]             = profile.get("country", "")
    feats["summary"]             = profile.get("summary", "")
    feats["headline"]            = profile.get("headline", "")

    # ── Domain-specific feature groups ───────────────────────────────────────
    feats.update(_career_features(career, jd_config))
    feats.update(_skill_features(skills, career, jd_config))
    feats.update(_behavioral_features(signals))
    feats.update(_context_features(profile, signals))

    # ── Hard-filter penalty (0 = disqualified, 0<x≤1 = penalty weight) ───────
    feats["hard_filter_score"] = _hard_filter_score(feats, jd_config)

    # ── Text blob for Stage-3 semantic embedding ──────────────────────────────
    feats["candidate_text"] = _build_text(candidate)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Career features
# ─────────────────────────────────────────────────────────────────────────────

def _career_features(career: list, jd_config: dict) -> dict:
    ai_title_kws  = jd_config.get("ai_ml_title_keywords", [])
    consulting_set = {c.lower() for c in jd_config["hard_disqualifiers"]["consulting_firms"]}
    prod_kws      = jd_config.get("production_evidence_keywords", [])
    research_kws  = jd_config.get("research_anti_patterns", [])

    total_months       = 0
    ai_months          = 0
    consulting_months  = 0
    product_months     = 0
    current_role_is_ai = False

    for job in career:
        title   = job.get("title", "").lower()
        company = job.get("company", "").lower()
        desc    = job.get("description", "").lower()
        dur     = int(job.get("duration_months", 0))
        size    = job.get("company_size", "")
        is_curr = bool(job.get("is_current", False))

        total_months += dur

        # AI/ML role: check title OR dense ML signals in description
        title_is_ai = any(kw in title for kw in ai_title_kws)
        ml_desc_hits = sum(1 for kw in [
            "machine learning", "deep learning", "neural network", "nlp",
            "recommendation", "ranking", "retrieval", "embedding", "llm",
            "transformer", "vector search", "fine-tun"
        ] if kw in desc)
        if title_is_ai or ml_desc_hits >= 3:
            ai_months += dur
            if is_curr:
                current_role_is_ai = True

        # Consulting detection
        is_consulting = any(cf in company for cf in consulting_set)
        if is_consulting:
            consulting_months += dur

        # Product company: not consulting + small-to-large startup/scaleup
        if not is_consulting and size in ("51-200", "201-500", "501-1000", "1001-5000"):
            product_months += dur

    # Production evidence score from all career descriptions
    all_desc = " ".join(j.get("description", "") for j in career).lower()
    prod_hits     = sum(1 for kw in prod_kws   if kw in all_desc)
    research_hits = sum(1 for kw in research_kws if kw in all_desc)
    prod_raw      = (prod_hits - 0.35 * research_hits) / max(len(prod_kws) * 0.25, 1)
    prod_evidence = max(0.0, min(prod_raw, 1.0))

    consulting_frac    = consulting_months / max(total_months, 1)
    # Penalty is softer when candidate also has product-company experience
    product_mitigates  = product_months / max(total_months, 1)
    consulting_penalty = max(0.30, 1.0 - consulting_frac * (1.0 - product_mitigates * 0.5))

    return {
        "total_career_months":    total_months,
        "ai_career_months":       ai_months,
        "ai_career_fraction":     ai_months / max(total_months, 1),
        "consulting_fraction":    consulting_frac,
        "consulting_penalty":     consulting_penalty,
        "product_months":         product_months,
        "prod_evidence_score":    prod_evidence,
        "current_role_is_ai":     current_role_is_ai,
        "career_text":            all_desc,
        "num_roles":              len(career),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skill features
# ─────────────────────────────────────────────────────────────────────────────

_PROFICIENCY_W = {"beginner": 0.25, "intermediate": 0.50, "advanced": 0.80, "expert": 1.00}

# Hand-coded variant map for career corroboration
_SKILL_VARIANTS: dict = {
    "embeddings":      ["embedding", "vector", "semantic search", "similarity search"],
    "faiss":           ["vector search", "ann", "similarity search", "approximate nearest"],
    "milvus":          ["vector database", "vector db", "vector store"],
    "pinecone":        ["vector database", "vector db", "vector store"],
    "qdrant":          ["vector database", "vector db"],
    "weaviate":        ["vector database", "vector db"],
    "chromadb":        ["vector database", "vector db"],
    "nlp":             ["natural language", "text processing", "transformer", "language model"],
    "pytorch":         ["deep learning", "neural network", "model training", "torch"],
    "tensorflow":      ["deep learning", "neural network", "model training"],
    "python":          ["pandas", "numpy", "scripting", "python"],
    "lora":            ["fine-tuning", "finetuning", "peft", "parameter efficient"],
    "qlora":           ["fine-tuning", "quantization", "peft"],
    "recommendation":  ["recommender", "collaborative filtering", "item ranking", "personalization"],
    "rag":             ["retrieval augmented", "retrieval-augmented", "document retrieval"],
    "bm25":            ["sparse retrieval", "inverted index", "keyword search", "tf-idf"],
    "reranking":       ["cross-encoder", "two-stage retrieval", "rerank", "listwise"],
    "ndcg":            ["ranking evaluation", "relevance", "search quality", "offline eval"],
    "elasticsearch":   ["search engine", "inverted index", "full-text search", "lucene"],
    "opensearch":      ["search engine", "full-text search", "elasticsearch"],
    "kafka":           ["message queue", "streaming", "event stream", "pub-sub"],
    "spark":           ["distributed processing", "pyspark", "big data", "mapreduce"],
}


def _get_variants(name: str) -> list:
    name_l = name.lower()
    for key, variants in _SKILL_VARIANTS.items():
        if key in name_l:
            return variants
    return []


def _skill_features(skills: list, career: list, jd_config: dict) -> dict:
    must_have_specs = jd_config.get("must_have_skills", [])
    nice_set = {s.lower() for s in jd_config.get("nice_to_have_skills", [])}

    career_text = " ".join(
        j.get("description", "") + " " + j.get("title", "")
        for j in career
    ).lower()

    must_matched  = 0
    nice_matched  = 0
    total_w       = 0.0
    top_skills    = []

    for skill in skills:
        name  = skill.get("name", "")
        namel = name.lower()
        prof  = skill.get("proficiency", "beginner")
        end   = int(skill.get("endorsements", 0))
        dur   = int(skill.get("duration_months", 1))

        # Base weight: proficiency × social proof × usage depth
        base_w = (
            _PROFICIENCY_W.get(prof, 0.25)
            * math.log1p(end)
            * math.sqrt(max(dur, 1))
        )

        # Anti-stuffing: halve weight if skill not corroborated in any job description
        corroborated = namel in career_text or any(
            v in career_text for v in _get_variants(namel)
        )
        corr_mult = 1.0 if corroborated else 0.45

        effective_w = base_w * corr_mult

        # Must-have match
        is_must = False
        for spec in must_have_specs:
            vlist = [v.lower() for v in spec.get("variants", [spec["canonical"]])]
            if any(v in namel for v in vlist) or any(namel in v for v in vlist if len(v) > 4):
                is_must = True
                must_matched += 1
                break

        # Nice-to-have match
        is_nice = any(n in namel for n in nice_set)
        if is_nice:
            nice_matched += 1

        multiplier = 2.5 if is_must else (1.3 if is_nice else 0.4)
        total_w += effective_w * multiplier

        if is_must or effective_w > 0.8:
            top_skills.append({"name": name, "proficiency": prof, "score": effective_w})

    # Normalize: max possible if all must-haves matched at expert+long+well-endorsed
    normalizer = max(len(must_have_specs) * 2.5 * 3.0, 1.0)

    return {
        "skill_weighted_score": min(total_w / normalizer, 1.0),
        "must_have_matched":    min(must_matched, len(must_have_specs)),
        "nice_have_matched":    nice_matched,
        "top_skills":           sorted(top_skills, key=lambda x: x["score"], reverse=True)[:6],
        "total_skills":         len(skills),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral features
# ─────────────────────────────────────────────────────────────────────────────

def _behavioral_features(signals: dict) -> dict:
    today = date.today()

    raw_last = signals.get("last_active_date", "2020-01-01")
    try:
        last_active = date.fromisoformat(raw_last)
    except (ValueError, TypeError):
        last_active = date(2020, 1, 1)

    days_inactive = max((today - last_active).days, 0)
    recency_decay = math.exp(-days_inactive / 90.0)   # half-life = 90 days

    np_days = int(signals.get("notice_period_days", 90))
    if   np_days <= 15: np_score = 1.00
    elif np_days <= 30: np_score = 0.90
    elif np_days <= 60: np_score = 0.65
    elif np_days <= 90: np_score = 0.45
    else:               np_score = 0.20

    gh_raw   = signals.get("github_activity_score", -1)
    gh_score = float(gh_raw) / 100.0 if gh_raw >= 0 else 0.25   # neutral if not linked

    rr  = float(signals.get("recruiter_response_rate", 0.0))
    icr = float(signals.get("interview_completion_rate", 0.0))

    # Skill assessment scores from Redrob platform
    assess_dict = signals.get("skill_assessment_scores", {})
    if assess_dict:
        assess_avg = sum(assess_dict.values()) / len(assess_dict)
        assess_score = assess_avg / 100.0
    else:
        assess_score = 0.35   # neutral: not assessed ≠ bad

    return {
        "days_inactive":             days_inactive,
        "recency_decay":             recency_decay,
        "recruiter_response_rate":   rr,
        "interview_completion_rate": icr,
        "np_score":                  np_score,
        "notice_period_days":        np_days,
        "open_to_work":              bool(signals.get("open_to_work_flag", False)),
        "gh_score":                  gh_score,
        "github_activity_score":     float(gh_raw),
        "assess_score":              assess_score,
        "profile_completeness":      float(signals.get("profile_completeness_score", 0.0)),
        "saved_by_recruiters_30d":   int(signals.get("saved_by_recruiters_30d", 0)),
        "offer_acceptance_rate":     float(signals.get("offer_acceptance_rate", -1)),
        "expected_salary_min":       float(signals.get("expected_salary_range_inr_lpa", {}).get("min", 0)),
        "expected_salary_max":       float(signals.get("expected_salary_range_inr_lpa", {}).get("max", 0)),
        "preferred_work_mode":       signals.get("preferred_work_mode", "flexible"),
        "willing_to_relocate":       bool(signals.get("willing_to_relocate", False)),
        "verified_email":            bool(signals.get("verified_email", False)),
        "verified_phone":            bool(signals.get("verified_phone", False)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Context features
# ─────────────────────────────────────────────────────────────────────────────

_PREFERRED_CITIES = frozenset([
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "new delhi", "delhi ncr", "navi mumbai",
    "thane", "greater noida",
])

_EDU_TIER_MAP = {
    "tier_1": 1.00, "tier_2": 0.80,
    "tier_3": 0.60, "tier_4": 0.40, "unknown": 0.50,
}


def _context_features(profile: dict, signals: dict) -> dict:
    loc_lower     = profile.get("location", "").lower()
    country       = profile.get("country", "").lower()
    will_relocate = bool(signals.get("willing_to_relocate", False))

    in_preferred = any(city in loc_lower for city in _PREFERRED_CITIES)
    in_india     = country in ("india", "in")

    if   in_preferred:                 loc_score = 1.00
    elif in_india and will_relocate:   loc_score = 0.75
    elif in_india:                     loc_score = 0.50
    elif will_relocate:                loc_score = 0.35
    else:                              loc_score = 0.10

    yoe = float(profile.get("years_of_experience", 0))
    if   6.0 <= yoe <= 8.0:  yoe_score = 1.00
    elif 5.0 <= yoe <= 9.0:  yoe_score = 0.85
    elif 4.0 <= yoe <= 12.0: yoe_score = 0.65
    elif 3.0 <= yoe <= 15.0: yoe_score = 0.45
    else:                    yoe_score = 0.20

    edu = profile.get("education", []) if "education" in profile else []
    best_edu = max(
        (_EDU_TIER_MAP.get(e.get("tier", "unknown"), 0.50) for e in edu),
        default=0.40
    )

    return {
        "location_score":    loc_score,
        "yoe_fit_score":     yoe_score,
        "edu_tier_score":    best_edu,
        "in_preferred_city": in_preferred,
        "in_india":          in_india,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hard filter score
# ─────────────────────────────────────────────────────────────────────────────

def _hard_filter_score(feats: dict, jd_config: dict) -> float:
    """
    Returns 0.0  → candidate is eliminated (definite non-fit).
    Returns >0.0 → penalty multiplier applied to composite score.
    """
    disq          = jd_config["hard_disqualifiers"]
    mismatch_list = [t.lower() for t in disq.get("domain_mismatch_titles", [])]
    title_lower   = feats["current_title"].lower()

    # Hard disqualify: too junior
    if feats["years_of_experience"] < float(disq.get("min_years_experience", 2.0)):
        return 0.0

    # Hard disqualify: completely unreachable
    max_inactive = int(disq.get("max_inactive_days_hard", 365))
    if feats["days_inactive"] > max_inactive and not feats["open_to_work"]:
        return 0.0

    # Domain mismatch: non-AI title AND essentially zero AI career history
    is_mismatch = any(t in title_lower for t in mismatch_list)
    if is_mismatch and feats["ai_career_fraction"] < 0.05:
        return 0.0

    # Build penalty multiplier for candidates who pass but need downweighting
    penalty = 1.0

    if is_mismatch:
        # Has some AI experience but current role is off-domain
        penalty *= 0.35

    # Consulting-only career (already encoded as consulting_penalty in career features)
    penalty *= feats["consulting_penalty"]

    # Hard location miss: not India + won't relocate
    if not feats["in_india"] and not feats["willing_to_relocate"]:
        penalty *= 0.25

    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# Text builder for semantic embedding
# ─────────────────────────────────────────────────────────────────────────────

def _build_text(candidate: dict) -> str:
    """Concatenate key text fields for all-MiniLM-L6-v2 encoding."""
    profile = candidate.get("profile", {})
    parts   = [
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]
    # Current + most recent roles first
    career = sorted(
        candidate.get("career_history", []),
        key=lambda j: (j.get("is_current", False), j.get("duration_months", 0)),
        reverse=True,
    )
    for job in career[:4]:
        parts.append(
            f"{job.get('title','')} at {job.get('company','')}: {job.get('description','')}"
        )
    skills_str = ", ".join(s["name"] for s in candidate.get("skills", []))
    parts.append(f"Skills: {skills_str}")

    text = " ".join(p for p in parts if p)
    return text[:3000]   # cap to avoid embedding token overflow
