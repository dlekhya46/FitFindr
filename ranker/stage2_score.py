"""
Stage 2: Multi-dimensional weighted composite scoring.

Four independent dimensions → weighted composite → apply hard-filter penalty.

  role_fit              (35%)  Skills + career trajectory + AI domain coverage
  technical_depth       (30%)  Production evidence + GitHub + assessments
  behavioral_availability (20%)  Engagement signals + recency + notice period
  context_fit           (15%)  Location + YoE + education tier
"""


def compute_composite_score(feats: dict, jd_config: dict) -> float:
    """
    Compute and store sub-scores, then return the weighted composite.
    Writes sub_scores back into feats for use by the reasoning generator.
    """
    weights = jd_config["scoring_weights"]

    rf  = _role_fit(feats, jd_config)
    td  = _technical_depth(feats)
    ba  = _behavioral_availability(feats)
    ctx = _context_fit(feats)

    feats["sub_scores"] = {
        "role_fit":    round(rf,  4),
        "tech_depth":  round(td,  4),
        "behavioral":  round(ba,  4),
        "context":     round(ctx, 4),
    }

    raw = (
        weights["role_fit"]               * rf
        + weights["technical_depth"]      * td
        + weights["behavioral_availability"] * ba
        + weights["context_fit"]          * ctx
    )

    # Apply hard-filter penalty (consulting fraction, domain partial mismatch, etc.)
    return min(raw * feats["hard_filter_score"], 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Role Fit  (35%)
# ─────────────────────────────────────────────────────────────────────────────

def _role_fit(feats: dict, jd_config: dict) -> float:
    total_must = len(jd_config["must_have_skills"])

    # Skill match (weighted by proficiency, endorsements, duration, corroboration)
    skill_score = feats["skill_weighted_score"]

    # Must-have coverage ratio — the single most important role-fit signal
    must_coverage = feats["must_have_matched"] / max(total_must, 1)

    # AI career fraction (cap normalisation at 60% of career = 1.0)
    ai_fraction = min(feats["ai_career_fraction"] / 0.60, 1.0)

    # Nice-to-have overlap
    nice_score = min(feats["nice_have_matched"] / 4.0, 1.0)

    # Bonus: current role is in AI/ML
    current_ai_bonus = 0.10 if feats["current_role_is_ai"] else 0.0

    score = (
        0.30 * skill_score
        + 0.40 * must_coverage
        + 0.20 * ai_fraction
        + 0.10 * nice_score
    ) + current_ai_bonus

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Technical Depth  (30%)
# ─────────────────────────────────────────────────────────────────────────────

def _technical_depth(feats: dict) -> float:
    # Production evidence mined from career descriptions
    prod  = feats["prod_evidence_score"]

    # GitHub activity: -1 means not linked → neutral 0.25
    gh = feats["gh_score"]

    # Redrob platform skill assessment scores
    assess = feats["assess_score"]

    # Profile completeness is a weak proxy for engagement quality
    completeness = feats["profile_completeness"] / 100.0

    score = (
        0.45 * prod
        + 0.25 * gh
        + 0.20 * assess
        + 0.10 * completeness
    )
    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral Availability  (20%)
# ─────────────────────────────────────────────────────────────────────────────

def _behavioral_availability(feats: dict) -> float:
    score = (
        0.30 * feats["recruiter_response_rate"]
        + 0.25 * feats["recency_decay"]
        + 0.20 * feats["interview_completion_rate"]
        + 0.15 * feats["np_score"]
        + 0.05 * float(feats["open_to_work"])
        + 0.05 * feats["gh_score"]
    )
    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Context Fit  (15%)
# ─────────────────────────────────────────────────────────────────────────────

def _context_fit(feats: dict) -> float:
    score = (
        0.50 * feats["location_score"]
        + 0.35 * feats["yoe_fit_score"]
        + 0.15 * feats["edu_tier_score"]
    )
    return min(score, 1.0)
