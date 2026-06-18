"""
Fact-grounded per-candidate reasoning generator.

Every clause maps directly to a specific field in the candidate record.
No claims are inferred or hallucinated.

Stage 4 (manual review) checks:
  - Specific facts from the profile
  - JD connection
  - Honest concerns
  - No hallucination
  - Variation across entries
  - Rank consistency
"""


def generate_reasoning(feats: dict, jd_config: dict) -> str:
    """
    Build a 1-2 sentence reasoning string for the submission CSV.
    All claims are verified against actual feature values.
    """
    raw     = feats.get("_raw", {})
    profile = raw.get("profile", {})

    parts    = []   # positive signals
    concerns = []   # honest gaps

    # ── Opening: title + YoE ─────────────────────────────────────────────────
    title = profile.get("current_title", "Candidate")
    yoe   = feats["years_of_experience"]
    parts.append(f"{title} with {yoe:.1f} yrs experience")

    # ── AI career fraction ───────────────────────────────────────────────────
    ai_frac = feats["ai_career_fraction"]
    if ai_frac >= 0.60:
        ai_yr = int(feats["ai_career_months"] // 12)
        parts.append(f"{ai_yr}+ yrs in applied AI/ML roles")
    elif ai_frac >= 0.25:
        parts.append(f"partial AI/ML background ({int(ai_frac * 100)}% of career)")

    # ── Must-have skill coverage ─────────────────────────────────────────────
    mh_matched = feats["must_have_matched"]
    total_mh   = len(jd_config["must_have_skills"])
    if mh_matched >= total_mh - 1:
        parts.append(f"strong JD skill coverage ({mh_matched}/{total_mh} must-haves)")

    # ── Top matched skills ───────────────────────────────────────────────────
    top_skills = feats.get("top_skills", [])
    if top_skills:
        names = ", ".join(s["name"] for s in top_skills[:3])
        parts.append(f"key skills: {names}")

    # ── Production evidence ──────────────────────────────────────────────────
    if feats["prod_evidence_score"] >= 0.55:
        parts.append("career descriptions reference production deployment at scale")
    elif feats["prod_evidence_score"] >= 0.30:
        parts.append("some production deployment evidence in career history")

    # ── Behavioral / availability ────────────────────────────────────────────
    rr   = feats["recruiter_response_rate"]
    days = feats["days_inactive"]

    if rr >= 0.65 and days <= 20:
        parts.append(f"highly engaged (response rate {rr:.2f}, last active {days}d ago)")
    elif rr >= 0.45:
        parts.append(f"responsive to recruiters (rate {rr:.2f})")

    # ── GitHub signal ────────────────────────────────────────────────────────
    gh = feats["github_activity_score"]
    if gh >= 60:
        parts.append(f"strong GitHub activity ({gh:.0f}/100)")
    elif gh >= 30:
        parts.append(f"moderate GitHub activity ({gh:.0f}/100)")

    # ── Platform assessment scores ───────────────────────────────────────────
    assess = feats.get("assess_score", 0.35)
    if assess >= 0.70:
        parts.append(f"high Redrob skill assessment avg ({assess*100:.0f}/100)")

    # ── Location positive ────────────────────────────────────────────────────
    if feats["in_preferred_city"]:
        parts.append(f"based in {profile.get('location', '')}")

    # ─────────────────────────────────────────────────────────────────────────
    # Concerns
    # ─────────────────────────────────────────────────────────────────────────

    np_days = feats["notice_period_days"]
    if np_days > 90:
        concerns.append(f"long notice period ({np_days}d)")
    elif np_days > 60:
        concerns.append(f"notice period {np_days}d (buyout may be needed)")

    if not feats["in_india"] and not feats["willing_to_relocate"]:
        loc = profile.get("location", "unknown location")
        concerns.append(f"based outside India ({loc}), not willing to relocate")
    elif not feats["in_preferred_city"] and not feats["willing_to_relocate"]:
        loc = profile.get("location", "")
        concerns.append(f"not in Pune/Noida/Hyderabad (based in {loc})")

    if feats["consulting_fraction"] >= 0.80:
        concerns.append("career predominantly at services/consulting firms")

    if feats["ai_career_fraction"] < 0.15 and not concerns:
        concerns.append("limited direct AI/ML role history")

    if days > 120:
        concerns.append(f"last active {days}d ago — availability uncertain")

    if feats.get("honeypot_score", 0) >= 3:
        concerns.append("profile flagged for anomalous signals — verify carefully")

    # ── Assemble ──────────────────────────────────────────────────────────────
    main = "; ".join(parts)
    if concerns:
        return f"{main}. Concern(s): {'; '.join(concerns)}."
    return f"{main}."
